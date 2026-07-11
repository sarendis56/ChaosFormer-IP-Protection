"""Controlled retraining attacks for the preregistered layer-subset study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from tqdm import tqdm

from src.experiments.layer_subset_common import (
    LayerVariantBank,
    SubsetSpec,
    imagenet_files,
    layer_subset_descriptors,
    load_imagenet_split,
    load_vision_model,
    make_eval_loader,
    stratified_indices,
)
from src.utils.vision_backbone_utils import get_layer_weight_views


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_subset(args, num_layers: int) -> SubsetSpec:
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["subsets"][args.model]
    if args.subset_name.endswith("_top"):
        k = int(args.subset_name.split("_", 1)[0][1:])
        impacts_path = (
            args.screen_dir
            / args.model.replace("/", "__")
            / "single_layer_impacts.json"
        )
        impacts = json.loads(impacts_path.read_text())
        ranking = [
            row["layer"]
            for row in sorted(impacts, key=lambda row: (-row["accuracy_drop"], row["layer"]))
        ]
        return SubsetSpec(args.subset_name, "top", tuple(sorted(ranking[:k])))
    for row in rows:
        if row["name"] == args.subset_name:
            return SubsetSpec(
                row["name"],
                row["family"],
                tuple(row["layers"]),
                row.get("seed"),
            )
    raise KeyError(f"Unknown preregistered subset: {args.subset_name}")


def image_size(processor) -> tuple[int, int]:
    size = processor.size
    if "height" in size and "width" in size:
        return int(size["height"]), int(size["width"])
    edge = int(size.get("shortest_edge", 224))
    return edge, edge


def make_train_loader(args, processor, seed: int):
    dataset = load_imagenet_split(args.data_dir, "train")
    subset_size = int(round(args.train_fraction * len(dataset)))
    index_dir = args.data_dir / "subsets"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"train_{args.train_fraction:.4f}_seed{args.data_seed}.npy"
    if index_path.exists():
        indices = np.load(index_path).astype(np.int64).tolist()
    else:
        indices = stratified_indices(dataset["label"], subset_size, args.data_seed)
        np.save(index_path, np.asarray(indices, dtype=np.int32))
    if len(indices) != subset_size:
        raise RuntimeError(f"Expected {subset_size} train indices, found {len(indices)}")
    dataset = dataset.select(indices)

    height, width = image_size(processor)
    mean = processor.image_mean
    std = processor.image_std
    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (height, width),
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        images = torch.stack([augmentation(row["image"].convert("RGB")) for row in batch])
        labels = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long)
        return {"pixel_values": images, "labels": labels}

    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
        generator=generator,
        worker_init_fn=worker_init,
        drop_last=True,
    )
    return loader, indices, len(dataset)


def make_batch_mixer(num_classes: int):
    return transforms_v2.RandomChoice(
        [
            transforms_v2.MixUp(num_classes=num_classes, alpha=0.8),
            transforms_v2.CutMix(num_classes=num_classes, alpha=1.0),
        ]
    )


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if targets.ndim == 1:
        return F.cross_entropy(logits, targets, label_smoothing=smoothing)
    if smoothing > 0:
        targets = targets * (1.0 - smoothing) + smoothing / targets.shape[1]
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    count = 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(pixel_values=images).logits
        predictions = logits.topk(5, dim=1).indices
        correct = predictions.eq(labels[:, None])
        correct1 += int(correct[:, :1].sum().item())
        correct5 += int(correct.any(dim=1).sum().item())
        count += len(labels)
    return {
        "top1": correct1 / count,
        "top5": correct5 / count,
        "examples": count,
    }


def layer_gradient_norms(bank: LayerVariantBank) -> list[float]:
    values = []
    for layer in bank.layers:
        total = torch.zeros((), device=next(layer.parameters()).device)
        for parameter in layer.parameters():
            if parameter.grad is not None:
                total += parameter.grad.detach().float().square().sum()
        values.append(float(total.sqrt().item()))
    return values


@torch.no_grad()
def weight_displacements(bank: LayerVariantBank) -> list[float]:
    values = []
    for index, layer in enumerate(bank.layers):
        views = get_layer_weight_views(layer)
        difference = torch.zeros((), device=next(layer.parameters()).device)
        baseline = torch.zeros_like(difference)
        for name, weight in views.attention.items():
            difference += (weight.float() - bank.clean[index].attention[name].float()).square().sum()
            baseline += bank.clean[index].attention[name].float().square().sum()
        for name, weight in views.ffn.items():
            difference += (weight.float() - bank.clean[index].ffn[name].float()).square().sum()
            baseline += bank.clean[index].ffn[name].float().square().sum()
        values.append(float((difference.sqrt() / baseline.sqrt().clamp_min(1e-20)).item()))
    return values


def reinitialize_selected(model, bank: LayerVariantBank, selected: tuple[int, ...]) -> None:
    initializer = getattr(model, "_init_weights", None)
    if initializer is None:
        raise RuntimeError("Model does not expose the Hugging Face _init_weights initializer")
    for index in selected:
        bank.layers[index].apply(initializer)


def cosine_schedule(step: int, total_steps: int, warmup_steps: int, min_factor: float = 0.01) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / denominator)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, processor = load_vision_model(args.model, device, args.model_path)
    bank = LayerVariantBank(model, args.encryption_seed, device)
    subset = resolve_subset(args, len(bank.layers))
    bank.apply(subset.layers)
    if args.attacker == "oracle_reinit":
        reinitialize_selected(model, bank, subset.layers)
    # Keep all stochastic training choices identical across blind and oracle conditions.
    seed_everything(args.seed)

    train_loader, train_indices, train_examples = make_train_loader(args, processor, args.seed)
    validation_loader, validation_indices = make_eval_loader(
        args.data_dir,
        "validation",
        processor,
        50000,
        args.eval_batch_size,
        args.num_workers,
        args.data_seed,
    )

    run_name = f"{args.model.replace('/', '__')}__{subset.name}__{args.attacker}__seed{args.seed}"
    run_dir = args.output_dir / args.phase / run_name
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.force:
        print(f"Completed result exists: {result_path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    initial = evaluate(model, validation_loader, device)
    num_classes = int(model.config.num_labels)
    mixer = make_batch_mixer(num_classes)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-4 if "deit-small" in args.model else 5e-5
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total_steps, warmup_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        examples = 0
        gradient_samples: list[list[float]] = []
        progress = tqdm(train_loader, desc=f"{run_name} epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress):
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            images, mixed_labels = mixer(images, labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixel_values=images).logits
                loss = soft_cross_entropy(logits, mixed_labels, args.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if step < args.gradient_samples_per_epoch:
                gradient_samples.append(layer_gradient_norms(bank))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = len(labels)
            loss_total += float(loss.item()) * batch_size
            examples += batch_size
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        validation = evaluate(model, validation_loader, device)
        gradient_mean = (
            np.asarray(gradient_samples, dtype=np.float64).mean(axis=0).tolist()
            if gradient_samples
            else []
        )
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": loss_total / examples,
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation,
            "layer_gradient_norm_mean": gradient_mean,
        }
        history.append(epoch_row)
        partial = {
            "status": "running",
            "model": args.model,
            "subset": subset.to_dict(),
            "attacker": args.attacker,
            "seed": args.seed,
            "initial_validation": initial,
            "history": history,
        }
        atomic_json(run_dir / "progress.json", partial)
        if args.checkpoint:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                run_dir / "last.pt",
            )

    displacement = weight_displacements(bank)
    selected_set = set(subset.layers)
    selected_displacement = [value for i, value in enumerate(displacement) if i in selected_set]
    clean_displacement = [value for i, value in enumerate(displacement) if i not in selected_set]
    curve = [initial["top1"]] + [row["validation"]["top1"] for row in history]
    recovery_auc = float(np.trapezoid(curve, dx=1.0) / args.epochs)
    result = {
        "status": "complete",
        "phase": args.phase,
        "model": args.model,
        "model_path": str(args.model_path) if args.model_path else None,
        "subset": subset.to_dict(),
        "subset_descriptors": layer_subset_descriptors(subset.layers, len(bank.layers)),
        "attacker": args.attacker,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "encryption_seed": args.encryption_seed,
        "train_fraction": args.train_fraction,
        "train_examples": train_examples,
        "train_indices_sha256": __import__("hashlib").sha256(
            np.asarray(train_indices, dtype=np.int32).tobytes()
        ).hexdigest(),
        "validation_indices_sha256": __import__("hashlib").sha256(
            np.asarray(validation_indices, dtype=np.int32).tobytes()
        ).hexdigest(),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": 0.8,
            "cutmix_alpha": 1.0,
            "mixup_probability": 1.0,
            "grad_clip": args.grad_clip,
            "warmup_epochs": warmup_epochs,
            "scheduler": "per-step cosine",
            "precision": "fp16 autocast with fp32 parameters",
        },
        "initial_validation": initial,
        "history": history,
        "recovery_auc": recovery_auc,
        "layer_relative_weight_displacement": displacement,
        "selected_layer_displacement_mean": float(np.mean(selected_displacement)),
        "unencrypted_layer_displacement_mean": float(np.mean(clean_displacement)),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    atomic_json(result_path, result)
    progress_path = run_dir / "progress.json"
    if progress_path.exists():
        progress_path.unlink()
    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--attacker", choices=("blind", "oracle_reinit"), default="blind")
    parser.add_argument("--phase", choices=("short", "full", "oracle", "vit_confirm"), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/peichun/imagenet-1k"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/layer_subset_retraining"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/layer_subset_mechanism/preregistration.json"),
    )
    parser.add_argument(
        "--screen-dir",
        type=Path,
        default=Path("results/layer_subset_mechanism"),
    )
    parser.add_argument("--data-seed", type=int, default=20260711)
    parser.add_argument("--encryption-seed", type=int, default=20260711)
    parser.add_argument("--train-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--gradient-samples-per-epoch", type=int, default=10)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

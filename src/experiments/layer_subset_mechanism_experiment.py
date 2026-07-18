"""Measure why layer placement affects ChaosFormer retraining robustness.

The experiment controls the encryption key across strategies, records Random-K
subsets before observing outcomes, and measures both task damage and layerwise
representation disruption.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.experiments.layer_subset_common import (
    LayerVariantBank,
    SubsetSpec,
    layer_subset_descriptors,
    load_vision_model,
    make_eval_loader,
    write_preregistration,
)
from src.utils.vision_backbone_utils import get_transformer_layers


DEFAULT_MODELS = (
    "facebook/deit-small-patch16-224",
    "google/vit-base-patch16-224",
)


def cache_batches(loader) -> list[dict[str, torch.Tensor]]:
    batches = []
    for batch in tqdm(loader, desc="Preprocessing evaluation images"):
        batches.append(
            {
                "pixel_values": batch["pixel_values"].contiguous(),
                "labels": batch["labels"].contiguous(),
            }
        )
    return batches


@torch.inference_mode()
def collect_outputs(model, batches, device: torch.device, hidden: bool) -> dict:
    logits_parts = []
    label_parts = []
    hidden_parts: list[list[torch.Tensor]] | None = None
    for batch in batches:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        output = model(pixel_values=pixels, output_hidden_states=hidden)
        logits_parts.append(output.logits.detach().float().cpu())
        label_parts.append(batch["labels"])
        if hidden:
            states = output.hidden_states[1:]
            if hidden_parts is None:
                hidden_parts = [[] for _ in states]
            for index, state in enumerate(states):
                hidden_parts[index].append(state[:, 0].detach().float().cpu())
    result = {
        "logits": torch.cat(logits_parts),
        "labels": torch.cat(label_parts),
    }
    if hidden_parts is not None:
        result["hidden"] = [torch.cat(parts) for parts in hidden_parts]
    return result


def accuracy(outputs: dict) -> float:
    return float((outputs["logits"].argmax(dim=-1) == outputs["labels"]).float().mean().item())


def linear_cka(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    x_gpu = x.to(device)
    y_gpu = y.to(device)
    x_gpu = x_gpu - x_gpu.mean(dim=0, keepdim=True)
    y_gpu = y_gpu - y_gpu.mean(dim=0, keepdim=True)
    cross = x_gpu.T @ y_gpu
    xx = x_gpu.T @ x_gpu
    yy = y_gpu.T @ y_gpu
    numerator = torch.linalg.matrix_norm(cross).square()
    denominator = torch.linalg.matrix_norm(xx) * torch.linalg.matrix_norm(yy)
    return float((numerator / denominator.clamp_min(1e-20)).item())


def compare_outputs(clean: dict, encrypted: dict, device: torch.device, include_hidden: bool) -> dict:
    clean_logits = clean["logits"]
    encrypted_logits = encrypted["logits"]
    clean_probs = clean_logits.softmax(dim=-1)
    kl = F.kl_div(
        encrypted_logits.log_softmax(dim=-1),
        clean_probs,
        reduction="batchmean",
    )
    metrics = {
        "accuracy": accuracy(encrypted),
        "prediction_agreement": float(
            (clean_logits.argmax(dim=-1) == encrypted_logits.argmax(dim=-1)).float().mean().item()
        ),
        "clean_to_encrypted_kl": float(kl.item()),
    }
    if include_hidden:
        cosines = []
        ckas = []
        for clean_hidden, encrypted_hidden in zip(clean["hidden"], encrypted["hidden"]):
            cosines.append(
                float(F.cosine_similarity(clean_hidden, encrypted_hidden, dim=-1).mean().item())
            )
            ckas.append(linear_cka(clean_hidden, encrypted_hidden, device))
        metrics.update(
            {
                "layerwise_cls_cosine": cosines,
                "layerwise_linear_cka": ckas,
                "mean_cosine_damage": float(np.mean([1.0 - value for value in cosines])),
                "mean_cka_damage": float(np.mean([1.0 - value for value in ckas])),
            }
        )
    return metrics


def add_top_k_specs(
    bank: LayerVariantBank,
    model,
    batches,
    clean,
    device: torch.device,
    k_values: list[int],
) -> tuple[list[SubsetSpec], list[dict]]:
    impacts = []
    for layer_index in tqdm(range(len(bank.layers)), desc="Single-layer impact"):
        bank.apply([layer_index])
        outputs = collect_outputs(model, batches, device, hidden=False)
        impacts.append(
            {
                "layer": layer_index,
                "accuracy": accuracy(outputs),
                "accuracy_drop": accuracy(clean) - accuracy(outputs),
            }
        )
    ranking = [
        row["layer"]
        for row in sorted(impacts, key=lambda row: (-row["accuracy_drop"], row["layer"]))
    ]
    specs = [
        SubsetSpec(
            name=f"k{k}_top",
            family="top",
            layers=tuple(sorted(ranking[:k])),
        )
        for k in k_values
    ]
    return specs, impacts


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_screen(args) -> None:
    device = torch.device(args.device)
    model, processor = load_vision_model(args.model, device, args.model_path)
    num_layers = len(get_transformer_layers(model))
    if num_layers != 12:
        raise ValueError(f"Preregistration assumes 12 blocks, found {num_layers}")

    manifest = json.loads(args.manifest.read_text())
    registered = [
        SubsetSpec(
            name=row["name"],
            family=row["family"],
            layers=tuple(row["layers"]),
            seed=row.get("seed"),
        )
        for row in manifest["subsets"][args.model]
    ]
    loader, indices = make_eval_loader(
        args.data_dir,
        "validation",
        processor,
        args.num_examples,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    batches = cache_batches(loader)
    model.eval()
    clean = collect_outputs(model, batches, device, hidden=True)
    baseline = {
        "accuracy": accuracy(clean),
        "num_examples": len(clean["labels"]),
        "sample_indices": indices,
    }
    output_dir = args.output_dir / args.model.replace("/", "__")
    write_json(output_dir / "baseline.json", baseline)

    bank = LayerVariantBank(model, args.seed, device)
    top_specs, impacts = add_top_k_specs(
        bank, model, batches, clean, device, list(manifest["k_values"])
    )
    write_json(output_dir / "single_layer_impacts.json", impacts)

    all_specs = top_specs + registered
    summary = []
    for spec in tqdm(all_specs, desc="Layer subsets"):
        result_path = output_dir / "subsets" / f"{spec.name}.json"
        if result_path.exists() and not args.force:
            summary.append(json.loads(result_path.read_text()))
            continue
        bank.apply(spec.layers)
        started = time.perf_counter()
        encrypted = collect_outputs(model, batches, device, hidden=True)
        row = {
            **spec.to_dict(),
            "k": len(spec.layers),
            "model": args.model,
            "num_examples": args.num_examples,
            "elapsed_seconds": time.perf_counter() - started,
            **layer_subset_descriptors(spec.layers, num_layers),
            **compare_outputs(clean, encrypted, device, include_hidden=True),
        }
        write_json(result_path, row)
        summary.append(row)
    bank.apply([])
    write_json(output_dir / "screen_summary.json", summary)


def run_enumeration(args) -> None:
    device = torch.device(args.device)
    model, processor = load_vision_model(args.model, device, args.model_path)
    loader, indices = make_eval_loader(
        args.data_dir,
        "validation",
        processor,
        args.enumeration_examples,
        args.batch_size,
        args.num_workers,
        args.seed + 1,
    )
    batches = cache_batches(loader)
    clean = collect_outputs(model, batches, device, hidden=False)
    bank = LayerVariantBank(model, args.seed, device)
    output_dir = args.output_dir / args.model.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "subset_enumeration.jsonl"
    completed = set()
    if path.exists() and not args.force:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["k"], tuple(row["layers"])))
    mode = "w" if args.force else "a"
    with path.open(mode) as handle:
        for k in args.k_values:
            subsets = combinations(range(len(bank.layers)), k)
            total = math.comb(len(bank.layers), k)
            for layers in tqdm(subsets, total=total, desc=f"Enumerating K={k}"):
                if (k, layers) in completed:
                    continue
                bank.apply(layers)
                outputs = collect_outputs(model, batches, device, hidden=False)
                row = {
                    "model": args.model,
                    "k": k,
                    "layers": list(layers),
                    "num_examples": args.enumeration_examples,
                    **layer_subset_descriptors(layers, len(bank.layers)),
                    **compare_outputs(clean, outputs, device, include_hidden=False),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
    write_json(
        output_dir / "enumeration_metadata.json",
        {
            "model": args.model,
            "num_examples": args.enumeration_examples,
            "sample_indices": indices,
            "k_values": args.k_values,
            "baseline_accuracy": accuracy(clean),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("manifest", "screen", "enumerate"), required=True)
    parser.add_argument("--model", choices=DEFAULT_MODELS, default=DEFAULT_MODELS[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/imagenet-1k"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/layer_subset_mechanism"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/layer_subset_mechanism/preregistration.json"),
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--num-examples", type=int, default=5000)
    parser.add_argument("--enumeration-examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "manifest":
        write_preregistration(
            args.manifest,
            models=DEFAULT_MODELS,
            k_values=args.k_values,
            random_count=args.random_count,
            seed=args.seed,
        )
    elif args.action == "screen":
        run_screen(args)
    else:
        run_enumeration(args)


if __name__ == "__main__":
    main()

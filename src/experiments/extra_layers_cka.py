"""Compute hidden-representation (linear CKA) damage for increasing numbers of encrypted ViT-B layers (2 base layers + 0..4 extra security layers), to be paired with the retraining attack accuracies in manuscript Table 16.

Usage (from repo root):
    python src/experiments/extra_layers_cka.py --device cuda:0
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.experiments.layer_subset_common import (
    LayerVariantBank,
    load_vision_model,
    make_eval_loader,
)
from src.experiments.layer_subset_mechanism_experiment import (
    accuracy,
    cache_batches,
    collect_outputs,
    compare_outputs,
)


def rank_by_accuracy_drop(impacts: list[dict]) -> list[int]:
    return [row["layer"] for row in sorted(impacts, key=lambda row: (-row["accuracy_drop"], row["layer"]))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/vit-base-patch16-224")
    parser.add_argument(
        "--model-path",
        type=Path,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/imagenet-1k"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/layer_subset_mechanism/extra_layers_cka.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-examples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, processor = load_vision_model(args.model, device, args.model_path)
    loader, _ = make_eval_loader(
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
    clean_acc = accuracy(clean)
    print(f"Clean accuracy: {clean_acc:.4f}")

    bank = LayerVariantBank(model, args.seed, device)

    # Top-k ranking from single-layer impact (same procedure as the mechanism study).
    impacts = []
    for layer_index in tqdm(range(len(bank.layers)), desc="Single-layer impact"):
        bank.apply([layer_index])
        outputs = collect_outputs(model, batches, device, hidden=False)
        encrypted_accuracy = accuracy(outputs)
        impacts.append(
            {
                "layer": layer_index,
                "accuracy": encrypted_accuracy,
                "accuracy_drop": clean_acc - encrypted_accuracy,
            }
        )
    ranking = rank_by_accuracy_drop(impacts)
    print(f"Top-k ranking: {ranking}")

    # Encrypt top-2 .. top-6 (2 base layers + 0..4 extra), measure representation damage.
    rows = []
    for k in [2, 3, 4, 5, 6]:
        bank.apply(ranking[:k])
        encrypted = collect_outputs(model, batches, device, hidden=True)
        m = compare_outputs(clean, encrypted, device, include_hidden=True)
        rows.append(
            {
                "encrypted_layers": k,
                "extra_layers": k - 2,
                "layers": ranking[:k],
                "mean_cka_damage": m["mean_cka_damage"],
                "mean_cosine_damage": m["mean_cosine_damage"],
                "initial_accuracy": m["accuracy"],
            }
        )
        print(
            f"K={k} (extra={k - 2}): CKA damage={m['mean_cka_damage']:.4f}, "
            f"cosine damage={m['mean_cosine_damage']:.4f}, init acc={m['accuracy']:.4f}"
        )

    out = {
        "model": args.model,
        "seed": args.seed,
        "clean_accuracy": clean_acc,
        "ranking": ranking,
        "impacts": impacts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

"""ImageNet clean accuracy and partial-key recovery for vision transformers.

On all 50,000 validation images (HF AutoImageProcessor):
  * clean top-1 accuracy of each model;
  * K=6 encrypted accuracy when every ACM (attention) key is recovered but the
    FFN permutations stay encrypted, and vice versa.

  uv run python -m src.experiments.vision_partial_key_eval --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.experiments.layer_subset_common import (
    LayerVariantBank,
    load_vision_model,
    make_eval_loader,
)
from src.utils.vision_backbone_utils import get_layer_weight_views

SELECTIONS = {
    "last_k": (6, 7, 8, 9, 10, 11),
    "random_01": (0, 2, 5, 6, 7, 8),
}


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    correct = total = 0
    for batch in loader:
        logits = model(pixel_values=batch["pixel_values"].to(device, non_blocking=True)).logits
        correct += (logits.argmax(-1).cpu() == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return correct / total


@torch.no_grad()
def apply_mixed(bank: LayerVariantBank, selected, attention_encrypted: bool, ffn_encrypted: bool) -> None:
    for index, layer in enumerate(bank.layers):
        views = get_layer_weight_views(layer)
        hit = index in selected
        attn_src = bank.encrypted[index] if hit and attention_encrypted else bank.clean[index]
        ffn_src = bank.encrypted[index] if hit and ffn_encrypted else bank.clean[index]
        for name, value in views.attention.items():
            value.copy_(attn_src.attention[name])
        for name, value in views.ffn.items():
            value.copy_(ffn_src.ffn[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["google/vit-base-patch16-224", "facebook/deit-small-patch16-224"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/imagenet-1k"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", type=Path, default=Path("results/vision_partial_key.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    out = {"pipeline": "HF AutoImageProcessor(use_fast=True), ImageNet-1k validation, all 50,000 images, FP32",
           "models": {}}
    for name in args.models:
        model, processor = load_vision_model(name, device)
        loader, _ = make_eval_loader(args.data_dir, "validation", processor, 50_000, 256, 16, args.seed)
        entry = {"clean_top1": accuracy(model, loader, device), "partial_key": {}}
        print(name, "clean", entry["clean_top1"], flush=True)
        bank = LayerVariantBank(model, args.seed, device)
        for label, layers in SELECTIONS.items():
            row = {}
            for case, (attn, ffn) in {
                "all_encrypted": (True, True),
                "attention_recovered_ffn_encrypted": (False, True),
                "ffn_recovered_attention_encrypted": (True, False),
            }.items():
                apply_mixed(bank, set(layers), attn, ffn)
                row[case] = accuracy(model, loader, device)
                print(name, label, case, row[case], flush=True)
            entry["partial_key"][label] = {"layers": list(layers), **row}
        bank.apply(())  # restore clean weights
        out["models"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

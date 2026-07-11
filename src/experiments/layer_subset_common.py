"""Shared utilities for controlled layer-subset mechanism experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import hashlib
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.encryption import DualEncryption
from src.utils.vision_backbone_utils import get_layer_weight_views, get_transformer_layers


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    family: str
    layers: tuple[int, ...]
    seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def stable_int(text: str, modulo: int = 2**31) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % modulo


def layer_subset_descriptors(layers: Sequence[int], num_layers: int) -> dict[str, float | int]:
    selected = np.asarray(sorted(layers), dtype=np.float64)
    pairwise = [
        abs(float(selected[i] - selected[j]))
        for i in range(len(selected))
        for j in range(i + 1, len(selected))
    ]
    adjacent = sum(int(b - a == 1) for a, b in zip(selected[:-1], selected[1:]))
    thirds = {min(2, int(3 * layer / num_layers)) for layer in selected}
    return {
        "depth_span": int(selected[-1] - selected[0]) if len(selected) > 1 else 0,
        "mean_pairwise_distance": float(np.mean(pairwise)) if pairwise else 0.0,
        "adjacent_pairs": adjacent,
        "depth_thirds_covered": len(thirds),
    }


def _spread_layers(num_layers: int, k: int, fraction: float) -> tuple[int, ...]:
    edges = np.linspace(0, num_layers, k + 1, dtype=int)
    chosen: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        width = max(1, hi - lo)
        pos = lo + min(width - 1, int(fraction * width))
        chosen.append(min(num_layers - 1, pos))
    if len(set(chosen)) != k:
        raise RuntimeError(f"Could not construct {k} spread layers from {num_layers} blocks")
    return tuple(chosen)


def preregistered_subsets(
    num_layers: int,
    k_values: Iterable[int],
    random_count: int,
    seed: int,
) -> list[SubsetSpec]:
    """Generate structural controls and fixed random subsets without using outcomes."""
    specs: list[SubsetSpec] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()

    def add(k: int, name: str, family: str, layers: Sequence[int], item_seed: int | None = None) -> None:
        normalized = tuple(sorted(int(x) for x in layers))
        if len(normalized) != k or len(set(normalized)) != k:
            raise ValueError(f"Invalid subset {name}: {normalized}")
        key = (k, normalized)
        if key in seen:
            return
        seen.add(key)
        specs.append(SubsetSpec(name=name, family=family, layers=normalized, seed=item_seed))

    for k in k_values:
        add(k, f"k{k}_last", "last", range(num_layers - k, num_layers))
        starts = sorted({0, (num_layers - k) // 2, num_layers - k})
        for index, start in enumerate(starts):
            add(k, f"k{k}_cluster_{index}", "clustered", range(start, start + k))
        spread_candidates = [
            _spread_layers(num_layers, k, 0.1),
            _spread_layers(num_layers, k, 0.5),
        ]
        edges = np.linspace(0, num_layers, k + 1, dtype=int)
        alternating = tuple(
            int(lo if index % 2 == 0 else hi - 1)
            for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
        )
        spread_candidates.append(alternating)
        for index, layers in enumerate(spread_candidates):
            add(k, f"k{k}_spread_{index}", "spread", layers)

        rng = random.Random(stable_int(f"{seed}:k={k}"))
        generated = 0
        draw = 0
        while generated < random_count:
            draw_seed = rng.randrange(2**31)
            layers = tuple(sorted(random.Random(draw_seed).sample(range(num_layers), k)))
            before = len(specs)
            add(k, f"k{k}_random_{generated:02d}", "random", layers, draw_seed)
            if len(specs) > before:
                generated += 1
            draw += 1
            if draw > 10000:
                raise RuntimeError("Unable to generate unique random subsets")
    return specs


def write_preregistration(
    path: Path,
    *,
    models: Sequence[str],
    k_values: Sequence[int],
    random_count: int,
    seed: int,
    full_random_indices: Sequence[int] = (0, 1, 2),
) -> dict:
    """Write an outcome-independent manifest used by screening and retraining."""
    model_specs = {}
    for model in models:
        specs = preregistered_subsets(12, k_values, random_count, seed)
        model_specs[model] = [spec.to_dict() for spec in specs]
    manifest = {
        "seed": seed,
        "models": list(models),
        "num_layers_expected": 12,
        "k_values": list(k_values),
        "random_subset_count": random_count,
        "full_retraining_selection_rule": {
            "top_k": True,
            "clustered": "middle structural control",
            "spread": "middle structural control",
            "random_indices": list(full_random_indices),
            "selection_uses_screening_outcomes": False,
        },
        "training_protocol": {
            "short_deit": {"epochs": 5, "seeds": [3101], "k": 6},
            "full_deit": {"epochs": 20, "seeds": [4101, 4102, 4103], "k": 6},
            "oracle_deit": {"epochs": 5, "seeds": [3101], "k": 6},
            "vit_confirmation": {"epochs": 20, "seeds": [6101], "k": 6},
            "selection_uses_training_outcomes": False,
        },
        "subsets": model_specs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def imagenet_files(data_dir: Path, split: str) -> list[str]:
    files = sorted(str(p) for p in (data_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet shards found under {data_dir / 'data'}")
    return files


def load_imagenet_split(data_dir: Path, split: str):
    return load_dataset("parquet", data_files=imagenet_files(data_dir, split), split="train")


def stratified_indices(labels: Sequence[int], num_examples: int, seed: int) -> list[int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels_array)
    if num_examples >= len(labels_array):
        return list(range(len(labels_array)))
    rng = np.random.default_rng(seed)
    by_class = {int(c): np.flatnonzero(labels_array == c) for c in classes}
    quota, remainder = divmod(num_examples, len(classes))
    selected: list[int] = []
    for rank, cls in enumerate(classes):
        candidates = by_class[int(cls)].copy()
        rng.shuffle(candidates)
        take = min(len(candidates), quota + int(rank < remainder))
        selected.extend(int(x) for x in candidates[:take])
    if len(selected) < num_examples:
        used = set(selected)
        remaining = np.asarray([i for i in range(len(labels_array)) if i not in used])
        rng.shuffle(remaining)
        selected.extend(int(x) for x in remaining[: num_examples - len(selected)])
    rng.shuffle(selected)
    return selected[:num_examples]


def make_eval_loader(
    data_dir: Path,
    split: str,
    processor,
    num_examples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, list[int]]:
    dataset = load_imagenet_split(data_dir, split)
    indices = stratified_indices(dataset["label"], num_examples, seed)
    dataset = dataset.select(indices)

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        images = [row["image"].convert("RGB") for row in batch]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
        labels = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long)
        return {"pixel_values": pixels, "labels": labels}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    return loader, indices


def load_vision_model(model_name: str, device: torch.device, model_path: Path | None = None):
    source = str(model_path) if model_path is not None else model_name
    processor = AutoImageProcessor.from_pretrained(source, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(source)
    model.to(device).eval()
    return model, processor


@dataclass
class WeightVariant:
    attention: dict[str, torch.Tensor]
    ffn: dict[str, torch.Tensor]


class LayerVariantBank:
    """Keep clean and deterministically encrypted layer weights for fair subset comparisons."""

    def __init__(self, model, seed: int, device: torch.device):
        self.layers = list(get_transformer_layers(model))
        self.clean: list[WeightVariant] = []
        self.encrypted: list[WeightVariant] = []
        encryptor = DualEncryption.from_model(
            model,
            master_secret=f"layer-subset-mechanism:{seed}",
            device=str(device),
            dtype=next(model.parameters()).dtype,
            mode="basic",
        )
        for layer_index, layer in enumerate(self.layers):
            views = get_layer_weight_views(layer)
            clean_attention = {name: value.detach().clone() for name, value in views.attention.items()}
            clean_ffn = {name: value.detach().clone() for name, value in views.ffn.items()}
            self.clean.append(WeightVariant(clean_attention, clean_ffn))
            permutation_index = stable_int(f"{seed}:permutation:{layer_index}", len(encryptor.permutation_matrices))
            result = encryptor.encrypt_layer_weights(
                {name: value.clone() for name, value in clean_attention.items()},
                {name: value.clone() for name, value in clean_ffn.items()},
                permutation_matrix_idx=permutation_index,
                layer_idx=layer_index,
            )
            self.encrypted.append(
                WeightVariant(
                    {name: value.detach().clone() for name, value in result.encrypted_attention.items()},
                    {name: value.detach().clone() for name, value in result.encrypted_ffn.items()},
                )
            )

    @torch.no_grad()
    def apply(self, selected_layers: Sequence[int]) -> None:
        selected = set(selected_layers)
        for index, layer in enumerate(self.layers):
            source = self.encrypted[index] if index in selected else self.clean[index]
            views = get_layer_weight_views(layer)
            for name, value in views.attention.items():
                value.copy_(source.attention[name])
            for name, value in views.ffn.items():
                value.copy_(source.ffn[name])

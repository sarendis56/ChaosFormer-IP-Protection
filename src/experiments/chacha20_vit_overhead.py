"""Paired ViT-B inference overhead for legacy and ChaCha20 diffusion."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import ViTForImageClassification

from src.encryption.arnold_transform import arnold_numba, iarnold_numba, arnold_triton, iarnold_triton
from src.encryption.chacha20 import chacha20_xor_, derive_chacha20_material
from src.encryption.xor_encryption import get_stable_seed
from src.experiments.chacha20_diffusion_benchmark import legacy_splitmix_xor_


def legacy_splitmix_xor_cpu_(tensor: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """CPU equivalent of the legacy SplitMix diffusion kernel."""
    if tensor.element_size() == 4:
        view = tensor.detach().numpy().reshape(-1).view(np.int32)
    elif tensor.element_size() == 2:
        view = tensor.detach().numpy().reshape(-1).view(np.int16)
    else:
        view = tensor.detach().numpy().reshape(-1).view(np.int8)
    offsets = np.arange(view.size, dtype=np.uint64)
    value = (np.uint64(seed) + offsets)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    stream = value ^ (value >> np.uint64(31))
    view ^= stream.astype(view.dtype)
    return tensor


def _weights(model, selected_layers: list[int]):
    for layer_idx in selected_layers:
        layer = model.vit.encoder.layer[layer_idx]
        yield layer_idx, "query", "square", layer.attention.attention.query.weight
        yield layer_idx, "key", "square", layer.attention.attention.key.weight
        yield layer_idx, "value", "square", layer.attention.attention.value.weight
        yield layer_idx, "attention_output", "square", layer.attention.output.dense.weight
        yield layer_idx, "intermediate", "ffn_columns", layer.intermediate.dense.weight
        yield layer_idx, "output", "ffn_rows", layer.output.dense.weight


class ProtectedViT:
    def __init__(self, model, selected_layers: list[int], mode: str, secret: bytes, device: torch.device):
        self.model = model
        self.mode = mode
        self.device = device
        self.arnold_key = [5, 1, 1, 1, 2]
        self.arnold = arnold_triton if device.type == "cuda" else arnold_numba
        self.iarnold = iarnold_triton if device.type == "cuda" else iarnold_numba
        self.legacy_diffuse = legacy_splitmix_xor_ if device.type == "cuda" else legacy_splitmix_xor_cpu_
        self.specs = []
        for layer_idx, name, operation, parameter in _weights(model, selected_layers):
            permutation = inverse = None
            if operation != "square":
                dimension = parameter.shape[1] if operation == "ffn_columns" else parameter.shape[0]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(get_stable_seed(layer_idx, name, secret))
                permutation = torch.randperm(dimension, generator=generator).to(parameter.device)
                inverse = torch.argsort(permutation)
            key, nonce = derive_chacha20_material(layer_idx, name, secret)
            legacy_seed = get_stable_seed(layer_idx, name, secret)
            self.specs.append((parameter, operation, permutation, inverse, key, nonce, legacy_seed))

    def _diffuse(self, spec) -> None:
        parameter, _, _, _, key, nonce, legacy_seed = spec
        if self.mode == "legacy":
            self.legacy_diffuse(parameter.data, legacy_seed)
        else:
            chacha20_xor_(parameter.data, key, nonce)

    @torch.no_grad()
    def encrypt(self) -> None:
        for spec in self.specs:
            parameter, operation, permutation, _, _, _, _ = spec
            if operation == "square":
                parameter.copy_(self.arnold(parameter, self.arnold_key))
            elif operation == "ffn_columns":
                parameter.copy_(parameter[:, permutation].contiguous())
            else:
                parameter.copy_(parameter[permutation].contiguous())
            self._diffuse(spec)

    @torch.no_grad()
    def decrypt(self) -> None:
        for spec in self.specs:
            parameter, operation, _, inverse, _, _, _ = spec
            self._diffuse(spec)
            if operation == "square":
                parameter.copy_(self.iarnold(parameter, self.arnold_key))
            elif operation == "ffn_columns":
                parameter.copy_(parameter[:, inverse].contiguous())
            else:
                parameter.copy_(parameter[inverse].contiguous())

    @torch.inference_mode()
    def authorized_inference(self, inputs: torch.Tensor) -> None:
        self.decrypt()
        self.model(pixel_values=inputs)
        self.encrypt()


def _time(function, repetitions: int) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repetitions


def benchmark(args) -> dict[str, object]:
    device = torch.device(args.device)
    model = ViTForImageClassification.from_pretrained(
        args.model, local_files_only=args.local_files_only
    ).eval().to(device)
    inputs = torch.randn(
        args.batch_size, 3, 224, 224, device=device, dtype=torch.float32
    )
    clean_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    for _ in range(5):
        model(pixel_values=inputs)
    baseline_samples = [
        _time(lambda: model(pixel_values=inputs), args.repetitions)
        for _ in range(args.trials)
    ]

    aggregate = statistics.mean if args.aggregate == "mean" else statistics.median
    result: dict[str, object] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
        "model": args.model,
        "batch_size": args.batch_size,
        "repetitions_per_trial": args.repetitions,
        "trial_count": args.trials,
        "baseline_ms": aggregate(baseline_samples),
        "baseline_trials_ms": baseline_samples,
        "measurements": [],
    }
    secret = b"ChaosFormer paired benchmark model key"
    for layer_count in args.layers:
        entry: dict[str, object] = {"protected_layers": layer_count}
        for mode in ("legacy", "chacha20"):
            model.load_state_dict(clean_state)
            protected = ProtectedViT(model, list(range(layer_count)), mode, secret, device)
            protected.encrypt()
            for _ in range(3):
                protected.authorized_inference(inputs)
            samples = [
                _time(lambda: protected.authorized_inference(inputs), args.repetitions)
                for _ in range(args.trials)
            ]
            entry[f"{mode}_ms"] = aggregate(samples)
            entry[f"{mode}_trials_ms"] = samples
        baseline = float(result["baseline_ms"])
        entry["legacy_overhead"] = float(entry["legacy_ms"]) / baseline
        entry["chacha20_overhead"] = float(entry["chacha20_ms"]) / baseline
        entry["chacha20_vs_legacy"] = float(entry["chacha20_ms"]) / float(entry["legacy_ms"])
        result["measurements"].append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="google/vit-base-patch16-224",
    )
    parser.add_argument("--device", default="cuda", help="torch device, e.g. cuda or cpu")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="resolve --model only from a local path or the Hugging Face cache",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--aggregate",
        choices=("median", "mean"),
        default="median",
        help="aggregate per-trial timings with the median (default) or arithmetic mean",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/chacha20_vit_overhead.json")
    )
    args = parser.parse_args()
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

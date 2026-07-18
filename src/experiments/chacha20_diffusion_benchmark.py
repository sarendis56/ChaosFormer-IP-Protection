"""Paired GPU benchmark for the ChaCha20 diffusion revision.

The benchmark compares the new RFC 8439 implementation with the former
SplitMix-style diffusion kernel and measures the cost added to an ACM
encrypt/decrypt pair.  All reported values are ratios on the same GPU/run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

from src.encryption.arnold_transform import arnold_triton, iarnold_triton
from src.encryption.chacha20 import chacha20_xor_, derive_chacha20_material


@triton.jit
def _splitmix32(seed, index):
    value = (seed + index).to(tl.uint64)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB
    return (value ^ (value >> 31)).to(tl.int32)


@triton.jit
def _legacy_splitmix_kernel(data_ptr, n_elements, seed, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(data_ptr + offsets, mask=mask, other=0)
    stream = _splitmix32(seed, offsets).to(value.dtype)
    tl.store(data_ptr + offsets, value ^ stream, mask=mask)


def legacy_splitmix_xor_(tensor: torch.Tensor, seed: int = 42) -> torch.Tensor:
    if tensor.element_size() == 4:
        view = tensor.view(-1).view(torch.int32)
    elif tensor.element_size() == 2:
        view = tensor.view(-1).view(torch.int16)
    else:
        view = tensor.view(-1).view(torch.int8)
    n_elements = view.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _legacy_splitmix_kernel[grid](view, n_elements, seed, BLOCK_SIZE=1024)
    return tensor


def _elapsed_ms(function, repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repetitions


def benchmark_once(size: int, repetitions: int) -> dict[str, float | int]:
    tensor = torch.randn(size, size, device="cuda", dtype=torch.float16)
    key, nonce = derive_chacha20_material(0, "attention.query", 20260712)
    arnold_key = [5, 1, 1, 1, 2]

    for _ in range(10):
        legacy_splitmix_xor_(tensor)
        chacha20_xor_(tensor, key, nonce)
        encrypted = arnold_triton(tensor, arnold_key)
        iarnold_triton(encrypted, arnold_key)
        encrypted = arnold_triton(tensor, arnold_key, xor_seed=20260712)
        iarnold_triton(encrypted, arnold_key, xor_seed=20260712)

    splitmix_ms = _elapsed_ms(lambda: legacy_splitmix_xor_(tensor), repetitions)
    chacha_ms = _elapsed_ms(lambda: chacha20_xor_(tensor, key, nonce), repetitions)

    def base_pair() -> None:
        encrypted = arnold_triton(tensor, arnold_key)
        iarnold_triton(encrypted, arnold_key)

    def secure_pair() -> None:
        encrypted = arnold_triton(tensor, arnold_key, xor_seed=20260712)
        iarnold_triton(encrypted, arnold_key, xor_seed=20260712)

    base_pair_ms = _elapsed_ms(base_pair, repetitions)
    secure_pair_ms = _elapsed_ms(secure_pair, repetitions)
    return {
        "matrix_size": size,
        "weights": size * size,
        "repetitions": repetitions,
        "legacy_splitmix_ms": splitmix_ms,
        "chacha20_ms": chacha_ms,
        "chacha20_vs_splitmix": chacha_ms / splitmix_ms,
        "acm_pair_ms": base_pair_ms,
        "acm_chacha20_pair_ms": secure_pair_ms,
        "secure_vs_base_pair": secure_pair_ms / base_pair_ms,
    }


def benchmark(size: int, repetitions: int, trials: int) -> dict[str, object]:
    samples = [benchmark_once(size, repetitions) for _ in range(trials)]
    timing_fields = (
        "legacy_splitmix_ms",
        "chacha20_ms",
        "chacha20_vs_splitmix",
        "acm_pair_ms",
        "acm_chacha20_pair_ms",
        "secure_vs_base_pair",
    )
    medians = {
        field: statistics.median(float(sample[field]) for sample in samples)
        for field in timing_fields
    }
    return {
        "matrix_size": size,
        "weights": size * size,
        "repetitions_per_trial": repetitions,
        "trial_count": trials,
        "median": medians,
        "trials": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[768, 1024, 1280])
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("results/chacha20_diffusion.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "measurements": [
            benchmark(size, args.repetitions, args.trials) for size in args.sizes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

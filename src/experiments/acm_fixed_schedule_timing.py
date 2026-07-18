"""Paired timing study for variable- and fixed-schedule ACM exponentiation.

The benchmark separates the small host-side 2x2 matrix exponentiation from a
complete GPU ACM encrypt/decrypt round trip. Reported comparisons are ratios so
the results remain meaningful when run on hardware different from the paper's
submitted configuration.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import triton

from src.encryption.arnold_transform import (
    _matrix_power_mod,
    _matrix_power_mod_fixed_schedule,
    arnold_triton,
    iarnold_triton,
)


BASE_MATRIX = np.array([[1, 1], [1, 2]], dtype=np.int64)


def _summary(samples_ns: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ns, dtype=np.float64)
    return {
        "median_ns": float(np.median(values)),
        "p05_ns": float(np.percentile(values, 5)),
        "p95_ns": float(np.percentile(values, 95)),
    }


def _ratio_summary(
    samples: dict[int, dict[str, list[float]]], powers: list[int]
) -> dict[str, object]:
    per_power = {
        str(power): float(
            np.median(samples[power]["fixed"])
            / np.median(samples[power]["original"])
        )
        for power in powers
    }
    ratios = np.asarray(list(per_power.values()), dtype=np.float64)
    return {
        "fixed_over_original_by_power": per_power,
        "median_fixed_over_original": float(np.median(ratios)),
        "geomean_fixed_over_original": float(np.exp(np.mean(np.log(ratios)))),
        "min_fixed_over_original": float(np.min(ratios)),
        "max_fixed_over_original": float(np.max(ratios)),
    }


def _leakage_summary(
    samples: dict[int, dict[str, list[float]]], powers: list[int]
) -> dict[str, dict[str, float]]:
    products = np.asarray(
        [power.bit_length() + power.bit_count() for power in powers],
        dtype=np.float64,
    )
    result: dict[str, dict[str, float]] = {}
    for mode in ("original", "fixed"):
        medians = np.asarray(
            [np.median(samples[power][mode]) for power in powers],
            dtype=np.float64,
        )
        correlation = float(np.corrcoef(products, medians)[0, 1])
        result[mode] = {
            "max_over_min_median": float(np.max(medians) / np.min(medians)),
            "correlation_with_original_product_count": correlation,
        }
    return result


def benchmark_host(
    powers: list[int], blocks: int, inner_loops: int, seed: int
) -> dict[int, dict[str, list[float]]]:
    functions: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
        "original": _matrix_power_mod,
        "fixed": _matrix_power_mod_fixed_schedule,
    }
    samples = {
        power: {"original": [], "fixed": []} for power in powers
    }
    rng = random.Random(seed)

    for power in powers:
        for function in functions.values():
            for _ in range(20):
                function(BASE_MATRIX, power, 768)

    jobs = [(power, mode) for power in powers for mode in functions]
    for _ in range(blocks):
        rng.shuffle(jobs)
        for power, mode in jobs:
            function = functions[mode]
            start_ns = time.perf_counter_ns()
            for _ in range(inner_loops):
                function(BASE_MATRIX, power, 768)
            elapsed_ns = time.perf_counter_ns() - start_ns
            samples[power][mode].append(elapsed_ns / inner_loops)

    return samples


def benchmark_gpu(
    powers: list[int],
    matrix_size: int,
    warmup: int,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[int, dict[str, list[float]]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    matrix = torch.randn(
        matrix_size,
        matrix_size,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    samples = {
        power: {"original": [], "fixed": []} for power in powers
    }

    for power in powers:
        key = [power, 1, 1, 1, 2]
        original = arnold_triton(matrix, key, fixed_schedule=False)
        fixed = arnold_triton(matrix, key, fixed_schedule=True)
        restored = iarnold_triton(fixed, key, fixed_schedule=True)
        torch.cuda.synchronize(device)
        if not torch.equal(original, fixed):
            raise RuntimeError(f"fixed ACM differs from original for power={power}")
        if not torch.equal(restored, matrix):
            raise RuntimeError(f"fixed ACM round trip failed for power={power}")

    for mode in (False, True):
        for power in (powers[0], powers[-1]):
            key = [power, 1, 1, 1, 2]
            for _ in range(warmup):
                encrypted = arnold_triton(matrix, key, fixed_schedule=mode)
                iarnold_triton(encrypted, key, fixed_schedule=mode)
    torch.cuda.synchronize(device)

    rng = random.Random(seed)
    jobs = [
        (power, mode)
        for power in powers
        for mode in ("original", "fixed")
    ]
    for _ in range(repeats):
        rng.shuffle(jobs)
        for power, mode in jobs:
            key = [power, 1, 1, 1, 2]
            fixed_schedule = mode == "fixed"
            torch.cuda.synchronize(device)
            start_ns = time.perf_counter_ns()
            encrypted = arnold_triton(
                matrix, key, fixed_schedule=fixed_schedule
            )
            iarnold_triton(
                encrypted, key, fixed_schedule=fixed_schedule
            )
            torch.cuda.synchronize(device)
            samples[power][mode].append(time.perf_counter_ns() - start_ns)

    return samples


def _serialize_samples(
    samples: dict[int, dict[str, list[float]]], powers: list[int]
) -> dict[str, object]:
    return {
        "per_power": {
            str(power): {
                mode: {
                    **_summary(samples[power][mode]),
                    "samples_ns": samples[power][mode],
                }
                for mode in ("original", "fixed")
            }
            for power in powers
        },
        "ratios": _ratio_summary(samples, powers),
        "timing_dependence": _leakage_summary(samples, powers),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--matrix-size", type=int, default=768)
    parser.add_argument("--host-blocks", type=int, default=21)
    parser.add_argument("--host-inner-loops", type=int, default=500)
    parser.add_argument("--gpu-warmup", type=int, default=10)
    parser.add_argument("--gpu-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/acm_fixed_schedule/timing_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU round-trip benchmark")
    if args.matrix_size < 1 or args.host_blocks < 1 or args.host_inner_loops < 1:
        raise ValueError("matrix size and host timing counts must be positive")
    if args.gpu_warmup < 0 or args.gpu_repeats < 1:
        raise ValueError("GPU warmup must be nonnegative and repeats must be positive")

    powers = list(range(3, 35))
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    host_samples = benchmark_host(
        powers, args.host_blocks, args.host_inner_loops, args.seed
    )
    gpu_samples = benchmark_gpu(
        powers,
        args.matrix_size,
        args.gpu_warmup,
        args.gpu_repeats,
        args.seed,
        device,
    )

    result = {
        "experiment": "acm_fixed_schedule_timing",
        "supported_iteration_range": [powers[0], powers[-1]],
        "fixed_schedule": {
            "rounds": 6,
            "matrix_products_per_exponentiation": 12,
            "claim_scope": "fixed algorithmic schedule, not machine-level constant time",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
            "matrix_size": args.matrix_size,
            "dtype": "float16",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        "configuration": {
            "host_blocks": args.host_blocks,
            "host_inner_loops": args.host_inner_loops,
            "gpu_warmup": args.gpu_warmup,
            "gpu_repeats": args.gpu_repeats,
            "seed": args.seed,
        },
        "host_exponentiation": _serialize_samples(host_samples, powers),
        "gpu_encrypt_decrypt_round_trip": _serialize_samples(gpu_samples, powers),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    host_ratio = result["host_exponentiation"]["ratios"]
    gpu_ratio = result["gpu_encrypt_decrypt_round_trip"]["ratios"]
    host_dependence = result["host_exponentiation"]["timing_dependence"]
    gpu_dependence = result["gpu_encrypt_decrypt_round_trip"]["timing_dependence"]
    print(json.dumps({
        "output": str(args.output),
        "host_fixed_over_original": host_ratio,
        "gpu_round_trip_fixed_over_original": gpu_ratio,
        "host_timing_dependence": host_dependence,
        "gpu_round_trip_timing_dependence": gpu_dependence,
    }, indent=2))


if __name__ == "__main__":
    main()

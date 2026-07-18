"""Aggregate independent ACM fixed-schedule timing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [json.loads(path.read_text()) for path in args.inputs]
    if any(run.get("experiment") != "acm_fixed_schedule_timing" for run in runs):
        raise ValueError("all inputs must be ACM fixed-schedule timing runs")

    gpu_run_ratios = [
        run["gpu_encrypt_decrypt_round_trip"]["ratios"]
        ["median_fixed_over_original"]
        for run in runs
    ]
    gpu_power_ratios = [
        ratio
        for run in runs
        for ratio in run["gpu_encrypt_decrypt_round_trip"]["ratios"]
        ["fixed_over_original_by_power"].values()
    ]
    host_run_ratios = [
        run["host_exponentiation"]["ratios"]["median_fixed_over_original"]
        for run in runs
    ]

    def dependence(component: str, mode: str, metric: str) -> list[float]:
        return [
            run[component]["timing_dependence"][mode][metric]
            for run in runs
        ]

    summary = {
        "experiment": "acm_fixed_schedule_timing_cross_run_summary",
        "inputs": [str(path) for path in args.inputs],
        "run_count": len(runs),
        "gpu_encrypt_decrypt_round_trip": {
            "run_median_fixed_over_original": gpu_run_ratios,
            "run_median_ratio_range": _range(gpu_run_ratios),
            "all_power_ratio_range": _range(gpu_power_ratios),
            "fixed_max_over_min_median_range": _range(
                dependence(
                    "gpu_encrypt_decrypt_round_trip",
                    "fixed",
                    "max_over_min_median",
                )
            ),
        },
        "host_exponentiation": {
            "run_median_fixed_over_original": host_run_ratios,
            "run_median_ratio_range": _range(host_run_ratios),
            "original_max_over_min_median_range": _range(
                dependence(
                    "host_exponentiation",
                    "original",
                    "max_over_min_median",
                )
            ),
            "fixed_max_over_min_median_range": _range(
                dependence(
                    "host_exponentiation",
                    "fixed",
                    "max_over_min_median",
                )
            ),
            "original_product_count_correlation_range": _range(
                dependence(
                    "host_exponentiation",
                    "original",
                    "correlation_with_original_product_count",
                )
            ),
            "fixed_product_count_correlation_range": _range(
                dependence(
                    "host_exponentiation",
                    "fixed",
                    "correlation_with_original_product_count",
                )
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

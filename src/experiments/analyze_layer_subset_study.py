"""Summarize representation, enumeration, and retraining mechanism results."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import spearmanr


MODELS = (
    "facebook__deit-small-patch16-224",
    "google__vit-base-patch16-224",
)
DESCRIPTORS = (
    "depth_span",
    "mean_pairwise_distance",
    "adjacent_pairs",
    "depth_thirds_covered",
)


def read_json(path: Path):
    return json.loads(path.read_text())


def aggregate(rows: list[dict], fields: tuple[str, ...]) -> dict:
    result = {"count": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        result[field] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def correlation(rows: list[dict], x_field: str, y_field: str) -> dict | None:
    x = np.asarray([row[x_field] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_field] for row in rows], dtype=np.float64)
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    statistic, pvalue = spearmanr(x, y)
    return {
        "spearman_rho": float(statistic),
        "pvalue": float(pvalue),
        "count": len(rows),
    }


def screen_analysis(root: Path) -> tuple[dict, dict[tuple[str, str], dict]]:
    baseline = read_json(root / "baseline.json")
    rows = read_json(root / "screen_summary.json")
    by_name = {(row["model"], row["name"]): row for row in rows}
    families = defaultdict(list)
    for row in rows:
        families[(row["k"], row["family"])].append(row)
    family_summary = {
        f"k{k}:{family}": aggregate(
            group,
            ("accuracy", "mean_cosine_damage", "mean_cka_damage", "prediction_agreement"),
        )
        for (k, family), group in sorted(families.items())
    }
    random_correlations = {}
    for k in (4, 6):
        random_rows = [row for row in rows if row["k"] == k and row["family"] == "random"]
        random_correlations[f"k{k}"] = {
            descriptor: {
                metric: correlation(random_rows, descriptor, metric)
                for metric in ("accuracy", "mean_cosine_damage", "mean_cka_damage")
            }
            for descriptor in DESCRIPTORS
        }
    return (
        {
            "baseline_accuracy": baseline["accuracy"],
            "num_examples": baseline["num_examples"],
            "family_summary": family_summary,
            "random_subset_correlations": random_correlations,
        },
        by_name,
    )


def enumeration_analysis(root: Path) -> dict:
    path = root / "subset_enumeration.jsonl"
    if not path.exists():
        return {"status": "missing"}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {"status": "complete" if len(rows) == 1419 else "partial", "count": len(rows)}
    for k in (4, 6):
        group = [row for row in rows if row["k"] == k]
        expected = 495 if k == 4 else 924
        if not group:
            result[f"k{k}"] = {"count": 0, "expected": expected, "status": "pending"}
            continue
        result[f"k{k}"] = {
            "count": len(group),
            "expected": expected,
            "metrics": aggregate(group, ("accuracy", "prediction_agreement", "clean_to_encrypted_kl")),
            "descriptor_correlations": {
                descriptor: {
                    metric: correlation(group, descriptor, metric)
                    for metric in ("accuracy", "prediction_agreement", "clean_to_encrypted_kl")
                }
                for descriptor in DESCRIPTORS
            },
            "lowest_accuracy_subsets": sorted(group, key=lambda row: row["accuracy"])[:5],
            "highest_accuracy_subsets": sorted(group, key=lambda row: -row["accuracy"])[:5],
        }
    return result


def retraining_analysis(root: Path, screen_rows: dict[tuple[str, str], dict]) -> dict:
    result_files = sorted(root.rglob("result.json")) if root.exists() else []
    rows = [read_json(path) for path in result_files]
    summary = {"count": len(rows), "phases": {}}
    by_phase = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)
    for phase, group in sorted(by_phase.items()):
        compact = []
        for row in group:
            final_top1 = row["history"][-1]["validation"]["top1"]
            compact.append(
                {
                    "model": row["model"],
                    "name": row["subset"]["name"],
                    "family": row["subset"]["family"],
                    "attacker": row["attacker"],
                    "seed": row["seed"],
                    "initial_top1": row["initial_validation"]["top1"],
                    "final_top1": final_top1,
                    "recovery_auc": row["recovery_auc"],
                }
            )
        summary["phases"][phase] = compact

    short = by_phase.get("short", [])
    joined = []
    for row in short:
        key = (row["model"], row["subset"]["name"])
        if key not in screen_rows:
            continue
        joined.append(
            {
                **screen_rows[key],
                "final_top1": row["history"][-1]["validation"]["top1"],
                "recovery_auc": row["recovery_auc"],
            }
        )
    summary["short_screen_correlations"] = {
        metric: {
            target: correlation(joined, metric, target)
            for target in ("final_top1", "recovery_auc")
        }
        for metric in (
            "accuracy",
            "mean_cosine_damage",
            "mean_cka_damage",
            *DESCRIPTORS,
        )
    }

    blind = {
        (row["model"], row["subset"]["name"], row["seed"]): row
        for row in by_phase.get("short", [])
    }
    oracle_deltas = []
    for row in by_phase.get("oracle", []):
        key = (row["model"], row["subset"]["name"], row["seed"])
        if key not in blind:
            continue
        blind_row = blind[key]
        oracle_deltas.append(
            {
                "model": row["model"],
                "name": row["subset"]["name"],
                "seed": row["seed"],
                "blind_final_top1": blind_row["history"][-1]["validation"]["top1"],
                "oracle_final_top1": row["history"][-1]["validation"]["top1"],
                "oracle_minus_blind": (
                    row["history"][-1]["validation"]["top1"]
                    - blind_row["history"][-1]["validation"]["top1"]
                ),
            }
        )
    summary["oracle_deltas"] = oracle_deltas
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism-root",
        type=Path,
        default=Path("results/layer_subset_mechanism"),
    )
    parser.add_argument(
        "--retraining-root",
        type=Path,
        default=Path("results/layer_subset_retraining"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/layer_subset_mechanism/analysis_summary.json"),
    )
    args = parser.parse_args()

    summary = {"models": {}}
    all_screen_rows = {}
    for model in MODELS:
        root = args.mechanism_root / model
        screen, rows = screen_analysis(root)
        all_screen_rows.update(rows)
        summary["models"][model] = {
            "screen": screen,
            "enumeration": enumeration_analysis(root),
        }
    summary["retraining"] = retraining_analysis(args.retraining_root, all_screen_rows)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

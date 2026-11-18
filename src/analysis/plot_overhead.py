"""Overhead figures from vit_latency_sweep results.

Bars: median baseline forward pass + end-to-end protection cost (median protected - median baseline);
labels: median protected / median baseline.

  uv run python -m src.analysis.plot_overhead --gpu results/latency/gpu.json --cpu results/latency/cpu.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FWD, PROT = "#6fdc9a", "#e8756a"


def panel(ax, rows, key, mode, xlabel):
    xs = [str(r[key]) for r in rows]
    base = [r["baseline_ms"] for r in rows]
    extra = [r[f"{mode}_ms"] - r["baseline_ms"] for r in rows]
    ax.bar(xs, base, color=FWD, label="Forward pass (unprotected baseline)")
    ax.bar(xs, [max(e, 0) for e in extra], bottom=base, color=PROT, label="Protection cost (end-to-end)")
    top = max(b + max(e, 0) for b, e in zip(base, extra, strict=True))
    for x, b, e, r in zip(xs, base, extra, rows, strict=True):
        ax.text(x, b + max(e, 0) + 0.01 * top, f"{r[f'{mode}_overhead']:.3f}×", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, top * 1.12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=Path, default=Path("results/latency/gpu.json"))
    p.add_argument("--cpu", type=Path, default=Path("results/latency/cpu.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/latency"))
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for dev, path in {"gpu": args.gpu, "cpu": args.cpu}.items():
        rows = json.loads(path.read_text())["rows"]
        layers = [r for r in rows if r["sweep"] == "layers"]
        iters = [r for r in rows if r["sweep"] == "iterations"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        panel(axes[0], iters, "n", "base", "Number of ACM Iterations ($K=6$)")
        panel(axes[1], layers, "K", "base", "Number of Encrypted Layers ($n=5$)")
        axes[0].legend(loc="lower center", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out / f"base_{dev}.pdf")
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        panel(ax, layers, "K", "secure", "Number of Encrypted Layers (Batch Size 64)")
        ax.legend(loc="lower center", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out / f"secure_{dev}.pdf")


if __name__ == "__main__":
    main()

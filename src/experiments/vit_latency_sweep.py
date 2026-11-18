"""End-to-end ViT-B latency for baseline, ChaosFormer and ChaosFormer-S (ChaCha20).

Two protected pipelines:
  layerwise  ``TransformerCipher.authorized_inference``: each protected layer is decrypted
             into shared plaintext buffers immediately before it runs and discarded after.
  reencrypt  decrypt all protected layers in place -> forward -> re-encrypt them.
Every configuration is timed with the same boundary: synchronize, R x protected pass,
synchronize.
Within each trial, the baseline and every distinct protected config run in rotating order;
each config is reported as the median over trials relative to the shared baseline.

  python -m src.experiments.vit_latency_sweep --device cuda:0 --pipeline reencrypt --output results/latency/gpu.json
  python -m src.experiments.vit_latency_sweep --device cpu --pipeline layerwise --output results/latency/cpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import ViTConfig, ViTForImageClassification

from src.encryption.transformer_cipher import TransformerCipher

SECRET = b"chaosformer-latency"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, reps, device) -> float:
    sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(device)
    return (time.perf_counter() - t0) * 1e3 / reps


def timed_protected(cipher, fwd, reps, device, pipeline) -> float:
    """Time ``reps`` protected passes; weights are encrypted before and after."""
    if pipeline == "layerwise":
        with cipher.authorized_inference():
            fwd()  # warm-up
            return timed(fwd, reps, device)

    @torch.inference_mode()
    def run():
        for layer in cipher.layer_specs:
            cipher.decrypt_layer(layer)
        fwd()
        for layer in cipher.layer_specs:
            cipher.encrypt_layer(layer)

    run()  # warm-up
    return timed(run, reps, device)


def measure(model, x, configs, trials, reps, device, pipeline="layerwise") -> list[dict]:
    """Time every config of a sweep against one shared baseline.

    Within each trial the baseline and all protected (config, mode) runs execute in a
    rotating order, so slow drift of the machine affects every config equally.
    """
    fwd = torch.inference_mode()(lambda: model(pixel_values=x))
    ciphers = {(i, mode): TransformerCipher(model, secure=mode == "secure", selected_layers=cfg["layers"],
                                            seed=SECRET, arnold_key=(cfg["n"], 1, 1, 1, 2))
               for i, cfg in enumerate(configs) for mode in ("base", "secure")}
    names = ["baseline", *ciphers]
    samples = {name: [] for name in names}
    for trial in range(trials):
        shift = trial % len(names)
        for name in names[shift:] + names[:shift]:
            if name == "baseline":
                samples[name].append(timed(fwd, reps, device))
                continue
            cipher = ciphers[name]  # one scheme applied at a time; weights end in plaintext
            cipher.encrypt_all()
            samples[name].append(timed_protected(cipher, fwd, reps, device, pipeline))
            cipher.decrypt_all()
    med = {k: statistics.median(v) for k, v in samples.items()}
    rows = []
    for i, cfg in enumerate(configs):
        row = dict(cfg, baseline_ms=med["baseline"], baseline_trials_ms=samples["baseline"])
        for mode in ("base", "secure"):
            row |= {f"{mode}_ms": med[i, mode], f"{mode}_trials_ms": samples[i, mode],
                    f"{mode}_overhead": med[i, mode] / med["baseline"]}
        rows.append(row)
    return rows


def environment(args, device, reps) -> dict:
    import transformers
    import triton

    cpu = subprocess.run("lscpu | grep 'Model name'", check=False, shell=True, capture_output=True, text=True).stdout
    cpu = cpu.split(":")[-1].strip()
    return {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu,
            "cpu": cpu, "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "triton": triton.__version__,
            "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads(),
            "batch_size": args.batch_size, "dtype": "float32", "trials": args.trials, "reps_per_trial": reps,
            "pipeline": args.pipeline, "boundary": "sync; reps x protected pass; sync; inference_mode for all",
            "aggregate": "median over trials; all configs share the baseline and run in rotating order"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--trials", type=int, default=7)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--sweeps", nargs="+", default=["layers", "iterations"])
    p.add_argument("--pipeline", choices=["layerwise", "reencrypt"], default="layerwise")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    reps = args.reps if device.type == "cuda" else max(1, args.reps // 5)

    try:
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    except OSError:
        model = ViTForImageClassification(ViTConfig())  # latency only depends on shapes
    model.to(device).eval()
    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(3):
            model(pixel_values=x)

    sweeps = {"layers": [{"sweep": "layers", "K": k, "n": 5} for k in (2, 4, 6, 8, 10, 12)],
              "iterations": [{"sweep": "iterations", "K": 6, "n": n} for n in (3, 5, 8, 12, 16, 32)]}
    # Configs shared by several sweeps (K=6, n=5) are measured once and reported in each.
    # Last-K; latency depends on K, not on which layers.
    configs = [cfg for name in args.sweeps for cfg in sweeps[name]]
    unique = list({(c["K"], c["n"]): {"K": c["K"], "n": c["n"], "layers": list(range(12 - c["K"], 12))}
                   for c in configs}.values())
    measured = {(r["K"], r["n"]): r for r in measure(model, x, unique, args.trials, reps, device, args.pipeline)}
    rows = [{"sweep": cfg["sweep"]} | measured[cfg["K"], cfg["n"]] for cfg in configs]
    for row in rows:
        print(json.dumps({k: row[k] for k in ("sweep", "K", "n", "baseline_ms", "base_ms", "secure_ms",
                                              "base_overhead", "secure_overhead")}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"environment": environment(args, device, reps), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

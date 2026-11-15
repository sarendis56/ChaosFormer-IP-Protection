"""End-to-end ViT-B latency for baseline, ChaosFormer and ChaosFormer-S (ChaCha20).

Every configuration is timed with the same boundary: synchronize,
R x [decrypt selected layers -> forward -> re-encrypt], synchronize. On CUDA, decrypt and re-encrypt
each replay a captured CUDA graph. No per-stage synchronization inside the timed
region. Baseline and protected trials alternate (forward/reverse order) and
the median over trials is reported. A separate, explicitly instrumented pass
(per-stage synchronization) gives a stage breakdown only.

On CPU, set NUMBA_THREADING_LAYER=workqueue: the default layer busy-waits on
the cores PyTorch uses once the ACM kernels have run.

  python -m src.experiments.vit_latency_sweep --device cuda:0 --output results/latency/gpu.json
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

from src.encryption.arnold_transform import (
    arnold_numba,
    arnold_triton,
    iarnold_numba,
    iarnold_triton,
)
from src.encryption.chacha20 import chacha20_xor_, derive_chacha20_material
from src.encryption.transformer_cipher import graph_replays
from src.encryption.xor_encryption import get_stable_seed

SECRET = b"chaosformer-latency"


def _weights(model, layers):
    for i in layers:
        layer = model.vit.encoder.layer[i]
        attention = layer.attention.attention
        yield i, "query", "square", attention.query.weight
        yield i, "key", "square", attention.key.weight
        yield i, "value", "square", attention.value.weight
        yield i, "attention_output", "square", layer.attention.output.dense.weight
        yield i, "intermediate", "ffn_columns", layer.intermediate.dense.weight
        yield i, "output", "ffn_rows", layer.output.dense.weight


class Scheme:
    """In-place encrypt/decrypt of the selected ViT layers.

    mode 'base': ACM on attention matrices, row/column permutation on FFN matrices.
    mode 'secure': base followed by ChaCha20 diffusion (removed first on decrypt).
    """

    def __init__(self, model, layers, mode: str, n: int, device: torch.device):
        self.secure = mode == "secure"
        self.arnold_key = [n, 1, 1, 1, 2]
        gpu = device.type == "cuda"
        self.arnold, self.iarnold = (arnold_triton, iarnold_triton) if gpu else (arnold_numba, iarnold_numba)
        self.specs = []
        for i, name, op, parameter in _weights(model, layers):
            perm = inv = None
            if op != "square":
                dim = parameter.shape[1] if op == "ffn_columns" else parameter.shape[0]
                generator = torch.Generator().manual_seed(get_stable_seed(i, name, SECRET))
                perm = torch.randperm(dim, generator=generator).to(parameter.device)
                inv = torch.argsort(perm)
            self.specs.append((parameter, op, perm, inv, *derive_chacha20_material(i, name, SECRET)))

    @staticmethod
    def _permute(parameter, op, index):
        parameter.copy_(parameter[:, index] if op == "ffn_columns" else parameter[index])

    @torch.no_grad()
    def encrypt(self) -> None:
        for parameter, op, perm, _, key, nonce in self.specs:
            if op == "square":
                parameter.copy_(self.arnold(parameter, self.arnold_key))
            else:
                self._permute(parameter, op, perm)
            if self.secure:
                chacha20_xor_(parameter.data, key, nonce)

    @torch.no_grad()
    def decrypt(self) -> None:
        for parameter, op, _, inv, key, nonce in self.specs:
            if self.secure:
                chacha20_xor_(parameter.data, key, nonce)
            if op == "square":
                parameter.copy_(self.iarnold(parameter, self.arnold_key))
            else:
                self._permute(parameter, op, inv)


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


@torch.inference_mode()
def breakdown(model, scheme, x, reps, device) -> dict:
    """Instrumented pass: per-stage synchronization (inflates totals on GPU)."""
    stages = {"decrypt_ms": scheme.decrypt, "forward_ms": lambda: model(pixel_values=x), "encrypt_ms": scheme.encrypt}
    acc = dict.fromkeys(stages, 0.0)
    for _ in range(reps):
        for key, fn in stages.items():
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            acc[key] += (time.perf_counter() - t0) * 1e3
    return {k: v / reps for k, v in acc.items()}


def measure(model, x, layers, n, trials, reps, device) -> dict:
    fwd = torch.inference_mode()(lambda: model(pixel_values=x))
    schemes = {mode: Scheme(model, layers, mode, n, device) for mode in ("base", "secure")}
    with torch.inference_mode():  # weights start and end in plaintext, so encrypt first
        graphs = {}
        for mode, s in schemes.items():
            s.encrypt()
            steps = [s.decrypt, s.encrypt]
            graphs[mode] = graph_replays(steps) if device.type == "cuda" else steps
            s.decrypt()
    samples = {name: [] for name in ("baseline", *schemes)}
    for trial in range(trials):
        for name in samples if trial % 2 == 0 else reversed(samples):
            if name == "baseline":
                samples[name].append(timed(fwd, reps, device))
                continue
            s = schemes[name]
            decrypt, encrypt = graphs[name]

            def run(decrypt=decrypt, encrypt=encrypt):
                decrypt()
                fwd()
                encrypt()

            with torch.inference_mode():  # one scheme applied at a time; weights end in plaintext
                s.encrypt()
                run()  # warm-up
                samples[name].append(timed(run, reps, device))
                s.decrypt()
    med = {k: statistics.median(v) for k, v in samples.items()}
    row = {f"{k}_ms": v for k, v in med.items()}
    row |= {f"{k}_trials_ms": v for k, v in samples.items()}
    row |= {f"{m}_overhead": med[m] / med["baseline"] for m in schemes}
    for mode, s in schemes.items():
        with torch.inference_mode():
            s.encrypt()
        row[f"{mode}_instrumented_breakdown"] = breakdown(model, s, x, reps, device)
        with torch.inference_mode():
            s.decrypt()
    return row


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
            "boundary": "sync; reps x (decrypt selected -> forward -> re-encrypt); sync; inference_mode for all",
            "aggregate": "median over trials; trials alternate order"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--trials", type=int, default=7)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--sweeps", nargs="+", default=["layers", "iterations"])
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

    configs = []
    if "layers" in args.sweeps:
        configs += [{"sweep": "layers", "K": k, "n": 5} for k in (2, 4, 6, 8, 10, 12)]
    if "iterations" in args.sweeps:
        configs += [{"sweep": "iterations", "K": 6, "n": n} for n in (3, 5, 8, 12, 16, 32)]

    rows = []
    for cfg in configs:
        layers = list(range(12 - cfg["K"], 12))  # Last-K; latency depends on K, not on which layers
        row = cfg | {"layers": layers} | measure(model, x, layers, cfg["n"], args.trials, reps, device)
        print(json.dumps({k: row[k] for k in ("sweep", "K", "n", "baseline_ms", "base_ms", "secure_ms",
                                              "base_overhead", "secure_overhead")}), flush=True)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"environment": environment(args, device, reps), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Time one encryption + decryption round of a whole ViT checkpoint per cipher.

Whole-file baselines (AES-CBC/CTR/GCM with AES-NI via OpenSSL, TEA, XOR) process every
weight byte; ChaosFormer processes only its K protected layers. Key setup is excluded.
GPU rows time only device work (weights already on the GPU). Methods run in rotating
order within each trial; each value is the median over trials.

  python -m src.experiments.cipher_comparison --output results/cipher_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from transformers import ViTConfig, ViTForImageClassification

from src.encryption.transformer_cipher import TransformerCipher

KEY = bytes(range(16))
IV = bytes(16)
MODELS = {  # name: (config, protected layers K); latency depends only on shapes
    "ViT-B": (ViTConfig(), 6),
    "ViT-L": (ViTConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                        intermediate_size=4096), 7),
}


def aes_round(data: bytes, mode: str) -> bytes:
    def run(m, buf, enc):
        c = Cipher(algorithms.AES(KEY), m)
        ctx = c.encryptor() if enc else c.decryptor()
        return ctx.update(buf) + ctx.finalize()

    if mode == "GCM":
        ct = Cipher(algorithms.AES(KEY), modes.GCM(IV[:12])).encryptor()
        out = ct.update(data) + ct.finalize()
        return run(modes.GCM(IV[:12], ct.tag), out, False)
    m = modes.CBC(IV) if mode == "CBC" else modes.CTR(IV)
    return run(m, run(m, data, True), False)


TEA_DELTA = np.uint32(0x9E3779B9)
TEA_KEY = np.frombuffer(KEY, dtype="<u4")


def tea(words: np.ndarray, decrypt: bool, chunk: int = 1 << 22) -> None:
    """In-place 32-cycle TEA (ECB) over uint32 pairs, vectorized over blocks."""
    k0, k1, k2, k3 = TEA_KEY
    pairs = words.reshape(-1, 2)
    with np.errstate(over="ignore"):
        for s in range(0, len(pairs), chunk):
            v0, v1 = pairs[s:s + chunk, 0].copy(), pairs[s:s + chunk, 1].copy()
            if decrypt:
                total = np.uint32(TEA_DELTA * np.uint32(32))
                for _ in range(32):
                    v1 -= ((v0 << 4) + k2) ^ (v0 + total) ^ ((v0 >> 5) + k3)
                    v0 -= ((v1 << 4) + k0) ^ (v1 + total) ^ ((v1 >> 5) + k1)
                    total -= TEA_DELTA
            else:
                total = np.uint32(0)
                for _ in range(32):
                    total += TEA_DELTA
                    v0 += ((v1 << 4) + k0) ^ (v1 + total) ^ ((v1 >> 5) + k1)
                    v1 += ((v0 << 4) + k2) ^ (v0 + total) ^ ((v0 >> 5) + k3)
            pairs[s:s + chunk, 0], pairs[s:s + chunk, 1] = v0, v1


def flat_bytes(model) -> np.ndarray:
    return torch.cat([p.detach().cpu().reshape(-1).view(torch.uint8) for p in model.state_dict().values()]).numpy()


def methods(model, k: int, device: torch.device) -> dict:
    """name -> zero-arg callable doing one encrypt + decrypt round."""
    data = flat_bytes(model)
    raw = data.tobytes() + bytes(-len(data) % 16)  # pad to whole AES blocks
    words = data[: len(data) // 8 * 8].view(np.uint32).copy()
    key64 = np.frombuffer(KEY, dtype=np.uint64)
    u64 = data[: len(data) // 16 * 16].view(np.uint64).copy().reshape(-1, 2)
    out = {f"AES-{m}": (lambda m=m: aes_round(raw, m)) for m in ("CBC", "CTR", "GCM")}
    out["TEA"] = lambda: (tea(words, False), tea(words, True))
    out["XOR"] = lambda: (np.bitwise_xor(u64, key64, out=u64), np.bitwise_xor(u64, key64, out=u64))
    model.to(device)
    layers = list(range(model.config.num_hidden_layers - k, model.config.num_hidden_layers))
    for name, secure in (("ChaosFormer", False), ("ChaosFormer-S", True)):
        cipher = TransformerCipher(model, secure=secure, selected_layers=layers, seed=b"chaosformer-comparison",
                                   arnold_key=(3, 1, 1, 1, 2))
        out[name] = lambda c=cipher: (c.encrypt_all(), c.decrypt_all())
    if device.type == "cuda":
        g = torch.from_numpy(u64.view(np.int64)).to(device)
        gk = torch.from_numpy(key64.view(np.int64)).to(device)
        out["XOR (GPU)"] = lambda: (g.bitwise_xor_(gk), g.bitwise_xor_(gk))
    return out


def measure(fns: dict, trials: int, device: torch.device) -> dict:
    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    names = list(fns)
    for fn in fns.values():
        fn()  # warm-up
    samples = {n: [] for n in names}
    for trial in range(trials):
        shift = trial % len(names)
        for n in names[shift:] + names[:shift]:
            samples[n].append(timed(fns[n]))
    return {n: {"median_s": statistics.median(v), "trials_s": v} for n, v in samples.items()}


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return os.uname().machine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    rows = {}
    for name in args.models:
        cfg, k = MODELS[name]
        torch.manual_seed(0)
        model = ViTForImageClassification(cfg).eval()
        fns = methods(model, k, device)
        if device.type == "cuda":  # GPU table: device rows only
            fns = {n: f for n, f in fns.items() if n in ("XOR (GPU)", "ChaosFormer", "ChaosFormer-S")}
        rows[name] = {"model_mb": len(flat_bytes(model)) / 2**20, "K": k, **measure(fns, args.trials, device)}
        print(name, {n: round(v["median_s"], 4) for n, v in rows[name].items() if isinstance(v, dict)}, flush=True)
        del model, fns
    env = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name(),
           "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "trials": args.trials,
           "acm_iterations": 3, "aggregate": "median over trials; methods in rotating order"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"environment": env, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Min-entropy accounting and a debiased fuzzy extractor for the SRAM-PUF benchmark.

  * Accounting of the code-offset extractor in puf_fuzzy_extractor.py under the
    measured response bias (i.i.d.-bias model).
  * A von Neumann-debiased code-offset extractor (Maes et al., CHES 2015) whose
    kept bits are unbiased under that model, so the n-k leaked bits leave k bits.
  * Reconstruction timing with warm-up and repeated runs; pin the process to one
    core (e.g. taskset -c 2) for stable numbers.

  uv run python -m src.analysis.puf_entropy --dataset-root <Raw_data_UNO>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import bchlib
import numpy as np

from src.analysis.puf_fuzzy_extractor import FuzzyExtractor

SWEEPS = {
    "temperature": ("RoomTemp_UNO_10Boards_Temp/Resp_UNO_AmbientTemp_nMeas15.npy",
                    "CollectedDataAcrossTemperature_UNO_10Boards/GResp_UNO_temp_all.npy", 38),
    "voltage": ("RoomTemp_UNO_8Boards_Volt/Resp_UNO_AmbientVolt_nMeas15.npy",
                "CollectedDataAcrossVoltage_UNO_8Boards/GResp_UNO_Volt_all.npy", 25),
}


class DebiasedExtractor:
    """Von Neumann pair debiasing -> repetition -> BCH code-offset (uniform k-bit secret)."""

    def __init__(self, secret_bytes: int, bch_t: int, bch_m: int = 10, repetition: int = 3):
        self.secret_bytes, self.repetition = secret_bytes, repetition
        self.bch = bchlib.BCH(bch_t, m=bch_m)
        self.codeword_bits = (secret_bytes + self.bch.ecc_bytes) * 8
        self.required_bits = self.codeword_bits * repetition

    def enroll(self, reference, reliability, secret):
        order = np.argsort(-reliability, kind="stable")
        first, second = order[0::2], order[1::2]
        keep = reference[first] != reference[second]  # discordant pairs
        kept = first[keep]
        if kept.size < self.required_bits:
            return None
        idx = kept[: self.required_bits]  # helper: positions only (value-independent under i.i.d. bias)
        code = np.frombuffer(secret + self.bch.encode(secret), dtype=np.uint8)
        helper = reference[idx] ^ np.repeat(np.unpackbits(code), self.repetition)
        return {"idx": idx, "helper": helper, "digest": hashlib.sha256(secret).digest(), "kept": int(kept.size)}

    def reconstruct(self, observation, enr):
        noisy = observation[enr["idx"]] ^ enr["helper"]
        bits = (noisy.reshape(self.codeword_bits, self.repetition).sum(1) > self.repetition // 2).astype(np.uint8)
        packed = np.packbits(bits)
        data = bytearray(packed[: self.secret_bytes])
        ecc = bytearray(packed[self.secret_bytes:])
        if self.bch.decode(data, ecc) < 0:
            return None
        self.bch.correct(data, ecc)
        secret = bytes(data)
        return hashlib.sha256(secret).digest()[:16] if hashlib.sha256(secret).digest() == enr["digest"] else None


def load(root: Path, sweep: str):
    amb, sw, n = SWEEPS[sweep]
    a = np.load(root / amb).reshape(-1, 8192, 15).astype(np.uint8)
    s = np.load(root / sw).reshape(a.shape[0], n, 8192).astype(np.uint8)
    p1 = a.mean(2)
    return (p1 >= 0.5).astype(np.uint8), np.maximum(p1, 1 - p1), s


def reconstruct(extractor, observation, enrollment):
    """Recovered key or None (the published extractor also returns diagnostics)."""
    key = extractor.reconstruct(observation, enrollment)
    return key if isinstance(extractor, DebiasedExtractor) else key[0]


def evaluate(extractor, root: Path, rng) -> dict:
    out = {}
    for sweep in SWEEPS:
        ref, rel, s = load(root, sweep)
        enrs = []
        for d in range(ref.shape[0]):
            secret = rng.integers(0, 256, extractor.secret_bytes, dtype=np.uint8).tobytes()
            enrs.append(extractor.enroll(ref[d], rel[d], secret))
        if any(e is None for e in enrs):
            out[sweep] = {"error": "insufficient debiased bits"}
            continue
        ok = [reconstruct(extractor, s[d, c], enrs[d]) is not None
              for d in range(ref.shape[0]) for c in range(s.shape[1])]
        fa = [reconstruct(extractor, s[o, s.shape[1] // 2], enrs[t]) is not None
              for t in range(ref.shape[0]) for o in range(ref.shape[0]) if o != t]
        out[sweep] = {"successes": int(sum(ok)), "trials": len(ok), "false_accepts": int(sum(fa)),
                      "false_accept_trials": len(fa)}
        if isinstance(extractor, DebiasedExtractor):
            out[sweep]["min_kept_debiased_bits"] = min(e["kept"] for e in enrs)
    return out


def timing(extractor, root: Path, rng, warmup: int, reps: int) -> dict:
    ref, rel, s = load(root, "temperature")
    obs, enrs = [], []
    for d in range(ref.shape[0]):
        secret = rng.integers(0, 256, extractor.secret_bytes, dtype=np.uint8).tobytes()
        enrs.append(extractor.enroll(ref[d], rel[d], secret))
        obs += [(s[d, c], d) for c in range(s.shape[1])]
    samples = []
    for i in range(warmup + reps):
        o, d = obs[i % len(obs)]
        t0 = time.perf_counter_ns()
        extractor.reconstruct(o, enrs[d])
        if i >= warmup:
            samples.append((time.perf_counter_ns() - t0) / 1e3)
    a = np.asarray(samples)
    return {"warmup": warmup, "repetitions": reps, "median_us": float(np.median(a)),
            "p95_us": float(np.percentile(a, 95)), "mean_us": float(a.mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--reps", type=int, default=10_000)
    p.add_argument("--output", type=Path, default=Path("results/puf_fuzzy_extractor/entropy.json"))
    args = p.parse_args()

    cpu = subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout
    cpu = cpu.split(":")[-1].strip()
    env = {"cpu": cpu, "machine": platform.machine(), "python": platform.python_version(),
           "numpy": version("numpy"), "bchlib": version("bchlib"), "threads": "single process, pinned with taskset"}

    # Entropy accounting of the published configuration under the measured bias.
    bias = {}
    for sweep in SWEEPS:
        ref, _, _ = load(args.dataset_root, sweep)
        p1 = float(ref.mean())
        bias[sweep] = {"fraction_ones": p1, "min_entropy_per_bit_iid": float(-np.log2(max(p1, 1 - p1)))}
    h = min(v["min_entropy_per_bit_iid"] for v in bias.values())
    pub = FuzzyExtractor()
    n, k = pub.required_puf_bits, pub.secret_bytes * 8
    accounting = {"published": {"n_bits": n, "k_bits": k, "leakage_n_minus_k": n - k,
                                "residual_lower_bound_bits": h * n - (n - k), "h_per_bit_used": h}}

    variants = {
        "published_rep5_bch63": pub,
        "debiased_rep3_k128_t45": DebiasedExtractor(16, 45),
        "debiased_rep3_k256_t33": DebiasedExtractor(32, 33),
    }
    results = {}
    for name, ex in variants.items():
        rng = np.random.default_rng(20260710)
        entry = {"reliability": evaluate(ex, args.dataset_root, rng)}
        entry["timing"] = timing(ex, args.dataset_root, np.random.default_rng(1), 200, args.reps)
        if isinstance(ex, DebiasedExtractor):
            n2, k2 = ex.required_bits, ex.secret_bytes * 8
            entry["accounting"] = {"n_bits": n2, "k_bits": k2, "leakage_n_minus_k": n2 - k2,
                                   "residual_bits_iid_bias_model": n2 - (n2 - k2),
                                   "puf_bits_scanned": 8192}
        results[name] = entry
        print(name, json.dumps(entry), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"environment": env, "bias": bias, "accounting": accounting,
                                       "variants": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()

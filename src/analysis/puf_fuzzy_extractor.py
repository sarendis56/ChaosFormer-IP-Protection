"""Benchmark a code-offset fuzzy extractor on public SRAM PUF measurements.

The expected dataset is the Arduino UNO corpus released with:
K. Pratihar et al., "Enhancing SRAM-Based PUF Reliability Through
Machine Learning-Aided Calibration Techniques," IEEE TCAD, 2024.

The extractor binds a uniformly random 128-bit secret to a stable subset of an
8,192-bit SRAM response. It uses a shortened BCH code followed by an odd-length
repetition code. Public helper data reveals no secret by itself; SHA-256 provides
privacy amplification after successful reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import bchlib
import numpy as np


@dataclass
class Enrollment:
    indices: np.ndarray
    helper_data: bytes
    key_digest: bytes
    secret_bytes: int
    codeword_bytes: int
    repetition: int


class FuzzyExtractor:
    """Code-offset fuzzy extractor with BCH and repetition decoding."""

    def __init__(
        self,
        *,
        secret_bytes: int = 16,
        bch_m: int = 10,
        bch_t: int = 63,
        repetition: int = 5,
    ) -> None:
        if repetition <= 0 or repetition % 2 == 0:
            raise ValueError("repetition must be a positive odd integer")
        self.secret_bytes = secret_bytes
        self.bch = bchlib.BCH(bch_t, m=bch_m)
        self.repetition = repetition
        self.codeword_bytes = secret_bytes + self.bch.ecc_bytes
        self.codeword_bits = self.codeword_bytes * 8
        self.required_puf_bits = self.codeword_bits * repetition

    @staticmethod
    def _as_bits(response: np.ndarray) -> np.ndarray:
        bits = np.asarray(response, dtype=np.uint8).reshape(-1)
        if np.any(bits > 1):
            raise ValueError("PUF response must contain only binary values")
        return bits

    def enroll(
        self,
        reference: np.ndarray,
        reliability: np.ndarray,
        secret: bytes,
    ) -> Enrollment:
        if len(secret) != self.secret_bytes:
            raise ValueError(f"secret must contain {self.secret_bytes} bytes")
        reference_bits = self._as_bits(reference)
        reliability_values = np.asarray(reliability, dtype=float).reshape(-1)
        if reference_bits.size != reliability_values.size:
            raise ValueError("reference and reliability sizes differ")
        if reference_bits.size < self.required_puf_bits:
            raise ValueError("PUF response is too short for this code")

        # Stable ordering makes enrollment exactly reproducible when ties occur.
        indices = np.argsort(-reliability_values, kind="stable")[
            : self.required_puf_bits
        ].astype(np.uint16)
        codeword = np.frombuffer(
            secret + self.bch.encode(secret), dtype=np.uint8
        )
        encoded_bits = np.repeat(np.unpackbits(codeword), self.repetition)
        helper_bits = reference_bits[indices] ^ encoded_bits
        return Enrollment(
            indices=indices,
            helper_data=np.packbits(helper_bits).tobytes(),
            key_digest=hashlib.sha256(secret).digest(),
            secret_bytes=self.secret_bytes,
            codeword_bytes=self.codeword_bytes,
            repetition=self.repetition,
        )

    def reconstruct(
        self, observation: np.ndarray, enrollment: Enrollment
    ) -> tuple[bytes | None, int]:
        observation_bits = self._as_bits(observation)
        helper_bits = np.unpackbits(
            np.frombuffer(enrollment.helper_data, dtype=np.uint8)
        )[: self.required_puf_bits]
        noisy_encoded = observation_bits[enrollment.indices] ^ helper_bits
        votes = noisy_encoded.reshape(self.codeword_bits, self.repetition).sum(axis=1)
        decoded_bits = (votes > self.repetition // 2).astype(np.uint8)
        decoded = np.packbits(decoded_bits)

        data = bytearray(decoded[: self.secret_bytes])
        ecc = bytearray(decoded[self.secret_bytes : self.codeword_bytes])
        corrected = self.bch.decode(data, ecc)
        if corrected < 0:
            return None, corrected
        self.bch.correct(data, ecc)
        secret = bytes(data)
        if hashlib.sha256(secret).digest() != enrollment.key_digest:
            return None, corrected
        return hashlib.sha256(secret).digest()[:16], corrected


def _pairwise_uniqueness(responses: np.ndarray) -> float:
    distances = [
        np.mean(responses[i] != responses[j])
        for i, j in combinations(range(responses.shape[0]), 2)
    ]
    return float(np.mean(distances))


def _condition_records(
    name: str,
    ambient_path: Path,
    sweep_path: Path,
    conditions: Iterable[float],
    extractor: FuzzyExtractor,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    conditions = np.asarray(list(conditions), dtype=float)
    ambient = np.load(ambient_path).reshape(-1, 8192, 15).astype(np.uint8)
    sweep = np.load(sweep_path).reshape(ambient.shape[0], len(conditions), 8192)
    sweep = sweep.astype(np.uint8)

    probability_one = ambient.mean(axis=2)
    references = (probability_one >= 0.5).astype(np.uint8)
    reliability = np.maximum(probability_one, 1.0 - probability_one)
    ambient_ber = np.mean(ambient != references[:, :, None], axis=(1, 2))

    enrollments: list[Enrollment] = []
    secrets: list[bytes] = []
    for device in range(ambient.shape[0]):
        secret = rng.integers(
            0, 256, extractor.secret_bytes, dtype=np.uint8
        ).tobytes()
        secrets.append(secret)
        enrollments.append(
            extractor.enroll(references[device], reliability[device], secret)
        )

    rows: list[dict] = []
    latencies_us: list[float] = []
    for device, enrollment in enumerate(enrollments):
        selected_reference = references[device, enrollment.indices]
        for condition_index, condition in enumerate(conditions):
            observation = sweep[device, condition_index]
            start = time.perf_counter_ns()
            key, corrected = extractor.reconstruct(observation, enrollment)
            latency_us = (time.perf_counter_ns() - start) / 1_000.0
            latencies_us.append(latency_us)
            rows.append(
                {
                    "sweep": name,
                    "device": device,
                    "condition": float(condition),
                    "raw_ber": float(
                        np.mean(
                            observation[enrollment.indices] != selected_reference
                        )
                    ),
                    "success": int(key is not None),
                    "corrected_errors": int(corrected),
                    "reconstruction_us": latency_us,
                }
            )

    false_accepts = 0
    false_trials = 0
    reference_index = int(np.argmin(np.abs(conditions - (24.0 if name == "temperature" else 5.0))))
    for target, enrollment in enumerate(enrollments):
        for other in range(ambient.shape[0]):
            if target == other:
                continue
            false_trials += 1
            key, _ = extractor.reconstruct(sweep[other, reference_index], enrollment)
            false_accepts += int(key is not None)

    successes = np.asarray([row["success"] for row in rows], dtype=float)
    raw_bers = np.asarray([row["raw_ber"] for row in rows], dtype=float)
    summary = {
        "sweep": name,
        "devices": int(ambient.shape[0]),
        "conditions": int(len(conditions)),
        "trials": int(len(rows)),
        "successes": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "raw_ber_mean": float(raw_bers.mean()),
        "raw_ber_max": float(raw_bers.max()),
        "ambient_ber_mean": float(ambient_ber.mean()),
        "uniqueness": _pairwise_uniqueness(references),
        "false_accepts": int(false_accepts),
        "false_accept_trials": int(false_trials),
        "reconstruction_us_median": float(np.median(latencies_us)),
        "reconstruction_us_p95": float(np.percentile(latencies_us, 95)),
    }
    return summary, rows


def benchmark(dataset_root: Path, output_dir: Path) -> dict:
    extractor = FuzzyExtractor()
    rng = np.random.default_rng(20260710)
    configs = [
        (
            "temperature",
            dataset_root
            / "RoomTemp_UNO_10Boards_Temp"
            / "Resp_UNO_AmbientTemp_nMeas15.npy",
            dataset_root
            / "CollectedDataAcrossTemperature_UNO_10Boards"
            / "GResp_UNO_temp_all.npy",
            np.r_[[-22.5, -20.0], np.arange(-18.5, 70.0, 2.5)],
        ),
        (
            "voltage",
            dataset_root
            / "RoomTemp_UNO_8Boards_Volt"
            / "Resp_UNO_AmbientVolt_nMeas15.npy",
            dataset_root
            / "CollectedDataAcrossVoltage_UNO_8Boards"
            / "GResp_UNO_Volt_all.npy",
            np.arange(3.8, 6.21, 0.1).round(1),
        ),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    rows: list[dict] = []
    for name, ambient, sweep, conditions in configs:
        missing = [str(path) for path in (ambient, sweep) if not path.exists()]
        if missing:
            raise FileNotFoundError("missing dataset files: " + ", ".join(missing))
        summary, condition_rows = _condition_records(
            name, ambient, sweep, conditions, extractor, rng
        )
        summaries.append(summary)
        rows.extend(condition_rows)

    with (output_dir / "puf_fuzzy_extractor_trials.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "dataset": "Pratihar et al. Arduino UNO SRAM PUF corpus",
        "extractor": {
            "key_bits": extractor.secret_bytes * 8,
            "bch_m": extractor.bch.m,
            "bch_t": extractor.bch.t,
            "bch_ecc_bits": extractor.bch.ecc_bits,
            "repetition": extractor.repetition,
            "puf_bits_consumed": extractor.required_puf_bits,
            "helper_data_bytes": extractor.required_puf_bits // 8,
            "selection_index_bytes": extractor.required_puf_bits * 2,
        },
        "results": summaries,
    }
    (output_dir / "puf_fuzzy_extractor_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/puf_fuzzy_extractor")
    )
    args = parser.parse_args()
    print(json.dumps(benchmark(args.dataset_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

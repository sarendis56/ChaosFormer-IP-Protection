"""Wav2Vec2/LibriSpeech effectiveness and overhead evaluation."""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.encryption.transformer_cipher import TransformerCipher


@dataclass
class ASREvaluation:
    word_error_rate: float
    elapsed_seconds: float
    utterances: int
    finite_logits_fraction: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _read_audio(audio: dict) -> tuple[np.ndarray, int]:
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
    samples, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32), int(sampling_rate)


def prepare_batches(
    processor: Wav2Vec2Processor,
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_samples: int,
    batch_size: int,
    dataset_file: Path | None = None,
) -> tuple[list[dict], dict]:
    if dataset_file is not None:
        dataset = load_dataset(
            "parquet", data_files={split: str(dataset_file)}, split=split
        )
    elif dataset_name == "openslr/librispeech_asr":
        # Address only the requested parquet. Loading the repository dataset
        # builder eagerly downloads every split in the selected configuration.
        parquet_url = (
            f"hf://datasets/{dataset_name}/{dataset_config}/{split}/0000.parquet"
        )
        dataset = load_dataset(
            "parquet", data_files={split: parquet_url}, split=split
        )
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    dataset = dataset.cast_column("audio", Audio(decode=False))

    records: list[tuple[np.ndarray, str]] = []
    total_seconds = 0.0
    for example in dataset:
        samples, sampling_rate = _read_audio(example["audio"])
        if sampling_rate != 16_000:
            raise ValueError(f"expected 16 kHz audio, found {sampling_rate}")
        records.append((samples, example["text"]))
        total_seconds += len(samples) / sampling_rate

    # Length bucketing reduces padding while preserving corpus-level WER.
    records.sort(key=lambda record: len(record[0]))
    batches = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        encoded = processor(
            [record[0] for record in chunk],
            sampling_rate=16_000,
            padding=True,
            return_tensors="pt",
        )
        batches.append(
            {
                "inputs": dict(encoded),
                "references": [record[1] for record in chunk],
            }
        )
    metadata = {
        "utterances": len(records),
        "audio_seconds": total_seconds,
        "batches": len(batches),
    }
    return batches, metadata


@torch.inference_mode()
def evaluate_asr(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    batches: list[dict],
    device: torch.device,
) -> ASREvaluation:
    predictions: list[str] = []
    references: list[str] = []
    finite = logits_count = 0
    model_dtype = next(model.parameters()).dtype

    _sync(device)
    start = time.perf_counter()
    for batch in batches:
        inputs = {}
        for name, value in batch["inputs"].items():
            if name == "input_values":
                inputs[name] = value.to(device=device, dtype=model_dtype)
            else:
                inputs[name] = value.to(device=device)
        logits = model(**inputs).logits
        finite += int(torch.isfinite(logits).sum())
        logits_count += logits.numel()
        token_ids = logits.argmax(dim=-1).cpu()
        predictions.extend(processor.batch_decode(token_ids))
        references.extend(batch["references"])
    _sync(device)

    return ASREvaluation(
        word_error_rate=float(wer(references, predictions)),
        elapsed_seconds=time.perf_counter() - start,
        utterances=len(references),
        finite_logits_fraction=finite / logits_count,
    )


def evaluate_repeated(
    model,
    processor,
    batches,
    device: torch.device,
    repeats: int,
) -> tuple[ASREvaluation, list[float]]:
    runs = [evaluate_asr(model, processor, batches, device) for _ in range(repeats)]
    first = runs[0]
    for run in runs[1:]:
        if not math.isclose(run.word_error_rate, first.word_error_rate, rel_tol=1e-12):
            raise RuntimeError("WER changed across timing repeats")
        if run.finite_logits_fraction != first.finite_logits_fraction:
            raise RuntimeError("finite-logit fraction changed across timing repeats")
    elapsed = [run.elapsed_seconds for run in runs]
    return (
        ASREvaluation(
            first.word_error_rate,
            statistics.median(elapsed),
            first.utterances,
            first.finite_logits_fraction,
        ),
        elapsed,
    )


def selected_layers(total: int, layer_count: str) -> list[int] | None:
    if layer_count == "all":
        return None
    count = int(layer_count)
    if not 1 <= count <= total:
        raise ValueError(f"layers must be between 1 and {total}")
    if count == 1:
        return [total // 2]
    return torch.linspace(0, total - 1, count).round().int().tolist()


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_source = str(args.model_path) if args.model_path else args.model
    processor = Wav2Vec2Processor.from_pretrained(model_source)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_source, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    batches, dataset_metadata = prepare_batches(
        processor,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.samples,
        batch_size=args.batch_size,
        dataset_file=args.dataset_file,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warm up CUDA and lazy kernels before timing.
    evaluate_asr(model, processor, batches[:1], device)
    baseline, baseline_trials = evaluate_repeated(
        model, processor, batches, device, args.timing_repeats
    )

    total_layers = len(model.wav2vec2.encoder.layers)
    selected = selected_layers(total_layers, args.layers)
    cipher = TransformerCipher(
        model,
        secure=args.mode == "secure",
        selected_layers=selected,
    )

    _sync(device)
    start = time.perf_counter()
    cipher.encrypt_all()
    encryption_seconds = time.perf_counter() - start
    unauthorized = evaluate_asr(model, processor, batches, device)

    with cipher.authorized_inference():
        authorized, authorized_trials = evaluate_repeated(
            model, processor, batches, device, args.timing_repeats
        )

    _sync(device)
    start = time.perf_counter()
    cipher.decrypt_all()
    decryption_seconds = time.perf_counter() - start

    return {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "mode": args.mode,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_layers": total_layers,
        "encrypted_layers": total_layers if selected is None else len(selected),
        "encrypted_parameters": cipher.encrypted_parameter_count,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "dataset_metadata": dataset_metadata,
        "baseline": asdict(baseline),
        "unauthorized": asdict(unauthorized),
        "authorized": asdict(authorized),
        "timing_repeats": args.timing_repeats,
        "baseline_elapsed_trials": baseline_trials,
        "authorized_elapsed_trials": authorized_trials,
        "authorized_overhead_ratio": authorized.elapsed_seconds / baseline.elapsed_seconds,
        "encryption_seconds": encryption_seconds,
        "decryption_seconds": decryption_seconds,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="optional local checkpoint path while retaining --model as the result label",
    )
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="optional local parquet for the requested dataset split",
    )
    parser.add_argument("--mode", choices=["base", "secure"], default="base")
    parser.add_argument("--layers", default="6")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timing_repeats < 1:
        parser.error("batch size and timing repeats must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

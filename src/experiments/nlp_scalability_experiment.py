"""NLP effectiveness and billion-parameter scalability experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.encryption.transformer_cipher import TransformerCipher


@dataclass
class Evaluation:
    metric_name: str
    metric_value: float
    elapsed_seconds: float
    samples: int
    finite_logits_fraction: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sst2_batches(tokenizer, max_samples: int, batch_size: int) -> Iterator[dict]:
    dataset = load_dataset("glue", "sst2", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def collate(examples):
        batch = tokenizer(
            [example["sentence"] for example in examples],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([example["label"] for example in examples])
        return batch

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


def _causal_batches(
    tokenizer, max_sequences: int, sequence_length: int
) -> list[dict]:
    dataset = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="test", streaming=True
    )
    token_ids: list[int] = []
    required = max_sequences * sequence_length + 1
    for example in dataset:
        text = example["text"]
        if not text.strip():
            continue
        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(token_ids) >= required:
            break
    if len(token_ids) < required:
        raise RuntimeError("WikiText stream did not provide enough tokens")
    batches = []
    for offset in range(0, max_sequences * sequence_length, sequence_length):
        ids = torch.tensor(token_ids[offset : offset + sequence_length + 1])
        batches.append(
            {
                "input_ids": ids[:-1].unsqueeze(0),
                "labels": ids[1:].unsqueeze(0),
            }
        )
    return batches


@torch.inference_mode()
def evaluate_sst2(model, batches, device: torch.device) -> Evaluation:
    correct = total = finite = logits_count = 0
    _sync(device)
    start = time.perf_counter()
    for batch in batches:
        labels = batch["labels"].to(device)
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }
        logits = model(**inputs).logits
        finite += int(torch.isfinite(logits).sum())
        logits_count += logits.numel()
        correct += int((logits.argmax(dim=-1) == labels).sum())
        total += labels.numel()
    _sync(device)
    return Evaluation(
        "accuracy",
        correct / total,
        time.perf_counter() - start,
        total,
        finite / logits_count,
    )


@torch.inference_mode()
def evaluate_causal(model, batches, device: torch.device) -> Evaluation:
    loss_sum = tokens = finite = logits_count = 0
    _sync(device)
    start = time.perf_counter()
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        finite_mask = torch.isfinite(logits)
        finite += int(finite_mask.sum())
        logits_count += logits.numel()
        # Replace non-finite logits only to make the unauthorized failure metric
        # numerically reportable. Its finite fraction remains reported separately.
        safe_logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        loss = F.cross_entropy(
            safe_logits.reshape(-1, safe_logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        loss_sum += float(loss)
        tokens += labels.numel()
    _sync(device)
    mean_loss = loss_sum / tokens
    perplexity = math.exp(min(mean_loss, 80.0))
    return Evaluation(
        "perplexity",
        perplexity,
        time.perf_counter() - start,
        len(batches),
        finite / logits_count,
    )


def _selected_layers(total: int, layer_count: str) -> list[int] | None:
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.task == "sst2":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, dtype=dtype
        ).to(device)
        batches = list(_sst2_batches(tokenizer, args.samples, args.batch_size))
        evaluator = evaluate_sst2
    else:
        if args.random_init:
            config = AutoConfig.from_pretrained(args.model)
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                with torch.device(device):
                    model = AutoModelForCausalLM.from_config(config)
            finally:
                torch.set_default_dtype(previous_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=dtype, low_cpu_mem_usage=True
            ).to(device)
        batches = _causal_batches(tokenizer, args.samples, args.sequence_length)
        evaluator = evaluate_causal
    model.eval()

    # Warm up lazy CUDA kernels without including compilation in measurements.
    warmup = batches[:1]
    evaluator(model, warmup, device)
    baseline = evaluator(model, batches, device)

    if hasattr(model, "roberta"):
        total_layers = len(model.roberta.encoder.layer)
    else:
        total_layers = len(model.model.decoder.layers)
    selected = _selected_layers(total_layers, args.layers)
    cipher = TransformerCipher(
        model,
        secure=args.mode == "secure",
        selected_layers=selected,
    )

    _sync(device)
    start = time.perf_counter()
    cipher.encrypt_all()
    encryption_seconds = time.perf_counter() - start
    unauthorized = evaluator(model, batches, device)

    with cipher.authorized_inference():
        authorized = evaluator(model, batches, device)

    _sync(device)
    start = time.perf_counter()
    cipher.decrypt_all()
    decryption_seconds = time.perf_counter() - start

    result = {
        "model": args.model,
        "task": args.task,
        "mode": args.mode,
        "random_init": args.random_init,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_layers": total_layers,
        "encrypted_layers": (
            total_layers if selected is None else len(selected)
        ),
        "encrypted_parameters": cipher.encrypted_parameter_count,
        "dtype": str(dtype),
        "sequence_length": args.sequence_length if args.task == "causal" else 128,
        "baseline": asdict(baseline),
        "unauthorized": asdict(unauthorized),
        "authorized": asdict(authorized),
        "authorized_overhead_ratio": (
            authorized.elapsed_seconds / baseline.elapsed_seconds
        ),
        "encryption_seconds": encryption_seconds,
        "decryption_seconds": decryption_seconds,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["sst2", "causal"], required=True)
    parser.add_argument("--mode", choices=["base", "secure"], default="secure")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="instantiate the official architecture without pretrained weights; "
        "intended only for runtime and memory scaling measurements",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

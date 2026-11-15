"""Retraining attacks on a protected RoBERTa classifier (SST-2 or 20 Newsgroups).

The attacker holds the protected checkpoint and 20% of the task's training set
(a 10% slice of that is held out for learning-rate selection) and fine-tunes all
parameters. Accuracy is reported on the SST-2 validation set or the 20NG test set.
The 20NG victim is produced first with --setup victim (full training set).

Setups:
  origin           unprotected victim checkpoint (upper bound)
  randinit         same architecture, random initialization
  public           public roberta-base + new head (extended-threat baseline)
  base_k6          ChaosFormer, 6 layers encrypted
  secure_all       ChaosFormer-S, all 12 layers; attacker uses the same
                   checkpoint stabilization as the ViT pipeline
  secure_all_reinit ChaosFormer-S, all 12 layers; attacker re-initializes the
                   encrypted matrices and keeps the plaintext remainder
  base_k6_reinit   ChaosFormer, 6 layers; same re-initialization attack

  uv run python -m src.experiments.roberta_retraining_attack --task 20ng --setup base_k6_reinit --lr 3e-5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.attacks.train import _stabilize_advanced_checkpoint_
from src.encryption.transformer_cipher import TransformerCipher

PUBLIC = "FacebookAI/roberta-base"
# task -> (victim checkpoint, dataset, config, text field, eval split, max length, classes)
TASKS = {
    "sst2": ("textattack/roberta-base-SST-2", "glue", "sst2", "sentence", "validation", 128, 2),
    "20ng": ("results/victims/roberta-base-20ng", "SetFit/20_newsgroups", None,
             "text", "test", 256, 20),
}
BASE_LAYERS = [0, 2, 4, 7, 9, 11]  # same as the K=6 effectiveness row (linspace)
SETUPS = ["victim", "origin", "randinit", "public", "base_k6", "base_k6_reinit", "secure_all", "secure_all_reinit"]


def build_model(setup: str, seed: int, task: str):
    victim, num_labels = TASKS[task][0], TASKS[task][6]
    torch.manual_seed(seed)
    if setup in ("victim", "public"):  # the victim is trained from the public checkpoint
        return AutoModelForSequenceClassification.from_pretrained(PUBLIC, num_labels=num_labels)
    if setup == "randinit":
        return AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(victim))
    model = AutoModelForSequenceClassification.from_pretrained(victim, dtype=torch.float32)
    if setup == "origin":
        return model
    secure = setup.startswith("secure")
    cipher = TransformerCipher(model, secure=secure, selected_layers=None if secure else BASE_LAYERS)
    cipher.encrypt_all()
    if setup == "secure_all":
        _stabilize_advanced_checkpoint_(model, logging.getLogger(__name__))
    elif setup.endswith("_reinit"):
        std = model.config.initializer_range
        with torch.no_grad():
            for layer in cipher.layer_specs:
                for spec in layer.weights:
                    spec.parameter.normal_(0.0, std)
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--setup", required=True, choices=SETUPS)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--task", choices=list(TASKS), default="sst2")
    p.add_argument("--output-dir", type=Path, help="default: results/roberta_retraining/<task>")
    args = p.parse_args()
    args.output_dir = args.output_dir or Path("results/roberta_retraining") / args.task
    device = torch.device(args.device)

    victim, dataset, config, field, eval_split, max_len, _ = TASKS[args.task]
    tokenizer = AutoTokenizer.from_pretrained(PUBLIC)
    data = load_dataset(dataset, config) if config else load_dataset(dataset)
    # victim = model owner trains on the full training set; attacker gets a fixed fraction
    fraction = 1.0 if args.setup == "victim" else args.fraction
    attacker = data["train"].shuffle(seed=1234).select(range(int(fraction * len(data["train"]))))
    split = attacker.train_test_split(test_size=0.1, seed=1234)

    def loader(ds, shuffle):
        def collate(rows):
            enc = tokenizer([r[field] for r in rows], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc["labels"] = torch.tensor([r["label"] for r in rows])
            return enc
        g = torch.Generator().manual_seed(args.seed)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate, generator=g)

    train_loader = loader(split["train"], True)
    dev_loader = loader(split["test"], False)
    val_loader = loader(data[eval_split], False)

    model = build_model(args.setup, args.seed, args.task).to(device)
    nonfinite = sum(int(not torch.isfinite(p).all()) for p in model.parameters())
    initial_val = evaluate(model, val_loader, device) if nonfinite == 0 else float("nan")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    torch.manual_seed(args.seed)
    history = []
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch + 1,
                        "dev": evaluate(model, dev_loader, device),
                        "val": evaluate(model, val_loader, device)})
        print(args.setup, args.lr, args.seed, history[-1], flush=True)

    if args.setup == "victim":  # final-epoch victim: the owner's recipe, not attacker-tuned
        model.save_pretrained(victim)
        tokenizer.save_pretrained(victim)
    best = max(history, key=lambda h: h["dev"])
    out = {**vars(args), "output_dir": str(args.output_dir), "initial_val": initial_val,
           "nonfinite_tensors_at_start": nonfinite, "attacker_train": len(split["train"]),
           "attacker_dev": len(split["test"]), "history": history,
           "best_dev": best["dev"], "val_at_best_dev": best["val"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.setup}_lr{args.lr:g}_seed{args.seed}.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
import random
from tqdm import tqdm
from datetime import datetime
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
import argparse
import json
from typing import Optional, Tuple

# Add project root to sys.path to allow importing from src (namespace package)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.dataset_utils import get_dataset

def _resolve_attack_dir(model_path: str, attacks_dir: str = "results/attacks") -> str:
    """Resolve model_path to a concrete attack directory.
    If model_path is 'original', return as-is. If it is an existing path, return it.
    Otherwise treat model_path as a prefix and return the latest results/attacks/<prefix>* directory.
    """
    if model_path == "original":
        return model_path
    p = Path(model_path)
    if p.exists() and p.is_dir():
        return str(p.resolve())
    # Treat as prefix: find latest directory under attacks_dir that starts with this prefix
    attacks_path = Path(attacks_dir)
    if not attacks_path.exists():
        raise FileNotFoundError(f"Attacks directory not found: {attacks_path}")
    matching = sorted(
        (d for d in attacks_path.iterdir() if d.is_dir() and d.name.startswith(model_path)),
        key=lambda x: x.name,
    )
    if not matching:
        raise FileNotFoundError(
            f"No attack directory matching prefix '{model_path}' in {attacks_path}. "
            "Use a full path or a prefix like attack_deit_basic_top-k_k6_rate0.2"
        )
    return str(matching[-1].resolve())


def fine_tune(model_path, model_name, dataset_name, epoch_num, lr_rate, weight_decay, device, batch_size=64,
              attacks_dir="results/attacks"):
    # Resolve attack directory by prefix if needed
    resolved_path = _resolve_attack_dir(model_path, attacks_dir=attacks_dir)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resolved_path == "original":
        setup_name = "original"
    else:
        setup_name = Path(resolved_path).name  # e.g. attack_deit_basic_top-k_k6_rate0.2_timestamp

    experiment_name = f"finetune_{dataset_name}_{setup_name}_{timestamp}"
    results_dir = Path("results/finetune") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = results_dir / "finetune.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting fine-tuning on {dataset_name}")
    logger.info(f"Model source: {resolved_path}")

    # Use stronger defaults when source checkpoint was produced by random-init pretraining.
    is_randinit_source = (resolved_path != "original") and Path(resolved_path).name.startswith("randinit_")
    if epoch_num is None:
        epoch_num = 30 if is_randinit_source else 10
    if lr_rate is None:
        lr_rate = 2e-4 if is_randinit_source else 1e-4
    if weight_decay is None:
        weight_decay = 0.05
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    logger.info(
        f"Fine-tune hparams: epochs={epoch_num}, lr={lr_rate}, wd={weight_decay}, "
        f"batch_size={batch_size}, randinit_source={is_randinit_source}"
    )

    device_obj = torch.device(device)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    # Load model
    if resolved_path == "original":
        logger.info("Loading original pretrained model...")
        model = AutoModelForImageClassification.from_pretrained(model_name)
    else:
        logger.info(f"Loading retrained model from {resolved_path}...")
        model = AutoModelForImageClassification.from_pretrained(model_name)
        state_dict_path = Path(resolved_path) / "best_model.pth"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Could not find best_model.pth in {resolved_path}")
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

    # Prepare dataset
    # Dataset-specific args are parsed in __main__ and passed via globals on purpose to keep changes localized
    train_set, test_set, num_classes = get_dataset(
        dataset_name,
        processor,
        imagenet100_path=getattr(fine_tune, "_imagenet100_path", None),
        officehome_source=getattr(fine_tune, "_officehome_source", None),
        officehome_target=getattr(fine_tune, "_officehome_target", None),
        domainnet_domain=getattr(fine_tune, "_domainnet_domain", "clipart"),
    )
    
    # Replace the head for the downstream task
    if model.num_labels != num_classes:
        logger.info(f"Replacing classifier head: {model.num_labels} -> {num_classes}")
        from src.utils.vision_backbone_utils import replace_classifier_head
        replace_classifier_head(model, num_classes)
        model.num_labels = num_classes

    model.to(device_obj)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    total_steps = epoch_num * len(train_loader)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_acc = 0.0
    
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_num}"):
            images, labels = images.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device_obj), labels.to(device_obj)
                outputs = model(images).logits
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Test Acc={acc:.2f}%, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if acc > best_acc:
            best_acc = acc

    logger.info(f"Fine-tuning complete. Best Test Accuracy: {best_acc:.2f}%")
    
    # Save final results summary
    results = {
        'model_path': resolved_path,
        'dataset': dataset_name,
        'best_acc': best_acc,
        'epochs': epoch_num,
        'lr': lr_rate,
    }
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a retrained model on downstream tasks.')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the attack results directory (containing best_model.pth) or "original"')
    parser.add_argument('--dataset', type=str,
                        choices=['cifar100', 'caltech101', 'caltech256', 'gtsrb', 'cub200',
                                 'imagenet100', 'officehome', 'domainnet_clipart'],
                        required=True,
                        help='Downstream dataset to fine-tune on')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224',
                        help='Base model architecture')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Number of fine-tuning epochs (auto: 30 for randinit source, else 10)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (auto: 1e-4 for randinit source, else 3e-5)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--imagenet100_path', type=str, default=None,
                        help='Path to ImageNet-100 root (expects train/ and val/ or test/ subfolders)')
    parser.add_argument('--officehome_source', type=str, default=None,
                        help='OfficeHome source domain for training (art, clipart, product, realworld)')
    parser.add_argument('--officehome_target', type=str, default=None,
                        help='OfficeHome target domain for testing (art, clipart, product, realworld)')
    parser.add_argument('--domainnet_domain', type=str, default='clipart',
                        help='DomainNet domain to filter (default: clipart)')
    parser.add_argument('--attacks_dir', type=str, default='results/attacks',
                        help='Directory containing attack result dirs (used when --model_path is a prefix)')

    args = parser.parse_args()

    fine_tune._imagenet100_path = args.imagenet100_path
    fine_tune._officehome_source = args.officehome_source
    fine_tune._officehome_target = args.officehome_target
    fine_tune._domainnet_domain = args.domainnet_domain

    fine_tune(args.model_path, args.model_name, args.dataset, args.epoch, args.lr,
              args.weight_decay, args.device, args.batch_size, attacks_dir=args.attacks_dir)

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torchvision.models as models
from torch.utils.data import DataLoader
from typing import Any, Optional, cast
from collections.abc import Sized

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.dataset_utils import get_dataset

PRESETS = {
    "cifar100": {
        "model_name": "convnext_tiny",
        "epochs": 200,
        "lr": 1e-3,
        "batch_size": 128,
        "weight_decay": 0.05,
        "warmup_epochs": 20,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "adamw",
    },
    "caltech256": {
        "model_name": "convnext_tiny",
        "epochs": 150,
        "lr": 0.05,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "sgd",
    },
    "cub200": {
        "model_name": "convnext_tiny",
        "epochs": 200,
        "lr": 0.05,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "sgd",
    },
    "imagenet100": {
        "model_name": "convnext_tiny",
        "epochs": 200,
        "lr": 0.1,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "grad_accum": 2,
        "optimizer": "sgd",
    },
    "officehome": {
        "model_name": "convnext_tiny",
        "epochs": 150,
        "lr": 0.05,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "sgd",
    },
    "domainnet_clipart": {
        "model_name": "convnext_tiny",
        "epochs": 120,
        "lr": 0.1,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "grad_accum": 2,
        "optimizer": "sgd",
    },
}

FINETUNE_PRESETS = {
    # Fine-tuning defaults for downstream transfer from pretrain.py checkpoints.
    "cifar100": {
        "model_name": "convnext_tiny",
        "epochs": 30,
        "lr": 2e-4,
        "batch_size": 128,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "adamw",
    },
    "caltech256": {
        "model_name": "convnext_tiny",
        "epochs": 30,
        "lr": 2e-4,
        "batch_size": 128,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "adamw",
    },
    "cub200": {
        "model_name": "convnext_tiny",
        "epochs": 30,
        "lr": 2e-4,
        "batch_size": 64,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "adamw",
    },
    "imagenet100": {
        "model_name": "convnext_tiny",
        "epochs": 30,
        "lr": 2e-4,
        "batch_size": 128,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "label_smoothing": 0.1,
        "grad_accum": 2,
        "optimizer": "adamw",
    },
    "officehome": {
        "model_name": "convnext_tiny",
        "epochs": 30,
        "lr": 2e-4,
        "batch_size": 128,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "label_smoothing": 0.1,
        "grad_accum": 1,
        "optimizer": "adamw",
    },
    "domainnet_clipart": {
        "model_name": "convnext_tiny",
        "epochs": 30,
        "lr": 2e-4,
        "batch_size": 128,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "label_smoothing": 0.1,
        "grad_accum": 2,
        "optimizer": "adamw",
    },
}

ARCHITECTURE_PRESETS = {
    "efficientnet_b0": {
        # EfficientNet-B0 usually converges more stably with AdamW + smaller LR than SGD defaults.
        "scratch": {
            "optimizer": "adamw",
            "lr": 8e-4,
            "weight_decay": 0.02,
            "warmup_epochs": 10,
        },
        # After ImageNet pretraining, use a gentler fine-tuning regime.
        "pretrained": {
            "optimizer": "adamw",
            "lr": 2e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 3,
        },
    },
}


def _extract_state_dict_and_config(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
        if isinstance(state_dict, dict):
            return state_dict, config if isinstance(config, dict) else {}
    if isinstance(checkpoint, dict):
        # Fallback: raw state_dict saved directly.
        tensor_values = all(isinstance(v, torch.Tensor) for v in checkpoint.values())
        if tensor_values:
            return checkpoint, {}
    raise ValueError("Unsupported checkpoint format. Expected a dict with 'model_state_dict' or a raw state_dict.")


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Support DataParallel checkpoints that prefix parameter names with "module."
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_model_name_from_checkpoint(pretrained_path: Optional[str]) -> Optional[str]:
    if not pretrained_path:
        return None
    checkpoint_path = Path(pretrained_path)
    if not checkpoint_path.exists():
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        _, config = _extract_state_dict_and_config(checkpoint)
        model_name = config.get("model_name")
        return str(model_name) if model_name is not None else None
    except Exception:
        return None


def get_cnn_model(model_name: str, num_classes: int):
    """Initialize a CNN model from torchvision (random weights)."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        fc = cast(nn.Linear, model.fc)
        model.fc = nn.Linear(fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        fc = cast(nn.Linear, model.fc)
        model.fc = nn.Linear(fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(weights=None)
        fc = cast(nn.Linear, model.fc)
        model.fc = nn.Linear(fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        fc = cast(nn.Linear, model.classifier[6])
        model.classifier[6] = nn.Linear(fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        fc = cast(nn.Linear, model.classifier[1])
        model.classifier[1] = nn.Linear(fc.in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        fc = cast(nn.Linear, model.classifier[2])
        model.classifier[2] = nn.Linear(fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        fc = cast(nn.Linear, model.classifier[1])
        model.classifier[1] = nn.Linear(fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported CNN architecture: {model_name}")
    return model


def posttrain(
    dataset_name: str,
    model_name: str = "resnet50",
    epochs: int = 100,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    grad_accum: int = 1,
    optimizer_name: str = "sgd",
    amp: bool = True,
    pretrained_path: Optional[str] = None,
    device: str = "cuda",
    output_dir: str = "results/from_scratch",
    **dataset_kwargs
):
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_mode = "finetune" if pretrained_path else "scratch"
    experiment_name = f"{run_mode}_{dataset_name}_{model_name}_{timestamp}"
    results_dir = Path(output_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = results_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CNN training on {dataset_name}")
    logger.info(
        f"Model: {model_name}, Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}, "
        f"Accum: {grad_accum}, LabelSmooth: {label_smoothing}"
    )

    device_obj = torch.device(device)
    
    # Use shared default dataset transforms (same path as fine_tune.py).
    train_set, test_set, num_classes = get_dataset(
        dataset_name,
        None,
        **dataset_kwargs
    )
    train_set_sized = cast(Sized, train_set)
    test_set_sized = cast(Sized, test_set)
    logger.info(f"Dataset loaded. Classes: {num_classes}, Train size: {len(train_set_sized)}, Test size: {len(test_set_sized)}")

    # Initialize model
    if pretrained_path:
        logger.info(f"Loading pretrained model from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        pretrained_state_dict, ckpt_config = _extract_state_dict_and_config(checkpoint)
        pretrained_state_dict = _normalize_state_dict_keys(pretrained_state_dict)
        ckpt_model_name = ckpt_config.get("model_name")
        if ckpt_model_name and ckpt_model_name != model_name:
            logger.warning(
                f"Checkpoint model_name={ckpt_model_name} differs from requested model_name={model_name}. "
                "Proceeding with requested architecture and loading matching tensors only."
            )

        # Build target model and load matching tensors (backbone transfer; head may differ in class count).
        model = get_cnn_model(model_name, num_classes)
        model_state = model.state_dict()
        pretrained_state = {
            k: v for k, v in pretrained_state_dict.items()
            if k in model_state and v.size() == model_state[k].size()
        }

        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        logger.info(f"Loaded {len(pretrained_state)} tensors from checkpoint.")
        if missing:
            logger.info(f"Missing keys (expected when classifier shape differs): {missing}")
        if unexpected:
            logger.info(f"Unexpected keys ignored from checkpoint: {unexpected}")
    else:
        logger.info(f"Initializing {model_name} with {num_classes} classes (random weights)...")
        model = get_cnn_model(model_name, num_classes)
    
    model.to(device_obj)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Choose optimizer and adjust learning rate if needed
    # Adam/AdamW typically need 10-50x smaller learning rates than SGD
    effective_lr = lr
    opt_lower = optimizer_name.lower()
    if opt_lower == "adamw":
        if lr > 1e-3:
            # Scale down learning rate for AdamW if it's too high
            effective_lr = max(1e-3, lr / 50.0)
            logger.warning(
                f"Learning rate {lr:.2e} is too high for AdamW. Automatically scaling down to {effective_lr:.2e} "
                f"to prevent training instability. AdamW typically needs 10-50x smaller learning rates than SGD."
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)
    elif opt_lower == "adam":
        if lr > 1e-3:
            # Scale down learning rate for Adam if it's too high
            effective_lr = max(1e-3, lr / 50.0)
            logger.warning(
                f"Learning rate {lr:.2e} is too high for Adam. Automatically scaling down to {effective_lr:.2e} "
                f"to prevent training instability. Adam typically needs 10-50x smaller learning rates than SGD."
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=weight_decay)
    else:
        # Standard SGD with momentum is often better for training CNNs from scratch
        optimizer = torch.optim.SGD(model.parameters(), lr=effective_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    
    if effective_lr != lr:
        logger.info(f"Using effective learning rate: {effective_lr:.2e} (requested: {lr:.2e})")
    logger.info(f"Optimizer: {optimizer_name}, Effective LR: {effective_lr:.2e}, Weight Decay: {weight_decay}")
    
    use_amp = amp and device_obj.type == "cuda"
    scaler = torch.GradScaler(device_obj.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    
    # Cosine schedule with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        step = -1
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(device_obj), labels.to(device_obj)
            with torch.autocast(device_type=device_obj.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss = loss / max(1, grad_accum)
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * max(1, grad_accum)
            pbar.set_postfix({'loss': loss.item() * max(1, grad_accum)})

        if step >= 0 and (step + 1) % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        epoch_loss = running_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device_obj), labels.to(device_obj)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Test Acc={acc:.2f}%, LR={current_lr:.2e}")
        
        scheduler.step()
        
        history.append({'epoch': epoch+1, 'loss': epoch_loss, 'acc': acc})
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            logger.info(f"New best accuracy: {best_acc:.2f}%")

    logger.info(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    
    # Save results
    results = {
        'config': {
            'dataset': dataset_name,
            'model': model_name,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'label_smoothing': label_smoothing,
            'grad_accum': grad_accum,
            'optimizer': optimizer_name,
            'amp': amp,
            'pretrained_path': pretrained_path,
        },
        'best_acc': best_acc,
        'history': history
    }
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-train a CNN (fine-tune when --pretrained_path is provided).')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar100', 'caltech256', 'cub200', 'imagenet100', 'officehome', 'domainnet_clipart'],
                        help='Dataset to train on')
    parser.set_defaults(use_preset=True)
    parser.add_argument('--no-preset', dest='use_preset', action='store_false',
                        help='Disable dataset-specific training preset')
    parser.add_argument('--model_name', type=str, default=None,
                        choices=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'mobilenet_v2', 'convnext_tiny', 'efficientnet_b0'],
                        help='CNN architecture (overrides preset)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default for SGD)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='Warmup epochs')
    parser.add_argument('--label_smoothing', type=float, default=None, help='Label smoothing')
    parser.add_argument('--grad_accum', type=int, default=None, help='Gradient accumulation steps')
    parser.add_argument('--optimizer', type=str, default=None, choices=['sgd', 'adam', 'adamw'], help='Optimizer to use')
    parser.add_argument('--amp', action='store_true', default=True, help='Enable AMP')
    parser.add_argument('--no-amp', dest='amp', action='store_false', help='Disable AMP')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Dataset specific
    parser.add_argument('--imagenet100_path', type=str, default=None)
    parser.add_argument('--officehome_source', type=str, default=None)
    parser.add_argument('--officehome_target', type=str, default=None)
    parser.add_argument('--domainnet_domain', type=str, default='clipart')
    
    args = parser.parse_args()

    requested_model_name = args.model_name
    inferred_model_name = infer_model_name_from_checkpoint(args.pretrained_path)
    model_name = requested_model_name or inferred_model_name
    if model_name is None:
        model_name = PRESETS.get(args.dataset, {}).get("model_name", "resnet50")

    effective = {}
    phase = "pretrained" if args.pretrained_path else "scratch"
    if args.use_preset:
        if phase == "pretrained":
            effective.update(FINETUNE_PRESETS.get(args.dataset, {}))
        else:
            effective.update(PRESETS.get(args.dataset, {}))
        arch_preset = ARCHITECTURE_PRESETS.get(model_name, {}).get(phase, {})
        effective.update(arch_preset)

    # Apply CLI overrides after presets.
    cli_overrides = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "label_smoothing": args.label_smoothing,
        "grad_accum": args.grad_accum,
        "optimizer": args.optimizer,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            effective[key] = value

    effective["model_name"] = effective.get("model_name", model_name)
    if phase == "pretrained":
        effective.setdefault("epochs", 30)
        effective.setdefault("lr", 2e-4)
        effective.setdefault("batch_size", 128)
        effective.setdefault("weight_decay", 0.05)
        effective.setdefault("warmup_epochs", 3)
        effective.setdefault("label_smoothing", 0.1)
        effective.setdefault("grad_accum", 1)
        effective.setdefault("optimizer", "adamw")
    else:
        effective.setdefault("epochs", 100)
        effective.setdefault("lr", 0.1)
        effective.setdefault("batch_size", 128)
        effective.setdefault("weight_decay", 1e-4)
        effective.setdefault("warmup_epochs", 5)
        effective.setdefault("label_smoothing", 0.1)
        effective.setdefault("grad_accum", 1)
        effective.setdefault("optimizer", "sgd")

    posttrain(
        dataset_name=args.dataset,
        model_name=effective["model_name"],
        epochs=effective["epochs"],
        lr=effective["lr"],
        weight_decay=effective["weight_decay"],
        batch_size=effective["batch_size"],
        warmup_epochs=effective["warmup_epochs"],
        label_smoothing=effective["label_smoothing"],
        grad_accum=effective["grad_accum"],
        optimizer_name=effective["optimizer"],
        amp=args.amp,
        pretrained_path=args.pretrained_path,
        device=args.device,
        imagenet100_path=args.imagenet100_path,
        officehome_source=args.officehome_source,
        officehome_target=args.officehome_target,
        domainnet_domain=args.domainnet_domain
    )

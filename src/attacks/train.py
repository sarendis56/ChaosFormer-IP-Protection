import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import logging
from tqdm import tqdm
from datetime import datetime
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from src.attacks.ImageNetEval import ImageNetValidationDataset, load_synset_mapping, validate_model
import argparse
import json


class AugmentedImageNetDataset(Dataset):
    def __init__(self, root_dir, processor, augment=True, augment_strength='medium'):
        """
        Args:
            root_dir: Path to ImageNet training directory
            processor: HuggingFace image processor
            augment: Whether to apply data augmentation
            augment_strength: Strength of augmentation ('light', 'medium', 'strong')
        """
        self.root_dir = root_dir
        self.processor = processor
        self.augment = augment

        # Load ImageNet training data
        self.imagenet_dataset = ImageFolder(root=root_dir)

        # Define augmentation transforms based on strength
        if augment:
            if augment_strength == 'light':
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                ])
            elif augment_strength == 'medium':
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                ])
            elif augment_strength == 'strong':
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandAugment(num_ops=2, magnitude=9),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ])
            else:
                raise ValueError(f"Unknown augment_strength: {augment_strength}")
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.imagenet_dataset)

    def __getitem__(self, idx):
        image, label = self.imagenet_dataset[idx]

        # Apply augmentation if enabled
        if self.augment and self.augment_transform is not None:
            image = self.augment_transform(image)

        # Process with HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        return pixel_values, label


def _model_prefix_for_setup(model_name: str) -> str:
    """Return 'deit_', 'vit_', or '' for directory matching (same as experiment naming)."""
    if not model_name:
        return ""
    name_lower = model_name.lower()
    if "deit" in name_lower:
        return "deit_"
    if "vit" in name_lower:
        return "vit_"
    return ""


def _configure_attack_logger(log_file: Path) -> logging.Logger:
    """Create per-run logger that logs to file + stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def find_latest_checkpoint(results_dir, setup_prefix, model_name=None):
    """Find the latest checkpoint directory matching the setup prefix.
    Matches both legacy names (e.g. basic_top-k_k6_...) and model-prefixed names
    (e.g. deit_basic_top-k_k6_..., vit_basic_top-k_k6_...) when model_name is given.
    """
    results_path = Path(results_dir)
    # Prefixes to match: user's setup and, if model given, model-prefixed version
    prefixes = [setup_prefix]
    if model_name:
        prefix = _model_prefix_for_setup(model_name)
        if prefix:
            prefixes.append(prefix + setup_prefix)
    matching_dirs = []

    for d in results_path.iterdir():
        if not d.is_dir():
            continue
        if not any(d.name.startswith(p) for p in prefixes):
            continue
        checkpoint_path = d / "checkpoints" / "final" / "model"
        if checkpoint_path.exists():
            matching_dirs.append(d)

    if not matching_dirs:
        return None

    matching_dirs.sort(key=lambda x: x.name)
    return matching_dirs[-1] / "checkpoints" / "final" / "model"


def _infer_model_scale(model_name: str) -> str:
    """Infer a rough model scale from the HF id (tiny/small/base)."""
    name = (model_name or "").lower()
    if "tiny" in name:
        return "tiny"
    if "small" in name:
        return "small"
    if "base" in name:
        return "base"
    return "unknown"

def _is_advanced_setup_name(setup_name: str | None) -> bool:
    if not setup_name:
        return False
    return "advanced" in setup_name.lower()


def _strip_timestamp_suffix(setup_name: str) -> str:
    """Strip trailing _YYYYMMDD_HHMMSS when present, otherwise keep full name."""
    parts = setup_name.split("_")
    if (
        len(parts) >= 3
        and len(parts[-2]) == 8
        and len(parts[-1]) == 6
        and parts[-2].isdigit()
        and parts[-1].isdigit()
    ):
        return "_".join(parts[:-2])
    return setup_name


def _auto_hparams_for_model(
    model_name: str,
    rand_init: bool = False,
    scratch_recipe: bool = False,
    advanced_setup: bool = False,
) -> dict:
    """Default hparams by model scale and training mode.

    - rand_init=False: conservative defaults for encrypted checkpoints.
    - rand_init=True or scratch_recipe=True: stronger defaults for training from scratch.
    """
    scale = _infer_model_scale(model_name)
    if rand_init:
        # ViT/DeiT scratch training generally needs much larger LR than fine-tuning.
        if scale == "tiny":
            return {"lr": 3e-4, "weight_decay": 5e-2, "batch_size": 256, "grad_clip": 1.0}
        if scale == "small":
            return {"lr": 2e-4, "weight_decay": 5e-2, "batch_size": 192, "grad_clip": 1.0}
        if scale == "base":
            return {"lr": 2e-4, "weight_decay": 5e-2, "batch_size": 128, "grad_clip": 1.0}
        return {"lr": 1e-4, "weight_decay": 5e-2, "batch_size": 128, "grad_clip": 1.0}
    if advanced_setup or scratch_recipe:
        # Fully-noised advanced checkpoints are less stable than clean scratch init.
        # Use empirically stable defaults for advanced mode (still override-able via CLI).
        if scale == "tiny":
            return {"lr": 1e-4, "weight_decay": 5e-2, "batch_size": 256, "grad_clip": 0.5}
        if scale == "small":
            return {"lr": 7e-5, "weight_decay": 5e-2, "batch_size": 192, "grad_clip": 0.5}
        if scale == "base":
            return {"lr": 3e-5, "weight_decay": 5e-2, "batch_size": 128, "grad_clip": 0.3}
        return {"lr": 3e-5, "weight_decay": 5e-2, "batch_size": 128, "grad_clip": 0.3}

    if scale == "tiny":
        return {"lr": 5e-5, "weight_decay": 1e-3, "batch_size": 128, "grad_clip": 1.0}
    if scale == "small":
        return {"lr": 1e-5, "weight_decay": 1e-3, "batch_size": 64, "grad_clip": 1.0}
    if scale == "base":
        return {"lr": 5e-6, "weight_decay": 1e-3, "batch_size": 32, "grad_clip": 1.0}
    return {"lr": 1e-5, "weight_decay": 1e-3, "batch_size": 64, "grad_clip": 1.0}


def _auto_recipe_for_model(
    model_name: str,
    rand_init: bool = False,
    scratch_recipe: bool = False,
    advanced_setup: bool = False,
) -> dict:
    """Default data/loss regularization recipe by mode."""
    scale = _infer_model_scale(model_name)
    if rand_init:
        # Scratch ViT/DeiT on subset data benefits significantly from stronger regularization.
        if scale in {"tiny", "small", "base"}:
            return {
                "augment_strength": "strong",
                "label_smoothing": 0.1,
                "mixup_alpha": 0.8,
                "cutmix_alpha": 1.0,
                "mixup_prob": 1.0,
            }
        return {
            "augment_strength": "strong",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.4,
            "cutmix_alpha": 0.5,
            "mixup_prob": 0.8,
        }
    if advanced_setup or scratch_recipe:
        # Encrypted advanced checkpoints can still benefit from mild regularization.
        return {
            "augment_strength": "medium",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.2,
            "cutmix_alpha": 0.5,
            "mixup_prob": 0.5,
        }
    return {
        "augment_strength": "medium",
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "mixup_prob": 0.0,
    }


def _auto_epochs_for_model(model_name: str, rand_init: bool = False) -> int:
    """Default epoch count by model scale/training mode."""
    scale = _infer_model_scale(model_name)
    if rand_init:
        if scale in {"small", "base"}:
            return 30
        if scale == "tiny":
            return 25
        return 20
    return 20


def _has_nonfinite_gradients(model: torch.nn.Module) -> bool:
    """Return True if any parameter gradient is NaN/Inf."""
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


def _sanitize_gradients_(model: torch.nn.Module, value_clip: float | None = None) -> bool:
    """In-place sanitize gradients. Returns True if any non-finite grad was found."""
    found_nonfinite = False
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            found_nonfinite = True
            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
        if value_clip is not None and value_clip > 0:
            p.grad.data.clamp_(min=-float(value_clip), max=float(value_clip))
    return found_nonfinite


def _stabilize_advanced_checkpoint_(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Stabilize extremely noisy checkpoints by rescaling large-weight tensors."""
    target_std = float(getattr(getattr(model, "config", None), "initializer_range", 0.02))
    max_allowed_std = target_std * 8.0
    clamped_tensors = 0
    rescaled_tensors = 0
    nonfinite_tensors = 0

    with torch.no_grad():
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not torch.isfinite(p.data).all():
                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
                nonfinite_tensors += 1
            if p.ndim >= 2:
                std = float(p.data.std().item())
                if std > max_allowed_std and std > 0:
                    p.data.mul_(target_std / std)
                    rescaled_tensors += 1
                p.data.clamp_(min=-1.0, max=1.0)
                clamped_tensors += 1

    # LayerNorm parameters should stay close to identity for stable training.
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            if m.weight is not None:
                m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()

    logger.info(
        "Advanced checkpoint stabilization: "
        f"nonfinite_tensors={nonfinite_tensors}, rescaled_tensors={rescaled_tensors}, "
        f"clamped_tensors={clamped_tensors}, target_std={target_std:.3g}"
    )


def _resolve_setup_name_raw(setup_label: str | None, model_path: Path | None) -> str:
    if setup_label is not None:
        return setup_label
    if model_path is not None:
        return model_path.parent.parent.parent.name
    return "unknown_setup"


def _build_experiment_name(setup_name_raw: str, model_name: str, rand_init: bool, rate: float, timestamp: str) -> tuple[str, bool]:
    setup_base = _strip_timestamp_suffix(setup_name_raw)
    advanced_setup = _is_advanced_setup_name(setup_name_raw)
    normalized_base = setup_base
    for p in ("deit_", "vit_"):
        if normalized_base.startswith(p):
            normalized_base = normalized_base[len(p):]
            break
    model_prefix = _model_prefix_for_setup(model_name)
    display_base = (model_prefix + normalized_base) if model_prefix else normalized_base
    init_prefix = "randinit_" if rand_init else "attack_"
    experiment_name = f"{init_prefix}{display_base}_rate{rate}_{timestamp}"
    return experiment_name, advanced_setup


def _load_training_model(model_path: Path | None, model_name: str, device: str, rand_init: bool, logger: logging.Logger):
    if rand_init:
        logger.info("Random-init mode: building model from config (no pretrained/encrypted weights).")
        config = AutoConfig.from_pretrained(model_name)
        return AutoModelForImageClassification.from_config(config)

    if model_path is None:
        raise ValueError("model_path cannot be None when rand_init is False")

    metadata_path = model_path.parent / "encryption_metadata.json"
    if metadata_path.exists():
        from src.utils.model_utils import load_encrypted_model
        logger.info(f"Loading encrypted model from {model_path.parent}...")
        model, _, _ = load_encrypted_model(model_path.parent, device=device)
        return model
    return AutoModelForImageClassification.from_pretrained(model_path)


def _build_scheduler(optimizer, lr_scheduler: str, epoch_num: int, lr_rate: float, lr_min_factor: float, lr_warmup_epochs: int):
    if lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch_num, eta_min=lr_rate * lr_min_factor
        )
    if lr_scheduler == 'cosine_warmup':
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(epoch):
            if epoch < lr_warmup_epochs:
                return epoch / lr_warmup_epochs
            progress = (epoch - lr_warmup_epochs) / (epoch_num - lr_warmup_epochs)
            return lr_min_factor + (1 - lr_min_factor) * 0.5 * (1 + np.cos(np.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    if lr_scheduler == 'step':
        step_size = max(1, epoch_num // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    if lr_scheduler == 'exponential':
        gamma = (lr_min_factor) ** (1.0 / epoch_num)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if lr_scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
    return None


def _reduce_lr(optimizer, factor: float, lr_floor: float) -> tuple[float, float]:
    current_lr = optimizer.param_groups[0]["lr"]
    new_lr = max(current_lr * factor, lr_floor)
    optimizer.param_groups[0]["lr"] = new_lr
    return current_lr, new_lr


def _build_batch_mixer(
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    logger: logging.Logger,
):
    """Build torchvision v2 MixUp/CutMix mixer."""
    if mixup_prob <= 0.0 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        return None

    mixers = []
    if mixup_alpha > 0.0:
        mixers.append(transforms_v2.MixUp(num_classes=num_classes, alpha=float(mixup_alpha)))
    if cutmix_alpha > 0.0:
        mixers.append(transforms_v2.CutMix(num_classes=num_classes, alpha=float(cutmix_alpha)))
    if not mixers:
        return None
    base_mixer = mixers[0] if len(mixers) == 1 else transforms_v2.RandomChoice(mixers)

    def _apply(images: torch.Tensor, labels: torch.Tensor):
        if np.random.rand() < float(mixup_prob):
            return base_mixer(images, labels)
        return images, labels

    logger.info("Using torchvision.transforms.v2 MixUp/CutMix backend.")
    return _apply


def _compute_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
) -> torch.Tensor:
    if targets.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long):
        return F.cross_entropy(logits, targets, label_smoothing=float(label_smoothing))

    # Soft-label cross-entropy for mixup/cutmix targets.
    soft_targets = targets
    if label_smoothing > 0:
        smooth = float(label_smoothing) / float(num_classes)
        soft_targets = soft_targets * (1.0 - float(label_smoothing)) + smooth
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def train(model_path, model_name, rate, epoch_num, lr_rate, weight_decay, device,
          use_augmentation=True, augment_strength: str | None = None, lr_scheduler='cosine',
          lr_warmup_epochs=0, lr_min_factor=0.01, output_dir='results/attacks',
          rand_init=False, train_dataset='imagenet1k', imagenet_train_root=None,
          batch_size: int | None = None, grad_clip: float = 1.0, auto_hparams: bool = True,
          setup_label: str | None = None, label_smoothing: float | None = None,
          mixup_alpha: float | None = None, cutmix_alpha: float | None = None,
          mixup_prob: float | None = None):
    
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    device_obj = torch.device(device)

    # Setup logging and output directory (name includes model prefix: deit_ / vit_)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_name_raw = _resolve_setup_name_raw(setup_label, model_path)
    experiment_name, advanced_setup = _build_experiment_name(
        setup_name_raw=setup_name_raw,
        model_name=model_name,
        rand_init=rand_init,
        rate=rate,
        timestamp=timestamp,
    )
    attack_results_dir = Path(output_dir) / experiment_name
    attack_results_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = attack_results_dir / "attack_training.log"
    logger = _configure_attack_logger(log_file)

    # Load model
    model = _load_training_model(model_path=model_path, model_name=model_name, device=device, rand_init=rand_init, logger=logger)

    if advanced_setup and (not rand_init):
        _stabilize_advanced_checkpoint_(model, logger)
    model.to(device_obj)

    # Auto-tune hyperparameters for different models unless explicitly overridden
    auto = _auto_hparams_for_model(
        model_name,
        rand_init=rand_init,
        scratch_recipe=advanced_setup,
        advanced_setup=advanced_setup,
    ) if auto_hparams else {}
    auto_recipe = _auto_recipe_for_model(
        model_name,
        rand_init=rand_init,
        scratch_recipe=advanced_setup,
        advanced_setup=advanced_setup,
    ) if auto_hparams else {}
    scale = _infer_model_scale(model_name)
    if epoch_num is None:
        epoch_num = _auto_epochs_for_model(model_name=model_name, rand_init=rand_init)
    # If caller left args at defaults, replace with safer scale-based defaults
    # NOTE: argparse cannot reliably tell if user passed a value; we treat "None" as "unset"
    if lr_rate is None:
        lr_rate = auto.get("lr", 1e-5)
    if weight_decay is None:
        weight_decay = auto.get("weight_decay", 1e-3)
    if batch_size is None:
        batch_size = int(auto.get("batch_size", 64))
    else:
        batch_size = int(batch_size)
    # grad_clip default can still be overridden via CLI
    if grad_clip is None:
        grad_clip = auto.get("grad_clip", 1.0)
    if augment_strength is None:
        augment_strength = str(auto_recipe.get("augment_strength", "medium"))
    if label_smoothing is None:
        label_smoothing = float(auto_recipe.get("label_smoothing", 0.0))
    if mixup_alpha is None:
        mixup_alpha = float(auto_recipe.get("mixup_alpha", 0.0))
    if cutmix_alpha is None:
        cutmix_alpha = float(auto_recipe.get("cutmix_alpha", 0.0))
    if mixup_prob is None:
        mixup_prob = float(auto_recipe.get("mixup_prob", 0.0))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    if mixup_alpha < 0.0:
        raise ValueError(f"mixup_alpha must be >= 0, got {mixup_alpha}")
    if cutmix_alpha < 0.0:
        raise ValueError(f"cutmix_alpha must be >= 0, got {cutmix_alpha}")
    if not (0.0 <= mixup_prob <= 1.0):
        raise ValueError(f"mixup_prob must be in [0, 1], got {mixup_prob}")

    # Advanced checkpoints (uniform-like noise) are numerically fragile at the start.
    # Use a short warmup by default unless user explicitly selected a different schedule.
    if advanced_setup and (not rand_init) and auto_hparams and lr_scheduler == 'cosine' and lr_warmup_epochs == 0:
        lr_scheduler = 'cosine_warmup'
        lr_warmup_epochs = 5
    if rand_init and auto_hparams and lr_scheduler == 'cosine' and lr_warmup_epochs == 0:
        lr_scheduler = 'cosine_warmup'
        lr_warmup_epochs = 5 if scale in {"small", "base"} else 3

    logger.info(f"Starting attack training on model: {model_path if model_path is not None else 'N/A (rand_init)'}")
    logger.info(f"Dataset subset ratio: {rate}")
    logger.info(f"Hyperparameters: epochs={epoch_num}, lr={lr_rate}, weight_decay={weight_decay}")
    logger.info(
        f"Training stability: batch_size={batch_size}, grad_clip={grad_clip}, "
        f"rand_init={rand_init}, advanced_setup={advanced_setup}, auto_hparams={auto_hparams}"
    )
    logger.info(
        f"Scheduler config: scheduler={lr_scheduler}, warmup_epochs={lr_warmup_epochs}, "
        f"lr_min_factor={lr_min_factor}"
    )
    logger.info(
        f"Regularization recipe: augment_strength={augment_strength}, label_smoothing={label_smoothing}, "
        f"mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}, mixup_prob={mixup_prob}"
    )

    # Load mappings
    logger.info("Loading synset mappings...")
    synset_to_name, validation_synsets = load_synset_mapping()

    # Create validation dataset
    logger.info("Creating validation dataset...")
    val_dir = Path('./dataset/imagenet/val_nolabel')
    if not val_dir.exists():
        raise FileNotFoundError(f"ImageNet validation directory not found at {val_dir}. Please ensure dataset is setup correctly.")
    val_dataset = ImageNetValidationDataset(str(val_dir), validation_synsets, processor)

    # Create augmented training dataset
    logger.info(f"Creating training dataset (augmentation: {use_augmentation}, strength: {augment_strength})...")
    if imagenet_train_root is not None:
        train_root = Path(imagenet_train_root)
    else:
        train_root = Path('./dataset/imagenet/train') if train_dataset == 'imagenet1k' else Path('./dataset/imagenet100/train')
    if not train_root.exists():
        raise FileNotFoundError(
            f"Training directory not found at {train_root}. "
            f"Set --imagenet_train_root or prepare the dataset directory."
        )
    
    imgnet_train = AugmentedImageNetDataset(
        root_dir=str(train_root),
        processor=processor,
        augment=use_augmentation,
        augment_strength=augment_strength
    )

    # Sample part of the dataset
    sample_size = max(1, int(round(rate * len(imgnet_train))))
    sample_size = min(sample_size, len(imgnet_train))
    logger.info(f"Subset sampling: {sample_size}/{len(imgnet_train)} examples ({100.0 * sample_size / len(imgnet_train):.2f}%)")
    indices = np.random.choice(len(imgnet_train), sample_size, replace=False).tolist()
    train_dataset = torch.utils.data.Subset(imgnet_train, indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    num_classes = int(getattr(getattr(model, "config", None), "num_labels", 1000))
    batch_mixer = _build_batch_mixer(
        num_classes=num_classes,
        mixup_alpha=float(mixup_alpha),
        cutmix_alpha=float(cutmix_alpha),
        mixup_prob=float(mixup_prob),
        logger=logger,
    )

    # Setup learning rate scheduler
    scheduler = _build_scheduler(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch_num=epoch_num,
        lr_rate=lr_rate,
        lr_min_factor=lr_min_factor,
        lr_warmup_epochs=lr_warmup_epochs,
    )
    
    best_top1_acc = 0.0
    best_model_path = attack_results_dir / "best_model.pth"
    lr_floor = max(float(lr_rate) * 0.05, 1e-6) if (advanced_setup and not rand_init) else 1e-8

    # Train the model
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_num}"):
            images, labels = images.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            loss_targets = labels
            if batch_mixer is not None:
                images, loss_targets = batch_mixer(images, labels)
            outputs = model(images).logits
            loss = _compute_classification_loss(
                logits=outputs,
                targets=loss_targets,
                num_classes=num_classes,
                label_smoothing=float(label_smoothing),
            )
            if not torch.isfinite(loss):
                # Skip update to avoid poisoning weights, and reduce LR to recover.
                current_lr, new_lr = _reduce_lr(optimizer, factor=0.5, lr_floor=lr_floor)
                logger.warning(
                    f"Non-finite loss detected (loss={loss.item()}). "
                    f"Skipping step and reducing LR: {current_lr:.2e} -> {new_lr:.2e}"
                )
                continue
            loss.backward()
            found_nonfinite_grad = _sanitize_gradients_(model, value_clip=(1.0 if advanced_setup else None))
            if found_nonfinite_grad or _has_nonfinite_gradients(model):
                current_lr, new_lr = _reduce_lr(optimizer, factor=0.8, lr_floor=lr_floor)
                logger.warning(
                    "Non-finite gradients detected. Sanitized gradients and reducing LR: "
                    f"{current_lr:.2e} -> {new_lr:.2e}"
                )
            if grad_clip and grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                if not torch.isfinite(total_norm):
                    current_lr, new_lr = _reduce_lr(optimizer, factor=0.8, lr_floor=lr_floor)
                    _sanitize_gradients_(model, value_clip=(1.0 if advanced_setup else None))
                    total_norm_retry = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                    if not torch.isfinite(total_norm_retry):
                        optimizer.zero_grad(set_to_none=True)
                        logger.warning(
                            f"Non-finite grad norm persisted after sanitize ({total_norm_retry}). "
                            f"Skipping step and reducing LR: {current_lr:.2e} -> {new_lr:.2e}"
                        )
                        continue
                    logger.warning(
                        f"Non-finite grad norm ({total_norm}) recovered after sanitize. Reduced LR: "
                        f"{current_lr:.2e} -> {new_lr:.2e}"
                    )
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        top1_acc, top5_acc, _ = validate_model(model, val_dataset, device_obj, synset_to_name)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(top1_acc)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Top-1 Acc={top1_acc:.2f}%, Top-5 Acc={top5_acc:.2f}%, LR={current_lr:.2e}")

        # Save the model with the best top1 accuracy
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best Top-1 accuracy: {best_top1_acc:.2f}%. Saved model.")

    # Save final results
    final_results = {
        'model_path': str(model_path),
        'rate': rate,
        'epoch_num': epoch_num,
        'lr_rate': lr_rate,
        'best_top1_acc': best_top1_acc,
        'timestamp': timestamp
    }
    with open(attack_results_dir / "attack_results.json", 'w') as f:
        json.dump(final_results, f, indent=4)
    
    logger.info(f"Attack training complete. Results saved to {attack_results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an Encrypted model on a subset of ImageNet.')
    parser.add_argument('--setup', type=str, default='basic_top-k_k4',
                        help='Prefix of the encryption setup to attack (e.g., basic_top-k_k4, advanced_all). Matching also tries model-prefixed dirs (e.g. deit_basic_top-k_k6).')
    parser.add_argument('--model', '--model_name', type=str, dest='model_name', default='google/vit-base-patch16-224',
                        help='Base model name for processor and checkpoint matching (e.g. facebook/deit-small-patch16-224)')
    parser.add_argument('--rate', type=float, default=0.2, help='Proportion of the dataset the attacker controls.')
    parser.add_argument('--epoch', type=int, default=None, help='Number of training epochs (default: auto by model)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: auto by model)')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (default: auto by model)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: auto by model)')
    parser.add_argument('--grad_clip', type=float, default=None, help='Global grad norm clip (0 to disable, default: auto by model)')
    parser.add_argument('--no_auto_hparams', action='store_true', help='Disable model-based hyperparameter defaults')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory where encrypted models are stored')
    parser.add_argument('--output_dir', type=str, default='results/attacks', help='Directory to save attack results')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--augment_strength', type=str, choices=['light', 'medium', 'strong'],
                        default=None, help='Strength of data augmentation (default: auto by mode/model)')
    parser.add_argument('--label_smoothing', type=float, default=None,
                        help='Label smoothing epsilon in [0,1) (default: auto by mode/model)')
    parser.add_argument('--mixup_alpha', type=float, default=None,
                        help='Mixup Beta(alpha, alpha); 0 disables mixup (default: auto by mode/model)')
    parser.add_argument('--cutmix_alpha', type=float, default=None,
                        help='CutMix Beta(alpha, alpha); 0 disables cutmix (default: auto by mode/model)')
    parser.add_argument('--mixup_prob', type=float, default=None,
                        help='Probability of applying mixup/cutmix per batch (default: auto by mode/model)')
    parser.add_argument('--lr_scheduler', type=str,
                        choices=['none', 'cosine', 'cosine_warmup', 'step', 'exponential', 'plateau'],
                        default='cosine', help='Learning rate scheduler type')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--lr_min_factor', type=float, default=0.01, help='Minimum learning rate factor')
    parser.add_argument('--rand_init', action='store_true', help='Randomly initialize the model before training')
    parser.add_argument('--train_dataset', type=str, choices=['imagenet1k', 'imagenet100'], default='imagenet1k',
                        help='Training dataset to simulate attacker pretraining')
    parser.add_argument('--imagenet_train_root', type=str, default=None,
                        help='Override training root directory (defaults to ./dataset/imagenet/train or ./dataset/imagenet100/train)')
    
    args = parser.parse_args()

    if args.rand_init:
        model_path = None
        print(f"Running random-init training (no checkpoint lookup). Setup label: {args.setup}")
    else:
        # Find the latest checkpoint for the specified setup (match plain or model-prefixed dir names)
        model_path = find_latest_checkpoint(args.results_dir, args.setup, model_name=args.model_name)
        
        if model_path is None:
            print(f"Error: Could not find any checkpoint matching setup prefix '{args.setup}' in {args.results_dir}")
            exit(1)
        
        print(f"Attacking latest checkpoint: {model_path}")

    use_augmentation = not args.no_augmentation
    train(model_path, args.model_name, args.rate, args.epoch, args.lr, args.weight_decay,
          args.device, use_augmentation, args.augment_strength, args.lr_scheduler,
          args.lr_warmup_epochs, args.lr_min_factor, args.output_dir, args.rand_init,
          args.train_dataset, args.imagenet_train_root, batch_size=args.batch_size,
          grad_clip=args.grad_clip, auto_hparams=(not args.no_auto_hparams),
          setup_label=args.setup, label_smoothing=args.label_smoothing,
          mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
          mixup_prob=args.mixup_prob)

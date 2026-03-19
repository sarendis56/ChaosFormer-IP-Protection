import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import logging
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from typing import Optional, cast

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.imagenet_eval import evaluate_topk_accuracy

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


def build_transforms(img_size=224, aug_policy="randaugment"):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
    ]
    if aug_policy == "randaugment":
        train_tf.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif aug_policy == "autoaugment":
        train_tf.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
    elif aug_policy == "trivialaugment":
        train_tf.append(transforms.TrivialAugmentWide())
    elif aug_policy == "none":
        pass
    else:
        raise ValueError(f"Unknown aug_policy: {aug_policy}")

    train_tf += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    if aug_policy != "none":
        train_tf.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)))

    val_tf = [
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(train_tf), transforms.Compose(val_tf)


def build_batch_mixer(num_classes: int, mixup_alpha: float, cutmix_alpha: float, mixup_prob: float):
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

    return _apply


def classification_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, label_smoothing: float) -> torch.Tensor:
    if targets.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long):
        return F.cross_entropy(logits, targets, label_smoothing=float(label_smoothing))
    soft_targets = targets
    if label_smoothing > 0:
        smooth = float(label_smoothing) / float(num_classes)
        soft_targets = soft_targets * (1.0 - float(label_smoothing)) + smooth
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def pretrain(
    model_name: str = "convnext_tiny",
    rate: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    batch_size: int = 128,
    warmup_epochs: int = 10,
    optimizer_name: str = "adamw",
    aug_policy: str = "randaugment",
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 1.0,
    amp: bool = True,
    device: str = "cuda",
    num_workers: Optional[int] = None,
    prefetch_factor: int = 4,
    channels_last: bool = True,
    tf32: bool = True,
    cudnn_benchmark: bool = True,
    imagenet_root: str = "./dataset/imagenet/train",
    output_dir: str = "results/pretrain"
):
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"pretrain_{model_name}_rate{rate}_{timestamp}"
    results_dir = Path(output_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = results_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CNN PRETRAINING on ImageNet (subset rate: {rate})")
    logger.info(f"Model: {model_name}, Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}")
    logger.info(
        f"Recipe: aug_policy={aug_policy}, label_smoothing={label_smoothing}, "
        f"mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}, mixup_prob={mixup_prob}"
    )

    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")
    
    # Prepare ImageNet transforms
    train_tf, val_tf = build_transforms(224, aug_policy=aug_policy)
    
    # Load ImageNet
    logger.info(f"Loading ImageNet from {imagenet_root}...")
    if not Path(imagenet_root).exists():
        raise FileNotFoundError(f"ImageNet root not found at {imagenet_root}")
    
    full_train_dataset = ImageFolder(root=imagenet_root, transform=train_tf)
    full_val_dataset = ImageFolder(root=imagenet_root, transform=val_tf)
    num_classes = len(full_train_dataset.classes)
    logger.info(f"Full ImageNet size: {len(full_train_dataset)}, Classes: {num_classes}")
    
    # Sample subset
    sample_size = int(rate * len(full_train_dataset))
    indices = np.random.choice(len(full_train_dataset), sample_size, replace=False).tolist()

    val_size = max(1, sample_size // 10)
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    logger.info(
        f"Subset size: {sample_size} (rate: {rate}), Train: {len(train_dataset)}, Val: {len(val_dataset)}"
    )
    
    if num_workers is None:
        cpu_count = os.cpu_count() or 8
        num_workers = min(16, max(4, cpu_count // 2))
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device_obj.type == "cuda"),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_obj.type == "cuda"),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        persistent_workers=persistent_workers,
    )
    logger.info(
        f"DataLoader: num_workers={num_workers}, prefetch_factor={prefetch_factor}, "
        f"persistent_workers={persistent_workers}, pin_memory={device_obj.type == 'cuda'}"
    )

    # Initialize model
    logger.info(f"Initializing {model_name} with {num_classes} classes...")
    model = get_cnn_model(model_name, num_classes)
    model.to(device_obj)
    batch_mixer = build_batch_mixer(
        num_classes=num_classes,
        mixup_alpha=float(mixup_alpha),
        cutmix_alpha=float(cutmix_alpha),
        mixup_prob=float(mixup_prob),
    )
    
    # Optimizer
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    
    use_amp = amp and device_obj.type == "cuda"
    scaler = torch.GradScaler(device_obj.type, enabled=use_amp)
    
    # Cosine schedule with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(pbar):
            if channels_last and device_obj.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            images = images.to(device_obj, non_blocking=True)
            labels = labels.to(device_obj, non_blocking=True)
            loss_targets = labels
            if batch_mixer is not None:
                images, loss_targets = batch_mixer(images, labels)
            
            with torch.autocast(device_type=device_obj.type, enabled=use_amp):
                outputs = model(images)
                loss = classification_loss(
                    logits=outputs,
                    targets=loss_targets,
                    num_classes=num_classes,
                    label_smoothing=float(label_smoothing),
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, LR={current_lr:.2e}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            acc = evaluate_topk_accuracy(model, val_loader, device_obj, topk=(1, 5))
            logger.info(f"Epoch {epoch+1}: Top-1 Acc={acc[1]:.2f}%, Top-5 Acc={acc[5]:.2f}%")

        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            ckpt_path = results_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'model_name': model_name,
                    'num_classes': num_classes,
                    'rate': rate,
                }
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save final model
    final_path = results_dir / "pretrained_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': model_name,
            'num_classes': num_classes,
            'rate': rate,
        }
    }, final_path)
    logger.info(f"Pretraining complete. Final model saved to {final_path}")
    return final_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain a CNN on a subset of ImageNet.')
    parser.add_argument('--model_name', type=str, default='convnext_tiny', help='CNN architecture')
    parser.add_argument('--rate', type=float, default=0.2, help='Proportion of ImageNet to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of pretraining epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=192, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'], help='Optimizer')
    parser.add_argument('--aug_policy', type=str, default='randaugment', choices=['randaugment', 'autoaugment', 'trivialaugment', 'none'],
                        help='Training augmentation policy')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing epsilon in [0,1)')
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help='MixUp beta alpha (0 to disable)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix beta alpha (0 to disable)')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability to apply MixUp/CutMix per batch')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader workers (default: auto)')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor per worker')
    parser.add_argument('--channels_last', action='store_true', default=True, help='Use channels_last memory format on CUDA')
    parser.add_argument('--no-channels_last', dest='channels_last', action='store_false', help='Disable channels_last')
    parser.add_argument('--tf32', action='store_true', default=True, help='Enable TF32 matmul/cuDNN on Ampere+ GPUs')
    parser.add_argument('--no-tf32', dest='tf32', action='store_false', help='Disable TF32')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=True, help='Enable cuDNN benchmark')
    parser.add_argument('--no-cudnn_benchmark', dest='cudnn_benchmark', action='store_false', help='Disable cuDNN benchmark')
    parser.add_argument('--imagenet_root', type=str, default='./dataset/imagenet/train', help='Path to ImageNet train folder')
    parser.add_argument('--output_dir', type=str, default='results/pretrain', help='Output directory')
    
    args = parser.parse_args()
    
    pretrain(
        model_name=args.model_name,
        rate=args.rate,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        aug_policy=args.aug_policy,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob,
        device=args.device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        channels_last=args.channels_last,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        imagenet_root=args.imagenet_root,
        output_dir=args.output_dir
    )

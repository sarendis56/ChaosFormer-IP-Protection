"""Quick CLI to evaluate a pretrained CNN checkpoint on ImageNet val.

Usage:
  uv run src/utils/eval_pretrained_imagenet.py --checkpoint <path_to_checkpoint.pth>
"""

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.experiments.pretrain import build_transforms, get_cnn_model
from src.utils.imagenet_eval import evaluate_topk_accuracy


def load_state_dict_from_checkpoint(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    # Fallback when a raw state_dict is saved directly.
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained CNN checkpoint on ImageNet validation (Top-1 / Top-5)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint (.pth)",
    )
    parser.add_argument(
        "--imagenet_val_root",
        type=str,
        default="dataset/imagenet/val",
        help="Path to ImageNet val root (class-subfolder layout for ImageFolder)",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Override model name in checkpoint config")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    val_root = Path(args.imagenet_val_root)
    if not val_root.exists():
        raise FileNotFoundError(f"ImageNet val root not found: {val_root}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    model_name = args.model_name or config.get("model_name", "convnext_tiny")
    num_classes = int(config.get("num_classes", 1000))

    _, val_tf = build_transforms(img_size=args.img_size, aug_policy="none")
    val_dataset = ImageFolder(root=str(val_root), transform=val_tf)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = get_cnn_model(model_name=model_name, num_classes=num_classes)
    state_dict = load_state_dict_from_checkpoint(checkpoint if isinstance(checkpoint, dict) else {})
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}")

    device = torch.device(args.device)
    model = model.to(device)

    acc = evaluate_topk_accuracy(model, val_loader, device, topk=(1, 5))

    print("\nEvaluation complete")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Num classes (from checkpoint): {num_classes}")
    print(f"ImageNet val samples: {len(val_dataset)}")
    print(f"Top-1 Accuracy: {acc[1]:.2f}%")
    print(f"Top-5 Accuracy: {acc[5]:.2f}%")


if __name__ == "__main__":
    main()

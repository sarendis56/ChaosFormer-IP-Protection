#!/usr/bin/env python3
"""
Usage:
    python demo.py img/img1.jpg img/img2.jpg

The script will:
    - load two images (you supply paths),
    - apply Arnold Cat Map (ACM) encryption,
    - then apply a diffusion-style XOR encryption that turns them into noise,
    - and display a 2x3 grid: [original | ACM | diffusion] per image.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Local imports
from src.encryption.arnold_transform import arnold_optimized, get_standard_key
from src.encryption.xor_encryption import xor_encrypt_decrypt

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 20
FONT_CANDIDATES = ["Calibri", "Carlito", "Liberation Sans", "DejaVu Sans", "Noto Sans CJK SC"]


def resolve_font_family() -> str:
    """
    Pick the first installed font from the preferred list.
    """
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in FONT_CANDIDATES:
        if font_name in installed_fonts:
            return font_name
    return "sans-serif"

def load_image(path: Path, side: int = 256) -> torch.Tensor:
    """
    Load an image, resize to square, and return a float tensor in [0, 1] with shape (H, W, C).
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((side, side), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C), [0,1]
    return torch.from_numpy(arr)  # (H, W, C), float32


def apply_acm(image_hw3: torch.Tensor) -> torch.Tensor:
    """
    Apply Arnold Cat Map (ACM) using the optimized implementation.

    Expects image as (H, W, C) float32 in [0,1].
    Returns encrypted image with same shape.
    """
    h, w, _ = image_hw3.shape
    if h != w:
        raise ValueError(f"ACM expects square image, got {h}x{w}")

    key = get_standard_key("strong")  # determinant-1 key, valid for any size
    enc = arnold_optimized(image_hw3, key)
    return enc


def apply_diffusion(image_hw3: torch.Tensor, layer_idx: int, name: str) -> torch.Tensor:
    """
    Apply a diffusion-style XOR encryption.

    We reuse the XOR weight encryption, which bitwise-scrambles the float32
    representation so the result visually looks like random noise.

    Expects and returns (H, W, C) float32.
    """
    # xor_encrypt_decrypt works on arbitrary float shapes.
    enc = xor_encrypt_decrypt(
        image_hw3.clone(),
        layer_idx=layer_idx,
        weight_name=name,
        in_place=False,
    )

    # For visualization, ignore the float values (which may be NaN/inf after bit-scrambling)
    # and instead interpret the underlying bits as uint32, then normalize to [0, 1].
    bits_int = enc.detach().cpu().view(torch.int32).numpy().astype(np.uint32)
    enc_np = bits_int.astype(np.float32) / np.float32(2**32 - 1)
    return torch.from_numpy(enc_np)


def to_numpy_img(image_hw3: torch.Tensor) -> np.ndarray:
    """
    Convert (H, W, C) float tensor in [0,1] to numpy uint8 RGB for plotting.
    """
    img = image_hw3.detach().cpu().clamp(0.0, 1.0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return img


def process_image(path: Path, layer_idx: int, name_prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load image and produce (original, acm, diffusion) triplet as numpy RGB arrays.
    """
    img = load_image(path)  # (H, W, C), float32 in [0,1]
    img_acm = apply_acm(img)
    img_diff = apply_diffusion(img_acm, layer_idx=layer_idx, name=f"{name_prefix}_diff")

    return to_numpy_img(img), to_numpy_img(img_acm), to_numpy_img(img_diff)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print("Usage: python demo.py path/to/img1 path/to/img2")
        return 1

    img_paths = [Path(p) for p in argv]
    for p in img_paths:
        if not p.exists():
            print(f"Error: {p} does not exist.")
            return 1

    # Process both images; use different layer_idx so XOR keys differ
    orig1, acm1, diff1 = process_image(img_paths[0], layer_idx=0, name_prefix="img1")
    orig2, acm2, diff2 = process_image(img_paths[1], layer_idx=1, name_prefix="img2")

    font_family = resolve_font_family()
    plt.rcParams["font.family"] = font_family

    # Plot 2x3 grid: rows = images, cols = [original, ACM, diffusion]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    titles = ["Original", "Permutation Only", "Permutation + Diffusion"]

    for row, (o, a, d, label) in enumerate(
        [(orig1, acm1, diff1, "Image 1"), (orig2, acm2, diff2, "Image 2")]
    ):
        imgs = [o, a, d]
        for col, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[row, col]
            ax.imshow(img)
            if row == 0:
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", fontfamily=font_family)
            if col == 0:
                ax.set_ylabel(label, fontsize=LABEL_FONTSIZE, fontweight="bold", fontfamily=font_family)
            ax.axis("off")

    plt.tight_layout()

    # Save figure to current working directory (PNG + PDF)
    out_png = Path.cwd() / "demo_output.png"
    out_pdf = Path.cwd() / "demo_output.pdf"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"Using font family: {font_family}")
    print(f"Saved demo figures to {out_png} and {out_pdf}")

    plt.show()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

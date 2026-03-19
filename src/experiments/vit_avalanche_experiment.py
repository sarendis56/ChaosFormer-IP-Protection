#!/usr/bin/env python3
import argparse
import hashlib
import hmac
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.encryption.arnold_transform import generate_arnold_key
from src.encryption.dual_encryption import DualEncryption, LayerEncryptionResult
from src.utils.vit_analyzer import VitEncryptionAnalyzer


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def hamming_distance_bits(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def flip_bits(src: bytes, num_bits: int, rng: np.random.Generator) -> bytes:
    out = bytearray(src)
    total_bits = len(out) * 8
    num_bits = min(max(0, num_bits), total_bits)
    for bit_pos in rng.choice(total_bits, size=num_bits, replace=False):
        byte_idx = int(bit_pos // 8)
        bit_idx = int(bit_pos % 8)
        out[byte_idx] ^= (1 << bit_idx)
    return bytes(out)


def generate_unique_bitflip_keys(
    src: bytes,
    num_bits: int,
    num_samples: int,
    rng: np.random.Generator,
    used_keys: set[bytes],
    max_attempts_per_sample: int = 100,
) -> List[bytes]:
    """Generate unique wrong keys for a specific requested Hamming distance."""
    keys: List[bytes] = []
    local_seen: set[bytes] = set()

    for _ in range(num_samples):
        accepted = False
        for _attempt in range(max_attempts_per_sample):
            candidate = flip_bits(src, num_bits, rng)
            if candidate == src:
                continue
            if candidate in used_keys or candidate in local_seen:
                continue
            local_seen.add(candidate)
            keys.append(candidate)
            accepted = True
            break
        if not accepted:
            break

    return keys


def response_to_key_bytes(response: np.ndarray) -> bytes:
    quantized = np.round(response * 1000).astype(np.int16)
    return hashlib.sha256(quantized.tobytes()).digest()


def derive_key_material(root_key_bytes: bytes, matrix_size: int) -> Dict[str, Any]:
    arnold_h = hmac.new(root_key_bytes, b"arnold", hashlib.sha256).digest()
    arnold_seed = int.from_bytes(arnold_h[:8], byteorder="big") % (2**32)
    arnold_key = generate_arnold_key(matrix_size, iterations=3, seed=arnold_seed)

    perm_password = hmac.new(root_key_bytes, b"permutation", hashlib.sha256).hexdigest()
    xor_h = hmac.new(root_key_bytes, b"xor", hashlib.sha256).digest()
    xor_seed_base = int.from_bytes(xor_h[:8], byteorder="big") % (2**32)

    return {
        "arnold_key": arnold_key,
        "password": perm_password,
        "xor_seed_base": xor_seed_base,
    }


def build_encryptor(model, device: str, root_key_bytes: bytes, num_permutation_matrices: int) -> DualEncryption:
    hidden = int(model.config.hidden_size) if hasattr(model, "config") and hasattr(model.config, "hidden_size") else 768
    km = derive_key_material(root_key_bytes=root_key_bytes, matrix_size=hidden)
    return DualEncryption.from_model(
        model=model,
        arnold_key=km["arnold_key"],
        password=km["password"],
        num_permutation_matrices=num_permutation_matrices,
        device=device,
        use_xor=False,
        xor_seed_base=km["xor_seed_base"],
        mode="basic",
    )


def get_num_layers(model) -> int:
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        return len(model.vit.encoder.layer)
    if hasattr(model, "deit") and hasattr(model.deit, "encoder"):
        return len(model.deit.encoder.layer)
    if hasattr(model, "beit") and hasattr(model.beit, "encoder"):
        return len(model.beit.encoder.layer)
    if hasattr(model, "encoder"):
        return len(model.encoder.layer)
    raise RuntimeError("Unsupported model backbone for avalanche experiment.")


@dataclass
class AvalancheRecord:
    variant: str
    distance_type: str
    distance: float
    accuracy: float
    requested_distance: int = -1
    noise_sigma: float = -1.0


def run_experiment(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    set_random_seeds(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    analyzer = VitEncryptionAnalyzer(
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_extra_layers=0,
        password=None,
        arnold_key=None,
        device=args.device,
        imagenet_path=args.imagenet_path,
        local_model_path=args.local_model_path,
    )

    num_layers = get_num_layers(analyzer.model)
    k = min(args.k, num_layers)
    selected_layers = sorted(rng.choice(num_layers, size=k, replace=False).tolist())
    logger.info(f"Selected random-k layers: {selected_layers}")

    if args.master_key:
        correct_root = hashlib.sha256(args.master_key.encode()).digest()
    else:
        correct_root = hashlib.sha256(f"avalanche-default-{args.random_seed}".encode()).digest()

    encryptor_ok = build_encryptor(
        model=analyzer.model,
        device=args.device,
        root_key_bytes=correct_root,
        num_permutation_matrices=args.num_permutation_matrices,
    )

    encrypted_records: Dict[int, LayerEncryptionResult] = {}

    for layer_idx in selected_layers:
        attention_weights, ffn_weights = analyzer._get_layer_weights(layer_idx)
        perm_idx = int(rng.integers(0, len(encryptor_ok.permutation_matrices)))
        enc = encryptor_ok.encrypt_layer_weights(
            attention_weights=attention_weights,
            ffn_weights=ffn_weights,
            permutation_matrix_idx=perm_idx,
            layer_idx=layer_idx,
        )
        encrypted_records[layer_idx] = LayerEncryptionResult(
            encrypted_attention={kname: tensor.detach().clone() for kname, tensor in enc.encrypted_attention.items()},
            encrypted_ffn={kname: tensor.detach().clone() for kname, tensor in enc.encrypted_ffn.items()},
            permutation_matrix_idx=perm_idx,
            arnold_key=enc.arnold_key.copy(),
        )
        analyzer._apply_encrypted_weights(
            layer_idx=layer_idx,
            encrypted_attention=encrypted_records[layer_idx].encrypted_attention,
            encrypted_ffn=encrypted_records[layer_idx].encrypted_ffn,
        )

    encrypted_metrics = analyzer.evaluator.evaluate_model(analyzer.model, analyzer.processor)
    encrypted_accuracy = encrypted_metrics.top1_accuracy / 100.0
    logger.info(f"Encrypted (ciphertext-only) accuracy: {encrypted_accuracy:.4f}")

    def restore_ciphertext() -> None:
        for li in selected_layers:
            enc = encrypted_records[li]
            analyzer._apply_encrypted_weights(li, enc.encrypted_attention, enc.encrypted_ffn)

    def evaluate_with_key(root_key: bytes) -> float:
        decryptor = build_encryptor(
            model=analyzer.model,
            device=args.device,
            root_key_bytes=root_key,
            num_permutation_matrices=args.num_permutation_matrices,
        )
        for li in selected_layers:
            dec_attn, dec_ffn = decryptor.decrypt_layer_weights(encrypted_records[li], layer_idx=li)
            analyzer._apply_encrypted_weights(li, dec_attn, dec_ffn)
        m = analyzer.evaluator.evaluate_model(analyzer.model, analyzer.processor)
        acc = m.top1_accuracy / 100.0
        restore_ciphertext()
        return acc

    records: List[AvalancheRecord] = []

    acc_ok = evaluate_with_key(correct_root)
    records.append(AvalancheRecord(variant="correct", distance_type="hamming_bits", distance=0.0, accuracy=acc_ok))
    logger.info(f"Correct-key accuracy: {acc_ok:.4f}")

    bit_distances = parse_csv_ints(args.bit_flips)
    used_wrong_keys: set[bytes] = set()
    for d in bit_distances:
        samples_for_d = args.samples_per_distance + (args.hamming1_extra_samples if d == 1 else 0)
        wrong_keys = generate_unique_bitflip_keys(
            src=correct_root,
            num_bits=d,
            num_samples=samples_for_d,
            rng=rng,
            used_keys=used_wrong_keys,
        )
        used_wrong_keys.update(wrong_keys)
        logger.info(f"Hamming distance {d}: generated {len(wrong_keys)} unique wrong keys")

        for s, wrong in enumerate(wrong_keys, start=1):
            acc = evaluate_with_key(wrong)
            records.append(
                AvalancheRecord(
                    variant=f"bitflip_{d}_sample_{s}",
                    distance_type="hamming_bits",
                    distance=float(hamming_distance_bits(correct_root, wrong)),
                    accuracy=acc,
                    requested_distance=d,
                )
            )

    base_response = rng.normal(loc=0.0, scale=1.0, size=args.puf_dim).astype(np.float32)
    for sigma in parse_csv_floats(args.noise_levels):
        for s in range(args.noise_samples):
            noise = rng.normal(loc=0.0, scale=sigma, size=base_response.shape).astype(np.float32)
            wrong = response_to_key_bytes(base_response + noise)
            acc = evaluate_with_key(wrong)
            records.append(
                AvalancheRecord(
                    variant=f"noise_{sigma}_sample_{s+1}",
                    distance_type="euclidean_response",
                    distance=float(np.linalg.norm(noise)),
                    accuracy=acc,
                    noise_sigma=float(sigma),
                )
            )

    restore_ciphertext()

    return {
        "model_name": args.model,
        "device": args.device,
        "initial_accuracy": analyzer.initial_accuracy,
        "ciphertext_accuracy": encrypted_accuracy,
        "selected_layers": selected_layers,
        "k": k,
        "records": [asdict(r) for r in records],
    }


def _build_plot_paths(base_png: Path) -> Dict[str, Path]:
    """Create descriptive output paths for separate bit/noise plots."""
    parent = base_png.parent
    stem = base_png.stem
    paths: Dict[str, Path] = {
        "bit_png": parent / f"{stem}_bit_flip_key_distance_broken_y_axis.png",
        "bit_pdf": parent / f"{stem}_bit_flip_key_distance_broken_y_axis.pdf",
        "noise_png": parent / f"{stem}_noisy_puf_response_distance.png",
        "noise_pdf": parent / f"{stem}_noisy_puf_response_distance.pdf",
    }
    return paths


def plot_results(payload: Dict[str, Any], out_png: Path) -> Dict[str, Path]:
    rows = payload["records"]
    bit_rows = [r for r in rows if r["distance_type"] == "hamming_bits"]
    noise_rows = [r for r in rows if r["distance_type"] == "euclidean_response"]

    paths = _build_plot_paths(out_png)
    # Paper-friendly defaults: shorter figures and larger fonts.
    label_fs = 18
    tick_fs = 16
    legend_fs = 16

    if bit_rows:
        fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.1], hspace=0.05)
        ax_bit_top = fig.add_subplot(gs[0, 0])
        ax_bit_bot = fig.add_subplot(gs[1, 0], sharex=ax_bit_top)

        correct_rows = [r for r in bit_rows if r.get("variant") == "correct"]
        wrong_rows = [r for r in bit_rows if r.get("variant") != "correct"]

        # Jitter wrong-key points so repeated trials at the same Hamming distance
        # do not fully overlap on a single pixel column.
        jitter_rng = np.random.default_rng(0)
        if wrong_rows:
            x_wrong = np.array([r["distance"] for r in wrong_rows], dtype=np.float32)
            y_wrong = np.array([r["accuracy"] * 100.0 for r in wrong_rows], dtype=np.float32)
            x_wrong_jittered = x_wrong + jitter_rng.uniform(-0.16, 0.16, size=len(x_wrong))
            ax_bit_top.scatter(
                x_wrong_jittered,
                y_wrong,
                s=22,
                alpha=0.45,
                color="tab:blue",
            )
            ax_bit_bot.scatter(
                x_wrong_jittered,
                y_wrong,
                s=24,
                alpha=0.6,
                color="tab:blue",
                label="Wrong key",
            )

        if correct_rows:
            x_ok = [r["distance"] for r in correct_rows]
            y_ok = [r["accuracy"] * 100.0 for r in correct_rows]
            ax_bit_top.scatter(x_ok, y_ok, s=72, marker="*", color="crimson", zorder=5, label="Correct key")
            ax_bit_bot.scatter(x_ok, y_ok, s=72, marker="*", color="crimson", zorder=5)

        # Broken y-axis ranges:
        # - top band around the correct-key accuracy (~80%)
        # - bottom band around wrong-key random-guess region.
        if correct_rows:
            y_correct = np.array([r["accuracy"] * 100.0 for r in correct_rows], dtype=np.float32)
            y_correct_med = float(np.median(y_correct))
        else:
            y_correct_med = 80.0
        top_lo = max(60.0, y_correct_med - 5.0)
        top_hi = min(100.0, y_correct_med + 5.0)
        if top_hi - top_lo < 2.0:
            top_lo, top_hi = max(60.0, y_correct_med - 2.0), min(100.0, y_correct_med + 2.0)

        if wrong_rows:
            y_wrong = np.array([r["accuracy"] * 100.0 for r in wrong_rows], dtype=np.float32)
            y_wrong_hi = float(np.percentile(y_wrong, 99))
            bot_hi = max(0.12, min(5.0, y_wrong_hi * 1.35 + 0.01))
        else:
            bot_hi = 0.2
        bot_lo = 0.0

        ax_bit_top.set_ylim(top_lo, top_hi)
        ax_bit_bot.set_ylim(bot_lo, bot_hi)

        # Axis cosmetics for broken-axis look.
        ax_bit_top.spines["bottom"].set_visible(False)
        ax_bit_bot.spines["top"].set_visible(False)
        ax_bit_top.tick_params(labelbottom=False, bottom=False)
        ax_bit_bot.xaxis.tick_bottom()

        # Diagonal break marks
        d = 0.012
        ax_bit_top.plot(
            (-d, +d), (-d, +d),
            transform=ax_bit_top.transAxes, color="k", clip_on=False, linewidth=0.9
        )
        ax_bit_top.plot(
            (1 - d, 1 + d), (-d, +d),
            transform=ax_bit_top.transAxes, color="k", clip_on=False, linewidth=0.9
        )
        ax_bit_bot.plot(
            (-d, +d), (1 - d, 1 + d),
            transform=ax_bit_bot.transAxes, color="k", clip_on=False, linewidth=0.9
        )
        ax_bit_bot.plot(
            (1 - d, 1 + d), (1 - d, 1 + d),
            transform=ax_bit_bot.transAxes, color="k", clip_on=False, linewidth=0.9
        )

        ax_bit_bot.set_xlabel("Hamming Distance (bits)")
        ax_bit_bot.set_ylabel("Top-1 Accuracy (%)")
        ax_bit_top.grid(alpha=0.25)
        ax_bit_bot.grid(alpha=0.3)
        ax_bit_top.legend(loc="best", fontsize=legend_fs)
        ax_bit_top.tick_params(axis="y", labelsize=tick_fs)
        ax_bit_bot.tick_params(axis="both", labelsize=tick_fs)
        ax_bit_bot.xaxis.label.set_fontsize(label_fs)
        ax_bit_bot.yaxis.label.set_fontsize(label_fs)
        fig.savefig(paths["bit_png"], dpi=220, bbox_inches="tight")
        fig.savefig(paths["bit_pdf"], bbox_inches="tight")
        plt.close(fig)
    if noise_rows:
        fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
        ax_noise = fig.add_subplot(111)
        x = [r["distance"] for r in noise_rows]
        y = [r["accuracy"] * 100.0 for r in noise_rows]
        ax_noise.scatter(x, y, s=30, alpha=0.85, color="tab:orange")
        ax_noise.set_xlabel("Euclidean Distance ||R'-R||_2")
        ax_noise.set_ylabel("Top-1 Accuracy (%)")
        ax_noise.grid(alpha=0.3)
        ax_noise.tick_params(axis="both", labelsize=tick_fs)
        ax_noise.xaxis.label.set_fontsize(label_fs)
        ax_noise.yaxis.label.set_fontsize(label_fs)
        fig.savefig(paths["noise_png"], dpi=220, bbox_inches="tight")
        fig.savefig(paths["noise_pdf"], bbox_inches="tight")
        plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avalanche-effect experiment for ViT encryption",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, default="google/vit-base-patch16-224")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--local-model-path", type=str, default=None)
    p.add_argument("--imagenet-path", type=str, default="dataset/imagenet/val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-permutation-matrices", type=int, default=6)
    p.add_argument("--k", type=int, default=6, help="Random-K encrypted layers")
    p.add_argument("--bit-flips", type=str, default="1,2,4,8,16")
    p.add_argument("--samples-per-distance", type=int, default=12)
    p.add_argument(
        "--hamming1-extra-samples",
        type=int,
        default=36,
        help="Additional samples specifically for requested Hamming distance 1",
    )
    p.add_argument("--noise-levels", type=str, default="0.0001,0.0005,0.001,0.005,0.01")
    p.add_argument("--noise-samples", type=int, default=12)
    p.add_argument("--puf-dim", type=int, default=256)
    p.add_argument("--master-key", type=str, default=None, help="Optional master key string")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--replot", action="store_true", help="Regenerate plot from existing avalanche_results.json")
    p.add_argument("--replot-json", type=str, default=None, help="Path to existing avalanche_results.json for --replot")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name or f"vit_avalanche_random-k_k{args.k}_{ts}"
    exp_dir = out_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("vit_avalanche")

    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.replot:
        if args.replot_json:
            json_path = Path(args.replot_json)
        else:
            json_path = exp_dir / "avalanche_results.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot replot: {json_path} not found. "
                "Use --replot-json to point to an existing avalanche_results.json."
            )
        with open(json_path, "r") as f:
            payload = json.load(f)
        plot_path = json_path.with_name("avalanche_plot.png")
        outputs = plot_results(payload, plot_path)
        print(f"Replot completed from: {json_path}")
        if outputs["bit_png"].exists():
            print(f"Bit-Flip Plot: {outputs['bit_png']}")
            print(f"Bit-Flip PDF: {outputs['bit_pdf']}")
        if outputs["noise_png"].exists():
            print(f"Noisy-PUF Plot: {outputs['noise_png']}")
            print(f"Noisy-PUF PDF: {outputs['noise_pdf']}")
        return 0

    payload = run_experiment(args, logger)
    with open(exp_dir / "avalanche_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    outputs = plot_results(payload, exp_dir / "avalanche_plot.png")

    correct_rows = [r for r in payload["records"] if r["distance_type"] == "hamming_bits" and r["distance"] == 0]
    wrong_rows = [r for r in payload["records"] if not (r["distance_type"] == "hamming_bits" and r["distance"] == 0)]
    correct_acc = float(np.mean([r["accuracy"] for r in correct_rows])) if correct_rows else 0.0
    wrong_acc = float(np.mean([r["accuracy"] for r in wrong_rows])) if wrong_rows else 0.0

    print(f"Experiment completed: {exp_dir}")
    print(f"Correct-key accuracy: {correct_acc:.2%}")
    print(f"Mean wrong-key accuracy: {wrong_acc:.2%}")
    print(f"Ciphertext-only accuracy: {payload['ciphertext_accuracy']:.2%}")
    print(f"Selected layers: {payload['selected_layers']}")
    print(f"Results: {exp_dir / 'avalanche_results.json'}")
    if outputs["bit_png"].exists():
        print(f"Bit-Flip Plot: {outputs['bit_png']}")
        print(f"Bit-Flip PDF: {outputs['bit_pdf']}")
    if outputs["noise_png"].exists():
        print(f"Noisy-PUF Plot: {outputs['noise_png']}")
        print(f"Noisy-PUF PDF: {outputs['noise_pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

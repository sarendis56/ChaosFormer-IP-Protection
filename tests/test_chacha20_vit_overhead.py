from argparse import Namespace

import torch
from transformers import ViTConfig, ViTForImageClassification

from src.experiments.chacha20_vit_overhead import benchmark


def test_baseline_and_protected_forwards_use_inference_mode(tmp_path, monkeypatch):
    model = ViTForImageClassification(
        ViTConfig(
            image_size=224,
            patch_size=112,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    )
    model.save_pretrained(tmp_path)
    modes = []
    forward = ViTForImageClassification.forward

    def observe_forward(self, *args, **kwargs):
        modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(ViTForImageClassification, "forward", observe_forward)
    result = benchmark(
        Namespace(
            model=str(tmp_path), local_files_only=True, device="cpu",
            batch_size=1, repetitions=1, trials=1, aggregate="median", layers=[1],
        )
    )
    assert modes and all(mode == (False, True) for mode in modes)
    assert result["baseline_ms"] > 0
    assert result["measurements"][0]["chacha20_ms"] > 0

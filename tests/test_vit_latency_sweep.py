import torch
from transformers import ViTConfig, ViTForImageClassification

from src.experiments.vit_latency_sweep import Scheme, measure


def _tiny_vit():
    config = ViTConfig(image_size=32, patch_size=16, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2)
    return ViTForImageClassification(config).eval()


def test_round_trip_restores_weights_and_encryption_changes_them():
    model = _tiny_vit()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    for mode in ("base", "secure"):
        scheme = Scheme(model, [0, 1], mode, 5, torch.device("cpu"))
        scheme.encrypt()
        assert any(not torch.equal(original[k], v) for k, v in model.state_dict().items())
        scheme.decrypt()
        assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())


def test_measure_leaves_plaintext_weights_and_reports_ratios():
    model = _tiny_vit()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    row = measure(model, torch.randn(2, 3, 32, 32), [1], 5, trials=2, reps=1, device=torch.device("cpu"))
    assert row["base_overhead"] > 0 and row["secure_overhead"] > 0
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())

import torch
from transformers import ViTConfig, ViTForImageClassification

from src.experiments.vit_latency_sweep import measure


import pytest


@pytest.mark.parametrize("pipeline", ["layerwise", "reencrypt"])
def test_measure_leaves_plaintext_weights_and_reports_ratios(pipeline):
    config = ViTConfig(image_size=32, patch_size=16, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2)
    model = ViTForImageClassification(config).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    configs = [{"layers": [1], "n": 5}, {"layers": [0, 1], "n": 3}]
    rows = measure(model, torch.randn(2, 3, 32, 32), configs, trials=3, reps=1, device=torch.device("cpu"),
                   pipeline=pipeline)
    assert len(rows) == 2 and rows[0]["baseline_ms"] == rows[1]["baseline_ms"]
    assert all(r["base_overhead"] > 0 and r["secure_overhead"] > 0 for r in rows)
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())

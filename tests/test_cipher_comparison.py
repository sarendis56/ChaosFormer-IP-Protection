import numpy as np
import torch
from transformers import ViTConfig, ViTForImageClassification

from src.experiments.cipher_comparison import aes_round, flat_bytes, methods, tea


def test_round_trips_restore_plaintext():
    data = np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8)
    for mode in ("CBC", "CTR", "GCM"):
        assert aes_round(data.tobytes(), mode) == data.tobytes()
    words = data.view(np.uint32).copy()
    tea(words, False)
    assert not np.array_equal(words, data.view(np.uint32))
    tea(words, True)
    assert np.array_equal(words, data.view(np.uint32))


def test_methods_leave_model_unchanged():
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=16)
    model = ViTForImageClassification(cfg).eval()
    before = flat_bytes(model).copy()
    for fn in methods(model, 1, torch.device("cpu")).values():
        fn()
    assert np.array_equal(flat_bytes(model), before)

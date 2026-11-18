import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification

from src.encryption.transformer_cipher import TransformerCipher

def same_bits(a, b):  # ciphertext may contain NaN bit patterns
    return torch.equal(a.view(torch.int32), b.view(torch.int32))

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("secure", [False, True])
def test_vit_layerwise_inference_matches_plaintext_and_keeps_ciphertext(device, secure):
    config = ViTConfig(image_size=32, patch_size=16, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=3, num_attention_heads=2)
    model = ViTForImageClassification(config).to(device).eval()
    x = torch.randn(2, 3, 32, 32, device=device)
    plain = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        reference = model(pixel_values=x).logits.clone()

    cipher = TransformerCipher(model, secure=secure, selected_layers=[0, 2])
    cipher.encrypt_all()
    stored = {k: v.clone() for k, v in model.state_dict().items()}
    assert any(not torch.equal(plain[k], v) for k, v in stored.items())
    with cipher.authorized_inference(), torch.inference_mode():
        for _ in range(2):
            assert torch.equal(model(pixel_values=x).logits, reference)
            assert all(same_bits(stored[k], v) for k, v in model.state_dict().items())

    cipher.decrypt_all()
    assert all(torch.equal(plain[k], v) for k, v in model.state_dict().items())

import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from src.encryption.transformer_cipher import TransformerCipher


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("secure", [False, True])
def test_graph_replayed_authorized_inference_matches_plaintext(secure):
    config = RobertaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, vocab_size=100)
    model = RobertaForSequenceClassification(config).cuda().eval()
    x = torch.randint(0, 100, (4, 16), device="cuda")
    plain = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        reference = model(input_ids=x).logits.clone()

    cipher = TransformerCipher(model, secure=secure)
    cipher.encrypt_all()
    with cipher.authorized_inference(), torch.inference_mode():
        for _ in range(2):  # replays must restore the encrypted state each pass
            assert torch.equal(model(input_ids=x).logits, reference)
    assert any(not torch.equal(plain[k], v) for k, v in model.state_dict().items())

    cipher.decrypt_all()
    assert all(torch.equal(plain[k], v) for k, v in model.state_dict().items())

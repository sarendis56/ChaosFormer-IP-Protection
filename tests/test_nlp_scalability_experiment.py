from src.experiments import nlp_scalability_experiment as experiment


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return {"input_ids": list(range(32))}


def test_causal_batches_honor_batch_size(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "load_dataset",
        lambda *args, **kwargs: iter([{"text": "enough tokens"}]),
    )

    batches = experiment._causal_batches(
        FakeTokenizer(), max_sequences=5, sequence_length=4, batch_size=2
    )

    assert [batch["input_ids"].shape for batch in batches] == [(2, 4), (2, 4), (1, 4)]
    assert batches[0]["input_ids"][0].tolist() == [0, 1, 2, 3]
    assert batches[0]["labels"][0].tolist() == [1, 2, 3, 4]



def test_wav2vec2_adapter_resolves_encoder_weights():
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    from src.encryption.transformer_cipher import TransformerCipher, transformer_layers

    config = Wav2Vec2Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=8,
        conv_dim=(8,),
        conv_stride=(2,),
        conv_kernel=(3,),
        vocab_size=8,
    )
    model = Wav2Vec2ForCTC(config)
    layers, family = transformer_layers(model)
    cipher = TransformerCipher(model, secure=False)

    assert family == "wav2vec2"
    assert len(layers) == 1
    assert len(cipher.layer_specs[0].weights) == 6
    assert cipher.encrypted_parameter_count == 4 * 8 * 8 + 2 * 8 * 16

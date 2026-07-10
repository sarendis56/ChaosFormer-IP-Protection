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

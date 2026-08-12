from types import SimpleNamespace

import torch
from torch import nn

from src.utils.vision_backbone_utils import apply_layer_weights


def _linear(rows: int, columns: int) -> nn.Linear:
    return nn.Linear(columns, rows, bias=False)


def test_apply_layer_weights_preserves_parameters():
    layer = SimpleNamespace(
        attention=SimpleNamespace(
            attention=SimpleNamespace(
                query=_linear(2, 2), key=_linear(2, 2), value=_linear(2, 2)
            ),
            output=SimpleNamespace(dense=_linear(2, 2)),
        ),
        intermediate=SimpleNamespace(dense=_linear(4, 2)),
        output=SimpleNamespace(dense=_linear(2, 4)),
    )
    parameters = {
        "query": layer.attention.attention.query.weight,
        "key": layer.attention.attention.key.weight,
        "value": layer.attention.attention.value.weight,
        "attention_output": layer.attention.output.dense.weight,
        "intermediate": layer.intermediate.dense.weight,
        "output": layer.output.dense.weight,
    }
    storage = {name: parameter.data_ptr() for name, parameter in parameters.items()}
    apply_layer_weights(
        layer,
        {
            "query": torch.full_like(parameters["query"], 1),
            "key": torch.full_like(parameters["key"], 2),
            "value": torch.full_like(parameters["value"], 3),
            "output": torch.full_like(parameters["attention_output"], 4),
        },
        {
            "intermediate": torch.full_like(parameters["intermediate"], 5),
            "output": torch.full_like(parameters["output"], 6),
        },
    )
    assert layer.attention.attention.query.weight is parameters["query"]
    assert layer.intermediate.dense.weight is parameters["intermediate"]
    assert parameters["query"].data_ptr() == storage["query"]
    assert parameters["intermediate"].data_ptr() == storage["intermediate"]
    assert torch.equal(parameters["query"], torch.ones_like(parameters["query"]))
    assert torch.equal(parameters["intermediate"], torch.full_like(parameters["intermediate"], 5))

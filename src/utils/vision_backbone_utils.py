"""
Backbone utilities for HuggingFace vision transformer models.

This repo originally targeted ViT (`model.vit.encoder.layer[...]`). To support other
vision transformer backbones (e.g., BEiT), we centralize:
- locating the transformer encoder layers
- extracting/applying attention + FFN weights in a layer
- generic classifier head replacement for downstream fine-tuning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class UnsupportedBackboneError(RuntimeError):
    pass


def _getattr_chain(obj, chain: Sequence[str]):
    cur = obj
    for attr in chain:
        if not hasattr(cur, attr):
            raise AttributeError(f"Missing attribute {attr!r} while traversing {'.'.join(chain)}")
        cur = getattr(cur, attr)
    return cur


def get_transformer_layers(model) -> Iterable:
    """
    Return the module list (or list-like) of transformer encoder layers for a supported model.

    Supports HuggingFace image-classification models that expose one of:
    - model.vit.encoder.layer
    - model.beit.encoder.layer
    - model.deit.encoder.layer
    - model.encoder.layer
    """
    candidates: List[Tuple[str, Tuple[str, ...]]] = [
        ("vit", ("vit", "encoder", "layer")),
        ("beit", ("beit", "encoder", "layer")),
        ("deit", ("deit", "encoder", "layer")),
        ("encoder", ("encoder", "layer")),
    ]
    for _name, chain in candidates:
        try:
            layers = _getattr_chain(model, chain)
            # Basic sanity: must be indexable and have len()
            _ = len(layers)
            _ = layers[0]
            return layers
        except Exception:
            continue

    raise UnsupportedBackboneError(
        "Unsupported model backbone: expected an encoder layer stack at "
        "`model.(vit|beit|deit).encoder.layer` or `model.encoder.layer`."
    )


@dataclass(frozen=True)
class LayerWeightViews:
    attention: Dict[str, torch.Tensor]
    ffn: Dict[str, torch.Tensor]


def _expect_weight(layer, chain: Sequence[str]) -> torch.Tensor:
    w = _getattr_chain(layer, chain)
    if not isinstance(w, torch.Tensor):
        raise UnsupportedBackboneError(f"Expected tensor at {'.'.join(chain)}, got {type(w)}")
    return w


def get_layer_weight_views(layer) -> LayerWeightViews:
    """
    Extract *views* (not copies) of attention + FFN weights from a transformer layer.

    Currently supports ViT/DeiT/BEiT-style encoder blocks where weights live at:
    - attention: layer.attention.attention.{query,key,value}.weight and layer.attention.output.dense.weight
    - ffn: layer.intermediate.dense.weight and layer.output.dense.weight
    """
    # Attention (QKV + output projection)
    attn = {
        "query": _expect_weight(layer, ("attention", "attention", "query", "weight")),
        "key": _expect_weight(layer, ("attention", "attention", "key", "weight")),
        "value": _expect_weight(layer, ("attention", "attention", "value", "weight")),
        "output": _expect_weight(layer, ("attention", "output", "dense", "weight")),
    }
    ffn = {
        "intermediate": _expect_weight(layer, ("intermediate", "dense", "weight")),
        "output": _expect_weight(layer, ("output", "dense", "weight")),
    }
    return LayerWeightViews(attention=attn, ffn=ffn)


def apply_layer_weights(layer, encrypted_attention: Dict[str, torch.Tensor], encrypted_ffn: Dict[str, torch.Tensor]) -> None:
    """
    Apply encrypted weights back into a transformer layer.
    """
    _getattr_chain(layer, ("attention", "attention", "query")).weight.data = encrypted_attention["query"]
    _getattr_chain(layer, ("attention", "attention", "key")).weight.data = encrypted_attention["key"]
    _getattr_chain(layer, ("attention", "attention", "value")).weight.data = encrypted_attention["value"]
    _getattr_chain(layer, ("attention", "output", "dense")).weight.data = encrypted_attention["output"]

    _getattr_chain(layer, ("intermediate", "dense")).weight.data = encrypted_ffn["intermediate"]
    _getattr_chain(layer, ("output", "dense")).weight.data = encrypted_ffn["output"]


def get_classifier_in_features(model) -> int:
    """
    Try to infer classifier input dimension for an AutoModelForImageClassification model.
    """
    # Most HF vision classification models expose `classifier` as nn.Linear
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        return int(model.classifier.in_features)

    # Some models expose `head`
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return int(model.head.in_features)

    # As a fallback, inspect parameters of the last linear layer we can find
    last_linear = None
    for _name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return int(last_linear.in_features)

    # Final fallback: config.hidden_size is common
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)

    raise UnsupportedBackboneError("Unable to infer classifier input features for this model.")


def replace_classifier_head(model, num_classes: int) -> None:
    """
    Replace the classification head in-place for common HF vision models.
    """
    in_features = get_classifier_in_features(model)

    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), nn.Linear):
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
        model.head = nn.Linear(in_features, num_classes)
    else:
        # Last resort: attach a `classifier` attribute (some models won't use it)
        model.classifier = nn.Linear(in_features, num_classes)

    # Keep config consistent when present
    if hasattr(model, "config"):
        model.config.num_labels = int(num_classes)
        # id2label/label2id are optional, but downstream scripts expect num_labels to be correct

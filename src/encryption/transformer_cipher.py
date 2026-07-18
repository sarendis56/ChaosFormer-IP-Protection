"""Architecture-neutral ChaosFormer tensor cipher for transformer layers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from .arnold_transform import arnold_triton, iarnold_triton
from .chacha20 import MasterSecret, chacha20_xor_, derive_chacha20_material
from .xor_encryption import get_stable_seed


@dataclass
class WeightSpec:
    name: str
    parameter: nn.Parameter
    operation: str
    permutation: torch.Tensor | None = None
    inverse_permutation: torch.Tensor | None = None
    diffusion_key: bytes | None = None
    diffusion_nonce: bytes | None = None


@dataclass
class LayerSpec:
    index: int
    module: nn.Module
    weights: list[WeightSpec]


def _resolve(root: object, path: str) -> object:
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def transformer_layers(model: nn.Module) -> tuple[list[nn.Module], str]:
    if hasattr(model, "roberta"):
        return list(model.roberta.encoder.layer), "roberta"
    if (
        hasattr(model, "wav2vec2")
        and hasattr(model.wav2vec2, "encoder")
        and hasattr(model.wav2vec2.encoder, "layers")
    ):
        return list(model.wav2vec2.encoder.layers), "wav2vec2"
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return list(model.model.decoder.layers), "opt"
    raise TypeError(
        f"unsupported architecture {type(model).__name__}; "
        "expected RoBERTa, Wav2Vec2, or OPT-style separate Q/K/V projections"
    )


def _weight_layouts(family: str) -> list[tuple[str, str]]:
    if family == "roberta":
        return [
            ("attention.self.query.weight", "square"),
            ("attention.self.key.weight", "square"),
            ("attention.self.value.weight", "square"),
            ("attention.output.dense.weight", "square"),
            ("intermediate.dense.weight", "ffn_columns"),
            ("output.dense.weight", "ffn_rows"),
        ]
    if family == "wav2vec2":
        return [
            ("attention.q_proj.weight", "square"),
            ("attention.k_proj.weight", "square"),
            ("attention.v_proj.weight", "square"),
            ("attention.out_proj.weight", "square"),
            ("feed_forward.intermediate_dense.weight", "ffn_columns"),
            ("feed_forward.output_dense.weight", "ffn_rows"),
        ]
    if family == "opt":
        return [
            ("self_attn.q_proj.weight", "square"),
            ("self_attn.k_proj.weight", "square"),
            ("self_attn.v_proj.weight", "square"),
            ("self_attn.out_proj.weight", "square"),
            ("fc1.weight", "ffn_columns"),
            ("fc2.weight", "ffn_rows"),
        ]
    raise ValueError(f"unknown family {family}")


class TransformerCipher:
    """Reversible layer-wise permutation with optional bit diffusion."""

    def __init__(
        self,
        model: nn.Module,
        *,
        secure: bool,
        selected_layers: list[int] | None = None,
        seed: MasterSecret = 20260710,
        arnold_key: tuple[int, int, int, int, int] = (3, 1, 1, 1, 2),
    ) -> None:
        layers, family = transformer_layers(model)
        selected = (
            list(range(len(layers))) if selected_layers is None else selected_layers
        )
        invalid = [index for index in selected if not 0 <= index < len(layers)]
        if invalid:
            raise ValueError(f"invalid layer indices: {invalid}")
        self.secure = secure
        self.seed = seed
        self.arnold_key = list(arnold_key)
        self.encrypted = False
        self.layer_specs: list[LayerSpec] = []

        for index in selected:
            module = layers[index]
            specs: list[WeightSpec] = []
            for path, operation in _weight_layouts(family):
                parameter = _resolve(module, path)
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(f"{path} did not resolve to a parameter")
                if parameter.ndim != 2:
                    raise ValueError(f"{path} is not a matrix")
                if operation == "square" and parameter.shape[0] != parameter.shape[1]:
                    raise ValueError(f"{path} is not square: {tuple(parameter.shape)}")

                permutation = inverse = None
                diffusion_key = diffusion_nonce = None
                if operation != "square":
                    hidden_dimension = (
                        parameter.shape[1]
                        if operation == "ffn_columns"
                        else parameter.shape[0]
                    )
                    generator = torch.Generator(device="cpu")
                    stable = get_stable_seed(index, path, seed)
                    generator.manual_seed(stable)
                    permutation = torch.randperm(
                        hidden_dimension, generator=generator
                    ).to(parameter.device)
                    inverse = torch.argsort(permutation)
                if secure:
                    diffusion_key, diffusion_nonce = derive_chacha20_material(
                        index, path, seed
                    )
                specs.append(
                    WeightSpec(
                        name=path,
                        parameter=parameter,
                        operation=operation,
                        permutation=permutation,
                        inverse_permutation=inverse,
                        diffusion_key=diffusion_key,
                        diffusion_nonce=diffusion_nonce,
                    )
                )
            self.layer_specs.append(LayerSpec(index, module, specs))

    @property
    def encrypted_parameter_count(self) -> int:
        return sum(
            spec.parameter.numel()
            for layer in self.layer_specs
            for spec in layer.weights
        )

    @staticmethod
    def _synchronize(parameter: nn.Parameter) -> None:
        if parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)

    def _diffuse(self, layer_index: int, spec: WeightSpec) -> None:
        del layer_index  # Material is pre-derived for every selected tensor.
        if spec.diffusion_key is None or spec.diffusion_nonce is None:
            raise RuntimeError(f"missing diffusion material for {spec.name}")
        chacha20_xor_(
            spec.parameter.data, spec.diffusion_key, spec.diffusion_nonce
        )

    def _permute(self, spec: WeightSpec) -> None:
        weight = spec.parameter.data
        if spec.operation == "square":
            transformed = arnold_triton(weight, self.arnold_key)
        elif spec.operation == "ffn_columns":
            transformed = weight[:, spec.permutation].contiguous()
        else:
            transformed = weight[spec.permutation].contiguous()
        weight.copy_(transformed)

    def _unpermute(self, spec: WeightSpec) -> None:
        weight = spec.parameter.data
        if spec.operation == "square":
            transformed = iarnold_triton(weight, self.arnold_key)
        elif spec.operation == "ffn_columns":
            transformed = weight[:, spec.inverse_permutation].contiguous()
        else:
            transformed = weight[spec.inverse_permutation].contiguous()
        weight.copy_(transformed)

    @torch.no_grad()
    def encrypt_layer(self, layer: LayerSpec) -> None:
        for spec in layer.weights:
            self._permute(spec)
            if self.secure:
                self._diffuse(layer.index, spec)

    @torch.no_grad()
    def decrypt_layer(self, layer: LayerSpec) -> None:
        for spec in layer.weights:
            if self.secure:
                self._diffuse(layer.index, spec)
            self._unpermute(spec)

    @torch.no_grad()
    def encrypt_all(self) -> None:
        if self.encrypted:
            raise RuntimeError("model is already encrypted")
        for layer in self.layer_specs:
            self.encrypt_layer(layer)
        if self.layer_specs:
            self._synchronize(self.layer_specs[-1].weights[-1].parameter)
        self.encrypted = True

    @torch.no_grad()
    def decrypt_all(self) -> None:
        if not self.encrypted:
            raise RuntimeError("model is not encrypted")
        for layer in self.layer_specs:
            self.decrypt_layer(layer)
        if self.layer_specs:
            self._synchronize(self.layer_specs[-1].weights[-1].parameter)
        self.encrypted = False

    @contextmanager
    def authorized_inference(self) -> Iterator[None]:
        """Decrypt immediately before each protected layer and re-encrypt after it."""
        if not self.encrypted:
            raise RuntimeError("authorized inference expects encrypted-at-rest weights")
        handles: list[torch.utils.hooks.RemovableHandle] = []

        for layer in self.layer_specs:
            handles.append(
                layer.module.register_forward_pre_hook(
                    lambda _module, _inputs, layer=layer: self.decrypt_layer(layer)
                )
            )
            handles.append(
                layer.module.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: (
                        self.encrypt_layer(layer),
                        output,
                    )[1],
                    always_call=True,
                )
            )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

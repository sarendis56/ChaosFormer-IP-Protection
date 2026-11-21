"""Architecture-neutral ChaosFormer tensor cipher for transformer layers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from .arnold_transform import arnold_optimized, arnold_triton, iarnold_triton
from .chacha20 import MasterSecret, chacha20_xor_, derive_chacha20_material
from .keys import acm_key, ffn_permutation


@dataclass
class WeightSpec:
    name: str
    parameter: nn.Parameter
    operation: str
    acm_key: list[int] | None = None
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
    if hasattr(model, "vit"):
        return list(model.vit.encoder.layer), "vit"
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
        "expected ViT, RoBERTa, Wav2Vec2, or OPT-style separate Q/K/V projections"
    )


def _weight_layouts(family: str) -> list[tuple[str, str]]:
    if family == "vit":
        return [
            ("attention.attention.query.weight", "square"),
            ("attention.attention.key.weight", "square"),
            ("attention.attention.value.weight", "square"),
            ("attention.output.dense.weight", "square"),
            ("intermediate.dense.weight", "ffn_columns"),
            ("output.dense.weight", "ffn_rows"),
        ]
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
        arnold_key: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        """``arnold_key`` overrides the derived ACM keys (e.g. to sweep n); by default
        every attention matrix gets its own key from ``seed`` (``keys.py``)."""
        layers, family = transformer_layers(model)
        selected = (
            list(range(len(layers))) if selected_layers is None else selected_layers
        )
        invalid = [index for index in selected if not 0 <= index < len(layers)]
        if invalid:
            raise ValueError(f"invalid layer indices: {invalid}")
        self.secure = secure
        self.seed = seed
        self.encrypted = False
        self.layer_specs: list[LayerSpec] = []
        # CPU ACM: flat gather indices per shape, (encrypt, decrypt). The ACM is a fixed
        # permutation of positions, so one precomputed index replaces the per-call map.
        self._acm_index: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        for index in selected:
            module = layers[index]
            specs: list[WeightSpec] = []
            n_square = n_ffn = 0
            for path, operation in _weight_layouts(family):
                parameter = _resolve(module, path)
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(f"{path} did not resolve to a parameter")
                if parameter.ndim != 2:
                    raise ValueError(f"{path} is not a matrix")
                if operation == "square" and parameter.shape[0] != parameter.shape[1]:
                    raise ValueError(f"{path} is not square: {tuple(parameter.shape)}")
                key = None
                if operation == "square":
                    key = list(arnold_key) if arnold_key is not None else acm_key(
                        seed, index, n_square, parameter.shape[0])
                    n_square += 1
                    if not parameter.is_cuda:
                        self._cache_acm_index(tuple(parameter.shape), key)

                permutation = inverse = None
                diffusion_key = diffusion_nonce = None
                if operation != "square":
                    hidden_dimension = (
                        parameter.shape[1]
                        if operation == "ffn_columns"
                        else parameter.shape[0]
                    )
                    n_ffn += 1
                    permutation = ffn_permutation(
                        seed, index, n_ffn, hidden_dimension
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
                        acm_key=key,
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

    def _cache_acm_index(self, shape: tuple[int, int], key: list[int]) -> None:
        if (shape, *key) not in self._acm_index:
            positions = torch.arange(shape[0] * shape[1], dtype=torch.float64).view(shape)
            forward = arnold_optimized(positions, key, fixed_schedule=True).long()  # exact below 2**53
            self._acm_index[(shape, *key)] = (forward, torch.argsort(forward.view(-1)).view(shape))

    def _transform_into(self, spec: WeightSpec, source: torch.Tensor, out: torch.Tensor, *, inverse: bool) -> None:
        """Write the (inverse) permutation of ``source`` into ``out`` without temporaries."""
        if spec.operation == "square":
            if source.is_cuda:
                (iarnold_triton if inverse else arnold_triton)(source, spec.acm_key, fixed_schedule=True, out=out)
            else:
                torch.take(source, self._acm_index[(tuple(source.shape), *spec.acm_key)][inverse], out=out)
            return
        index = spec.inverse_permutation if inverse else spec.permutation
        if spec.operation == "ffn_rows":
            torch.index_select(source, 0, index, out=out)
        elif source.is_cuda:
            torch.index_select(source, 1, index, out=out)
        else:
            out.copy_(source[:, index])  # faster than CPU index_select along columns

    @torch.no_grad()
    def encrypt_layer(self, layer: LayerSpec) -> None:
        for spec in layer.weights:
            weight = spec.parameter.data
            out = torch.empty_like(weight)
            self._transform_into(spec, weight, out, inverse=False)
            weight.copy_(out)
            if self.secure:
                self._diffuse(layer.index, spec)

    @torch.no_grad()
    def decrypt_layer(self, layer: LayerSpec) -> None:
        for spec in layer.weights:
            if self.secure:
                self._diffuse(layer.index, spec)
            weight = spec.parameter.data
            out = torch.empty_like(weight)
            self._transform_into(spec, weight, out, inverse=True)
            weight.copy_(out)

    @torch.no_grad()
    def _decrypt_into(
        self, layer: LayerSpec, stored: list[torch.Tensor], plain: list[torch.Tensor], scratch: list[torch.Tensor]
    ) -> None:
        """Decrypt ``stored`` (left unchanged) into ``plain``."""
        for spec, source, out, work in zip(layer.weights, stored, plain, scratch, strict=True):
            if self.secure:
                work.copy_(source)
                chacha20_xor_(work, spec.diffusion_key, spec.diffusion_nonce)
                source = work
            self._transform_into(spec, source, out, inverse=True)

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
        """Decrypt each protected layer into shared buffers immediately before it runs.

        The stored ciphertext is never modified, so nothing is re-encrypted: while a
        layer runs its parameters point at the plaintext buffers, afterwards back at the
        ciphertext. The buffers are zeroed after the last protected layer and on exit,
        so at most one layer is in plaintext at a time. On CUDA each layer's decryption
        replays a captured CUDA graph.
        """
        if not self.encrypted:
            raise RuntimeError("authorized inference expects encrypted-at-rest weights")
        if not self.layer_specs:
            yield
            return
        shapes = [spec.parameter.shape for spec in self.layer_specs[0].weights]
        if any([spec.parameter.shape for spec in layer.weights] != shapes for layer in self.layer_specs):
            raise ValueError("authorized inference expects protected layers with equal weight shapes")
        with torch.inference_mode(False):  # parameters may not point at inference tensors
            plain = [torch.empty_like(spec.parameter) for spec in self.layer_specs[0].weights]
            scratch = [torch.empty_like(p) for p in plain] if self.secure else plain
        stored = [[spec.parameter.data for spec in layer.weights] for layer in self.layer_specs]
        steps: list[Callable[[], None]] = [
            lambda layer=layer, data=data: self._decrypt_into(layer, data, plain, scratch)
            for layer, data in zip(self.layer_specs, stored, strict=True)
        ]

        def zero_plaintext() -> None:
            for buffer in plain:
                buffer.zero_()

        steps.append(zero_plaintext)
        if plain[0].is_cuda:
            steps = graph_replays(steps)
        *decrypts, wipe = steps
        last = max(layer.index for layer in self.layer_specs)

        def attach(layer: LayerSpec, data: list[torch.Tensor], decrypt: Callable[[], None]):
            def before(_module, _inputs):
                decrypt()
                for spec, buffer in zip(layer.weights, plain, strict=True):
                    spec.parameter.data = buffer

            def after(_module, _inputs, output):
                for spec, tensor in zip(layer.weights, data, strict=True):
                    spec.parameter.data = tensor
                if layer.index == last:
                    wipe()
                return output

            return (
                layer.module.register_forward_pre_hook(before),
                layer.module.register_forward_hook(after, always_call=True),
            )

        handles = [
            handle
            for layer, data, decrypt in zip(self.layer_specs, stored, decrypts, strict=True)
            for handle in attach(layer, data, decrypt)
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            wipe()


def graph_replays(steps: list[Callable[[], None]]) -> list[Callable[[], None]]:
    """Capture each CUDA step as a graph and return the replay callables.

    Running ``steps`` in order must leave the weights unchanged (e.g. decrypt
    then encrypt), because they run once eagerly to warm up kernels. Replays
    must follow the capture order, since the graphs share one memory pool.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for step in steps:
            step()
    torch.cuda.current_stream().wait_stream(stream)
    pool = torch.cuda.graph_pool_handle()
    replays = []
    for step in steps:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):  # records kernels, does not run them
            step()
        replays.append(graph.replay)
    return replays

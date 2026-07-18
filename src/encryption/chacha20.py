"""ChaCha20 keystream generation for bitwise weight diffusion.

The GPU path generates the keystream in registers and XORs it directly into a
contiguous tensor.  No tensor-sized key is allocated.  The implementation uses
the IETF ChaCha20 state layout from RFC 8439 (256-bit key, 32-bit block counter,
and 96-bit nonce).
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from typing import Union

import numpy as np
import torch
import triton
import triton.language as tl


MasterSecret = Union[int, str, bytes, bytearray]


def _secret_bytes(secret: MasterSecret) -> bytes:
    if isinstance(secret, int):
        if secret < 0:
            raise ValueError("master secret must be non-negative")
        width = max(1, (secret.bit_length() + 7) // 8)
        return secret.to_bytes(width, "big")
    if isinstance(secret, str):
        return secret.encode("utf-8")
    return bytes(secret)


def derive_chacha20_material(
    layer_idx: int,
    weight_name: str,
    master_secret: MasterSecret,
) -> tuple[bytes, bytes]:
    """Derive a tensor-specific ChaCha20 key and nonce.

    The SHA-256 calls are a lightweight, domain-separated KDF for the software
    prototype.  A deployment can supply the same material from its system KDF.
    """

    context = (
        int(layer_idx).to_bytes(8, "big", signed=False)
        + len(weight_name.encode("utf-8")).to_bytes(4, "big")
        + weight_name.encode("utf-8")
    )
    secret = _secret_bytes(master_secret)
    root = hashlib.sha256(b"ChaosFormer/KDF/root/v1\x00" + secret).digest()
    key = hmac.new(
        root, b"ChaosFormer/ChaCha20/key/v1\x00" + context, hashlib.sha256
    ).digest()
    nonce = hmac.new(
        root, b"ChaosFormer/ChaCha20/nonce/v1\x00" + context, hashlib.sha256
    ).digest()[:12]
    return key, nonce


def _rotl32_host(value: int, shift: int) -> int:
    value &= 0xFFFFFFFF
    return ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF


def _quarter_round_host(
    state: list[int], a: int, b: int, c: int, d: int
) -> None:
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32_host(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32_host(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32_host(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32_host(state[b] ^ state[c], 7)


def chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """Return one RFC 8439 ChaCha20 block (64 bytes)."""

    if len(key) != 32:
        raise ValueError("ChaCha20 key must be 32 bytes")
    if len(nonce) != 12:
        raise ValueError("ChaCha20 nonce must be 12 bytes")
    if not 0 <= counter <= 0xFFFFFFFF:
        raise ValueError("ChaCha20 counter must fit in 32 bits")

    initial = [
        0x61707865,
        0x3320646E,
        0x79622D32,
        0x6B206574,
        *struct.unpack("<8I", key),
        counter,
        *struct.unpack("<3I", nonce),
    ]
    state = initial.copy()
    for _ in range(10):
        _quarter_round_host(state, 0, 4, 8, 12)
        _quarter_round_host(state, 1, 5, 9, 13)
        _quarter_round_host(state, 2, 6, 10, 14)
        _quarter_round_host(state, 3, 7, 11, 15)
        _quarter_round_host(state, 0, 5, 10, 15)
        _quarter_round_host(state, 1, 6, 11, 12)
        _quarter_round_host(state, 2, 7, 8, 13)
        _quarter_round_host(state, 3, 4, 9, 14)
    return struct.pack(
        "<16I", *((value + base) & 0xFFFFFFFF for value, base in zip(state, initial))
    )


@triton.jit
def _rotl32(value, shift: tl.constexpr):
    return ((value << shift) | (value >> (32 - shift))).to(tl.uint32)


@triton.jit
def _quarter_round(a, b, c, d):
    a = (a + b).to(tl.uint32)
    d = _rotl32(d ^ a, 16)
    c = (c + d).to(tl.uint32)
    b = _rotl32(b ^ c, 12)
    a = (a + b).to(tl.uint32)
    d = _rotl32(d ^ a, 8)
    c = (c + d).to(tl.uint32)
    b = _rotl32(b ^ c, 7)
    return a, b, c, d


@triton.jit
def _xor_stream_word(data_ptr, offsets, mask, stream):
    value = tl.load(data_ptr + offsets, mask=mask, other=0)
    tl.store(data_ptr + offsets, value ^ stream.to(tl.int32), mask=mask)


@triton.jit
def _chacha20_xor_kernel(
    data_ptr,
    n_words,
    k0,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    k7,
    n0,
    n1,
    n2,
    initial_counter,
    BLOCKS: tl.constexpr,
):
    block_ids = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    word_base = block_ids * 16
    active = word_base < n_words

    c0 = tl.full((BLOCKS,), 0x61707865, tl.uint32)
    c1 = tl.full((BLOCKS,), 0x3320646E, tl.uint32)
    c2 = tl.full((BLOCKS,), 0x79622D32, tl.uint32)
    c3 = tl.full((BLOCKS,), 0x6B206574, tl.uint32)
    c4 = tl.full((BLOCKS,), k0, tl.uint32)
    c5 = tl.full((BLOCKS,), k1, tl.uint32)
    c6 = tl.full((BLOCKS,), k2, tl.uint32)
    c7 = tl.full((BLOCKS,), k3, tl.uint32)
    c8 = tl.full((BLOCKS,), k4, tl.uint32)
    c9 = tl.full((BLOCKS,), k5, tl.uint32)
    c10 = tl.full((BLOCKS,), k6, tl.uint32)
    c11 = tl.full((BLOCKS,), k7, tl.uint32)
    c12 = (block_ids + initial_counter).to(tl.uint32)
    c13 = tl.full((BLOCKS,), n0, tl.uint32)
    c14 = tl.full((BLOCKS,), n1, tl.uint32)
    c15 = tl.full((BLOCKS,), n2, tl.uint32)

    x0, x1, x2, x3 = c0, c1, c2, c3
    x4, x5, x6, x7 = c4, c5, c6, c7
    x8, x9, x10, x11 = c8, c9, c10, c11
    x12, x13, x14, x15 = c12, c13, c14, c15

    for _ in range(10):
        x0, x4, x8, x12 = _quarter_round(x0, x4, x8, x12)
        x1, x5, x9, x13 = _quarter_round(x1, x5, x9, x13)
        x2, x6, x10, x14 = _quarter_round(x2, x6, x10, x14)
        x3, x7, x11, x15 = _quarter_round(x3, x7, x11, x15)
        x0, x5, x10, x15 = _quarter_round(x0, x5, x10, x15)
        x1, x6, x11, x12 = _quarter_round(x1, x6, x11, x12)
        x2, x7, x8, x13 = _quarter_round(x2, x7, x8, x13)
        x3, x4, x9, x14 = _quarter_round(x3, x4, x9, x14)

    y0 = (x0 + c0).to(tl.uint32)
    y1 = (x1 + c1).to(tl.uint32)
    y2 = (x2 + c2).to(tl.uint32)
    y3 = (x3 + c3).to(tl.uint32)
    y4 = (x4 + c4).to(tl.uint32)
    y5 = (x5 + c5).to(tl.uint32)
    y6 = (x6 + c6).to(tl.uint32)
    y7 = (x7 + c7).to(tl.uint32)
    y8 = (x8 + c8).to(tl.uint32)
    y9 = (x9 + c9).to(tl.uint32)
    y10 = (x10 + c10).to(tl.uint32)
    y11 = (x11 + c11).to(tl.uint32)
    y12 = (x12 + c12).to(tl.uint32)
    y13 = (x13 + c13).to(tl.uint32)
    y14 = (x14 + c14).to(tl.uint32)
    y15 = (x15 + c15).to(tl.uint32)

    # A program computes BLOCKS independent ChaCha blocks.  Each call below is
    # one word position across those blocks, avoiding redundant block work.
    _xor_stream_word(data_ptr, word_base + 0, active & (word_base + 0 < n_words), y0)
    _xor_stream_word(data_ptr, word_base + 1, active & (word_base + 1 < n_words), y1)
    _xor_stream_word(data_ptr, word_base + 2, active & (word_base + 2 < n_words), y2)
    _xor_stream_word(data_ptr, word_base + 3, active & (word_base + 3 < n_words), y3)
    _xor_stream_word(data_ptr, word_base + 4, active & (word_base + 4 < n_words), y4)
    _xor_stream_word(data_ptr, word_base + 5, active & (word_base + 5 < n_words), y5)
    _xor_stream_word(data_ptr, word_base + 6, active & (word_base + 6 < n_words), y6)
    _xor_stream_word(data_ptr, word_base + 7, active & (word_base + 7 < n_words), y7)
    _xor_stream_word(data_ptr, word_base + 8, active & (word_base + 8 < n_words), y8)
    _xor_stream_word(data_ptr, word_base + 9, active & (word_base + 9 < n_words), y9)
    _xor_stream_word(data_ptr, word_base + 10, active & (word_base + 10 < n_words), y10)
    _xor_stream_word(data_ptr, word_base + 11, active & (word_base + 11 < n_words), y11)
    _xor_stream_word(data_ptr, word_base + 12, active & (word_base + 12 < n_words), y12)
    _xor_stream_word(data_ptr, word_base + 13, active & (word_base + 13 < n_words), y13)
    _xor_stream_word(data_ptr, word_base + 14, active & (word_base + 14 < n_words), y14)
    _xor_stream_word(data_ptr, word_base + 15, active & (word_base + 15 < n_words), y15)


def _words_le(value: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(value) // 4}I", value)


def chacha20_xor_(
    tensor: torch.Tensor,
    key: bytes,
    nonce: bytes,
    *,
    initial_counter: int = 0,
) -> torch.Tensor:
    """XOR an arbitrary contiguous tensor with an RFC 8439 keystream in-place."""

    if len(key) != 32 or len(nonce) != 12:
        raise ValueError("ChaCha20 requires a 32-byte key and 12-byte nonce")
    if not tensor.is_contiguous():
        raise ValueError("ChaCha20 diffusion requires a contiguous tensor")
    if tensor.numel() == 0:
        return tensor
    if tensor.numel() * tensor.element_size() % 4:
        raise ValueError("tensor byte length must be divisible by four")

    n_words = tensor.numel() * tensor.element_size() // 4
    n_blocks = triton.cdiv(n_words, 16)
    if initial_counter < 0 or initial_counter + n_blocks > 2**32:
        raise ValueError("ChaCha20 counter space exhausted")

    key_words = _words_le(key)
    nonce_words = _words_le(nonce)
    if tensor.is_cuda:
        word_view = tensor.view(-1).view(torch.int32)
        blocks_per_program = 32
        grid = (triton.cdiv(n_blocks, blocks_per_program),)
        _chacha20_xor_kernel[grid](
            word_view,
            n_words,
            *key_words,
            *nonce_words,
            initial_counter,
            BLOCKS=blocks_per_program,
            num_warps=4,
        )
        return tensor

    raw = tensor.detach().view(torch.uint8).numpy().reshape(-1)
    stream = np.empty(raw.size, dtype=np.uint8)
    for block_index in range(n_blocks):
        block = chacha20_block(key, initial_counter + block_index, nonce)
        start = block_index * 64
        end = min(start + 64, raw.size)
        stream[start:end] = np.frombuffer(block, dtype=np.uint8)[: end - start]
    np.bitwise_xor(raw, stream, out=raw)
    return tensor

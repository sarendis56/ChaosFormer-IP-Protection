"""Key schedule of ChaosFormer (manuscript Secs. 4.2 and 4.2.3).

K_i = HKDF(K_mvs, i) is 512 bits: K_i^attn || K_i^FFN, 256 bits each.
K_i^attn holds one 64-bit block per attention matrix j in (Q, K, V, O):
29-bit b_p, 29-bit b_q, 6-bit b_n, giving p, q = 1 + (b mod (S-1)) and
n = 3 + (b_n mod 32). Each FFN matrix f in (1, 2) is shuffled by Fisher-Yates
with indices drawn from a ChaCha20 stream keyed by HKDF(K_i^FFN, f).
Diffusion material is K_t^diff || N_t = HKDF(K_mvs, i || id(t)).
"""

from __future__ import annotations

import hashlib
import hmac
from functools import lru_cache

import torch

from .chacha20 import MasterSecret, _secret_bytes, chacha20_block

ATTENTION_ORDER = ("query", "key", "value", "output")  # j = 0..3
N_MIN, N_VALUES = 3, 32


def hkdf(ikm: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-SHA-256 with an empty salt."""
    prk = hmac.new(b"\x00" * 32, ikm, hashlib.sha256).digest()
    out, block = b"", b""
    for counter in range(1, -(-length // 32) + 1):
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
    return out[:length]


def _encode(*fields: int | str) -> bytes:
    """Length-delimited encoding of the HKDF context fields."""
    out = b""
    for field in fields:
        raw = field.to_bytes(8, "big") if isinstance(field, int) else field.encode()
        out += len(raw).to_bytes(4, "big") + raw
    return out


@lru_cache(maxsize=None)
def layer_keys(secret: bytes, layer: int) -> tuple[bytes, bytes]:
    k_i = hkdf(secret, _encode("layer", layer), 64)
    return k_i[:32], k_i[32:]


def acm_key(secret: MasterSecret, layer: int, matrix: int, size: int) -> list[int]:
    """Arnold key [n, 1, p, q, pq+1 mod S] for attention matrix ``matrix`` of ``layer``."""
    block = int.from_bytes(layer_keys(_secret_bytes(secret), layer)[0][8 * matrix: 8 * matrix + 8], "big")
    b_p, b_q, b_n = block >> 35, (block >> 6) & (2**29 - 1), block & 63
    p, q = 1 + b_p % (size - 1), 1 + b_q % (size - 1)
    return [N_MIN + b_n % N_VALUES, 1, p, q, (p * q + 1) % size]


def ffn_permutation(secret: MasterSecret, layer: int, f: int, size: int) -> torch.Tensor:
    """Fisher-Yates permutation of ``size`` indices for FFN matrix f (1 or 2)."""
    key = hkdf(layer_keys(_secret_bytes(secret), layer)[1], _encode("ffn", f), 32)

    def words():
        counter = 0
        while True:
            block = chacha20_block(key, counter, b"\x00" * 12)
            yield from (int.from_bytes(block[k:k + 4], "little") for k in range(0, 64, 4))
            counter += 1

    stream, perm = words(), list(range(size))
    for k in range(size - 1, 0, -1):
        limit = 2**32 - 2**32 % (k + 1)  # reject the biased tail
        while (w := next(stream)) >= limit:
            pass
        j = w % (k + 1)
        perm[k], perm[j] = perm[j], perm[k]
    return torch.tensor(perm, dtype=torch.long)


def diffusion_material(secret: MasterSecret, layer: int, tensor_id: str) -> tuple[bytes, bytes]:
    """ChaCha20 key (256 bits) and nonce (96 bits) for one protected tensor."""
    out = hkdf(_secret_bytes(secret), _encode("diffusion", layer, tensor_id), 44)
    return out[:32], out[32:]


if __name__ == "__main__":  # self-check
    # RFC 5869 test case 3 (empty salt and info).
    assert hkdf(bytes.fromhex("0b" * 22), b"", 42).hex() == (
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8")
    for size in (384, 768):
        for j in range(4):
            n, a, b, c, d = acm_key(b"k", 0, j, size)
            assert 3 <= n <= 34 and 1 <= b < size and 1 <= c < size and (a * d - b * c) % size == 1
    perm = ffn_permutation(b"k", 0, 1, 768)
    assert sorted(perm.tolist()) == list(range(768))
    assert not torch.equal(perm, ffn_permutation(b"k", 0, 2, 768))
    assert diffusion_material(b"k", 0, "a") != diffusion_material(b"k", 1, "a")
    print("keys self-check passed")

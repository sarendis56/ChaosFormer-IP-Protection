"""
XOR-based encryption for neural network weights.

This module provides bit-level XOR encryption that transforms model weights
into white noise, providing an additional layer of protection on top of
Arnold Cat Map and permutation-based encryption.

The XOR encryption uses domain-separated ChaCha20 key generation where each
layer and weight tensor gets a unique key and nonce based on its position and
name.  The deployment master secret is supplied by the PUF/KDF pipeline.

Optimized for performance with:
- In-place operations to reduce memory allocation
- Local Generator to avoid global state synchronization
- Stable hashing for deterministic key generation across runs
"""

import torch
import hashlib
import triton
import triton.language as tl
from typing import Tuple, Optional, Union

from .chacha20 import MasterSecret, chacha20_xor_, derive_chacha20_material


def xor_encrypt_decrypt_numba(weight: torch.Tensor,
                              layer_idx: int,
                              weight_name: str,
                              seed_base: MasterSecret = 42):
    """Compatibility wrapper for in-place ChaCha20 diffusion on CPU or GPU."""
    if weight.is_cuda:
        # Use on-the-fly key generation to avoid shape mismatches
        return xor_encrypt_decrypt_triton(
            weight,
            layer_idx=layer_idx,
            weight_name=weight_name,
            seed_base=seed_base
        )

    key, nonce = derive_chacha20_material(layer_idx, weight_name, seed_base)
    return chacha20_xor_(weight, key, nonce)


@triton.jit
def xor_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x ^ y
    tl.store(x_ptr + offsets, output, mask=mask)


def xor_encrypt_decrypt_triton(weight: torch.Tensor, xor_key: Optional[torch.Tensor] = None,
                               layer_idx: Optional[int] = None, weight_name: Optional[str] = None,
                               seed_base: MasterSecret = 42):
    """
    Triton implementation of XOR encryption/decryption.
    Modifies weight in-place.
    
    Can use either a pre-allocated xor_key or generate a ChaCha20 keystream
    on-the-fly if xor_key is None.
    """
    if not weight.is_cuda:
        # Pre-allocated key case
        if xor_key is not None:
            return xor_encrypt_decrypt(weight, xor_key, in_place=True)
        # On-the-fly case
        return xor_encrypt_decrypt_numba(weight, layer_idx, weight_name, seed_base)
    
    n_elements = weight.numel()
    if xor_key is not None and weight.dtype == torch.float32 and xor_key.numel() == n_elements:
        weight_view = weight.view(torch.int32)
        xor_key_view = xor_key.view(weight_view.dtype)
        def grid(meta):
            return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        xor_kernel[grid](weight_view, xor_key_view, n_elements, BLOCK_SIZE=1024)
    else:
        if layer_idx is None or weight_name is None:
            raise ValueError("Either xor_key must be provided, or both layer_idx and weight_name")
        key, nonce = derive_chacha20_material(layer_idx, weight_name, seed_base)
        chacha20_xor_(weight, key, nonce)
        
    return weight


def get_stable_seed(layer_idx: int, weight_name: str, seed_base: MasterSecret) -> int:
    """
    Creates a run-independent deterministic seed using SHA256.
    
    This replaces Python's hash() which is randomized per-process by default,
    ensuring the same seed is generated across different runs.
    
    Args:
        layer_idx: Index of the layer
        weight_name: Name/identifier of the weight
        seed_base: Base seed value
        
    Returns:
        int: Deterministic seed value
    """
    content = f"{layer_idx}_{weight_name}_{seed_base}".encode('utf-8')
    return int(hashlib.sha256(content).hexdigest(), 16) % (2**32)


def generate_xor_key(weight_shape: Tuple[int, ...], 
                     layer_idx: int, 
                     weight_name: str,
                     device: Union[str, torch.device] = 'cuda',
                     seed_base: MasterSecret = 42) -> torch.Tensor:
    """
    Generate deterministic XOR key for a weight tensor (PUF-derived style).
    
    This function generates a deterministic random key based on the layer index
    and weight name, simulating PUF-derived keys that are reproducible but
    unique per weight tensor.
    
    Args:
        weight_shape: Shape of the weight tensor
        layer_idx: Index of the layer
        weight_name: Name/identifier of the weight (e.g., 'query', 'intermediate')
        device: Device to place the key tensor on ('cuda' or 'cpu')
        seed_base: Base seed value for key generation
        
    Returns:
        torch.Tensor: XOR key tensor as int32 for bit-level XOR operations
        
    Example:
        >>> key = generate_xor_key((768, 768), layer_idx=0, weight_name='query')
        >>> key.shape  # torch.Size([768, 768])
        >>> key.dtype  # torch.int32
    """
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    key_bytes, nonce = derive_chacha20_material(layer_idx, weight_name, seed_base)
    key = torch.empty(weight_shape, device=device_obj, dtype=torch.int32)
    key.zero_()
    chacha20_xor_(key, key_bytes, nonce)
    
    return key


def xor_encrypt_decrypt(weight: torch.Tensor, 
                       xor_key: Optional[torch.Tensor] = None,
                       layer_idx: Optional[int] = None,
                       weight_name: Optional[str] = None,
                       seed_base: MasterSecret = 42,
                       in_place: bool = False) -> torch.Tensor:
    """
    Apply XOR encryption/decryption to a float weight tensor.
    
    XOR is its own inverse, so this function works for both encryption and decryption.
    Uses optimized in-place operations when possible to reduce memory allocation.
    
    Args:
        weight: Weight tensor (float32)
        xor_key: XOR key tensor (int32) with same shape as weight. If None, generates key
                 using layer_idx and weight_name.
        layer_idx: Layer index for key generation (required if xor_key is None)
        weight_name: Weight name for key generation (required if xor_key is None)
        seed_base: Base seed for key generation (used if xor_key is None)
        in_place: If True, modifies weight tensor in-place. If False, returns a new tensor.
        
    Returns:
        torch.Tensor: Encrypted/decrypted weight tensor (float32) with same shape
        
    Raises:
        ValueError: If weight and key shapes don't match, or if xor_key is None but
                   layer_idx/weight_name are not provided
        
    Example:
        >>> weight = torch.randn(768, 768)
        >>> encrypted = xor_encrypt_decrypt(weight, layer_idx=0, weight_name='query')
        >>> decrypted = xor_encrypt_decrypt(encrypted, layer_idx=0, weight_name='query')
        >>> torch.allclose(weight, decrypted)  # True
    """
    # Generate key if not provided
    if xor_key is None:
        if layer_idx is None or weight_name is None:
            raise ValueError("Either xor_key must be provided, or both layer_idx and weight_name")
        # Ensure key is generated on the same device as weight
        device = weight.device
        xor_key = generate_xor_key(weight.shape, layer_idx, weight_name, device, seed_base)
    
    if weight.shape != xor_key.shape:
        raise ValueError(f"Weight shape {weight.shape} doesn't match key shape {xor_key.shape}")
    
    # Ensure key is on the same device as weight
    xor_key = xor_key.to(weight.device)
    
    # Convert float32 to int32 for bit-level XOR by reinterpreting bits
    if in_place:
        # In-place operation: modify weight directly
        weight_int = weight.view(torch.int32)
        weight_int.bitwise_xor_(xor_key)
        return weight
    else:
        # Out-of-place operation: create new tensor
        weight_int = weight.view(torch.int32).clone()
        weight_int.bitwise_xor_(xor_key)
        return weight_int.view(torch.float32)


def encrypt_weights_xor(weights: dict, 
                       layer_idx: int,
                       device: Union[str, torch.device] = 'cuda',
                       seed_base: int = 42,
                       in_place: bool = False) -> dict:
    """
    Encrypt multiple weight tensors using XOR encryption.
    
    Args:
        weights: Dictionary mapping weight names to tensors
        layer_idx: Index of the layer containing these weights
        device: Device for key generation
        seed_base: Base seed for key generation
        in_place: If True, modifies weights in-place. If False, returns new tensors.
        
    Returns:
        dict: Dictionary with encrypted weights (same keys as input)
        
    Example:
        >>> weights = {
        ...     'query': torch.randn(768, 768),
        ...     'key': torch.randn(768, 768)
        ... }
        >>> encrypted = encrypt_weights_xor(weights, layer_idx=0)
    """
    encrypted_weights = {}
    for weight_name, weight in weights.items():
        encrypted_weights[weight_name] = xor_encrypt_decrypt(
            weight, 
            layer_idx=layer_idx, 
            weight_name=weight_name,
            seed_base=seed_base,
            in_place=in_place
        )
    return encrypted_weights


def decrypt_weights_xor(encrypted_weights: dict,
                        layer_idx: int,
                        device: Union[str, torch.device] = 'cuda',
                        seed_base: int = 42,
                        in_place: bool = False) -> dict:
    """
    Decrypt multiple weight tensors using XOR decryption.
    
    Since XOR is its own inverse, this is identical to encryption.
    
    Args:
        encrypted_weights: Dictionary mapping weight names to encrypted tensors
        layer_idx: Index of the layer containing these weights
        device: Device for key generation
        seed_base: Base seed for key generation
        in_place: If True, modifies weights in-place. If False, returns new tensors.
        
    Returns:
        dict: Dictionary with decrypted weights (same keys as input)
    """
    return encrypt_weights_xor(encrypted_weights, layer_idx, device, seed_base, in_place)


def verify_xor_encryption_cycle(weight: torch.Tensor,
                                layer_idx: int,
                                weight_name: str,
                                tolerance: float = 1e-5,
                                seed_base: int = 42) -> bool:
    """
    Verify that XOR encryption followed by decryption recovers original weight.
    
    Args:
        weight: Original weight tensor
        layer_idx: Layer index for key generation
        weight_name: Weight name for key generation
        tolerance: Numerical tolerance for comparison
        seed_base: Base seed for key generation
        
    Returns:
        bool: True if encryption cycle is successful
        
    Example:
        >>> weight = torch.randn(768, 768)
        >>> is_valid = verify_xor_encryption_cycle(weight, 0, 'query')
        >>> print(is_valid)  # True
    """
    encrypted = xor_encrypt_decrypt(weight, layer_idx=layer_idx, weight_name=weight_name, seed_base=seed_base)
    decrypted = xor_encrypt_decrypt(encrypted, layer_idx=layer_idx, weight_name=weight_name, seed_base=seed_base)
    return torch.allclose(weight, decrypted, atol=tolerance)

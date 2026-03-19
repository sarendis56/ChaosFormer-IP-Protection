"""
Permutation-based encryption for Feed-Forward Network (FFN) weights.
- Password-based permutation matrix generation using HMAC-SHA256
- Support for different matrix dimensions (768x3072 for ViT-base)
- Invertible transformations for decryption
- Memory-efficient GPU operations
"""

import torch
import hmac
import hashlib
import numpy as np
from typing import Optional, Union


def generate_permutation_matrix(size: int = 768, 
                              seed: Optional[int] = 42, 
                              password: Optional[str] = None, 
                              index: Optional[int] = None) -> torch.Tensor:
    """
    Generate a permutation matrix for weight encryption.
    
    This function creates a square permutation matrix that can be used to
    scramble the rows or columns of neural network weight matrices. When a
    password is provided, it uses HMAC-SHA256 for cryptographically secure
    permutation generation.
    
    Args:
        size: Dimension of the square permutation matrix (default: 768 for ViT)
        seed: Random seed for reproducible generation (used when password is None)
        password: Password for HMAC-based secure generation
        index: Index for generating multiple different matrices from same password
        
    Returns:
        torch.Tensor: Square permutation matrix of shape (size, size)
        
    Example:
        >>> # Simple seed-based generation
        >>> P = generate_permutation_matrix(768, seed=42)
        >>> 
        >>> # Password-based secure generation
        >>> P = generate_permutation_matrix(768, password="my_secret", index=0)
    """
    if password is not None and index is not None:
        # Create HMAC object with the password
        h = hmac.new(password.encode(), digestmod=hashlib.sha256)
        # Update with the index to generate different matrices
        h.update(str(index).encode())
        # Get the hash value and convert to integer for seed
        seed = int.from_bytes(h.digest()[:8], byteorder='big')
        
    torch.manual_seed(seed)
    
    # Generate random permutation indices
    indices = torch.randperm(size)
    # Create permutation matrix
    P = torch.eye(size)  
    P = P[:, indices]
    return P


def encrypt_ffn_weight_row_permutation(weight_tensor: torch.Tensor,
                                     P_or_indices: Union[torch.Tensor, torch.LongTensor]) -> torch.Tensor:
    """
    Encrypt FFN weight matrix using row permutation.

    Args:
        weight_tensor: Weight matrix to encrypt, shape (hidden_size, intermediate_size)
        P_or_indices: Permutation matrix (square) or 1D tensor of permutation indices.
                      Indices are much faster as they avoid matrix multiplication.

    Returns:
        torch.Tensor: Encrypted weight matrix
    """
    if P_or_indices.dim() == 1:
        # Use efficient indexing: O(NM) instead of O(N^2 M)
        return weight_tensor[P_or_indices].contiguous()
    
    # Fallback to matrix multiplication for legacy support
    return (P_or_indices.to(weight_tensor.dtype) @ weight_tensor).contiguous()


def decrypt_ffn_weight_row_permutation(encrypted_tensor: torch.Tensor,
                                     P_or_indices: Union[torch.Tensor, torch.LongTensor]) -> torch.Tensor:
    """
    Decrypt FFN weight matrix using inverse row permutation.
    """
    if P_or_indices.dim() == 1:
        # For permutation indices, we need the inverse indices
        inv_indices = torch.zeros_like(P_or_indices)
        inv_indices[P_or_indices] = torch.arange(len(P_or_indices), device=P_or_indices.device)
        return encrypted_tensor[inv_indices].contiguous()
    
    # For permutation matrix P, P^(-1) = P^T
    return (P_or_indices.t().to(encrypted_tensor.dtype) @ encrypted_tensor).contiguous()


def encrypt_ffn_weight_col_permutation(weight_tensor: torch.Tensor,
                                     P_or_indices: Union[torch.Tensor, torch.LongTensor]) -> torch.Tensor:
    """
    Encrypt FFN weight matrix using column permutation.
    """
    if P_or_indices.dim() == 1:
        return weight_tensor[:, P_or_indices].contiguous()
    
    return (weight_tensor @ P_or_indices.to(weight_tensor.dtype)).contiguous()


def decrypt_ffn_weight_col_permutation(encrypted_tensor: torch.Tensor,
                                     P_or_indices: Union[torch.Tensor, torch.LongTensor]) -> torch.Tensor:
    """
    Decrypt FFN weight matrix using inverse column permutation.
    """
    if P_or_indices.dim() == 1:
        inv_indices = torch.zeros_like(P_or_indices)
        inv_indices[P_or_indices] = torch.arange(len(P_or_indices), device=P_or_indices.device)
        return encrypted_tensor[:, inv_indices].contiguous()
    
    return (encrypted_tensor @ P_or_indices.t().to(encrypted_tensor.dtype)).contiguous()


def generate_multiple_permutation_matrices(num_matrices: int = 6,
                                         size: int = 768,
                                         password: Optional[str] = None,
                                         device: str = 'cuda') -> list:
    """
    Generate multiple permutation matrices for enhanced security.
    
    This function creates a set of different permutation matrices that can be
    used to encrypt different layers with different permutations, increasing
    the overall security of the encryption scheme.
    
    Args:
        num_matrices: Number of permutation matrices to generate
        size: Dimension of each permutation matrix
        password: Password for HMAC-based generation (if None, uses index as seed)
        device: Device to place the matrices on ('cuda' or 'cpu')
        
    Returns:
        list: List of permutation matrices as torch.Tensors
        
    Example:
        >>> matrices = generate_multiple_permutation_matrices(
        ...     num_matrices=6, password="my_secret", device='cuda'
        ... )
        >>> len(matrices)  # 6
        >>> matrices[0].shape  # torch.Size([768, 768])
    """
    matrices = []
    for i in range(num_matrices):
        if password is not None:
            # Use password-based HMAC seeding
            P = generate_permutation_matrix(
                size=size, 
                password=password, 
                index=i
            )
        else:
            # Fallback to simple index-based seeding
            P = generate_permutation_matrix(
                size=size, 
                seed=i
            )
        matrices.append(P.to(device))
    return matrices


def verify_permutation_invertibility(P: torch.Tensor, 
                                   test_tensor: Optional[torch.Tensor] = None) -> bool:
    """
    Verify that a permutation matrix is properly invertible.
    
    Args:
        P: Permutation matrix to verify
        test_tensor: Optional test tensor to verify encryption/decryption cycle
        
    Returns:
        bool: True if the permutation is invertible
        
    Example:
        >>> P = generate_permutation_matrix(768)
        >>> is_invertible = verify_permutation_invertibility(P)
        >>> print(is_invertible)  # True
    """
    # Check if P @ P^T = I (identity matrix)
    identity_check = torch.allclose(P @ P.t(), torch.eye(P.shape[0], device=P.device))
    
    if test_tensor is not None and len(test_tensor.shape) == 2:
        # Test actual encryption/decryption cycle
        hidden_size = P.shape[0]

        # Check if this is a row permutation case (first dimension matches permutation matrix)
        if test_tensor.shape[0] == hidden_size:
            encrypted = encrypt_ffn_weight_row_permutation(test_tensor, P)
            decrypted = decrypt_ffn_weight_row_permutation(encrypted, P)
        # Check if this is a column permutation case (second dimension matches permutation matrix)
        elif test_tensor.shape[1] == hidden_size:
            encrypted = encrypt_ffn_weight_col_permutation(test_tensor, P)
            decrypted = decrypt_ffn_weight_col_permutation(encrypted, P)
        else:
            return identity_check

        cycle_check = torch.allclose(test_tensor, decrypted, atol=1e-6)
        return identity_check and cycle_check
    
    return identity_check

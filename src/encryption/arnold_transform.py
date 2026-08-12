"""
Arnold Cat Map (ACM) encryption implementation for neural network weight protection.

References:
    - Arnold, V.I. and Avez, A. (1968). Ergodic problems of classical mechanics.
    - Zhang, Y. et al. (2019). Image encryption using DNA addition combining with 
      chaotic maps. Mathematical and Computer Modelling.
"""

import torch
import numpy as np
import triton
import triton.language as tl
from numba import njit, prange
from typing import List, Union, Tuple, Optional, Dict


@njit(parallel=True)
def _arnold_numba_kernel(input_matrix, output_matrix, h, w, a_final, b_final, c_final, d_final):
    """Numba JIT kernel for Arnold Cat Map."""
    for i in prange(h):
        for j in range(w):
            new_j = (a_final * j + b_final * i) % w
            new_i = (c_final * j + d_final * i) % h
            output_matrix[new_i, new_j] = input_matrix[i, j]


def arnold_numba(
    matrix: torch.Tensor,
    key: List[int],
    fixed_schedule: bool = False,
) -> torch.Tensor:
    """Numba-optimized CPU implementation of Arnold Cat Map."""
    N, a, b, c, d = key
    h, w = matrix.shape[:2]
    
    # Compute the final transformation matrix with the selected schedule.
    transformation_matrix = np.array([[a, b], [c, d]], dtype=np.int64)
    power_fn = _matrix_power_mod_fixed_schedule if fixed_schedule else _matrix_power_mod
    final_matrix = power_fn(transformation_matrix, N, w)
    
    a_f, b_f = int(final_matrix[0, 0]), int(final_matrix[0, 1])
    c_f, d_f = int(final_matrix[1, 0]), int(final_matrix[1, 1])
    
    # Convert torch tensor to numpy for numba
    input_np = matrix.detach().cpu().numpy()
    output_np = np.empty_like(input_np)
    
    _arnold_numba_kernel(input_np, output_np, h, w, a_f, b_f, c_f, d_f)
    
    return torch.from_numpy(output_np).to(matrix.device)


def iarnold_numba(
    matrix: torch.Tensor,
    key: List[int],
    fixed_schedule: bool = False,
) -> torch.Tensor:
    """Numba-optimized CPU implementation of inverse Arnold Cat Map."""
    N, a, b, c, d = key
    h, w = matrix.shape[:2]
    
    inv_key_matrix = np.array([[d, -b], [-c, a]]) % w
    inv_key = [N, int(inv_key_matrix[0, 0]), int(inv_key_matrix[0, 1]), 
               int(inv_key_matrix[1, 0]), int(inv_key_matrix[1, 1])]
    
    return arnold_numba(matrix, inv_key, fixed_schedule=fixed_schedule)


@triton.jit
def arnold_gather_kernel(
    input_ptr,
    output_ptr,
    h, w,
    a_inv, b_inv, c_inv, d_inv,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each thread at (y_f, x_f) finds its source (y_s, x_s)
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    rf = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    cf = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    mask = (rf < h)[:, None] & (cf < w)[None, :]
    
    # Output coordinates (y_f, x_f) -> rf is y_f, cf is x_f
    x_f = cf[None, :]
    y_f = rf[:, None]
    
    # Source coordinates (y_s, x_s) using inverse transform
    x_s = (a_inv * x_f + b_inv * y_f) % w
    y_s = (c_inv * x_f + d_inv * y_f) % h
    
    # Load from input at (y_s, x_s)
    input_offsets = y_s * w + x_s
    val = tl.load(input_ptr + input_offsets, mask=mask)
    
    # Store to output at (y_f, x_f)
    output_offsets = rf[:, None] * w + cf[None, :]
    tl.store(output_ptr + output_offsets, val, mask=mask)


def arnold_triton(
    matrix: torch.Tensor,
    key: List[int],
    xor_seed: Optional[Union[int, str, bytes, bytearray]] = None,
    xor_context: str = "arnold-attention",
    fixed_schedule: bool = False,
) -> torch.Tensor:
    """
    Triton implementation of Arnold Cat Map with optional ChaCha20 diffusion.
    
    Args:
        matrix: Input matrix (CUDA tensor)
        key: Arnold key [N, a, b, c, d]
        xor_seed: Master secret for ChaCha20 diffusion. If None, diffusion is skipped.
        xor_context: Domain-separation label for the protected tensor.
        fixed_schedule: Use the six-round, key-independent exponentiation schedule
            for supported iteration counts in ``[0, 63]``.
    """
    if not matrix.is_cuda:
        # Fallback to optimized version for CPU
        result = arnold_optimized(matrix, key, fixed_schedule=fixed_schedule)
        if xor_seed is not None:
            from .xor_encryption import xor_encrypt_decrypt_triton
            xor_encrypt_decrypt_triton(
                result, layer_idx=0, weight_name=xor_context, seed_base=xor_seed
            )
        return result
        
    N, a, b, c, d = key
    h, w = matrix.shape[:2]
    
    # Compute M^N mod w with the selected schedule.
    transformation_matrix = np.array([[a, b], [c, d]], dtype=np.int64)
    power_fn = _matrix_power_mod_fixed_schedule if fixed_schedule else _matrix_power_mod
    final_matrix = power_fn(transformation_matrix, N, w)
    
    # Compute inverse of final_matrix mod w
    # det is 1, so inverse of [[A, B], [C, D]] is [[D, -B], [-C, A]]
    a_f, b_f = int(final_matrix[0, 0]), int(final_matrix[0, 1])
    c_f, d_f = int(final_matrix[1, 0]), int(final_matrix[1, 1])
    
    a_inv = int(d_f % w)
    b_inv = int((-b_f) % w)
    c_inv = int((-c_f) % w)
    d_inv = int(a_f % w)
    
    output = torch.empty_like(matrix)
    
    def grid(meta):
        return (
            triton.cdiv(h, meta['BLOCK_SIZE_H']),
            triton.cdiv(w, meta['BLOCK_SIZE_W']),
        )
    
    arnold_gather_kernel[grid](
        matrix, output,
        h, w,
        a_inv, b_inv, c_inv, d_inv,
        BLOCK_SIZE_H=16, BLOCK_SIZE_W=16
    )

    if xor_seed is not None:
        from .xor_encryption import xor_encrypt_decrypt_triton
        xor_encrypt_decrypt_triton(
            output, layer_idx=0, weight_name=xor_context, seed_base=xor_seed
        )
    
    return output


def iarnold_triton(
    matrix: torch.Tensor,
    key: List[int],
    xor_seed: Optional[Union[int, str, bytes, bytearray]] = None,
    xor_context: str = "arnold-attention",
    fixed_schedule: bool = False,
) -> torch.Tensor:
    """
    Inverse Arnold Cat Map with optional pre-permutation ChaCha20 removal.
    """
    N, a, b, c, d = key
    h, w = matrix.shape[:2]
    
    # Compute inverse transformation matrix mod w
    inv_key_matrix = np.array([[d, -b], [-c, a]]) % w
    inv_key = [N, int(inv_key_matrix[0, 0]), int(inv_key_matrix[0, 1]), 
               int(inv_key_matrix[1, 0]), int(inv_key_matrix[1, 1])]
    
    input_matrix = matrix
    if xor_seed is not None:
        from .xor_encryption import xor_encrypt_decrypt_triton
        input_matrix = matrix.clone()
        xor_encrypt_decrypt_triton(
            input_matrix,
            layer_idx=0,
            weight_name=xor_context,
            seed_base=xor_seed,
        )

    return arnold_triton(
        input_matrix,
        inv_key,
        xor_seed=None,
        xor_context=xor_context,
        fixed_schedule=fixed_schedule,
    )


def arnold(matrix: Union[torch.Tensor, np.ndarray], key: List[int]) -> torch.Tensor:
    """Apply the Arnold Cat Map transformation."""
    return arnold_optimized(matrix, key)


# Global cache for coordinate grids to avoid recreating them every time
# Key: (height, width, device_str), Value: (x_grid, y_grid)
_COORDINATE_GRID_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_coordinate_grids(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get coordinate grids, using cache if available.
    
    Args:
        h: Height of the matrix
        w: Width of the matrix
        device: Device to create grids on
        
    Returns:
        Tuple of (x_orig_grid, y_orig_grid) where:
        - x_orig_grid contains column indices (varies along width)
        - y_orig_grid contains row indices (varies along height)
    """
    device_str = str(device)
    cache_key = (h, w, device_str)
    
    if cache_key in _COORDINATE_GRID_CACHE:
        x_grid, y_grid = _COORDINATE_GRID_CACHE[cache_key]
        # Ensure grids are on the correct device
        if x_grid.device != device:
            x_grid = x_grid.to(device)
            y_grid = y_grid.to(device)
            _COORDINATE_GRID_CACHE[cache_key] = (x_grid, y_grid)
        return x_grid, y_grid
    
    # Create new grids
    y_orig_grid, x_orig_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.long),
        torch.arange(w, device=device, dtype=torch.long),
        indexing='ij'
    )
    
    # Cache them
    _COORDINATE_GRID_CACHE[cache_key] = (x_orig_grid, y_orig_grid)
    
    return x_orig_grid, y_orig_grid


def _matrix_power_mod(matrix: np.ndarray, power: int, mod: int) -> np.ndarray:
    """
    Compute matrix^power mod mod using binary exponentiation.
    
    This is an O(log power) operation instead of O(power).
    
    Args:
        matrix: 2x2 numpy array representing the transformation matrix
        power: Exponent to raise the matrix to
        mod: Modulus value
        
    Returns:
        np.ndarray: matrix^power mod mod
    """
    if power == 0:
        return np.array([[1, 0], [0, 1]], dtype=np.int64)
    if power == 1:
        return matrix % mod
    
    result = np.array([[1, 0], [0, 1]], dtype=np.int64)
    base = matrix.copy() % mod
    
    while power > 0:
        if power % 2 == 1:
            result = (result @ base) % mod
        base = (base @ base) % mod
        power //= 2
    
    return result


def _matrix_power_mod_fixed_schedule(
    matrix: np.ndarray,
    power: int,
    mod: int,
    schedule_bits: int = 6,
) -> np.ndarray:
    """Compute ``matrix**power mod mod`` with a key-independent operation schedule.

    Every round performs both a candidate multiply and a square, then selects the
    candidate with arithmetic masking. The default six rounds cover ChaosFormer's
    supported ACM iteration range of 3--34 and always execute 12 matrix products.
    This fixes the algorithmic schedule; it is not by itself a machine-level
    constant-time guarantee for the Python/NumPy runtime.
    """
    if schedule_bits < 1:
        raise ValueError("schedule_bits must be positive")
    if power < 0 or power >= 1 << schedule_bits:
        raise ValueError(
            f"power must be in [0, {(1 << schedule_bits) - 1}] for "
            f"a {schedule_bits}-bit fixed schedule"
        )

    result = np.array([[1, 0], [0, 1]], dtype=np.int64)
    base = np.asarray(matrix, dtype=np.int64) % mod

    for bit_index in range(schedule_bits):
        candidate = (result @ base) % mod
        squared = (base @ base) % mod
        bit = (power >> bit_index) & 1
        result = (bit * candidate + (1 - bit) * result) % mod
        base = squared

    return result


def arnold_optimized(
    matrix: Union[torch.Tensor, np.ndarray],
    key: List[int],
    fixed_schedule: bool = False,
) -> torch.Tensor:
    """
    Apply Arnold Cat Map transformation using matrix exponentiation optimization.
    
    This optimized version uses binary exponentiation to compute M^N mod W in O(log N)
    time instead of O(N), eliminating the coordinate transformation loop.
    
    The Arnold Cat Map is defined by the transformation:
    [x']   [a b] [x]
    [y'] = [c d] [y]  (mod N)
    
    Applying it N times is equivalent to:
    [x_final]   [a b]^N [x_start]
    [y_final] = [c d]   [y_start]  (mod W)
    
    Args:
        matrix: Input matrix to encrypt. Must be square (H x W x ...).
        key: Arnold key parameters [N, a, b, c, d] where:
            - N: Number of iterations
            - a, b, c, d: Transformation matrix parameters
            
    Returns:
        torch.Tensor: Encrypted matrix with same shape as input
        
    Raises:
        ValueError: If matrix is not square or key parameters are invalid
        
    Example:
        >>> matrix = torch.randn(768, 768)
        >>> key = [100, 1, 1, 1, 2]  # N=100 iterations, standard Arnold map
        >>> encrypted = arnold_optimized(matrix, key)  # Much faster than arnold() for large N
    """
    N, a, b, c, d = key
    h, w = matrix.shape[:2]

    if h != w:
        raise ValueError("Matrix must be square")
    if (a * d - b * c) % w != 1:
        raise ValueError("Invalid Arnold key: determinant must be 1 (mod w)")

    # Ensure tensor is a torch tensor and get its device
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.from_numpy(matrix)

    device = matrix.device

    # Compute the final transformation matrix with the selected schedule.
    transformation_matrix = np.array([[a, b], [c, d]], dtype=np.int64)
    power_fn = _matrix_power_mod_fixed_schedule if fixed_schedule else _matrix_power_mod
    final_matrix = power_fn(transformation_matrix, N, w)
    
    # Convert to torch tensor on the same device
    final_matrix_torch = torch.tensor(final_matrix, device=device, dtype=torch.long)
    a_final, b_final = final_matrix_torch[0, 0], final_matrix_torch[0, 1]
    c_final, d_final = final_matrix_torch[1, 0], final_matrix_torch[1, 1]

    # Get coordinate grids (using cache if available)
    x_orig_grid, y_orig_grid = _get_coordinate_grids(h, w, device)
    
    # Flatten coordinates for efficient processing
    # Note: With indexing='ij', x_orig_grid represents rows, y_orig_grid represents columns
    # This is because meshgrid with 'ij' returns (row_grid, col_grid) where:
    # - First grid (x_orig_grid): varies along columns, contains row indices
    # - Second grid (y_orig_grid): varies along rows, contains column indices
    x_coords_to_transform = x_orig_grid.flatten()  # Row indices (0 to h-1)
    y_coords_to_transform = y_orig_grid.flatten()  # Column indices (0 to w-1)

    # Apply the final transformation matrix ONCE (no loop needed!)
    # Arnold transform: [x'] = [a b] [x] mod w
    #                   [y']   [c d] [y] mod h
    final_dest_x = (a_final * x_coords_to_transform + b_final * y_coords_to_transform) % w
    final_dest_y = (c_final * x_coords_to_transform + d_final * y_coords_to_transform) % h

    # Calculate final destination indices for scatter operation
    # Row-major indexing is y * width + x.
    final_dest_flat_indices = (final_dest_y * w + final_dest_x).long()

    # Reshape original matrix and prepare output buffer
    original_matrix_flat = matrix.reshape(h * w, -1)
    output_flat = torch.zeros_like(original_matrix_flat)

    # Single efficient scatter: put original elements in their final positions
    output_flat.index_copy_(0, final_dest_flat_indices, original_matrix_flat)

    result = output_flat.reshape(matrix.shape)

    # Ensure the result is contiguous for saving
    return result.contiguous()


def iarnold_optimized(
    matrix: Union[torch.Tensor, np.ndarray],
    key: List[int],
    fixed_schedule: bool = False,
) -> torch.Tensor:
    """
    Apply inverse Arnold Cat Map transformation using optimized matrix exponentiation.
    
    This function computes the inverse transformation by calculating the
    inverse of the Arnold transformation matrix modulo the matrix width.
    
    Args:
        matrix: Encrypted matrix to decrypt. Must be square.
        key: Arnold key parameters [N, a, b, c, d] (same as used for encryption)
        
    Returns:
        torch.Tensor: Decrypted matrix with same shape as input
        
    Raises:
        ValueError: If key parameters are invalid
        
    Example:
        >>> encrypted_matrix = arnold_optimized(original_matrix, key)
        >>> decrypted_matrix = iarnold_optimized(encrypted_matrix, key)
        >>> torch.allclose(original_matrix, decrypted_matrix)  # Should be True
    """
    N, a, b, c, d = key
    h, w = matrix.shape[:2]
    
    if (a * d - b * c) % w != 1:
        raise ValueError("Invalid Arnold key: determinant must be 1 (mod w)")

    # Compute inverse transformation matrix
    inv_key_matrix = np.array([[d, -b],
                               [-c, a]]) % w
    inv_key = [N,
               inv_key_matrix[0, 0],
               inv_key_matrix[0, 1],
               inv_key_matrix[1, 0],
               inv_key_matrix[1, 1]]
    
    return arnold_optimized(matrix, inv_key, fixed_schedule=fixed_schedule)


def iarnold(matrix: Union[torch.Tensor, np.ndarray], key: List[int]) -> torch.Tensor:
    """Apply the inverse Arnold Cat Map transformation."""
    return iarnold_optimized(matrix, key)


def generate_arnold_key(matrix_size: int, iterations: Optional[int] = None, 
                       seed: int = None) -> List[int]:
    """
    Generate a valid Arnold Cat Map key for a given matrix size.
    
    The key consists of [N, a, b, c, d] where the transformation matrix
    M = [[a, b], [c, d]] must have det(M) ≡ 1 (mod matrix_size) to be invertible.
    
    We use a construction M = [[1, p], [q, pq + 1]] which always has det(M) = 1.
    
    Args:
        matrix_size: Size of the square matrix (width/height)
        iterations: Number of Arnold iterations to apply. If None, sampled from [3, 34].
        seed: Random seed for reproducible key generation
        
    Returns:
        List[int]: Valid Arnold key [N, a, b, c, d]
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    if iterations is None:
        iterations = rng.randint(3, 35) # Range [3, 34]
    
    # Use the construction [[1, p], [q, pq + 1]] to guarantee det = 1
    # p and q are sampled from [1, matrix_size - 1]
    p = rng.randint(1, matrix_size)
    q = rng.randint(1, matrix_size)
    
    a = 1
    b = p
    c = q
    d = (p * q + 1) % matrix_size
    
    return [iterations, int(a), int(b), int(c), int(d)]


def verify_arnold_invertibility(key: List[int], matrix_size: int) -> bool:
    """
    Verify that an Arnold key produces invertible transformations.
    
    Args:
        key: Arnold key parameters [N, a, b, c, d]
        matrix_size: Size of the matrix the key will be applied to
        
    Returns:
        bool: True if the key is valid and invertible
        
    Example:
        >>> key = [3, 1, 1, 1, 2]
        >>> is_valid = verify_arnold_invertibility(key, 768)
        >>> print(is_valid)  # True
    """
    N, a, b, c, d = key
    determinant = (a * d - b * c) % matrix_size
    return determinant == 1


# Standard Arnold keys for common use cases
STANDARD_ARNOLD_KEYS = {
    'default': [3, 1, 1, 1, 2],
    'strong': [5, 2, 3, 3, 5],
    'extra_strong': [11, 1, 3, 2, 7],
}


def get_standard_key(key_name: str = 'default') -> List[int]:
    """
    Get a predefined standard Arnold key.
    
    Args:
        key_name: Name of the standard key ('default', 'strong', 'extra_strong')
        
    Returns:
        List[int]: Standard Arnold key parameters
        
    Raises:
        KeyError: If key_name is not recognized
    """
    if key_name not in STANDARD_ARNOLD_KEYS:
        available_keys = list(STANDARD_ARNOLD_KEYS.keys())
        raise KeyError(f"Unknown key name '{key_name}'. Available: {available_keys}")
    
    return STANDARD_ARNOLD_KEYS[key_name].copy()

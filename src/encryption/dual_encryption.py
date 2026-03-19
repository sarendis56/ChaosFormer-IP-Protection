"""
Combines Arnold Cat Map (ACM) encryption for attention weights with permutation-based
encryption for Feed-Forward Network (FFN) weights.
"""

import torch
import numpy as np
import hmac
import hashlib
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import logging

from .arnold_transform import (
    arnold, iarnold, get_standard_key,
    arnold_triton, iarnold_triton, arnold_optimized, iarnold_optimized,
    generate_arnold_key
)
from .permutation import (
    generate_multiple_permutation_matrices,
    encrypt_ffn_weight_row_permutation,
    decrypt_ffn_weight_row_permutation
)
from .xor_encryption import (
    generate_xor_key,
    xor_encrypt_decrypt,
    xor_encrypt_decrypt_numba,
    xor_encrypt_decrypt_triton,
    get_stable_seed,
    encrypt_weights_xor,
    decrypt_weights_xor
)


@dataclass
class EncryptionConfig:
    """Configuration for dual/triple encryption system."""
    arnold_key: List[int]
    master_secret: Optional[str] = None  # Unified secret source (e.g., PUF)
    password: Optional[str] = None
    num_permutation_matrices: int = 6
    matrix_size: int = 768
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32
    use_xor: bool = False  # Enable XOR encryption as third layer
    xor_seed_base: int = 42  # Base seed for XOR key generation
    mode: str = 'basic'  # 'basic' or 'advanced'
    l_modular: int = 256  # Modular addition value for advanced mode


class LayerEncryptionResult(NamedTuple):
    """Result of encrypting a single transformer layer."""
    encrypted_attention: Dict[str, torch.Tensor]
    encrypted_ffn: Dict[str, torch.Tensor]
    permutation_matrix_idx: int
    arnold_key: List[int]


class DualEncryption:
    """
    Dual/Triple encryption system for Vision Transformer layers.
    
    This class provides a unified interface for encrypting and decrypting
    transformer layers using:
    - Arnold Cat Map (ACM) for attention weights
    - Permutation-based encryption for FFN weights
    - Optional XOR encryption for all weights (when use_xor=True)
    
    Attributes:
        config: Encryption configuration
        permutation_matrices: List of generated permutation matrices
        logger: Logger for tracking operations
    """
    
    def __init__(self,
                 master_secret: Optional[str] = None,
                 arnold_key: Optional[List[int]] = None,
                 password: Optional[str] = None,
                 num_permutation_matrices: int = 6,
                 matrix_size: int = 768,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 use_xor: bool = False,
                 xor_seed_base: int = 42,
                 mode: str = 'basic',
                 l_modular: int = 256):
        """
        Initialize the dual/triple encryption system.

        Args:
            master_secret: Unified secret source (e.g., software-simulated PUF).
                          If provided, other keys will be derived from it.
            arnold_key: Arnold Cat Map key parameters [N, a, b, c, d]
            password: Password for permutation matrix generation
            num_permutation_matrices: Number of permutation matrices to generate
            matrix_size: Size of permutation matrices (hidden_size of the model)
                        e.g., 768 for ViT-base, 1024 for ViT-large
            device: Device for tensor operations
            dtype: Data type for tensors
            use_xor: Enable XOR encryption as third layer (default: False)
            xor_seed_base: Base seed for XOR key generation (default: 42)
            mode: Encryption mode ('basic' or 'advanced')
            l_modular: Modular value for advanced mode addition
        """
        # Derive parameters from master_secret if provided (simulating PUF derivation)
        if master_secret is not None:
            # Derive Arnold key (N=3 iterations by default)
            if arnold_key is None:
                arnold_h = hmac.new(master_secret.encode(), b"arnold", hashlib.sha256).digest()
                # Ensure seed is within 32-bit range for reliability across platforms
                arnold_seed = int.from_bytes(arnold_h[:8], byteorder='big') % (2**32)
                arnold_key = generate_arnold_key(matrix_size, iterations=3, seed=arnold_seed)
            
            # Derive Permutation password
            if password is None:
                password = hmac.new(master_secret.encode(), b"permutation", hashlib.sha256).hexdigest()
            
            # Derive XOR seed base
            if xor_seed_base == 42:
                xor_h = hmac.new(master_secret.encode(), b"xor", hashlib.sha256).digest()
                xor_seed_base = int.from_bytes(xor_h[:8], byteorder='big') % (2**32)

        # Use default Arnold key if none provided and no master_secret
        if arnold_key is None:
            arnold_key = get_standard_key('default')
            
        self.config = EncryptionConfig(
            arnold_key=arnold_key,
            master_secret=master_secret,
            password=password,
            num_permutation_matrices=num_permutation_matrices,
            matrix_size=matrix_size,
            device=device,
            dtype=dtype,
            use_xor=use_xor,
            xor_seed_base=xor_seed_base
        )
        
        # Generate permutation matrices
        self.permutation_matrices = generate_multiple_permutation_matrices(
            num_matrices=num_permutation_matrices,
            size=matrix_size,
            password=password,
            device=device
        )
        
        # Convert to specified dtype
        for i, matrix in enumerate(self.permutation_matrices):
            self.permutation_matrices[i] = matrix.to(dtype=dtype)
        
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_model(cls,
                   model,
                   master_secret: Optional[str] = None,
                   arnold_key: Optional[List[int]] = None,
                   password: Optional[str] = None,
                   num_permutation_matrices: int = 6,
                   device: str = 'cuda',
                   dtype: torch.dtype = torch.float32,
                   use_xor: bool = False,
                   xor_seed_base: int = 42,
                   mode: str = 'basic',
                   l_modular: int = 256):
        """
        Create a DualEncryption instance with dimensions extracted from a model.

        Args:
            model: Vision Transformer model (ViTForImageClassification)
            master_secret: Unified secret source (e.g., PUF)
            arnold_key: Arnold Cat Map key parameters [N, a, b, c, d]
            password: Password for permutation matrix generation
            num_permutation_matrices: Number of permutation matrices to generate
            device: Device for tensor operations
            dtype: Data type for tensors
            use_xor: Enable XOR encryption as third layer
            xor_seed_base: Base seed for XOR key generation
            mode: Encryption mode ('basic' or 'advanced')
            l_modular: Modular value for advanced mode addition

        Returns:
            DualEncryption: Configured encryption system
        """
        # Extract hidden size from the model configuration
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            matrix_size = model.config.hidden_size
        else:
            # Fallback: extract from actual layer weights
            from ..utils.vision_backbone_utils import get_transformer_layers, get_layer_weight_views
            layers = get_transformer_layers(model)
            first_layer = layers[0]
            matrix_size = get_layer_weight_views(first_layer).attention["query"].shape[0]

        return cls(
            master_secret=master_secret,
            arnold_key=arnold_key,
            password=password,
            num_permutation_matrices=num_permutation_matrices,
            matrix_size=matrix_size,
            device=device,
            dtype=dtype,
            use_xor=use_xor,
            xor_seed_base=xor_seed_base
        )

    def encrypt_attention_weights(self,
                                attention_weights: Dict[str, torch.Tensor],
                                layer_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Encrypt attention weights using Arnold Cat Map, optionally with XOR or modular addition.
        
        Args:
            attention_weights: Dictionary containing attention weight tensors
                             (query, key, value, output)
            layer_idx: Layer index for XOR/modular key generation
        
        Returns:
            Dict[str, torch.Tensor]: Encrypted attention weights
        """
        encrypted_attention = {}
        
        for name, weight in attention_weights.items():
            # Apply permutation first (ACM for square matrices)
            if weight.shape[0] == weight.shape[1]:
                # Optimized GPU path
                if weight.is_cuda:
                    # For advanced mode, we can use XOR as a simulation of modular addition 
                    # if we don't have a dedicated modular addition kernel. 
                    # However, the user specifically asked for (Pi(W) + M) mod L.
                    # Let's implement modular addition.
                    
                    if self.config.mode == 'advanced':
                        # Advanced mode: Permutation + Modular Addition
                        permuted = arnold_triton(weight, self.config.arnold_key)
                        # Simulate PRF keystream M
                        xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                        torch.manual_seed(xor_seed)
                        M = torch.randn_like(permuted) * 0.1 # Scaled noise
                        # W_enc = (Pi(W) + M) mod L is for integer weights usually.
                        # For float weights, we can simulate it by adding noise.
                        # If the user means 8-bit quantization then modular addition makes sense.
                        # For now, let's follow the formula literally or simulate with addition.
                        encrypted_attention[name] = (permuted + M) % self.config.l_modular if self.config.l_modular != 0 else permuted + M
                    else:
                        # Basic mode: ACM (+ optional XOR)
                        xor_seed = None
                        if self.config.use_xor:
                            xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                        
                        encrypted_attention[name] = arnold_triton(
                            weight, 
                            self.config.arnold_key, 
                            xor_seed=xor_seed
                        )
                else:
                    # CPU fallback
                    if self.config.mode == 'advanced':
                        permuted = arnold_optimized(weight, self.config.arnold_key)
                        xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                        torch.manual_seed(xor_seed)
                        M = torch.randn_like(permuted) * 0.1
                        encrypted_attention[name] = (permuted + M) % self.config.l_modular if self.config.l_modular != 0 else permuted + M
                    else:
                        if self.config.use_xor:
                            weight = xor_encrypt_decrypt_numba(
                                weight, layer_idx=layer_idx, weight_name=name, seed_base=self.config.xor_seed_base
                            )
                        encrypted_attention[name] = arnold_optimized(weight, self.config.arnold_key)
            else:
                # Non-square matrices (if any in attention, though usually they are square)
                # Apply Knuth Shuffle (simulated by random permutation)
                encrypted_attention[name] = self._knuth_shuffle(weight, layer_idx, name)
            
        return encrypted_attention

    def _knuth_shuffle(self, weight: torch.Tensor, layer_idx: int, name: str) -> torch.Tensor:
        """Simulate Knuth Shuffle using deterministic random permutation."""
        seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
        torch.manual_seed(seed)
        
        # Row permutation for non-square matrices
        perm = torch.randperm(weight.shape[0], device=weight.device)
        permuted = weight[perm]
        
        if self.config.mode == 'advanced':
            M = torch.randn_like(permuted) * 0.1
            return (permuted + M) % self.config.l_modular if self.config.l_modular != 0 else permuted + M
        
        return permuted

    def encrypt_ffn_weights(self, 
                          ffn_weights: Dict[str, torch.Tensor],
                          permutation_matrix_idx: int = 0,
                          layer_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Encrypt FFN weights using row permutation (Knuth Shuffle), optionally with modular addition.
        
        Args:
            ffn_weights: Dictionary containing FFN weight tensors
                        (intermediate, output)
            permutation_matrix_idx: Index of permutation matrix to use
            layer_idx: Layer index for key generation
        
        Returns:
            Dict[str, torch.Tensor]: Encrypted FFN weights
        """
        encrypted_ffn = {}
        
        for name, weight in ffn_weights.items():
            # Knuth Shuffle (Permutation)
            # For FFN, we use the provided permutation matrices or generate on the fly
            if permutation_matrix_idx < len(self.permutation_matrices):
                perm_matrix = self.permutation_matrices[permutation_matrix_idx]
                
                if name == 'intermediate':
                    # intermediate_weight (hidden_size x intermediate_size)
                    # Row permutation on transpose to permute the hidden_size dimension
                    permuted = encrypt_ffn_weight_row_permutation(
                        weight.transpose(0, 1),
                        perm_matrix.to(dtype=weight.dtype)
                    ).transpose(0, 1)
                else:
                    # output_weight (intermediate_size x hidden_size)
                    # Row permutation directly
                    permuted = encrypt_ffn_weight_row_permutation(
                        weight,
                        perm_matrix.to(dtype=weight.dtype)
                    )
            else:
                # Fallback to Knuth Shuffle if index out of range
                permuted = self._knuth_shuffle(weight, layer_idx, name)
            
            if self.config.mode == 'advanced':
                xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                torch.manual_seed(xor_seed)
                M = torch.randn_like(permuted) * 0.1
                encrypted_ffn[name] = (permuted + M) % self.config.l_modular if self.config.l_modular != 0 else permuted + M
            else:
                # Basic mode XOR if enabled
                if self.config.use_xor:
                    permuted = xor_encrypt_decrypt_triton(
                        permuted, 
                        layer_idx=layer_idx, 
                        weight_name=name,
                        seed_base=self.config.xor_seed_base
                    )
                encrypted_ffn[name] = permuted
                
        return encrypted_ffn
    
    def decrypt_attention_weights(self, 
                                encrypted_attention: Dict[str, torch.Tensor],
                                layer_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Decrypt attention weights using inverse Arnold Cat Map, optionally with XOR or modular subtraction.
        
        Args:
            encrypted_attention: Dictionary containing encrypted attention weights
            layer_idx: Layer index for XOR/modular key generation
        
        Returns:
            Dict[str, torch.Tensor]: Decrypted attention weights
        """
        decrypted_attention = {}
        
        for name, weight in encrypted_attention.items():
            if weight.shape[0] == weight.shape[1]:
                # Optimized GPU path
                if weight.is_cuda:
                    if self.config.mode == 'advanced':
                        # Advanced mode: Modular Subtraction + Inverse Permutation
                        xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                        torch.manual_seed(xor_seed)
                        M = torch.randn_like(weight) * 0.1
                        # W_dec = (W_enc - M) mod L
                        subtracted = (weight - M) % self.config.l_modular if self.config.l_modular != 0 else weight - M
                        decrypted_attention[name] = iarnold_triton(subtracted, self.config.arnold_key)
                    else:
                        # Basic mode: Inverse ACM (+ optional XOR)
                        xor_seed = None
                        if self.config.use_xor:
                            xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                        
                        decrypted_attention[name] = iarnold_triton(
                            weight, 
                            self.config.arnold_key, 
                            xor_seed=xor_seed
                        )
                else:
                    # CPU fallback
                    if self.config.mode == 'advanced':
                        xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                        torch.manual_seed(xor_seed)
                        M = torch.randn_like(weight) * 0.1
                        subtracted = (weight - M) % self.config.l_modular if self.config.l_modular != 0 else weight - M
                        decrypted_attention[name] = iarnold_optimized(subtracted, self.config.arnold_key)
                    else:
                        decrypted_tensor = iarnold_optimized(weight, self.config.arnold_key)
                        if self.config.use_xor:
                            decrypted_tensor = xor_encrypt_decrypt_numba(
                                decrypted_tensor, layer_idx=layer_idx, weight_name=name, seed_base=self.config.xor_seed_base
                            )
                        decrypted_attention[name] = decrypted_tensor
            else:
                # Non-square matrices
                decrypted_attention[name] = self._knuth_unshuffle(weight, layer_idx, name)
            
        return decrypted_attention

    def _knuth_unshuffle(self, weight: torch.Tensor, layer_idx: int, name: str) -> torch.Tensor:
        """Simulate Knuth Unshuffle."""
        seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
        torch.manual_seed(seed)
        
        perm = torch.randperm(weight.shape[0], device=weight.device)
        inv_perm = torch.argsort(perm)
        
        if self.config.mode == 'advanced':
            M = torch.randn_like(weight) * 0.1
            weight = (weight - M) % self.config.l_modular if self.config.l_modular != 0 else weight - M
            
        return weight[inv_perm]

    def decrypt_ffn_weights(self, 
                          encrypted_ffn: Dict[str, torch.Tensor],
                          permutation_matrix_idx: int = 0,
                          layer_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Decrypt FFN weights using inverse row permutation, optionally with modular subtraction.
        
        Args:
            encrypted_ffn: Dictionary containing encrypted FFN weights
            permutation_matrix_idx: Index of permutation matrix used for encryption
            layer_idx: Layer index for key generation
        
        Returns:
            Dict[str, torch.Tensor]: Decrypted FFN weights
        """
        decrypted_ffn = {}
        
        for name, weight in encrypted_ffn.items():
            if self.config.mode == 'advanced':
                xor_seed = get_stable_seed(layer_idx, name, self.config.xor_seed_base)
                torch.manual_seed(xor_seed)
                M = torch.randn_like(weight) * 0.1
                weight = (weight - M) % self.config.l_modular if self.config.l_modular != 0 else weight - M
            
            if permutation_matrix_idx < len(self.permutation_matrices):
                perm_matrix = self.permutation_matrices[permutation_matrix_idx]
                
                if name == 'intermediate':
                    decrypted_ffn[name] = decrypt_ffn_weight_row_permutation(
                        weight.transpose(0, 1),
                        perm_matrix
                    ).transpose(0, 1)
                else:
                    decrypted_ffn[name] = decrypt_ffn_weight_row_permutation(
                        weight,
                        perm_matrix
                    )
            else:
                decrypted_ffn[name] = self._knuth_unshuffle(weight, layer_idx, name)
                
            # Basic mode XOR if enabled
            if self.config.mode == 'basic' and self.config.use_xor:
                decrypted_ffn[name] = xor_encrypt_decrypt_triton(
                    decrypted_ffn[name], 
                    layer_idx=layer_idx, 
                    weight_name=name,
                    seed_base=self.config.xor_seed_base
                )
                
        return decrypted_ffn
    
    def encrypt_layer_weights(self, 
                            attention_weights: Dict[str, torch.Tensor],
                            ffn_weights: Dict[str, torch.Tensor],
                            permutation_matrix_idx: int = 0,
                            layer_idx: int = 0) -> LayerEncryptionResult:
        """
        Encrypt both attention and FFN weights for a complete transformer layer.
        
        Args:
            attention_weights: Attention weight tensors
            ffn_weights: FFN weight tensors  
            permutation_matrix_idx: Index of permutation matrix to use
            layer_idx: Layer index for XOR key generation (if use_xor=True)
        
        Returns:
            LayerEncryptionResult: Complete encryption result
        """
        encrypted_attention = self.encrypt_attention_weights(attention_weights, layer_idx)
        encrypted_ffn = self.encrypt_ffn_weights(ffn_weights, permutation_matrix_idx, layer_idx)
        
        return LayerEncryptionResult(
            encrypted_attention=encrypted_attention,
            encrypted_ffn=encrypted_ffn,
            permutation_matrix_idx=permutation_matrix_idx,
            arnold_key=self.config.arnold_key.copy()
        )
    
    def decrypt_layer_weights(self, 
                            encryption_result: LayerEncryptionResult,
                            layer_idx: int = 0) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Decrypt both attention and FFN weights for a complete transformer layer.
        
        Args:
            encryption_result: Result from encrypt_layer_weights
            layer_idx: Layer index for XOR key generation (if use_xor=True)
        
        Returns:
            Tuple containing (decrypted_attention, decrypted_ffn)
        """
        decrypted_attention = self.decrypt_attention_weights(
            encryption_result.encrypted_attention,
            layer_idx
        )
        decrypted_ffn = self.decrypt_ffn_weights(
            encryption_result.encrypted_ffn,
            encryption_result.permutation_matrix_idx,
            layer_idx
        )
        
        return decrypted_attention, decrypted_ffn
    
    def verify_encryption_cycle(self, 
                              attention_weights: Dict[str, torch.Tensor],
                              ffn_weights: Dict[str, torch.Tensor],
                              permutation_matrix_idx: int = 0,
                              layer_idx: int = 0,
                              tolerance: float = 1e-5) -> bool:
        """
        Verify that encryption followed by decryption recovers original weights.
        
        Args:
            attention_weights: Original attention weights
            ffn_weights: Original FFN weights
            permutation_matrix_idx: Permutation matrix index to test
            layer_idx: Layer index for XOR key generation (if use_xor=True)
            tolerance: Numerical tolerance for comparison
        
        Returns:
            bool: True if encryption cycle is successful
        """
        # Store original weights
        original_attention = {k: v.clone() for k, v in attention_weights.items()}
        original_ffn = {k: v.clone() for k, v in ffn_weights.items()}
        
        # Encrypt
        encryption_result = self.encrypt_layer_weights(
            attention_weights, ffn_weights, permutation_matrix_idx, layer_idx
        )
        
        # Decrypt
        decrypted_attention, decrypted_ffn = self.decrypt_layer_weights(encryption_result, layer_idx)
        
        # Verify attention weights
        for name in original_attention:
            if not torch.allclose(original_attention[name], decrypted_attention[name], atol=tolerance):
                self.logger.warning(f"Attention weight {name} verification failed")
                return False
        
        # Verify FFN weights
        for name in original_ffn:
            if not torch.allclose(original_ffn[name], decrypted_ffn[name], atol=tolerance):
                self.logger.warning(f"FFN weight {name} verification failed")
                return False
        
        return True

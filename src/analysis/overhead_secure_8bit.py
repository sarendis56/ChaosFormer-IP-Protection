import torch
import numpy as np
import time
import hmac
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import ViTForImageClassification, ViTImageProcessor, BitsAndBytesConfig
from tqdm import tqdm
import sys
import os
import argparse
import glob
import random
from PIL import Image

# Check for bitsandbytes
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("Warning: bitsandbytes not found. 8-bit quantization will not be available.")

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Global configuration for number of runs
# Adjust these values to control evaluation repetition for different devices
NUM_RUNS_CPU = 3      # Number of runs for CPU evaluation
NUM_RUNS_GPU = 20     # Number of runs for GPU evaluation

from encryption.arnold_transform import (
    arnold, iarnold, arnold_optimized, iarnold_optimized, 
    arnold_triton, iarnold_triton, arnold_numba, iarnold_numba,
    generate_arnold_key
)
from encryption.xor_encryption import (
    generate_xor_key, xor_encrypt_decrypt, 
    xor_encrypt_decrypt_triton, xor_encrypt_decrypt_numba,
    get_stable_seed
)

def get_optimal_arnold_functions(device: torch.device):
    """Return the optimized Arnold Cat Map functions for the given device."""
    if device.type == 'cuda':
        return arnold_optimized, iarnold_optimized
    else:
        return arnold_numba, iarnold_numba

@dataclass
class TimingBreakdown:
    """Container for timing breakdown data."""
    total_time: float
    forward_time: float
    acm_enc_time: float = 0.0
    acm_dec_time: float = 0.0
    ffn_enc_time: float = 0.0
    ffn_dec_time: float = 0.0
    xor_enc_time: float = 0.0
    xor_dec_time: float = 0.0
    overhead_ratio: float = 1.0

    @property
    def encryption_time(self) -> float:
        return self.acm_enc_time + self.ffn_enc_time + self.xor_enc_time

    @property
    def decryption_time(self) -> float:
        return self.acm_dec_time + self.ffn_dec_time + self.xor_dec_time

    @property
    def total_overhead_time(self) -> float:
        return self.encryption_time + self.decryption_time


class InferenceOverheadAnalyzer:
    """Analyzer for measuring triple encryption inference overhead (ACM + FFN + XOR)."""

    # Keep ACM iteration sweep centralized so analysis, CSV export, and plots stay consistent.
    # If you change this list, the x-axis labels and CSV `n_iterations` will follow automatically.
    ACM_ITERATIONS = [3, 5, 8, 12, 16, 32]

    # Global font size configuration for all plots - centralized for easy maintenance
    # To change any font size, simply update the values here!
    FONT_SIZES = {
        'main_title': 55,      # Main figure title
        'axis_labels': 48,     # X and Y axis labels
        'subtitles': 40,       # Subplot titles (a, b, c, d)
        'ratio_text': 40,      # Overhead ratio text above bars
        'tick_labels': 28,     # X and Y tick labels
        'legend': 40           # Legend text
    }

    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = "auto", batch_size: int = 1, skip_imagenet: bool = False, use_triton: bool = False, master_secret: Optional[str] = None, load_in_8bit: bool = True):
        """Initialize the analyzer with model and device.

        Args:
            model_name: Name of the ViT model to use
            device: Device to run on ('auto', 'cuda', or 'cpu')
            batch_size: Batch size for inference (default: 1 for single sequential inference)
            skip_imagenet: Skip loading ImageNet dataset (useful for plot regeneration)
            use_triton: Use Triton kernels for encryption/decryption
            master_secret: Unified master secret (PUF simulation root)
            load_in_8bit: Load model in 8-bit quantization using bitsandbytes
        """
        self.model_name = model_name
        self.use_triton = use_triton
        self.master_secret = master_secret
        self.load_in_8bit = load_in_8bit

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if self.use_triton and self.device.type != 'cuda':
            print("Warning: Triton kernels only supported on CUDA. Disabling Triton.")
            self.use_triton = False

        # IMPORTANT: Triton launches on the *active* CUDA device. If the user selects
        # cuda:N, ensure the active device matches so Triton can access tensor pointers.
        if self.device.type == "cuda":
            try:
                torch.cuda.set_device(self.device)
            except Exception:
                # Fallback for older PyTorch builds that only accept an integer.
                if self.device.index is not None:
                    torch.cuda.set_device(self.device.index)

        if self.load_in_8bit and not HAS_BNB:
            print("Warning: bitsandbytes not available. Disabling 8-bit loading.")
            self.load_in_8bit = False

        self.batch_size = batch_size  # Store batch size (default=1)

        # Initialize model and processor
        # Check if local model exists first, otherwise use HuggingFace Hub
        from pathlib import Path

        # Extract model identifier from HuggingFace model name
        if '/' in model_name:
            model_identifier = model_name.split('/')[-1]  # e.g., 'vit-base-patch16-224'
        else:
            model_identifier = model_name

        local_model_path = Path(f"model/{model_identifier}")

        # Configure quantization if requested
        quantization_config = None
        if self.load_in_8bit:
            print(f"Configuring 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        if local_model_path.exists() and local_model_path.is_dir():
            print(f"Loading local model from {local_model_path}...")
            if self.load_in_8bit and self.device.type == "cuda":
                device_map = str(self.device)
            else:
                device_map = None
            self.model = ViTForImageClassification.from_pretrained(
                str(local_model_path),
                device_map=device_map,
                quantization_config=quantization_config
            )
            if not self.load_in_8bit:
                self.model = self.model.to(self.device)
            self.processor = ViTImageProcessor.from_pretrained(str(local_model_path))
        else:
            print(f"Local model not found at {local_model_path}, loading {model_name} from HuggingFace Hub...")
            if self.load_in_8bit and self.device.type == "cuda":
                device_map = str(self.device)
            else:
                device_map = None
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=quantization_config
            )
            if not self.load_in_8bit:
                self.model = self.model.to(self.device)
            self.processor = ViTImageProcessor.from_pretrained(model_name)

            # Save model locally for future use
            try:
                print(f"Saving model to {local_model_path} for future use...")
                local_model_path.mkdir(parents=True, exist_ok=True)
                # Save to CPU first to avoid device issues
                cpu_model = ViTForImageClassification.from_pretrained(model_name)
                cpu_model.save_pretrained(local_model_path)
                self.processor.save_pretrained(local_model_path)
                print(f"Model saved successfully to {local_model_path}")
            except Exception as save_error:
                print(f"Warning: Could not save model locally: {save_error}")
                print("Model will be loaded from HuggingFace Hub on future runs")

        # Derive keys from master_secret if provided (simulating hardware codesign)
        if master_secret:
            # Derive Arnold key (assuming matrix_size 768 for ViT-base)
            arnold_h = hmac.new(master_secret.encode(), b"arnold", hashlib.sha256).digest()
            arnold_seed = int.from_bytes(arnold_h[:8], byteorder='big') % (2**32)
            self.derived_acm_key = generate_arnold_key(768, iterations=3, seed=arnold_seed)
            
            # Derive Permutation password
            self.derived_perm_password = hmac.new(master_secret.encode(), b"permutation", hashlib.sha256).hexdigest()
            
            # Derive XOR seed base
            xor_h = hmac.new(master_secret.encode(), b"xor", hashlib.sha256).digest()
            self.derived_xor_seed_base = int.from_bytes(xor_h[:8], byteorder='big') % (2**32)
            print(f"✓ Derived all keys from master secret: {master_secret[:4]}...{master_secret[-4:]}")
        else:
            self.derived_acm_key = [3, 1, 1, 1, 2]
            self.derived_perm_password = None
            self.derived_xor_seed_base = 42

        # Sample real images from ImageNet dataset (skip if only regenerating plots)
        if not skip_imagenet:
            self.sample_input = self._load_imagenet_samples(num_samples=1000, batch_size=self.batch_size)
        else:
            print("Skipping ImageNet dataset loading for plot regeneration...")
            self.sample_input = None
            self.imagenet_samples = None

        # Select optimal Arnold functions for this device
        self.arnold_encrypt, self.arnold_decrypt = get_optimal_arnold_functions(self.device)

        print(f"Initialized analyzer with model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")

    def _load_imagenet_samples(self, num_samples: int = 1000, batch_size: int = 1) -> torch.Tensor:
        """Load and preprocess samples from ImageNet validation dataset.

        Args:
            num_samples: Number of images to sample from ImageNet
            batch_size: Batch size for the returned tensor

        Returns:
            Preprocessed tensor of shape (batch_size, 3, 224, 224)
        """
        imagenet_path = Path("dataset/imagenet/val")

        if not imagenet_path.exists():
            print(f"Warning: ImageNet dataset not found at {imagenet_path}")
            print("Falling back to random tensor generation...")
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Get all image files from all class directories
        image_files = []
        for class_dir in imagenet_path.iterdir():
            if class_dir.is_dir():
                class_images = list(class_dir.glob("*.JPEG"))
                image_files.extend(class_images)

        if len(image_files) == 0:
            print("Warning: No JPEG images found in ImageNet dataset")
            print("Falling back to random tensor generation...")
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Set seed for reproducible sampling
        random.seed(42)

        # Sample random images
        sampled_files = random.sample(image_files, min(num_samples, len(image_files)))
        print(f"Sampled {len(sampled_files)} images from ImageNet dataset")

        # Load and preprocess images
        processed_images = []
        for img_path in tqdm(sampled_files, desc="Loading ImageNet samples"):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')

                # Preprocess using the ViT processor
                inputs = self.processor(images=image, return_tensors="pt")
                processed_images.append(inputs['pixel_values'].squeeze(0))

            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue

        if len(processed_images) == 0:
            print("Warning: Failed to load any images from ImageNet dataset")
            print("Falling back to random tensor generation...")
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Stack all images and create batches
        all_images = torch.stack(processed_images)

        # For inference, we'll cycle through these images to create batches
        # For now, just return the first batch_size images
        if batch_size <= len(all_images):
            batch_images = all_images[:batch_size]
        else:
            # If we need more images than we have, repeat some
            indices = torch.randint(0, len(all_images), (batch_size,))
            batch_images = all_images[indices]

        # Store all images for later use in different batch configurations
        self.imagenet_samples = all_images.to(self.device)

        return batch_images.to(self.device)

    def _get_sample_input(self, batch_size: int = None) -> torch.Tensor:
        """Get sample input tensor with specified batch size.

        Args:
            batch_size: Desired batch size. If None, uses self.batch_size

        Returns:
            Tensor of shape (batch_size, 3, 224, 224)
        """
        if batch_size is None:
            batch_size = self.batch_size

        # If we have ImageNet samples, use them
        if hasattr(self, 'imagenet_samples') and self.imagenet_samples is not None:
            if batch_size <= len(self.imagenet_samples):
                return self.imagenet_samples[:batch_size]
            else:
                # If we need more images than we have, repeat some
                indices = torch.randint(0, len(self.imagenet_samples), (batch_size,))
                return self.imagenet_samples[indices]
        else:
            # Fallback to random tensor
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

    def _synchronize(self):
        """Synchronize device if CUDA."""
        if self.device.type == 'cuda':
            # Sync the selected device (not just the current default).
            torch.cuda.synchronize(self.device)

    def _time_operation(self, operation, num_runs: int = 1, warmup_runs: int = 3) -> float:
        """Generic timing function for any operation with warmup."""
        # Warm-up runs
        for _ in range(warmup_runs):
            operation()
        
        # Actual timing runs
        times = []
        for _ in range(num_runs):
            self._synchronize()
            start_time = time.perf_counter()
            operation()
            self._synchronize()
            times.append(time.perf_counter() - start_time)
        
        return np.mean(times)

    def _time_forward_pass(self, model: torch.nn.Module, input_tensor: torch.Tensor, num_runs: int = 1) -> float:
        """Measure forward pass time for a model."""
        def forward():
            with torch.no_grad():
                _ = model(input_tensor)
        return self._time_operation(forward, num_runs)

    def _time_acm_operation(self, matrix: torch.Tensor, key: List[int], is_encrypt: bool = True, num_runs: int = 1) -> float:
        """Measure ACM encryption/decryption time using optimized functions."""
        op = self.arnold_encrypt if is_encrypt else self.arnold_decrypt
        def acm_op():
            _ = op(matrix, key)
        return self._time_operation(acm_op, num_runs)

    def _time_ffn_operation(self, matrix: torch.Tensor, perm_indices: torch.Tensor, is_encrypt: bool = True, num_runs: int = 1) -> float:
        """Measure FFN permutation encryption/decryption time."""
        def ffn_op():
            if is_encrypt:
                _ = matrix[:, perm_indices]
            else:
                inv_perm = torch.argsort(perm_indices)
                _ = matrix[:, inv_perm]
        return self._time_operation(ffn_op, num_runs)

    def _time_realistic_encrypted_inference(self, encrypted_layers: List[int], acm_key: List[int], 
                                           num_runs: int = None, warmup_runs: int = 3) -> Dict[str, float]:
        """Time realistic encrypted inference with actual model weight manipulation.
        
        This follows the Simulation.ipynb methodology where we actually encrypt/decrypt
        model weights in-place during inference, providing realistic timing measurements.
        
        Returns separate timing for ACM and FFN components.
        """
        from collections import defaultdict
        
        # Set number of runs based on device if not specified
        if num_runs is None:
            num_runs = NUM_RUNS_GPU if self.device.type == 'cuda' else NUM_RUNS_CPU
        
        # Store original values of weights we will modify to avoid using load_state_dict
        # which is problematic for quantized models
        saved_weights_data = {}
        for layer_idx in encrypted_layers:
            layer = self.model.vit.encoder.layer[layer_idx]
            weights_to_save = {
                'query': layer.attention.attention.query.weight,
                'key': layer.attention.attention.key.weight,
                'value': layer.attention.attention.value.weight,
                'attention_output': layer.attention.output.dense.weight,
                'intermediate': layer.intermediate.dense.weight,
                'output': layer.output.dense.weight,
            }
            for name, param in weights_to_save.items():
                saved_weights_data[(layer_idx, name)] = param.data.clone()
        
        # Generate permutation indices for each layer DURING timing (not pre-computed)
        # This ensures fair benchmarking where everything is computed from scratch
        def generate_permutation_indices(layer_idx: int):
            """Generate permutation indices for a specific layer during timing."""
            layer = self.model.vit.encoder.layer[layer_idx]
            # Get FFN dimensions
            intermediate_weight = layer.intermediate.dense.weight
            output_weight = layer.output.dense.weight
            
            # Generate permutation indices (simulating PUF-derived keys)
            if self.derived_perm_password:
                h = hmac.new(self.derived_perm_password.encode(), str(layer_idx).encode(), hashlib.sha256)
                # Seed must be between 0 and 2**32 - 1 for torch.manual_seed
                seed = int.from_bytes(h.digest()[:8], byteorder='big') % (2**32)
                torch.manual_seed(seed)
            else:
                torch.manual_seed(42 + layer_idx)  # Deterministic for reproducibility
                
            return {
                'in': torch.randperm(intermediate_weight.shape[1], device=self.device),
                'hidden': torch.randperm(output_weight.shape[1], device=self.device)
            }

        # Get sample input for current batch size
        sample_input = self._get_sample_input()

        # Warmup runs - general model inference
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(sample_input)
        
        # ACM-specific warmup to avoid compilation overhead
        # Use a weight of the same type as the model weights (could be int8)
        sample_weight_param = self.model.vit.encoder.layer[0].attention.attention.query.weight
        sample_weight = torch.randn(768, 768, device=self.device).to(sample_weight_param.dtype)
        for _ in range(3):
            self._synchronize()
            _ = self.arnold_encrypt(sample_weight, acm_key)
            _ = self.arnold_decrypt(sample_weight, acm_key)
            self._synchronize()
        
        # Additional GPU warmup to ensure consistent performance
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._synchronize()
            
            for _ in range(2):
                _ = self.arnold_encrypt(sample_weight, acm_key)
                _ = self.arnold_decrypt(sample_weight, acm_key)
                self._synchronize()
            
            # Warm up FFN operations
            sample_ffn = torch.randn(768, 768, device=self.device).to(sample_weight_param.dtype)
            sample_perms = torch.randperm(768, device=self.device)
            
            for _ in range(3):
                inv_perms = torch.zeros_like(sample_perms)
                inv_perms[sample_perms] = torch.arange(len(sample_perms), device=sample_perms.device, dtype=sample_perms.dtype)
                _ = sample_ffn[:, inv_perms]
                _ = sample_ffn[:, sample_perms]
                self._synchronize()
            
            # Warm up XOR operations
            sample_xor_weight = torch.randn(768, 768, device=self.device).to(sample_weight_param.dtype)
            
            for _ in range(3):
                # Use optimized version for warmup (on-the-fly key)
                xor_encrypt_decrypt_triton(sample_xor_weight, layer_idx=0, weight_name='test',
                                           seed_base=42)
                xor_encrypt_decrypt_triton(sample_xor_weight, layer_idx=0, weight_name='test',
                                           seed_base=42)  # Second call to decrypt back
                self._synchronize()
            
        


        # Measure baseline inference (no encryption)
        baseline_times = []
        for _ in range(num_runs):
            self._synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(sample_input)
            
            self._synchronize()
            baseline_times.append(time.perf_counter() - start_time)

        # Measure encrypted inference with realistic sequential processing
        encrypted_times = []
        acm_decrypt_times = []
        acm_encrypt_times = []
        ffn_decrypt_times = []
        ffn_encrypt_times = []
        xor_decrypt_times = []
        xor_encrypt_times = []
        
        for run_idx in range(num_runs):
            # Reset model weights to original data
            for (layer_idx, name), original_data in saved_weights_data.items():
                layer = self.model.vit.encoder.layer[layer_idx]
                if name == 'query': layer.attention.attention.query.weight.data.copy_(original_data)
                elif name == 'key': layer.attention.attention.key.weight.data.copy_(original_data)
                elif name == 'value': layer.attention.attention.value.weight.data.copy_(original_data)
                elif name == 'attention_output': layer.attention.output.dense.weight.data.copy_(original_data)
                elif name == 'intermediate': layer.intermediate.dense.weight.data.copy_(original_data)
                elif name == 'output': layer.output.dense.weight.data.copy_(original_data)
            
            run_acm_decrypt_time = 0
            run_acm_encrypt_time = 0
            run_ffn_decrypt_time = 0
            run_ffn_encrypt_time = 0
            run_xor_decrypt_time = 0
            run_xor_encrypt_time = 0
            
            self._synchronize()
            inference_start = time.perf_counter()
            
            # Decryption phase - decrypt all layers before inference
            for layer_idx in encrypted_layers:
                layer = self.model.vit.encoder.layer[layer_idx]
                perms = generate_permutation_indices(layer_idx)
                
                # Time XOR decryption (applied to all weights)
                self._synchronize()
                start = time.perf_counter()
                
                # XOR encrypt all weights in the layer
                all_weights = [
                    ('query', layer.attention.attention.query.weight.data),
                    ('key', layer.attention.attention.key.weight.data),
                    ('value', layer.attention.attention.value.weight.data),
                    ('attention_output', layer.attention.output.dense.weight.data),
                    ('intermediate', layer.intermediate.dense.weight.data),
                    ('output', layer.output.dense.weight.data),
                ]
                
                # Skip XOR timing for attention weights if using fused triton
                # since they will be handled by the ACM kernel
                xor_weights_to_time = all_weights
                if self.use_triton:
                    xor_weights_to_time = [
                        ('intermediate', layer.intermediate.dense.weight.data),
                        ('output', layer.output.dense.weight.data),
                    ]
                
                # Check if we need to regenerate keys (fair benchmarking)
                for weight_name, w in xor_weights_to_time:
                    if self.use_triton and w.is_cuda:
                        # Use optimized on-the-fly XOR (CUDA only)
                        xor_encrypt_decrypt_triton(w, layer_idx=layer_idx, weight_name=weight_name, seed_base=self.derived_xor_seed_base)
                    else:
                        xor_encrypt_decrypt_numba(w, layer_idx=layer_idx, weight_name=weight_name, seed_base=self.derived_xor_seed_base)
                
                self._synchronize()
                run_xor_decrypt_time += time.perf_counter() - start
                
                # Time attention decryption (ACM)
                self._synchronize()
                start = time.perf_counter()
                
                attention_weights = [
                    ('query', layer.attention.attention.query.weight.data),
                    ('key', layer.attention.attention.key.weight.data),
                    ('value', layer.attention.attention.value.weight.data),
                    ('attention_output', layer.attention.output.dense.weight.data)
                ]
                
                for name, w in attention_weights:
                    if w.shape[0] == w.shape[1]:  # Only square matrices for ACM
                        if self.use_triton and w.is_cuda:
                            # Use fused inverse ACM + XOR
                            xor_seed = get_stable_seed(layer_idx, name, seed_base=self.derived_xor_seed_base)
                            w.copy_(iarnold_triton(w, self.derived_acm_key, xor_seed=xor_seed))
                        else:
                            w.copy_(self.arnold_decrypt(w, self.derived_acm_key))
                
                self._synchronize()
                run_acm_decrypt_time += time.perf_counter() - start
                
                # Time FFN decryption
                self._synchronize()
                start = time.perf_counter()
                
                # Fast O(n) inverse permutation
                inv_perm_in = torch.zeros_like(perms['in'])
                inv_perm_hidden = torch.zeros_like(perms['hidden'])
                inv_perm_in[perms['in']] = torch.arange(len(perms['in']), device=perms['in'].device, dtype=perms['in'].dtype)
                inv_perm_hidden[perms['hidden']] = torch.arange(len(perms['hidden']), device=perms['hidden'].device, dtype=perms['hidden'].dtype)
                
                layer.intermediate.dense.weight.data = layer.intermediate.dense.weight.data[:, inv_perm_in]
                layer.output.dense.weight.data = layer.output.dense.weight.data[:, inv_perm_hidden]
                
                self._synchronize()
                run_ffn_decrypt_time += time.perf_counter() - start

            # Perform inference on decrypted model
            with torch.no_grad():
                _ = self.model(sample_input)
            
            # Re-encryption phase - encrypt all layers after inference
            for layer_idx in encrypted_layers:
                layer = self.model.vit.encoder.layer[layer_idx]
                perms = generate_permutation_indices(layer_idx)
                
                # Time FFN re-encryption
                self._synchronize()
                start = time.perf_counter()
                
                layer.intermediate.dense.weight.data = layer.intermediate.dense.weight.data[:, perms['in']]
                layer.output.dense.weight.data = layer.output.dense.weight.data[:, perms['hidden']]
                
                self._synchronize()
                run_ffn_encrypt_time += time.perf_counter() - start
                
                # Time attention re-encryption (ACM)
                self._synchronize()
                start = time.perf_counter()
                
                attention_weights = [
                    ('query', layer.attention.attention.query.weight.data),
                    ('key', layer.attention.attention.key.weight.data),
                    ('value', layer.attention.attention.value.weight.data),
                    ('attention_output', layer.attention.output.dense.weight.data)
                ]
                for name, w in attention_weights:
                    # if w.shape[0] == w.shape[1]:  # Only square matrices for ACM
                    if self.use_triton and w.is_cuda:
                        # Use fused ACM + XOR
                        xor_seed = get_stable_seed(layer_idx, name, seed_base=self.derived_xor_seed_base)
                        w.copy_(arnold_triton(w, self.derived_acm_key, xor_seed=xor_seed))
                    else:
                        w.copy_(self.arnold_encrypt(w, self.derived_acm_key))
                
                self._synchronize()
                run_acm_encrypt_time += time.perf_counter() - start
                
                # Time XOR encryption (applied to all weights)
                self._synchronize()
                start = time.perf_counter()
                
                # XOR encrypt all weights in the layer
                all_weights = [
                    ('query', layer.attention.attention.query.weight.data),
                    ('key', layer.attention.attention.key.weight.data),
                    ('value', layer.attention.attention.value.weight.data),
                    ('attention_output', layer.attention.output.dense.weight.data),
                    ('intermediate', layer.intermediate.dense.weight.data),
                    ('output', layer.output.dense.weight.data),
                ]
                
                # Skip XOR timing for attention weights if using fused triton
                # since they were already handled by the ACM kernel
                xor_weights_to_time = all_weights
                if self.use_triton:
                    xor_weights_to_time = [
                        ('intermediate', layer.intermediate.dense.weight.data),
                        ('output', layer.output.dense.weight.data),
                    ]
                
                for weight_name, w in xor_weights_to_time:
                    if self.use_triton and w.is_cuda:
                        # Use optimized on-the-fly XOR (CUDA only)
                        xor_encrypt_decrypt_triton(w, layer_idx=layer_idx, weight_name=weight_name, seed_base=self.derived_xor_seed_base)
                    else:
                        xor_encrypt_decrypt_numba(w, layer_idx=layer_idx, weight_name=weight_name, seed_base=self.derived_xor_seed_base)
                
                self._synchronize()
                run_xor_encrypt_time += time.perf_counter() - start
            
            self._synchronize()
            total_inference_time = time.perf_counter() - inference_start
            
            encrypted_times.append(total_inference_time)
            acm_decrypt_times.append(run_acm_decrypt_time)
            acm_encrypt_times.append(run_acm_encrypt_time)
            ffn_decrypt_times.append(run_ffn_decrypt_time)
            ffn_encrypt_times.append(run_ffn_encrypt_time)
            xor_decrypt_times.append(run_xor_decrypt_time)
            xor_encrypt_times.append(run_xor_encrypt_time)

        # Restore original model weights
        for (layer_idx, name), original_data in saved_weights_data.items():
            layer = self.model.vit.encoder.layer[layer_idx]
            if name == 'query': layer.attention.attention.query.weight.data.copy_(original_data)
            elif name == 'key': layer.attention.attention.key.weight.data.copy_(original_data)
            elif name == 'value': layer.attention.attention.value.weight.data.copy_(original_data)
            elif name == 'attention_output': layer.attention.output.dense.weight.data.copy_(original_data)
            elif name == 'intermediate': layer.intermediate.dense.weight.data.copy_(original_data)
            elif name == 'output': layer.output.dense.weight.data.copy_(original_data)
        
        return {
            'baseline_time': np.mean(baseline_times),
            'encrypted_time': np.mean(encrypted_times),
            'acm_decrypt_time': np.mean(acm_decrypt_times),
            'acm_encrypt_time': np.mean(acm_encrypt_times),
            'ffn_decrypt_time': np.mean(ffn_decrypt_times),
            'ffn_encrypt_time': np.mean(ffn_encrypt_times),
            'xor_decrypt_time': np.mean(xor_decrypt_times),
            'xor_encrypt_time': np.mean(xor_encrypt_times),
            'total_overhead': (np.mean(acm_decrypt_times) + np.mean(acm_encrypt_times) + 
                             np.mean(ffn_decrypt_times) + np.mean(ffn_encrypt_times) +
                             np.mean(xor_decrypt_times) + np.mean(xor_encrypt_times))
        }




    def _timing_result_to_breakdown(self, timing_result: Dict[str, float]) -> TimingBreakdown:
        """Convert timing result dictionary to TimingBreakdown."""
        return TimingBreakdown(
            total_time=timing_result['encrypted_time'],
            forward_time=timing_result['baseline_time'],
            acm_enc_time=timing_result['acm_encrypt_time'],
            acm_dec_time=timing_result['acm_decrypt_time'],
            ffn_enc_time=timing_result['ffn_encrypt_time'],
            ffn_dec_time=timing_result['ffn_decrypt_time'],
            xor_enc_time=timing_result['xor_encrypt_time'],
            xor_dec_time=timing_result['xor_decrypt_time'],
            overhead_ratio=timing_result['encrypted_time'] / timing_result['baseline_time']
        )

    def analyze_layers_overhead(self, max_layers: int = 12) -> List[TimingBreakdown]:
        """Analyze overhead vs Number of Encrypted Layers using realistic methodology."""
        print(f"\nAnalyzing layers overhead (max_layers={max_layers})...")

        key = [3, 1, 1, 1, 2]  # Fixed ACM key (n=3)
        layer_counts = [2, 4, 6, 8, 10, 12]
        
        results = []
        for num_layers in tqdm(layer_counts):
            encrypted_layers = list(range(num_layers))
            timing_result = self._time_realistic_encrypted_inference(encrypted_layers, key, num_runs=None)
            results.append(self._timing_result_to_breakdown(timing_result))

        return results

    def _create_stacked_bar_chart(self, ax, results: List[TimingBreakdown], x_labels: List[int], 
                                   x_label: str, show_legend: bool = True):
        """Create a stacked bar chart from TimingBreakdown results."""
        FONT_SIZES = self.FONT_SIZES
        colors = {
            'Forward Pass': '#2ecc71',
            'ACM Encryption': '#e74c3c',
            'ACM Decryption': '#c0392b',
            'FFN Encryption': '#3498db',
            'FFN Decryption': '#2980b9',
            'XOR Encryption': '#9b59b6',      # Purple
            'XOR Decryption': '#8e44ad'       # Darker purple
        }

        forward_times = [r.forward_time * 1000 for r in results]
        acm_enc_times = [r.acm_enc_time * 1000 for r in results]
        acm_dec_times = [r.acm_dec_time * 1000 for r in results]
        ffn_enc_times = [r.ffn_enc_time * 1000 for r in results]
        ffn_dec_times = [r.ffn_dec_time * 1000 for r in results]
        xor_enc_times = [r.xor_enc_time * 1000 for r in results]
        xor_dec_times = [r.xor_dec_time * 1000 for r in results]

        bar_width = 0.8
        x_positions = np.arange(len(x_labels))

        # Stack bars: Forward Pass -> XOR Decrypt -> ACM Decrypt -> FFN Decrypt -> FFN Encrypt -> ACM Encrypt -> XOR Encrypt
        ax.bar(x_positions, forward_times, width=bar_width, label='Forward Pass', 
               color=colors['Forward Pass'], alpha=0.7)
        ax.bar(x_positions, xor_dec_times, width=bar_width, bottom=forward_times, 
               label='XOR Decryption', color=colors['XOR Decryption'], alpha=0.7)
        ax.bar(x_positions, acm_dec_times, width=bar_width,
               bottom=[f + xd for f, xd in zip(forward_times, xor_dec_times)],
               label='ACM Decryption', color=colors['ACM Decryption'], alpha=0.7)
        ax.bar(x_positions, ffn_dec_times, width=bar_width,
               bottom=[f + xd + ad for f, xd, ad in zip(forward_times, xor_dec_times, acm_dec_times)],
               label='FFN Decryption', color=colors['FFN Decryption'], alpha=0.7)
        ax.bar(x_positions, ffn_enc_times, width=bar_width,
               bottom=[f + xd + ad + fd for f, xd, ad, fd in zip(forward_times, xor_dec_times, acm_dec_times, ffn_dec_times)],
               label='FFN Encryption', color=colors['FFN Encryption'], alpha=0.7)
        ax.bar(x_positions, acm_enc_times, width=bar_width,
               bottom=[f + xd + ad + fd + fe for f, xd, ad, fd, fe in zip(forward_times, xor_dec_times, acm_dec_times, ffn_dec_times, ffn_enc_times)],
               label='ACM Encryption', color=colors['ACM Encryption'], alpha=0.7)
        ax.bar(x_positions, xor_enc_times, width=bar_width,
               bottom=[f + xd + ad + fd + fe + ae for f, xd, ad, fd, fe, ae in zip(forward_times, xor_dec_times, acm_dec_times, ffn_dec_times, ffn_enc_times, acm_enc_times)],
               label='XOR Encryption', color=colors['XOR Encryption'], alpha=0.7)

        # Add overhead ratios above bars
        total_times = [f + xd + ad + fd + fe + ae + xe for f, xd, ad, fd, fe, ae, xe in 
                      zip(forward_times, xor_dec_times, acm_dec_times, ffn_dec_times, ffn_enc_times, acm_enc_times, xor_enc_times)]
        for i, (total_time, forward_time) in enumerate(zip(total_times, forward_times)):
            overhead_ratio = total_time / forward_time if forward_time > 0 else 0
            ax.text(i, total_time + 0.05, f'{overhead_ratio:.3f}×' if overhead_ratio > 0 else 'N/A',
                   ha='center', va='bottom', fontsize=FONT_SIZES['ratio_text'], fontweight='bold')

        ax.set_xlabel(x_label, fontsize=FONT_SIZES['axis_labels'])
        ax.set_ylabel('Time (ms)', fontsize=FONT_SIZES['axis_labels'])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=FONT_SIZES['tick_labels'])
        ax.tick_params(axis='y', labelsize=FONT_SIZES['tick_labels'])
        ax.set_ylim(0, max(total_times) * 1.15)

        if show_legend:
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZES['legend'])
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_edgecolor('lightgray')
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)

    def create_overhead_breakdown_plot(self,
                                     acm_results: List[TimingBreakdown],
                                     layers_results: List[TimingBreakdown],
                                     acm_results_batch64: List[TimingBreakdown],
                                     layers_results_batch64: List[TimingBreakdown],
                                     device_name: str = "GPU",
                                     output_dir: str = "results/analysis"):
        """Create comprehensive overhead breakdown plot with batch size comparison."""
        print("\nCreating overhead breakdown plot...")

        FONT_SIZES = self.FONT_SIZES
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        n_layers = [2, 4, 6, 8, 10, 12]

        # Two subplots: batch size 1 and batch size 64, both vs encrypted layers
        fig, axes = plt.subplots(1, 2, figsize=(36, 8))
        plt.subplots_adjust(top=1, hspace=0.3, wspace=0.2)

        self._create_stacked_bar_chart(axes[0], layers_results, n_layers, 
                                       'Number of Encrypted Layers (Batch Size 1)', show_legend=True)
        self._create_stacked_bar_chart(axes[1], layers_results_batch64, n_layers, 
                                       'Number of Encrypted Layers (Batch Size 64)', show_legend=False)

        plt.tight_layout()

        # Save both PNG and PDF versions
        png_path = f'{output_dir}/inference_overhead_breakdown_{device_name.lower()}.png'
        pdf_path = f'{output_dir}/inference_overhead_breakdown_{device_name.lower()}.pdf'

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

        print(f"✓ Saved: {png_path}")
        print(f"✓ Saved: {pdf_path}")

        # Save data as well (layers only)
        self.save_overhead_data(layers_results, output_dir, device_name)

    def _timing_to_dict(self, result: TimingBreakdown, **extra_fields) -> dict:
        """Convert TimingBreakdown to dictionary for CSV export."""
        return {
            'forward_time_ms': result.forward_time * 1000,
            'acm_enc_time_ms': result.acm_enc_time * 1000,
            'acm_dec_time_ms': result.acm_dec_time * 1000,
            'ffn_enc_time_ms': result.ffn_enc_time * 1000,
            'ffn_dec_time_ms': result.ffn_dec_time * 1000,
            'xor_enc_time_ms': result.xor_enc_time * 1000,
            'xor_dec_time_ms': result.xor_dec_time * 1000,
            'total_time_ms': result.total_time * 1000,
            'overhead_ratio': result.overhead_ratio,
            **extra_fields
        }

    def save_overhead_data(self,
                          layers_results: List[TimingBreakdown],
                          output_dir: str,
                          device_name: str = "GPU"):
        """Save the overhead analysis data (layers experiment only)."""
        import pandas as pd

        layer_counts = [2, 4, 6, 8, 10, 12]
        
        layers_data = [self._timing_to_dict(r, n_layers=n) 
                       for r, n in zip(layers_results, layer_counts)]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        layers_filename = f'layers_overhead_{device_name.lower()}.csv'
        
        pd.DataFrame(layers_data).to_csv(f'{output_dir}/{layers_filename}', index=False)

        print(f"✓ Saved data: {output_dir}/{layers_filename}")

    def create_batch64_only_plot(self,
                                layers_results_batch64: List[TimingBreakdown],
                                device_name: str = "GPU",
                                output_dir: str = "results/analysis"):
        """Create a separate figure with only batch size 64 plots (layers experiment only)."""
        print("\nCreating batch size 64 only plot (layers only)...")

        FONT_SIZES = self.FONT_SIZES
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        fig, ax = plt.subplots(1, 1, figsize=(18, 8))
        plt.subplots_adjust(top=1, hspace=0.3, wspace=0.2)

        n_layers = [2, 4, 6, 8, 10, 12]

        self._create_stacked_bar_chart(ax, layers_results_batch64, n_layers, 
                                       'Number of Encrypted Layers (Batch Size 64)', show_legend=True)

        plt.tight_layout()

        # Save both PNG and PDF versions
        png_path = f'{output_dir}/inference_overhead_batch64_only_{device_name.lower()}.png'
        pdf_path = f'{output_dir}/inference_overhead_batch64_only_{device_name.lower()}.pdf'

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

        print(f"✓ Saved: {png_path}")
        print(f"✓ Saved: {pdf_path}")

    def _load_results_from_csv(self, csv_path: Path) -> List[TimingBreakdown]:
        """Load TimingBreakdown results from CSV file."""
        import pandas as pd
        
        if not csv_path.exists():
            return None
        
        df = pd.read_csv(csv_path)
        results = []
        for _, row in df.iterrows():
            forward_time = max(row['forward_time_ms'] / 1000.0, 1e-8)
            # Handle both old format (without XOR) and new format (with XOR)
            xor_enc_time = row.get('xor_enc_time_ms', 0.0) / 1000.0
            xor_dec_time = row.get('xor_dec_time_ms', 0.0) / 1000.0
            results.append(TimingBreakdown(
                total_time=row['total_time_ms'] / 1000.0,
                forward_time=forward_time,
                acm_enc_time=row['acm_enc_time_ms'] / 1000.0,
                acm_dec_time=row['acm_dec_time_ms'] / 1000.0,
                ffn_enc_time=row['ffn_enc_time_ms'] / 1000.0,
                ffn_dec_time=row['ffn_dec_time_ms'] / 1000.0,
                xor_enc_time=xor_enc_time,
                xor_dec_time=xor_dec_time,
                overhead_ratio=row['overhead_ratio']
            ))
        return results

    def regenerate_plots_from_csv(self, device_name: str = "gpu", output_dir: str = "results/analysis", include_batch64: bool = True):
        """Regenerate plots from existing CSV files without rerunning the experiments (layers only)."""
        from pathlib import Path

        print(f"\nRegenerating plots from CSV files for {device_name.upper()} (layers only)...")

        base_path = Path(output_dir)
        layers_csv = base_path / f"layers_overhead_{device_name.lower()}.csv"

        if not layers_csv.exists():
            print(f"Error: CSV file not found in {output_dir}")
            print(f"Expected file: {layers_csv.name}")
            return

        layers_results = self._load_results_from_csv(layers_csv)

        layers_results_batch64 = None

        if include_batch64:
            layers_csv_b64 = base_path / f"layers_overhead_{device_name.lower()}_batch64.csv"
            
            if layers_csv_b64.exists():
                layers_results_batch64 = self._load_results_from_csv(layers_csv_b64)
                print("✓ Found batch size 64 data")
            else:
                print("⚠ Batch size 64 CSV file not found, using only batch size 1 data")
                include_batch64 = False

        device_upper = device_name.upper()
        if include_batch64 and layers_results_batch64:
            self.create_overhead_breakdown_plot(
                acm_results=None,
                layers_results=layers_results,
                acm_results_batch64=None,
                layers_results_batch64=layers_results_batch64,
                device_name=device_upper,
                output_dir=output_dir
            )
            self.create_batch64_only_plot(
                layers_results_batch64,
                device_upper, output_dir
            )
        else:
            self.create_overhead_breakdown_plot(
                acm_results=None,
                layers_results=layers_results,
                acm_results_batch64=None,
                layers_results_batch64=layers_results,
                device_name=device_upper,
                output_dir=output_dir
            )

        print(f"✓ Successfully regenerated plots for {device_upper}")

    def run_complete_analysis(self, output_dir: str = "results/analysis"):
        """Run the complete overhead analysis with batch size comparison."""
        print(f"Starting complete inference overhead analysis (Master Secret: {self.master_secret is not None})...")

        # Store current batch size
        original_batch_size = self.batch_size
        
        # Analyze with batch size 1 (single inference)
        self.batch_size = 1
        print(f"\nRunning analysis with batch_size={self.batch_size}...")

        # Sample input will be handled by _get_sample_input() method
        
        # Analyze layers overhead only (ACM iterations experiment removed)
        layers_results = self.analyze_layers_overhead(max_layers=12)

        # Save batch size 1 results to CSV
        device_name = "GPU" if self.device.type == 'cuda' else "CPU"
        self.save_overhead_data(layers_results, output_dir, device_name)

        # Now analyze with batch size 64
        self.batch_size = 64
        print(f"\nRunning analysis with batch_size={self.batch_size}...")

        # Sample input will be handled by _get_sample_input() method
        
        # Analyze layers overhead with batch size 64
        layers_results_batch64 = self.analyze_layers_overhead(max_layers=12)

        # Save batch size 64 results to CSV with suffix
        self.save_overhead_data(layers_results_batch64, output_dir, f"{device_name}_batch64")
        
        # Restore original batch size
        self.batch_size = original_batch_size

        # Create comprehensive plot with both batch sizes (layers only)
        device_name = "GPU" if self.device.type == 'cuda' else "CPU"
        self.create_overhead_breakdown_plot(
            acm_results=None,
            layers_results=layers_results,
            acm_results_batch64=None,
            layers_results_batch64=layers_results_batch64,
            device_name=device_name,
            output_dir=output_dir
        )

        # Create separate batch size 64 only plot (layers only)
        self.create_batch64_only_plot(
            layers_results_batch64,
            device_name, output_dir
        )

        print("\nInference overhead analysis complete!")
        print(f"Results saved to: {output_dir}/")


def run_device_analysis(device: str, use_triton: bool = False, master_secret: Optional[str] = None):
    """Run analysis for a specific device."""
    print(f"\n{'='*60}")
    print(f"Running analysis on {device.upper()} (Triton: {use_triton}, Master Secret: {master_secret is not None})")
    print(f"{'='*60}")
    
    # Initialize analyzer with batch_size=1 (single sequential inference)
    # This simulates the real-world scenario where each inference processes
    # one input and requires sequential encryption/decryption of all layers
    print(f"Initializing analyzer with batch_size=1 (sequential inference) on {device.upper()}...")
    analyzer = InferenceOverheadAnalyzer(device=device, batch_size=1, use_triton=use_triton, master_secret=master_secret)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


def regenerate_plots(device_name: str = "gpu", output_dir: str = "results/analysis", include_batch64: bool = True):
    """Regenerate plots from existing CSV files."""
    analyzer = InferenceOverheadAnalyzer(device="cpu", batch_size=1, skip_imagenet=True)
    analyzer.regenerate_plots_from_csv(device_name, output_dir, include_batch64)

def main():
    """Main function to run the inference overhead analysis on both GPU and CPU.

    To adjust the number of evaluation runs:
    - Modify NUM_RUNS_CPU for CPU evaluation
    - Modify NUM_RUNS_GPU for GPU evaluation

    To regenerate plots from existing CSV files, use:
    regenerate_plots(device_name="gpu", output_dir="results/analysis")

    Command line usage:
    python inference_overhead_analysis.py --replot --device gpu --output-dir results/analysis
    python inference_overhead_analysis.py --replot --device cuda:1 --output-dir results/analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inference Overhead Analysis for Triple Encryption (ACM + FFN + XOR)')
    parser.add_argument('--replot', action='store_true',
                       help='Regenerate plots from existing CSV files instead of running new experiments')
    parser.add_argument('--device', type=str, default='gpu',
                       help='Device for computation (gpu, cpu, cuda, cuda:0, cuda:1, etc.)')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                       help='Directory containing CSV files or where to save results (default: results/analysis)')
    parser.add_argument('--include-batch64', action='store_true', default=True,
                       help='Include batch size 64 data if available (default: True)')
    parser.add_argument('--triton', action='store_true',
                       help='Use Triton kernels for encryption/decryption (CUDA only)')
    parser.add_argument('--master-secret', type=str, default="puf_sim_root_secret_2026",
                       help='Unified master secret (PUF simulation root)')

    args = parser.parse_args()

    # Handle CSV regeneration mode
    if args.replot:
        print("Regenerating plots from CSV files...")
        # For CSV regeneration, normalize device name for file lookup
        if args.device.startswith('cuda') or args.device == 'gpu':
            device_name = 'gpu'
        else:
            device_name = 'cpu'
        regenerate_plots(
            device_name=device_name,
            output_dir=args.output_dir,
            include_batch64=args.include_batch64
        )
        return

    # Validate and normalize the requested device
    requested_device = args.device
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"✓ Multiple CUDA devices available: {torch.cuda.device_count()} devices")
    print("✓ CPU available")

    if requested_device == 'gpu':
        device_to_run = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_to_run == 'cpu':
            print("Warning: GPU requested but CUDA not available, using CPU")
    elif requested_device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("Error: CUDA requested but not available")
            return
        if ':' in requested_device:
            device_id = int(requested_device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                print(f"Error: CUDA device {device_id} not available. Available devices: 0-{torch.cuda.device_count()-1}")
                return
        device_to_run = requested_device
    elif requested_device == 'cpu':
        device_to_run = 'cpu'
    else:
        print(f"Error: Invalid device '{requested_device}'. Use 'gpu', 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.")
        return

    print(f"\nRunning analysis on device: {device_to_run.upper()}")
    print(f"Configuration: CPU runs={NUM_RUNS_CPU}, GPU runs={NUM_RUNS_GPU}")

    # Run analysis on the specified device
    try:
        run_device_analysis(device_to_run, use_triton=args.triton, master_secret=args.master_secret)
    except Exception as e:
        print(f"Error running analysis on {device_to_run.upper()}: {str(e)}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

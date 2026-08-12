from __future__ import annotations

import unittest

import torch

from src.encryption.arnold_transform import arnold_triton, iarnold_triton
from src.encryption.chacha20 import (
    chacha20_block,
    chacha20_xor_,
    derive_chacha20_material,
)


class ChaCha20DiffusionTest(unittest.TestCase):
    def test_rfc8439_block_vector(self) -> None:
        key = bytes(range(32))
        nonce = bytes.fromhex("000000090000004a00000000")
        expected = bytes.fromhex(
            "10f1e7e4d13b5915500fdd1fa32071c4"
            "c7d1f4c733c068030422aa9ac3d46c4e"
            "d2826446079faa0914c2d705d98b02a2"
            "b5129cd1de164eb9cbd083e8a2503c4e"
        )
        self.assertEqual(chacha20_block(key, 1, nonce), expected)

    def test_cpu_round_trip_and_domain_separation(self) -> None:
        original = torch.arange(256, dtype=torch.int32)
        first = original.clone()
        second = original.clone()
        key_a, nonce_a = derive_chacha20_material(3, "attention.query", 12345)
        key_b, nonce_b = derive_chacha20_material(4, "attention.query", 12345)

        chacha20_xor_(first, key_a, nonce_a)
        chacha20_xor_(second, key_b, nonce_b)
        self.assertFalse(torch.equal(first, original))
        self.assertFalse(torch.equal(first, second))

        chacha20_xor_(first, key_a, nonce_a)
        self.assertTrue(torch.equal(first, original))

    def test_transformer_cipher_secure_round_trip(self) -> None:
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

        from src.encryption.transformer_cipher import TransformerCipher

        config = Wav2Vec2Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_conv_pos_embedding_groups=2,
            num_conv_pos_embeddings=8,
            conv_dim=(8,),
            conv_stride=(2,),
            conv_kernel=(3,),
            vocab_size=8,
        )
        model = Wav2Vec2ForCTC(config)
        cipher = TransformerCipher(model, secure=True)
        originals = {
            spec.name: spec.parameter.detach().clone()
            for spec in cipher.layer_specs[0].weights
        }

        cipher.encrypt_all()
        self.assertTrue(
            any(
                not torch.equal(spec.parameter, originals[spec.name])
                for spec in cipher.layer_specs[0].weights
            )
        )
        cipher.decrypt_all()
        self.assertTrue(
            all(
                torch.equal(spec.parameter, originals[spec.name])
                for spec in cipher.layer_specs[0].weights
            )
        )

    def test_legacy_dual_encryption_secure_cpu_round_trip(self) -> None:
        from src.encryption.dual_encryption import DualEncryption

        for mode, use_xor in (("basic", True), ("advanced", False)):
            cipher = DualEncryption(
                master_secret="test-device-root",
                num_permutation_matrices=1,
                matrix_size=8,
                device="cpu",
                mode=mode,
                use_xor=use_xor,
            )
            attention = {
                name: torch.randn(8, 8)
                for name in ("query", "key", "value", "output")
            }
            ffn = {
                "intermediate": torch.randn(16, 8),
                "output": torch.randn(8, 16),
            }

            self.assertTrue(
                cipher.verify_encryption_cycle(attention, ffn, layer_idx=2)
            )

    def test_dual_encryption_cpu_uses_device_independent_attention_material(self) -> None:
        from src.encryption.dual_encryption import DualEncryption

        cipher = DualEncryption(
            master_secret="test-device-root",
            num_permutation_matrices=0,
            matrix_size=8,
            device="cpu",
            mode="basic",
            use_xor=True,
        )
        original = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        expected = arnold_triton(
            original,
            cipher.config.arnold_key,
            xor_seed=cipher.diffusion_secret,
            xor_context="attention:2:query",
        )
        actual = cipher.encrypt_attention_weights(
            {"query": original}, layer_idx=2
        )["query"]
        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_matches_cpu_and_round_trips_fp16(self) -> None:
        original = torch.arange(1024, dtype=torch.float16)
        key, nonce = derive_chacha20_material(6, "fc1.weight", b"puf response")

        cpu = original.clone()
        gpu = original.cuda()
        chacha20_xor_(cpu, key, nonce)
        chacha20_xor_(gpu, key, nonce)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(gpu.cpu().view(torch.int16), cpu.view(torch.int16))
        )

        chacha20_xor_(gpu, key, nonce)
        self.assertTrue(torch.equal(gpu.cpu(), original))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_arnold_chacha20_round_trip_is_bit_exact(self) -> None:
        original = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        arnold_key = [5, 1, 1, 1, 2]
        encrypted = arnold_triton(original, arnold_key, xor_seed=123456)
        decrypted = iarnold_triton(encrypted, arnold_key, xor_seed=123456)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(original.view(torch.int16), decrypted.view(torch.int16))
        )


if __name__ == "__main__":
    unittest.main()

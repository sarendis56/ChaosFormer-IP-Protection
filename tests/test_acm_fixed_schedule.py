import numpy as np
import torch
import unittest

from src.encryption.arnold_transform import (
    _matrix_power_mod,
    _matrix_power_mod_fixed_schedule,
    arnold_optimized,
    iarnold_optimized,
)


class FixedScheduleTest(unittest.TestCase):
    def test_matches_binary_exponentiation_for_all_six_bit_powers(self):
        matrix = np.array([[1, 1], [1, 2]], dtype=np.int64)

        for modulus in (17, 64, 768):
            for power in range(64):
                expected = _matrix_power_mod(matrix, power, modulus)
                actual = _matrix_power_mod_fixed_schedule(matrix, power, modulus)
                np.testing.assert_array_equal(actual, expected)

    def test_rejects_powers_outside_six_bit_range(self):
        matrix = np.array([[1, 1], [1, 2]], dtype=np.int64)

        for power in (-1, 64):
            with self.assertRaises(ValueError):
                _matrix_power_mod_fixed_schedule(matrix, power, 64)

    def test_preserves_existing_cpu_acm_outputs(self):
        matrix = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)

        for power in (3, 5, 8, 12, 16, 24, 32, 34):
            key = [power, 1, 1, 1, 2]
            original_encrypted = arnold_optimized(matrix, key)
            fixed_encrypted = arnold_optimized(
                matrix, key, fixed_schedule=True
            )
            original_inverse = iarnold_optimized(original_encrypted, key)
            fixed_inverse = iarnold_optimized(
                fixed_encrypted, key, fixed_schedule=True
            )

            self.assertTrue(torch.equal(fixed_encrypted, original_encrypted))
            self.assertTrue(torch.equal(fixed_inverse, original_inverse))
            self.assertTrue(torch.equal(original_inverse, matrix))


if __name__ == "__main__":
    unittest.main()

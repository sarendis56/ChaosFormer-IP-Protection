import torch

from src.encryption.permutation import generate_permutation_matrix
from src.encryption.dual_encryption import DualEncryption


def test_permutation_generation_does_not_change_global_rng():
    torch.manual_seed(7)
    expected = torch.rand(4)
    torch.manual_seed(7)
    generate_permutation_matrix(size=8, seed=42)
    assert torch.equal(torch.rand(4), expected)


def test_fallback_permutation_does_not_change_global_rng():
    cipher = DualEncryption(matrix_size=2, num_permutation_matrices=0, device="cpu")
    weight = torch.arange(8).reshape(4, 2)
    torch.manual_seed(7)
    expected = torch.rand(4)
    torch.manual_seed(7)
    cipher._knuth_shuffle(weight, 1, "weight")
    assert torch.equal(torch.rand(4), expected)

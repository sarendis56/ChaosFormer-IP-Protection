import hashlib

import numpy as np

import importlib.util
from pathlib import Path

_module_path = Path(__file__).parents[1] / "src" / "analysis" / "puf_fuzzy_extractor.py"
_spec = importlib.util.spec_from_file_location("puf_fuzzy_extractor", _module_path)
_module = importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)
FuzzyExtractor = _module.FuzzyExtractor


def test_fuzzy_extractor_corrects_noise_and_rejects_other_device():
    extractor = FuzzyExtractor()
    rng = np.random.default_rng(7)
    reference = rng.integers(0, 2, 8192, dtype=np.uint8)
    reliability = np.linspace(1.0, 0.5, 8192)
    secret = bytes(range(16))
    enrollment = extractor.enroll(reference, reliability, secret)

    observation = reference.copy()
    selected = enrollment.indices.reshape(extractor.codeword_bits, extractor.repetition)
    observation[selected[:, :2].reshape(-1)] ^= 1

    key, corrected = extractor.reconstruct(observation, enrollment)
    assert key == hashlib.sha256(secret).digest()[:16]
    assert corrected == 0

    other_device = 1 - reference
    key, _ = extractor.reconstruct(other_device, enrollment)
    assert key is None

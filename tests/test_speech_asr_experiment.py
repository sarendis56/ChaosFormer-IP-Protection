import io

import numpy as np
import soundfile as sf

from src.experiments.speech_asr_experiment import _read_audio, selected_layers


def test_read_audio_from_embedded_bytes():
    expected = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, expected, 16_000, format="WAV", subtype="FLOAT")

    actual, sampling_rate = _read_audio({"bytes": buffer.getvalue(), "path": None})

    assert sampling_rate == 16_000
    np.testing.assert_allclose(actual, expected)


def test_selected_speech_layers_are_spread_across_encoder():
    assert selected_layers(12, "6") == [0, 2, 4, 7, 9, 11]
    assert selected_layers(12, "all") is None

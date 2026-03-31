import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys


def _make_tts_module():
    import importlib
    if "backend.tts" in sys.modules:
        del sys.modules["backend.tts"]
    import backend.tts as tts
    return tts


def test_get_voices_returns_list():
    tts = _make_tts_module()
    voices = tts.get_voices()
    assert isinstance(voices, list)
    assert len(voices) > 0
    assert all("id" in v and "language" in v and "gender" in v for v in voices)


def test_get_voices_includes_english():
    tts = _make_tts_module()
    voices = tts.get_voices()
    en_voices = [v for v in voices if v["language"] == "en"]
    assert len(en_voices) >= 3


def test_generate_full_returns_bytes():
    tts = _make_tts_module()
    mock_model = MagicMock()
    fake_audio = np.zeros(24000, dtype=np.float32)
    mock_result = MagicMock()
    mock_result.audio = fake_audio
    mock_model.generate.return_value = [mock_result]

    with patch("backend.tts._model", mock_model):
        result = tts.generate_full("Hello", "casual_male")

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_generate_stream_yields_bytes():
    tts = _make_tts_module()
    mock_model = MagicMock()
    fake_audio = np.zeros(4800, dtype=np.float32)
    mock_result = MagicMock()
    mock_result.audio = fake_audio
    mock_model.generate.return_value = [mock_result, mock_result]

    with patch("backend.tts._model", mock_model):
        chunks = list(tts.generate_stream("Hello", "casual_male"))

    assert len(chunks) == 2
    assert all(isinstance(c, bytes) for c in chunks)
    # Each chunk must be float32 PCM: 4800 samples × 4 bytes = 19200 bytes
    assert all(len(c) == 4800 * 4 for c in chunks)

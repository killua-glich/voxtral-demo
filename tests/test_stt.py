import pytest
from unittest.mock import MagicMock, patch, mock_open
import sys


def _make_stt_module():
    """Import stt freshly after mocks are in place."""
    import importlib
    if "backend.stt" in sys.modules:
        del sys.modules["backend.stt"]
    import backend.stt as stt
    return stt


def test_transcribe_returns_expected_keys():
    stt = _make_stt_module()
    mock_model = MagicMock()
    mock_processor = MagicMock()

    mock_processor.apply_transcrition_request.return_value = {
        "input_ids": MagicMock(shape=(1, 10))
    }
    mock_model.generate.return_value = [list(range(15))]
    mock_processor.decode.return_value = "hello world"

    with patch("backend.stt._model", mock_model), \
         patch("backend.stt._processor", mock_processor), \
         patch("backend.stt._current_model_key", "mini"), \
         patch("tempfile.NamedTemporaryFile", mock_open()), \
         patch("os.unlink"):
        result = stt.transcribe(b"fake", "test.wav")

    assert "transcript" in result
    assert "language" in result
    assert "duration_s" in result


def test_transcribe_strips_whitespace():
    stt = _make_stt_module()
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.apply_transcrition_request.return_value = {
        "input_ids": MagicMock(shape=(1, 10))
    }
    mock_model.generate.return_value = [list(range(15))]
    mock_processor.decode.return_value = "  hello world  "

    with patch("backend.stt._model", mock_model), \
         patch("backend.stt._processor", mock_processor), \
         patch("backend.stt._current_model_key", "mini"), \
         patch("tempfile.NamedTemporaryFile", mock_open()), \
         patch("os.unlink"):
        result = stt.transcribe(b"fake", "test.wav")

    assert result["transcript"] == "hello world"


def test_load_model_skips_if_same_key():
    stt = _make_stt_module()
    with patch("backend.stt._current_model_key", "mini"), \
         patch("backend.stt.VoxtralForConditionalGeneration") as mock_cls:
        stt.load_model("mini")
        mock_cls.from_pretrained.assert_not_called()

import sys
from unittest.mock import MagicMock
import pytest

@pytest.fixture(autouse=True)
def mock_mlx(monkeypatch):
    """Prevent actual model loading during tests."""
    mocks = {
        "mlx": MagicMock(),
        "mlx.core": MagicMock(),
        "mlx_voxtral": MagicMock(),
        "mlx_audio": MagicMock(),
        "mlx_audio.tts": MagicMock(),
        "mlx_audio.tts.utils": MagicMock(),
    }
    for key, value in mocks.items():
        monkeypatch.setitem(sys.modules, key, value)
    yield

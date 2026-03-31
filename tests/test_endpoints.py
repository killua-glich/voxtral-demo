import io
import pytest
from unittest.mock import patch


def _make_app():
    import sys
    # Only evict backend.main so that patches already applied to backend.stt /
    # backend.tts remain in effect when the request is processed.
    for key in list(sys.modules.keys()):
        if key == "backend.main":
            del sys.modules[key]
    from fastapi.testclient import TestClient
    import backend.main as main
    return TestClient(main.app)


def test_voices_returns_list():
    with patch("backend.tts.get_voices", return_value=[
        {"id": "casual_male", "language": "en", "gender": "male"}
    ]):
        client = _make_app()
        response = client.get("/voices")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["id"] == "casual_male"


def test_transcribe_success():
    fake_result = {"transcript": "hello", "language": "en", "duration_s": 1.0}
    with patch("backend.stt.transcribe", return_value=fake_result):
        client = _make_app()
        audio = io.BytesIO(b"RIFF fake wav data")
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", audio, "audio/wav")},
        )
    assert response.status_code == 200
    assert response.json()["transcript"] == "hello"


def test_transcribe_rejects_bad_format():
    client = _make_app()
    bad_file = io.BytesIO(b"not audio")
    response = client.post(
        "/transcribe",
        files={"audio": ("file.exe", bad_file, "application/octet-stream")},
    )
    assert response.status_code == 400


def test_tts_success():
    with patch("backend.tts.generate_full", return_value=b"RIFF fake wav"):
        client = _make_app()
        response = client.post("/tts", data={"text": "hello", "voice": "casual_male"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")


def test_tts_rejects_empty_text():
    client = _make_app()
    response = client.post("/tts", data={"text": "", "voice": "casual_male"})
    assert response.status_code == 400

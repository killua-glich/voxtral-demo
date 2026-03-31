import io
import os

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load as load_tts_model

MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
SAMPLE_RATE = 24000

VOICES = [
    {"id": "casual_male",     "language": "en", "gender": "male"},
    {"id": "casual_female",   "language": "en", "gender": "female"},
    {"id": "cheerful_female", "language": "en", "gender": "female"},
    {"id": "neutral_male",    "language": "en", "gender": "male"},
    {"id": "neutral_female",  "language": "en", "gender": "female"},
    {"id": "fr_male",         "language": "fr", "gender": "male"},
    {"id": "fr_female",       "language": "fr", "gender": "female"},
    {"id": "es_male",         "language": "es", "gender": "male"},
    {"id": "es_female",       "language": "es", "gender": "female"},
    {"id": "de_male",         "language": "de", "gender": "male"},
    {"id": "de_female",       "language": "de", "gender": "female"},
    {"id": "it_male",         "language": "it", "gender": "male"},
    {"id": "it_female",       "language": "it", "gender": "female"},
    {"id": "pt_male",         "language": "pt", "gender": "male"},
    {"id": "pt_female",       "language": "pt", "gender": "female"},
    {"id": "nl_male",         "language": "nl", "gender": "male"},
    {"id": "nl_female",       "language": "nl", "gender": "female"},
    {"id": "ar_male",         "language": "ar", "gender": "male"},
    {"id": "hi_male",         "language": "hi", "gender": "male"},
    {"id": "hi_female",       "language": "hi", "gender": "female"},
]

_model = None


def load_model() -> None:
    global _model
    if _model is None:
        _model = load_tts_model(MODEL_ID)


def get_voices() -> list[dict]:
    return VOICES


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), SAMPLE_RATE, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


def _build_kwargs(text: str, voice: str, speed: float, ref_audio_path: str | None, ref_text: str | None) -> dict:
    kwargs: dict = {"text": text, "voice": voice}
    if speed != 1.0:
        kwargs["speed"] = speed
    if ref_audio_path:
        kwargs["ref_audio"] = ref_audio_path
    if ref_text:
        kwargs["ref_text"] = ref_text
    return kwargs


def generate_full(
    text: str,
    voice: str = "casual_male",
    speed: float = 1.0,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
) -> bytes:
    load_model()
    kwargs = _build_kwargs(text, voice, speed, ref_audio_path, ref_text)
    chunks = [np.array(result.audio) for result in _model.generate(**kwargs)]
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return _to_wav_bytes(audio)


def generate_stream(
    text: str,
    voice: str = "casual_male",
    speed: float = 1.0,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
):
    """Yield raw float32 PCM bytes per chunk at 24 kHz."""
    load_model()
    kwargs = _build_kwargs(text, voice, speed, ref_audio_path, ref_text)
    for result in _model.generate(**kwargs):
        yield np.array(result.audio).astype(np.float32).tobytes()

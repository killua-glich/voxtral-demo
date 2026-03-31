import os
import tempfile

from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

MODELS = {
    "mini": "mistralai/Voxtral-Mini-4B-Realtime-2602",
    "small": "mistralai/Voxtral-Small-24B-2507",
}

_current_model_key: str | None = None
_model = None
_processor = None


def load_model(key: str = "mini") -> None:
    global _current_model_key, _model, _processor
    if _current_model_key == key:
        return
    model_id = MODELS[key]
    _model = VoxtralForConditionalGeneration.from_pretrained(model_id)
    _processor = VoxtralProcessor.from_pretrained(model_id)
    _current_model_key = key


def transcribe(
    audio_bytes: bytes,
    filename: str,
    language: str = "auto",
    temperature: float = 0.0,
    diarize: bool = False,
    model_key: str = "mini",
) -> dict:
    load_model(model_key)

    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        lang = None if language == "auto" else language
        inputs = _processor.apply_transcrition_request(
            language=lang,
            audio=tmp_path,
        )
        outputs = _model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=temperature,
        )
        transcript = _processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return {
            "transcript": transcript.strip(),
            "language": language,
            "duration_s": 0.0,
        }
    finally:
        os.unlink(tmp_path)

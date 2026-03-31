import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

import backend.stt as stt
import backend.tts as tts

app = FastAPI(title="Voxtral Demo API")

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def _check_audio_ext(filename: str) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )


@app.get("/voices")
def list_voices():
    return tts.get_voices()


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form("mini"),
    language: str = Form("auto"),
    temperature: float = Form(0.0),
    diarize: bool = Form(False),
):
    _check_audio_ext(audio.filename or ".wav")
    if model not in ("mini", "small"):
        raise HTTPException(status_code=400, detail="model must be 'mini' or 'small'")

    audio_bytes = await audio.read()
    result = stt.transcribe(
        audio_bytes=audio_bytes,
        filename=audio.filename or "audio.wav",
        language=language,
        temperature=temperature,
        diarize=diarize,
        model_key=model,
    )
    return result


@app.post("/tts")
async def text_to_speech(
    text: str = Form(""),
    voice: str = Form("casual_male"),
    language: str = Form("en"),
    speed: float = Form(1.0),
    stream: bool = Form(False),
    ref_audio: UploadFile | None = File(None),
    ref_text: str = Form(""),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(status_code=400, detail="speed must be between 0.5 and 2.0")

    ref_audio_path: str | None = None
    tmp_ref: str | None = None

    if ref_audio is not None:
        _check_audio_ext(ref_audio.filename or ".wav")
        ref_bytes = await ref_audio.read()
        suffix = os.path.splitext(ref_audio.filename or ".wav")[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(ref_bytes)
            tmp_ref = f.name
        ref_audio_path = tmp_ref

    try:
        if stream:
            def pcm_generator():
                for chunk in tts.generate_stream(
                    text=text,
                    voice=voice,
                    speed=speed,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text or None,
                ):
                    yield chunk

            return StreamingResponse(pcm_generator(), media_type="application/octet-stream")
        else:
            wav_bytes = tts.generate_full(
                text=text,
                voice=voice,
                speed=speed,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text or None,
            )
            return Response(content=wav_bytes, media_type="audio/wav")
    finally:
        if tmp_ref:
            os.unlink(tmp_ref)

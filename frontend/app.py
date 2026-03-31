import io
import os
import time

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_voices() -> list[str]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/voices", timeout=5)
        resp.raise_for_status()
        return [v["id"] for v in resp.json()]
    except Exception:
        return ["casual_male", "casual_female", "neutral_male", "neutral_female"]


def _pcm_bytes_to_numpy(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)


def _wav_bytes_to_numpy(wav: bytes) -> tuple[int, np.ndarray]:
    buf = io.BytesIO(wav)
    data, sr = sf.read(buf, dtype="float32")
    return sr, data


# ---------------------------------------------------------------------------
# Tab: Transcribe
# ---------------------------------------------------------------------------

def run_transcribe(audio_input, model_choice, language, temperature, diarize):
    if audio_input is None:
        return "No audio provided."

    # Gradio gives (sample_rate, np.ndarray) for mic; file path for upload
    if isinstance(audio_input, tuple):
        sr, arr = audio_input
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        audio_bytes = buf.getvalue()
        filename = "recording.wav"
    else:
        with open(audio_input, "rb") as f:
            audio_bytes = f.read()
        filename = os.path.basename(audio_input)

    model_key = "mini" if "Mini" in model_choice else "small"

    try:
        resp = httpx.post(
            f"{BACKEND_URL}/transcribe",
            files={"audio": (filename, audio_bytes, "audio/wav")},
            data={
                "model": model_key,
                "language": language,
                "temperature": str(temperature),
                "diarize": str(diarize).lower(),
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["transcript"]
    except httpx.ConnectError:
        return "❌ Cannot reach backend. Is start.sh running?"
    except httpx.HTTPStatusError as e:
        return f"❌ Backend error {e.response.status_code}: {e.response.text}"


# ---------------------------------------------------------------------------
# Tab: Text to Speech
# ---------------------------------------------------------------------------

def run_tts(text, voice, speed, stream_mode, ref_audio):
    if not text.strip():
        return None, "Text is empty."

    ref_path = ref_audio if isinstance(ref_audio, str) else None
    ref_files = None
    if ref_path:
        ref_files = {"ref_audio": (os.path.basename(ref_path), open(ref_path, "rb"), "audio/wav")}

    data = {
        "text": text,
        "voice": voice,
        "speed": str(speed),
        "stream": "true" if stream_mode else "false",
    }

    start = time.perf_counter()
    status = ""

    try:
        if stream_mode:
            pcm_chunks = []
            first_chunk_t = None
            with httpx.stream("POST", f"{BACKEND_URL}/tts", data=data, files=ref_files, timeout=120) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    if first_chunk_t is None:
                        first_chunk_t = time.perf_counter() - start
                    pcm_chunks.append(chunk)
            total_t = time.perf_counter() - start
            raw = b"".join(pcm_chunks)
            audio_array = _pcm_bytes_to_numpy(raw)
            status = f"⚡ Streaming — first chunk: {first_chunk_t:.2f}s | total: {total_t:.2f}s"
            return (SAMPLE_RATE, audio_array), status
        else:
            resp = httpx.post(f"{BACKEND_URL}/tts", data=data, files=ref_files, timeout=120)
            resp.raise_for_status()
            total_t = time.perf_counter() - start
            sr, audio_array = _wav_bytes_to_numpy(resp.content)
            status = f"⏳ Non-streaming — total: {total_t:.2f}s"
            return (sr, audio_array), status
    except httpx.ConnectError:
        return None, "❌ Cannot reach backend. Is start.sh running?"
    except httpx.HTTPStatusError as e:
        return None, f"❌ Backend error {e.response.status_code}: {e.response.text}"
    finally:
        if ref_files:
            for _, (_, fh, _) in ref_files.items():
                fh.close()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

VOICES = _fetch_voices()

MODEL_CHOICES = [
    "Voxtral Mini 4B (fast, real-time)",
    "Voxtral Small 24B (high quality)",
]

LANGUAGE_CHOICES = ["auto", "en", "fr", "de", "es", "nl", "pt", "it", "hi", "ar"]

with gr.Blocks(title="Voxtral Demo") as demo:
    gr.Markdown("# Voxtral Demo\nLocal inference on Apple Silicon via MLX")

    with gr.Tab("🎙 Transcribe"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"], label="Audio input")
                model_dd = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
                lang_dd = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="Language")
                temp_sl = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature")
                diarize_cb = gr.Checkbox(label="Speaker diarization (Small model only)")
                transcribe_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                transcript_out = gr.Textbox(label="Transcript", lines=10, buttons=["copy"])

        transcribe_btn.click(
            run_transcribe,
            inputs=[audio_in, model_dd, lang_dd, temp_sl, diarize_cb],
            outputs=[transcript_out],
        )

    with gr.Tab("🔊 Text to Speech"):
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="Text", lines=5, placeholder="Enter text to synthesize…")
                voice_dd = gr.Dropdown(VOICES, value=VOICES[0] if VOICES else "casual_male", label="Voice")
                speed_sl = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                stream_toggle = gr.Checkbox(label="Streaming mode (compare latency)")
                ref_audio_in = gr.Audio(
                    sources=["upload"],
                    label="Custom voice (optional, 3–10s reference clip)",
                    type="filepath",
                )
                tts_btn = gr.Button("Generate Speech", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Output", type="numpy")
                status_out = gr.Textbox(label="Timing", lines=1, interactive=False)

        tts_btn.click(
            run_tts,
            inputs=[text_in, voice_dd, speed_sl, stream_toggle, ref_audio_in],
            outputs=[audio_out, status_out],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

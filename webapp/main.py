import gc
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import torch
import whisperx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from translator.main import _load_pipeline as _load_translator_pipeline
from whisperx.asr import FasterWhisperPipeline

from webapp.config import MODEL_CONFIGS, auto_select_config

# Load environment variables from .env file in workspace root
load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="WhisperX Web")


@app.exception_handler(Exception)
def _generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Cache loaded models by config key
_model_cache: dict[str, FasterWhisperPipeline] = {}


def _device_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": round(props.total_memory / (1024**3), 2),
        }
    return {"device": "cpu", "gpu_name": None, "vram_gb": 0}


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(config_key: str):
    if config_key in _model_cache:
        return _model_cache[config_key]

    cfg = MODEL_CONFIGS[config_key]
    device = _get_device()
    model = whisperx.load_model(
        cfg["name"], device, compute_type=cfg["compute_type"]
    )
    _model_cache[config_key] = model
    return model


def _translate_segments(segments: list[dict]) -> tuple[list[dict], float]:
    """Translate every segment's text in order using the local translator.

    Processes segments sequentially to preserve timing/order (queue-like).
    """
    if not segments:
        return [], 0.0

    pipe = _load_translator_pipeline()
    translated = []
    t0 = time.time()
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            translated_text = ""
        else:
            result = pipe(text, max_length=512, truncation=True)
            translated_text = result[0]["translation_text"]
        translated.append({**seg, "translated_text": translated_text})
    elapsed = round(time.time() - t0, 2)
    return translated, elapsed


@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/gpu")
def gpu_info():
    return _device_info()


@app.get("/api/models")
def list_models():
    info = _device_info()
    default = auto_select_config(info.get("gpu_name"), info.get("device"))
    return {"default": default, "configs": MODEL_CONFIGS, "gpu": info}


def _download_audio(url: str, suffix: str = ".bin") -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("仅支持 http/https 音频链接")
    ext = Path(parsed.path).suffix or suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        input_path = tmp.name
    try:
        urllib.request.urlretrieve(url, input_path)
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise RuntimeError(f"下载音频失败：{e}") from e
    return input_path


@app.post("/api/transcribe")
def transcribe(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    model_key: str = Form("auto"),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    hf_token: Optional[str] = Form(None),
    translate: bool = Form(True),
):
    start_time = time.time()

    if not file and not audio_url:
        raise HTTPException(status_code=400, detail="请上传音频文件或提供音频链接")

    # Prefer HF token from .env; fallback to form input
    effective_hf_token = os.getenv("HF_TOKEN") or hf_token

    # Save uploaded file or download from URL
    if file:
        ext = Path(file.filename or "upload").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.file.read())
            input_path = tmp.name
    else:
        input_path = _download_audio(audio_url)

    wav_path = ""
    try:
        # Convert to WhisperX-friendly WAV (16kHz mono)
        wav_path = input_path + ".wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Choose model config
        if model_key == "auto":
            model_key = auto_select_config(
                _device_info().get("gpu_name"), _device_info().get("device")
            )
        if model_key not in MODEL_CONFIGS:
            model_key = auto_select_config(
                _device_info().get("gpu_name"), _device_info().get("device")
            )

        cfg = MODEL_CONFIGS[model_key]
        device_str = _get_device()
        model = _load_model(model_key)

        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(
            audio, batch_size=cfg["batch_size"], language=language
        )

        detected_lang = language or result.get("language", "en")

        # Align timestamps (word-level)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device_str
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device_str,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optional speaker diarization
        diarize_segments = []
        if diarize:
            from whisperx.diarize import DiarizationPipeline

            if not cfg.get("supports_diarization", True):
                raise ValueError("当前模型配置不建议开启说话人分离")
            if not effective_hf_token:
                raise ValueError(
                    "hf_token is required for speaker diarization. "
                    "Set HF_TOKEN in .env or pass it via the form."
                )
            try:
                diarize_model = DiarizationPipeline(
                    model_name=cfg.get("diarize_model"),
                    token=effective_hf_token,
                    device=device_str,
                )
            except (GatedRepoError, RepositoryNotFoundError) as e:
                raise RuntimeError(
                    "无法访问 pyannote 说话人分离模型。请确保："
                    "1) 在 Hugging Face 上接受该模型用户协议；"
                    "2) 使用的 Token 有读取权限。"
                ) from e
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        full_text = " ".join(seg["text"].strip() for seg in result["segments"])

        # Translate every segment in original order to avoid timing confusion
        translated_segments = result["segments"]
        translation_time = 0.0
        if translate:
            translated_segments, translation_time = _translate_segments(
                result["segments"]
            )

        elapsed_time = round(time.time() - start_time, 2)

        return {
            "success": True,
            "model_key": model_key,
            "model": cfg["name"],
            "compute_type": cfg["compute_type"],
            "batch_size": cfg["batch_size"],
            "device": device_str,
            "language": detected_lang,
            "diarize_enabled": diarize,
            "diarize_model": cfg.get("diarize_model") if diarize else None,
            "supports_diarization": cfg.get("supports_diarization", True),
            "translate_enabled": translate,
            "translation_time": translation_time,
            "elapsed_time": elapsed_time,
            "text": full_text,
            "segments": translated_segments,
            "diarize_segments": diarize_segments if diarize else None,
        }

    finally:
        for p in (input_path, wav_path):
            if p and os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

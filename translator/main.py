import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import pipeline

from translator.config import MAX_LENGTH, MODEL_NAME

# 加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="Translator EN-ZH")


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

# 延迟加载，避免启动时长时间下载模型
_pipeline = None


def _device_info():
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        }
    return {"device": "cpu", "gpu_name": None, "vram_gb": 0}


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline(
            "translation",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=_get_device(),
            max_length=MAX_LENGTH,
        )
    return _pipeline


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待翻译的英文文本")


@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    info = _device_info()
    return {
        "success": True,
        "model": MODEL_NAME,
        "task": "translation_en_zh",
        **info,
    }


@app.post("/api/translate")
def translate(req: TranslateRequest):
    start_time = time.time()

    pipe = _load_pipeline()
    result = pipe(
        req.text.strip(),
        max_length=MAX_LENGTH,
        truncation=True,
    )
    translated_text = result[0]["translation_text"]
    elapsed_time = round(time.time() - start_time, 2)

    return {
        "success": True,
        "model": MODEL_NAME,
        "device": _device_info()["device"],
        "source_text": req.text,
        "translated_text": translated_text,
        "elapsed_time": elapsed_time,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

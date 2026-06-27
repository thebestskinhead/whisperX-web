"""GPU model recommendations for the WhisperX web app."""

MODEL_CONFIGS = {
    "t4": {
        "name": "large-v2",
        "compute_type": "float16",
        "batch_size": 4,
        "diarize_model": "pyannote/speaker-diarization-community-1",
        "supports_diarization": True,
        "description": "T4 推荐：Tensor Core FP16，支持说话人分离，16GB 显存充裕",
    },
    "p100": {
        "name": "large-v2",
        "compute_type": "float16",
        "batch_size": 2,
        "diarize_model": "pyannote/speaker-diarization-community-1",
        "supports_diarization": True,
        "description": "P100 可用：FP16 较慢但仍支持说话人分离；显存吃紧请改用 medium",
    },
    "medium": {
        "name": "medium",
        "compute_type": "float16",
        "batch_size": 4,
        "diarize_model": "pyannote/speaker-diarization-community-1",
        "supports_diarization": True,
        "description": "平衡速度与精度，T4/P100/CPU 均可，支持说话人分离",
    },
    "base": {
        "name": "base",
        "compute_type": "float16",
        "batch_size": 8,
        "diarize_model": "pyannote/speaker-diarization-community-1",
        "supports_diarization": True,
        "description": "速度最快、精度较低，适合 CPU 或快速预览，支持说话人分离",
    },
    "cpu": {
        "name": "base",
        "compute_type": "int8",
        "batch_size": 1,
        "diarize_model": "pyannote/speaker-diarization-community-1",
        "supports_diarization": False,
        "description": "纯 CPU fallback，int8 量化，不建议开启说话人分离",
    },
}


def auto_select_config(gpu_name: str, device: str) -> str:
    """Pick a default model config based on detected GPU."""
    name = (gpu_name or "").lower()
    if device == "cpu":
        return "cpu"
    if "t4" in name:
        return "t4"
    if "p100" in name:
        return "p100"
    # fallback for other 16GB+ cards
    return "medium"

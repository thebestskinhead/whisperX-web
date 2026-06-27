# WhisperX Web App

基于 WhisperX 的 Web 音频转录服务。支持上传 MP3/WAV/M4A 等音频，输出带时间戳的文本，可选说话人分离，可选逐句英译中。

## 文件结构

```
webapp/
├── config.py      # GPU 模型推荐配置
├── main.py        # FastAPI 后端
├── start.sh       # 启动脚本（自动设置 cuDNN 路径）
├── static/
│   └── index.html # 前端页面
└── README.md      # 本文档
```

## 快速启动

```bash
cd /workspace
uv sync --all-extras --dev
./webapp/start.sh
```

服务启动后访问：http://localhost:8000

## 配置环境变量

项目根目录的 `.env` 文件会被自动加载。如需开启说话人分离，请配置：

```bash
# /workspace/.env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

然后在 Hugging Face 上接受模型使用协议：
https://huggingface.co/pyannote/speaker-diarization-community-1

网页表单中的 HF Token 输入框可作为 `.env` 配置的覆盖项，留空则使用环境变量。

## 模型选择建议

| GPU | 推荐模型 | 精度 | batch_size | 说明 |
|-----|----------|------|------------|------|
| Tesla T4 | large-v2 | float16 | 4 | Tensor Core 加速，精度高 |
| Tesla P100 | large-v2 | float16 | 2 | 可用，但 FP16 无 Tensor Core 加速；可改用 medium 提速 |
| CPU | base | int8 | 1 | 纯 CPU fallback |

网页会根据检测到的 GPU 自动选择默认模型，也可手动切换。

## API 接口

- `GET /` - 前端页面
- `GET /api/gpu` - GPU 信息
- `GET /api/models` - 可用模型配置
- `POST /api/transcribe` - 上传音频并转录
  - `file`: 音频文件
  - `model_key`: 模型配置键（`auto`/`t4`/`p100`/`medium`/`base`/`cpu`）
  - `language`: 语言代码（如 `zh`、`en`，留空自动检测）
  - `translate`: 是否逐句翻译为中文（`true`/`false`，默认 `true`）
  - `diarize`: 是否启用说话人分离（`true`/`false`）
  - `hf_token`: Hugging Face Token（启用 diarize 时必需；优先读取 `.env` 中的 `HF_TOKEN`）
  - 返回字段包含 `elapsed_time`（总耗时）和 `translation_time`（翻译耗时），单位秒

## 命令行测试

```bash
export LD_LIBRARY_PATH="/workspace/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
curl -s http://localhost:8000/api/models
curl -s -F "file=@/path/to/audio.wav" -F "model_key=auto" http://localhost:8000/api/transcribe | python -m json.tool
```

## 注意事项

- 首次运行会从 Hugging Face / PyTorch 下载模型，请保持网络畅通。
- 如果启动时报 cuDNN 错误，`start.sh` 已自动设置 `LD_LIBRARY_PATH`。
- 说话人分离需要 Hugging Face Token 并同意 pyannote 模型用户协议。
- 翻译通过本地导入 `/workspace/translator` 完成，不经过 HTTP，按原文顺序逐句处理。

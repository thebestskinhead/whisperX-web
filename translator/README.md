# 英译中翻译后端

基于 `Helsinki-NLP/opus-mt-en-zh` 的轻量级 FastAPI 翻译服务，仅支持英文到中文翻译。

## 文件结构

```
translator/
├── __init__.py
├── config.py      # 模型配置
├── main.py        # FastAPI 后端
├── start.sh       # 启动脚本
├── static/
│   └── index.html # 简单测试页面
└── README.md      # 本文档
```

## 快速启动

```bash
cd /workspace
uv sync --all-extras --dev
./translator/start.sh
```

服务启动后访问：http://localhost:8001

## API 接口

- `GET /` - 前端测试页面
- `GET /api/health` - 服务与设备信息
- `POST /api/translate` - 翻译文本
  - 请求体：`{"text": "Hello, world."}`
  - 返回示例：
    ```json
    {
      "success": true,
      "model": "Helsinki-NLP/opus-mt-en-zh",
      "device": "cuda",
      "source_text": "Hello, world.",
      "translated_text": "你好，世界。",
      "elapsed_time": 0.12
    }
    ```

## 命令行测试

```bash
curl -s http://localhost:8001/api/health | python -m json.tool

curl -s -X POST http://localhost:8001/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine translation is fast and convenient."}' \
  | python -m json.tool
```

## 注意事项

- 首次运行会从 Hugging Face 下载模型，请保持网络畅通。
- 模型仅在第一次请求时加载，以缩短启动时间。
- 当前仅支持英文输入；输入为空会返回 400 错误。

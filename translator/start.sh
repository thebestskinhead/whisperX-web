#!/usr/bin/env bash
# 启动英译中翻译服务
set -e
cd "$(dirname "$0")/.."

exec uv run python -m translator.main "$@"

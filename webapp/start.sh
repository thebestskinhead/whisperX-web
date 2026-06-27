#!/usr/bin/env bash
# Start the WhisperX web app with proper cuDNN library path
set -e
cd "$(dirname "$0")/.."

CUDNN_LIB="/workspace/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH}"

exec uv run python -m webapp.main "$@"

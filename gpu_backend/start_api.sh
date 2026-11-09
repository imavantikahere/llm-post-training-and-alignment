#!/usr/bin/env bash
# Launch the FastAPI server in the current Python environment.
# On RunPod the boot wrapper activates the persistent /workspace virtualenv.
set -euo pipefail
cd "$(dirname "$0")"

: "${INFERENCE_API_KEY:?INFERENCE_API_KEY must be configured in the Pod environment}"
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llm-post-training}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

exec python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --workers 1

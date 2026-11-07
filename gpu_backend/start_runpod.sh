#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${INFERENCE_API_KEY:?Set INFERENCE_API_KEY in your shell before starting}"
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llm-post-training}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
python - <<'PY2'
import torch
assert torch.cuda.is_available(), "No CUDA GPU available on RunPod Pod"
print("GPU:", torch.cuda.get_device_name(0))
PY2
python - <<'PY2'
from inference_engine import CheckpointEngine
print(CheckpointEngine().check_adapters())
PY2
exec python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --workers 1

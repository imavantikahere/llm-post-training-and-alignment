#!/usr/bin/env bash
# RunPod custom ENTRYPOINT wrapper: normal image services + persistent FastAPI.
# A plain "Start command" may become arguments to /start.sh instead of running.
set -Eeuo pipefail

BACKEND=/workspace/llm-post-training-service/backend
RUNTIME=/workspace/llm-post-training-service/runtime
LOG_DIR=/workspace/llm-post-training-service/logs
VENV="$RUNTIME/venv"
export PROJECT_ROOT=/workspace/llm-post-training
export HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR" "$RUNTIME" "$HF_HOME"

(
  set -Eeuo pipefail
  echo "===== FastAPI boot $(date -u +%FT%TZ) ====="
  if [[ ! -v INFERENCE_API_KEY || -z "$INFERENCE_API_KEY" ]]; then
    echo "ERROR: INFERENCE_API_KEY is not set"
    exit 1
  fi
  for file in "$BACKEND/start_api.sh" "$BACKEND/requirements-api.txt"; do
    [[ -f "$file" ]] || { echo "ERROR: Missing $file"; exit 1; }
  done
  for stage in stage1_non_instruction_adapter stage2_domain_sft_adapter stage3_dpo_adapter; do
    [[ -f "$PROJECT_ROOT/models/$stage/adapter_config.json" ]] || {
      echo "ERROR: Missing adapter $stage"; exit 1;
    }
  done

  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Creating persistent Python virtual environment"
    python -m venv "$VENV"
  fi
  export VIRTUAL_ENV="$VENV"
  export PATH="$VENV/bin:$PATH"
  export PIP_CACHE_DIR="$RUNTIME/pip-cache"

  if ! python - <<'PY'
import importlib.util
mods = ('torch','transformers','peft','unsloth','bitsandbytes','accelerate','safetensors','fastapi','uvicorn')
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise RuntimeError("Missing persistent packages: " + ", ".join(missing))
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("Python environment OK; CUDA GPU:", torch.cuda.get_device_name(0))
PY
  then
    echo "Installing dependencies to persistent network-volume venv"
    python -m pip install -r "$BACKEND/requirements-api.txt"
    python - <<'PY'
import torch
from unsloth import FastLanguageModel
import peft, transformers, fastapi, uvicorn, bitsandbytes
assert torch.cuda.is_available(), "CUDA unavailable"
print("Inference dependencies verified")
PY
  fi

  cd "$BACKEND"
  echo "Starting API with Python: $(command -v python)"
  exec bash "$BACKEND/start_api.sh"
) >> "$LOG_DIR/fastapi-startup.log" 2>&1 &

# Preserve normal FileBrowser/Jupyter/SSH container services.
if [[ ! -f /start.sh ]]; then
  echo "ERROR: /start.sh missing; incompatible image" >&2
  exit 1
fi
exec bash /start.sh

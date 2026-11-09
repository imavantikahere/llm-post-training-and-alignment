#!/usr/bin/env bash
# Boot RunPod's normal services AND the persistent FastAPI inference service.
# Select this script as the Pod's container/Docker start command.
# It must be copied to /workspace/llm-post-training-service/backend/boot_pod.sh
# before configuring RunPod to invoke it.
set -Eeuo pipefail

BACKEND="/workspace/llm-post-training-service/backend"
LOG_DIR="/workspace/llm-post-training-service/logs"
mkdir -p "$LOG_DIR"
export PROJECT_ROOT="/workspace/llm-post-training"
export HF_HOME="/workspace/hf-cache"
export PYTHONUNBUFFERED=1

# Launch API bootstrap in background: do NOT delay the normal Jupyter/SSH services.
(
    set -Eeuo pipefail
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Qwen backend bootstrap"

    if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
        echo "ERROR: INFERENCE_API_KEY is missing. Set it in RunPod Pod environment variables."
        exit 1
    fi
    if [[ ! -d "$BACKEND" ]]; then
        echo "ERROR: Persistent backend folder is missing: $BACKEND"
        exit 1
    fi
    if [[ ! -f "$BACKEND/start_api.sh" ]]; then
        echo "ERROR: Expected $BACKEND/start_api.sh. Preserve the proven startup script."
        exit 1
    fi
    for stage in stage1_non_instruction_adapter stage2_domain_sft_adapter stage3_dpo_adapter; do
        if [[ ! -f "$PROJECT_ROOT/models/$stage/adapter_config.json" ]]; then
            echo "ERROR: Missing LoRA adapter configuration for $stage"
            exit 1
        fi
    done

    cd "$BACKEND"
    # Container-level pip packages may be absent after a fresh Pod is deployed.
    # Only install when missing, using the existing known backend requirements.
    if ! python - <<'PY'
import importlib.util
modules = ["uvicorn", "fastapi", "torch", "transformers", "peft", "unsloth", "bitsandbytes", "accelerate", "safetensors"]
missing = [module for module in modules if importlib.util.find_spec(module) is None]
if missing:
    raise RuntimeError("Missing Python modules: " + ", ".join(missing))
print("Backend dependencies present")
PY
    then
        if [[ ! -f "$BACKEND/requirements-api.txt" ]]; then
            echo "ERROR: Dependencies missing and requirements-api.txt not found in $BACKEND"
            exit 1
        fi
        echo "Installing backend requirements from persistent file..."
        python -m pip install -r "$BACKEND/requirements-api.txt"
    fi

    echo "Launching FastAPI on port 8000..."
    exec bash "$BACKEND/start_api.sh"
) >> "$LOG_DIR/fastapi-startup.log" 2>&1 &

# Keep the image's built-in lifecycle (Jupyter, SSH, Nginx).
# In the verified PyTorch template /start.sh remains running.
if [[ ! -f /start.sh ]]; then
    echo "ERROR: Base image /start.sh not found; check the selected RunPod template." >&2
    exit 1
fi
exec bash /start.sh

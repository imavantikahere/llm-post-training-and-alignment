# RunPod GitHub integration: Dockerfile path gpu_backend/serverless/Dockerfile
# Build context is the Git repository root (verify the build logs).
# Built worker uses its own dependencies, not the Pod venv.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROJECT_ROOT=/runpod-volume/llm-post-training \
    HF_HOME=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_MOUNT=/runpod-volume

WORKDIR /app

COPY gpu_backend/serverless/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Reuse existing, version-controlled inference engine without editing it.
COPY gpu_backend/inference_engine.py /app/inference_engine.py
COPY gpu_backend/serverless/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

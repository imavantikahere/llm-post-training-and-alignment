# GPU VM/container deployment. Match your host CUDA driver; verify on your target GPU.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY inference_engine.py backend.py ./
ENV PROJECT_ROOT=/models
EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

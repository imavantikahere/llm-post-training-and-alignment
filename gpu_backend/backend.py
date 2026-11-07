"""FastAPI service for a GPU VM or persistent GPU container.

Do not use a managed Colab runtime as a general public web-service host.
"""
from __future__ import annotations

import os
import secrets
import uuid
import threading
import time
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from inference_engine import CheckpointEngine, STAGES

app = FastAPI(title="Qwen Post-Training Inference API", docs_url=None, redoc_url=None)
engine = CheckpointEngine()


def authorize(authorization: str | None = Header(default=None)):
    expected = os.environ.get("INFERENCE_API_KEY", "")
    if not expected or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not secrets.compare_digest(authorization[7:], expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


class GenerateRequest(BaseModel):
    note: str = Field(min_length=20, max_length=3500)
    stages: list[str] = Field(default_factory=lambda: ["CPT", "SFT", "DPO"], min_length=1, max_length=4)
    max_new_tokens: int = Field(default=350, ge=64, le=512)


@app.get("/health", dependencies=[Depends(authorize)])
def health():
    return {"status": "ready", "gpu_stage_loaded": engine.stage, "supported_stages": STAGES}


@app.post("/generate", dependencies=[Depends(authorize)])
def generate(body: GenerateRequest):
    try:
        return engine.run(body.note, body.stages, body.max_new_tokens)
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex))
    except ValueError as ex:
        raise HTTPException(status_code=422, detail=str(ex))
    except RuntimeError as ex:
        raise HTTPException(status_code=503, detail=str(ex))


# Async jobs prevent long-running generation from exceeding RunPod HTTP proxy timeout.
# Single-process, in-memory queue: jobs are lost when the Pod restarts.
_jobs = {}
_jobs_lock = threading.Lock()
_job_running = False
JOB_TTL_SECONDS = 3600


def _run_job(job_id: str, body: GenerateRequest):
    global _job_running
    try:
        result = engine.run(body.note, body.stages, body.max_new_tokens)
        with _jobs_lock:
            _jobs[job_id].update(status="completed", result=result)
    except Exception as exc:
        # Do not disclose infrastructure configuration or secrets to API clients.
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:350]}")
    finally:
        with _jobs_lock:
            _job_running = False


@app.post("/jobs", dependencies=[Depends(authorize)], status_code=202)
def submit_job(body: GenerateRequest):
    global _job_running
    if any(stage not in STAGES for stage in body.stages):
        raise HTTPException(status_code=422, detail="Unknown stage")
    with _jobs_lock:
        if _job_running:
            raise HTTPException(status_code=429, detail="GPU busy; retry after current job")
        now = time.time()
        for job_id in list(_jobs):
            if now - _jobs[job_id]["created_at"] > JOB_TTL_SECONDS:
                del _jobs[job_id]
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "running", "created_at": now}
        _job_running = True
    threading.Thread(target=_run_job, args=(job_id, body), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/jobs/{job_id}", dependencies=[Depends(authorize)])
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        if time.time() - job["created_at"] > JOB_TTL_SECONDS:
            del _jobs[job_id]
            raise HTTPException(status_code=404, detail="Job expired")
        return {"job_id": job_id, **{k:v for k,v in job.items() if k != "created_at"}}

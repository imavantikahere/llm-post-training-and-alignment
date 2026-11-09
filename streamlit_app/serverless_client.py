"""RunPod Queue Serverless client. Secrets remain server-side in Streamlit."""
from __future__ import annotations

import re
import time
from typing import Callable

import requests

ROOT = "https://api.runpod.ai/v2"
POLL_SECONDS = 4
WAIT_SECONDS = 780
EXECUTION_TIMEOUT_MS = 600_000
TTL_MS = 1_200_000


def run_serverless_job(
    endpoint_id: str,
    api_key: str,
    note: str,
    stages: list[str],
    max_new_tokens: int,
    on_status: Callable[[str], None] | None = None,
) -> dict:
    """Submit one async job, poll its status, and return the engine's output."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{5,80}", endpoint_id):
        raise ValueError("Set a valid SERVERLESS_ENDPOINT_ID in Streamlit Secrets.")
    if not api_key or len(api_key) < 10:
        raise ValueError("Set SERVERLESS_API_KEY in Streamlit Secrets.")
    url = f"{ROOT}/{endpoint_id}"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "note": note,
            "stages": stages,
            "max_new_tokens": max_new_tokens,
        },
        "policy": {
            "executionTimeout": EXECUTION_TIMEOUT_MS,
            "ttl": TTL_MS,
        },
    }

    response = requests.post(
        url + "/run",
        headers=headers,
        json=payload,
        timeout=(10, 45),
    )
    if not response.ok:
        raise RuntimeError(
            f"Serverless job submission failed (HTTP {response.status_code}). "
            "Check endpoint ID, RunPod API key and endpoint state."
        )
    data = response.json()
    job_id = data.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{5,120}", job_id):
        raise RuntimeError("Serverless did not return a valid job ID.")

    deadline = time.monotonic() + WAIT_SECONDS
    last_message = None
    try:
        while time.monotonic() < deadline:
            state_response = requests.get(
                url + "/status/" + job_id,
                headers=headers,
                timeout=(10, 35),
            )
            state_response.raise_for_status()
            state = state_response.json()
            status = str(state.get("status", "UNKNOWN")).upper()
            if status == "COMPLETED":
                output = state.get("output")
                if not isinstance(output, dict) or not isinstance(output.get("outputs"), dict):
                    raise RuntimeError("Serverless worker returned an invalid inference result.")
                return output
            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                error = str(state.get("error", "Unknown worker error"))
                raise RuntimeError(
                    f"Serverless job {status.lower()}: {error[:300]}"
                )
            message = {
                "IN_QUEUE": "Waiting for an available Serverless GPU worker…",
                "IN_PROGRESS": "GPU worker is running your Qwen checkpoints…",
            }.get(status, "Waiting for Serverless job status…")
            if on_status is not None and message != last_message:
                on_status(message)
                last_message = message
            time.sleep(POLL_SECONDS)
        raise TimeoutError(
            "The Serverless job did not finish within the demo waiting limit. "
            "Check RunPod endpoint Requests and Workers."
        )
    except (TimeoutError, KeyboardInterrupt):
        # Best-effort cancel; not a guarantee that all GPU billing stops immediately.
        try:
            requests.post(url + "/cancel/" + job_id, headers=headers, timeout=(5, 12))
        except requests.RequestException:
            pass
        raise

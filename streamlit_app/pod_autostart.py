"""Normal RunPod Pod auto-resume for Streamlit (NOT RunPod Serverless).

Set in Streamlit Secrets:
RUNPOD_AUTO_START = true
RUNPOD_API_KEY = "<RunPod management API key>"
RUNPOD_POD_ID = "<existing stopped Pod ID>"
GPU_API_KEY = "<backend INFERENCE_API_KEY>"
RUNPOD_STOP_AFTER_REQUEST = true

IMPORTANT: This is a small demonstration controller, not a production billing
guarantee. A Streamlit process may be killed before the finally-stop executes.
Rate limits are per Streamlit server process, not globally durable.
A stopped Pod may be unable to resume if its original GPU slot is unavailable.
"""
from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import date
from urllib.parse import urlparse

import requests
import streamlit as st

API_BASE = "https://api.runpod.io/v2"
BOOT_SECONDS = 480
RETRY_SECONDS = 8
STARTS_PER_DAY = 3  # Soft per-process safety limit, not a billing hard cap.
JOBS_PER_DAY = 10

_GATE = threading.Lock()
_COUNTS_LOCK = threading.Lock()
_COUNTS = {"day": "", "starts": 0, "jobs": 0}


def _setting(name: str, default=""):
    try:
        value = st.secrets.get(name, os.environ.get(name, default))
    except (FileNotFoundError, KeyError):
        value = os.environ.get(name, default)
    return value


def _enabled(name: str, default=False) -> bool:
    value = _setting(name, default)
    return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"}


def _count(kind: str) -> None:
    with _COUNTS_LOCK:
        today = date.today().isoformat()
        if _COUNTS["day"] != today:
            _COUNTS.update(day=today, starts=0, jobs=0)
        upper = STARTS_PER_DAY if kind == "starts" else JOBS_PER_DAY
        if _COUNTS[kind] >= upper:
            raise RuntimeError(
                f"Today's demo {kind} limit reached; please try another day."
            )
        _COUNTS[kind] += 1


def _runpod(method: str, pod_id: str, key: str, action: str = "") -> dict:
    url = f"{API_BASE}/pods/{pod_id}" + (f"/{action}" if action else "")
    response = requests.request(
        method, url, headers={"Authorization": f"Bearer {key}"},
        timeout=(10, 30),
    )
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        raise RuntimeError(
            f"RunPod {action or 'status'} returned HTTP {response.status_code}. "
            "Check Pod availability, account credits, API permissions and Pod ID."
        )
    return data if isinstance(data, dict) else {}


def _ready(api_url: str, inference_key: str) -> bool:
    try:
        response = requests.get(
            api_url + "/health",
            headers={"Authorization": f"Bearer {inference_key}"},
            timeout=(4, 8),
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


@contextmanager
def pod_session(current_url: str, inference_key: str):
    """Serializes live jobs; resumes a stopped Pod and optionally stops it afterward.

    Do not expose auto-resume on an unrestricted, high-traffic public application.
    """
    if not _GATE.acquire(blocking=False):
        raise RuntimeError("A live GPU demonstration is already in progress. Please retry later.")

    started_here = False
    pod_id = ""
    management_key = ""
    try:
        _count("jobs")
        enabled = _enabled("RUNPOD_AUTO_START", False)
        if not enabled:
            # Existing behavior stays unchanged until explicitly configured.
            yield current_url.rstrip("/")
            return

        pod_id = str(_setting("RUNPOD_POD_ID")).strip()
        management_key = str(_setting("RUNPOD_API_KEY")).strip()
        if not re.fullmatch(r"[a-zA-Z0-9_-]{5,64}", pod_id):
            raise RuntimeError("RUNPOD_POD_ID missing or invalid in Streamlit Secrets.")
        if not management_key:
            raise RuntimeError("RUNPOD_API_KEY missing from Streamlit Secrets.")

        # The only allowed inference destination is the configured Pod.
        api_url = f"https://{pod_id}-8000.proxy.runpod.net"
        if _ready(api_url, inference_key):
            yield api_url
            return

        info = _runpod("GET", pod_id, management_key)
        # REST v2 returns desiredStatus for a Pod; handle status as fallback.
        status = str(info.get("desiredStatus") or info.get("status") or "").upper()
        if status not in {"RUNNING", "STARTING"}:
            _count("starts")
            try:
                _runpod("POST", pod_id, management_key, "start")
            except RuntimeError as ex:
                raise RuntimeError(
                    "RunPod could not start the stopped Pod. Its previous GPU may "
                    "no longer be available; attach the network volume to a replacement Pod "
                    "and update RUNPOD_POD_ID. " + str(ex)
                ) from ex
            started_here = True

        notice = st.empty()
        start_time = time.monotonic()
        while time.monotonic() - start_time < BOOT_SECONDS:
            if _ready(api_url, inference_key):
                notice.empty()
                yield api_url
                return
            notice.info("Starting RunPod GPU and loading the inference API. This may take a few minutes.")
            time.sleep(RETRY_SECONDS)
        notice.empty()
        raise RuntimeError(
            "The Pod did not become ready within 8 minutes. "
            "Check RunPod logs, port 8000, startup configuration and GPU capacity."
        )
    finally:
        if started_here and _enabled("RUNPOD_STOP_AFTER_REQUEST", True):
            try:
                _runpod("POST", pod_id, management_key, "stop")
                st.info("The automatically started RunPod Pod has been sent a stop request.")
            except Exception:
                st.error(
                    "IMPORTANT: RunPod automatic stop FAILED. "
                    "Open the RunPod dashboard and stop this Pod manually to avoid ongoing GPU charges."
                )
        _GATE.release()

"""RunPod Queue Serverless worker for the existing Qwen2.5-1.5B engine.

Reuses the untouched gpu_backend/inference_engine.py. The network volume mounted
at /runpod-volume contains models and HF cache from the normal-Pod deployment.
"""
import os
from pathlib import Path

import runpod
from inference_engine import CheckpointEngine, STAGES

VOLUME = Path(os.getenv("RUNPOD_VOLUME_MOUNT", "/runpod-volume"))
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(VOLUME / "llm-post-training")))
HF_HOME = Path(os.getenv("HF_HOME", str(VOLUME / "hf-cache")))
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

engine = CheckpointEngine(root=PROJECT_ROOT)


def handler(job):
    data = job.get("input", {})
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object.")
    note = data.get("note", "")
    stages = data.get("stages", ["SFT"])
    max_new_tokens = data.get("max_new_tokens", 350)

    if not isinstance(note, str) or not (20 <= len(note.strip()) <= 3500):
        raise ValueError("Note must contain 20–3500 characters.")
    if not isinstance(stages, list) or not (1 <= len(stages) <= 4):
        raise ValueError("Select 1 to 4 checkpoints.")
    if any(stage not in STAGES for stage in stages):
        raise ValueError("Unrecognized checkpoint.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer.")
    if not 64 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be 64–512.")

    missing = [
        str(PROJECT_ROOT / "models" / folder / "adapter_config.json")
        for folder in (
            "stage1_non_instruction_adapter",
            "stage2_domain_sft_adapter",
            "stage3_dpo_adapter",
        )
        if not (PROJECT_ROOT / "models" / folder / "adapter_config.json").is_file()
    ]
    if missing:
        raise RuntimeError(
            "LoRA model files unavailable. Attach the original network volume "
            "to this Serverless endpoint and verify the mount at /runpod-volume."
        )
    return engine.run(note.strip(), stages, max_new_tokens)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

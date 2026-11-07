#!/usr/bin/env python3
"""Download private LoRA adapters from 3 HF model repos to persistent RunPod volume.
Expected env: HF_TOKEN, CPT_REPO, SFT_REPO, DPO_REPO.
"""
import os, json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.getenv("PROJECT_ROOT", "/workspace/llm-post-training"))
model_dir = root / "models"
model_dir.mkdir(parents=True, exist_ok=True)
for stage, env_name, folder in [
    ("CPT", "CPT_REPO", "stage1_non_instruction_adapter"),
    ("SFT", "SFT_REPO", "stage2_domain_sft_adapter"),
    ("DPO", "DPO_REPO", "stage3_dpo_adapter"),
]:
    repo = os.environ.get(env_name)
    if not repo:
        raise SystemExit(f"Set {env_name} to your HF model repository ID")
    dest = model_dir / folder
    snapshot_download(
        repo_id=repo, repo_type="model", token=os.environ.get("HF_TOKEN"),
        local_dir=str(dest),
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "tokenizer*", "special_tokens_map.json", "README.md"],
    )
    config = dest / "adapter_config.json"
    weights = (dest / "adapter_model.safetensors")
    if not config.is_file() or not weights.is_file():
        raise SystemExit(f"{stage}: missing adapter_config.json or adapter_model.safetensors at {dest}")
    base = str(json.loads(config.read_text()).get("base_model_name_or_path", "")).lower()
    if "qwen2.5" not in base or "1.5b" not in base:
        raise SystemExit(f"{stage}: unexpected model family in adapter_config.json: {base}")
    print(stage, "ready:", dest)
print("All adapters downloaded successfully.")

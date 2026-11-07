"""GPU inference using saved Qwen2.5-1.5B LoRA adapters. No model training.

Loads *one* checkpoint at a time to fit a single Colab T4 or small GPU.
A stage's adapter is a complete exported PEFT checkpoint relative to its
recorded base; do not stack CPT + SFT + DPO adapters at inference.
"""
from __future__ import annotations

import gc
import json
import os
import threading
import time
from pathlib import Path

STAGES = ("Base Qwen", "CPT", "SFT", "DPO")
BASE_MODEL = "unsloth/Qwen2.5-1.5B-bnb-4bit"
TASK = (
    "Transform this technician note into a structured industrial maintenance report. "
    "Clearly distinguish observed evidence from unverified causes, "
    "identify missing information, and do not invent measurements or confirm faults.\n\n"
    "Technician note: "
)


def make_prompt(note: str) -> str:
    """Exact prompt wrapper from notebook 05; no chat template added."""
    return "### Instruction:\n" + TASK + note.strip() + "\n\n### Response:\n"


class CheckpointEngine:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.getenv("PROJECT_ROOT", "/content/drive/MyDrive/llm-post-training-experimental-study"))
        self.paths = {
            "Base Qwen": BASE_MODEL,
            "CPT": self.root / "models/stage1_non_instruction_adapter",
            "SFT": self.root / "models/stage2_domain_sft_adapter",
            "DPO": self.root / "models/stage3_dpo_adapter",
        }
        self.lock = threading.Lock()
        self.stage = None
        self.model = None
        self.tokenizer = None

    def check_adapters(self):
        """Reject missing, incomplete or wrong-model checkpoint folders."""
        for stage in STAGES[1:]:
            path = Path(self.paths[stage])
            config_path = path / "adapter_config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"{stage}: missing {config_path}")
            if not any((path / x).is_file() for x in ("adapter_model.safetensors", "adapter_model.bin")):
                raise FileNotFoundError(f"{stage}: missing adapter weights in {path}")
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            base = str(cfg.get("base_model_name_or_path", "")).lower()
            if "qwen2.5" not in base or "1.5b" not in base:
                raise ValueError(f"{stage} adapter uses unexpected base: {base!r}")
        return {stage: str(self.paths[stage]) for stage in STAGES}

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.stage = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load(self, stage: str):
        if stage not in STAGES:
            raise ValueError("Unknown checkpoint")
        if self.stage == stage and self.model is not None:
            return
        # Import Unsloth before Transformers/PEFT in new Python processes.
        from unsloth import FastLanguageModel
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable; select a CUDA GPU runtime")
        self.unload()
        path = str(self.paths[stage])
        if stage != "Base Qwen":
            config = Path(path) / "adapter_config.json"
            if not config.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {config}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        FastLanguageModel.for_inference(model)
        model.eval()
        self.model, self.tokenizer, self.stage = model, tokenizer, stage

    def _generate(self, note: str, max_new_tokens: int):
        import torch
        prompt = make_prompt(note)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        input_length = inputs["input_ids"].shape[-1]
        if input_length + max_new_tokens > 1024:
            raise ValueError(f"Note too long: prompt is {input_length} tokens; maximum total context is 1024")
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_token_ids = generated[0][input_length:]
        output = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        token_count = len(new_token_ids)
        # This is a *length-limit flag*, not proof of mid-sentence truncation.
        eos_reached = bool(token_count) and int(new_token_ids[-1]) == self.tokenizer.eos_token_id
        return {
            "response": output,
            "generated_tokens": token_count,
            "reached_token_limit": token_count >= max_new_tokens and not eos_reached,
            "latency_seconds": round(time.perf_counter() - start, 2),
        }

    def run(self, note: str, stages: list[str], max_new_tokens: int = 350):
        if not isinstance(note, str) or not (20 <= len(note.strip()) <= 3500):
            raise ValueError("Note must contain 20–3500 characters")
        if not stages or len(stages) > 4 or any(s not in STAGES for s in stages):
            raise ValueError("Select 1 to 4 valid checkpoints")
        if not 64 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 64 and 512")
        # Non-blocking lock: avoid multiple concurrent model loads on one GPU.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("GPU is busy processing another request. Retry shortly.")
        try:
            outputs = {}
            for stage in dict.fromkeys(stages):
                self.load(stage)
                outputs[stage] = self._generate(note, max_new_tokens)
            return {
                "note": note,
                "prompt_template": "notebook_05_exact",
                "max_new_tokens": max_new_tokens,
                "outputs": outputs,
                "model_family": "Qwen2.5-1.5B",
                "evaluation_type": "live_inference_unreviewed",
            }
        finally:
            self.lock.release()

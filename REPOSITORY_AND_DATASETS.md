# Repository files and Hugging Face dataset publication guide

This guide describes the **actual GitHub tree inspected on 9 November 2026**, not a hypothetical project skeleton.

## Current experiment files (already uploaded; do not duplicate)

| Path | What it is | Action |
|---|---|---|
| `notebooks/01_data_preprocessing.ipynb` | Prepare source and training datasets | Keep |
| `notebooks/02_non_instruction_finetuning.ipynb` | Domain CPT | Keep |
| `notebooks/03_instruction_sft.ipynb` | Instruction SFT | Keep |
| `notebooks/04_dpo_alignment.ipynb` | DPO preference training | Keep |
| `notebooks/05_comparative_evaluation.ipynb` | Four-checkpoint evaluation | Keep |
| `data/raw/industrial_maintenance_raw_300_paragraphs.txt` | Synthetic domain text | Keep if redistribution rights are clear |
| `data/raw/industrial_maintenance_raw.pdf` | Original document | Verify provenance/license before any additional redistribution |
| `data/instruction/industrial_maintenance_instruction_dataset.jsonl` | Source instruction examples | Keep |
| `data/preference/industrial_maintenance_dpo_preference_dataset.jsonl` | Source preference pairs | Keep |
| `data/processed/{continued_pretraining,sft_prepared,dpo_prepared}.jsonl` | Prepared model training inputs | Keep |
| `results/metrics/stage1_non_instruction_metrics.json` | Training log artifact | Keep |
| `streamlit_app/data/checkpoint_outputs.jsonl` | Saved generated test reports | Keep |
| `streamlit_app/data/evidence_reviewed.csv` | AI-assisted evidence review | Keep |
| `streamlit_app/data/error_reviewed.csv` | AI-assisted error review | Keep |
| `streamlit_app/app.py` | Public Streamlit demo | Keep |
| `streamlit_app/pod_autostart.py` | RunPod Pod control | Keep |
| `gpu_backend/backend.py` | FastAPI service | Keep |
| `gpu_backend/inference_engine.py` | Qwen / PEFT inference implementation | Keep |
| `gpu_backend/requirements-api.txt` | Inference dependency specification | Keep |
| `gpu_backend/boot_pod.sh` | Persistent environment/start wrapper | Keep |
| `gpu_backend/start_runpod.sh` | Manual backup startup path | Keep |
| `gpu_backend/download_adapters.py` | Optional adapter restore | Keep, never commit its token |
| `gpu_backend/test_endpoint.py` | Smoke test utility | Keep |
| `gpu_backend/Dockerfile` | Optional alternate image-based deployment | Keep, but document as distinct from the tested Pod setup |
| `gpu_backend/requirements-runpod.txt` and `requirements.txt` | Overlapping dependency specs | Consider consolidation into one canonical file; ensure no scripts silently rely on the old subset |

## Recommended next uploads, if you have these artifacts

1. **`results/metrics/`**: SFT and DPO training summaries (loss curves, optimizer/preference diagnostics), with clear labels about what they do *not* prove.
2. **`results/figures/`**: exported, readable benchmark figures (evidence recall, header presence, error counts and any DPO diagnostic plots) from existing notebooks.
3. **`results/evaluation_05/`**: reproducible evaluation manifest with scenario IDs, config, generation settings and review protocol, avoiding duplicate data if `streamlit_app/data/` already serves the CSVs.
4. **`docs/`**: a short reproducibility/environment note with Python, PyTorch, Transformers, PEFT, Unsloth and CUDA versions actually used, plus a changelog.
5. **`LICENSE`**: select a software license only after confirming which dependencies and bundled data allow redistribution. A code license does **not** automatically cover dataset inputs or the raw PDF.

None of these files should be invented. Publish them only if you have real outputs and rights to distribute them.

## Things that should NOT be uploaded to GitHub

- API keys, `.streamlit/secrets.toml`, `.env`, or Hugging Face read tokens.
- `/workspace/llm-post-training-service/runtime/venv`, pip caches, Qwen base-model cache or model weight files.
- The privately hosted CPT, SFT and DPO adapter binaries; link their Hugging Face **model** repos if you choose to make them available.
- `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, large notebook temporary outputs or infrastructure logs.
- Operational plant records, private equipment logs, non-public manuals or copyright-restricted datasets.

Two already-tracked files, `gpu_backend/__pycache__/backend.cpython-313.pyc` and `gpu_backend/__pycache__/inference_engine.cpython-313.pyc`, should be removed from Git history going forward in a separate cleanup. Adding `.gitignore` prevents *new* cache files but does not untrack these existing blobs.

## Where to put the Hugging Face datasets

Use a **new Hugging Face Dataset repository**, *not* your three existing Hugging Face Model repositories. Recommended suggested repo name (not created):

`imavantika/industrial-maintenance-post-training-synthetic`

Suggested layout:

```text
README.md                                 # Dataset card and structured metadata
cpt/continued_pretraining.jsonl
sft/sft_prepared.jsonl
dpo/dpo_prepared.jsonl
source/industrial_maintenance_raw_300_paragraphs.txt
```

**Mapping from GitHub:**

- GitHub `data/processed/continued_pretraining.jsonl` → HF `cpt/continued_pretraining.jsonl`.
- GitHub `data/processed/sft_prepared.jsonl` → HF `sft/sft_prepared.jsonl`.
- GitHub `data/processed/dpo_prepared.jsonl` → HF `dpo/dpo_prepared.jsonl`.
- GitHub `data/raw/industrial_maintenance_raw_300_paragraphs.txt` → optional HF `source/`.
- Source JSONLs under GitHub `data/instruction/` and `data/preference/` may also be shared under `source/` if you want users to regenerate prepared inputs.

Keep the evaluated outputs and AI-assisted review labels in GitHub `streamlit_app/data/` because the application loads them from there. If you mirror evaluation data to Hugging Face, use a **separate named subset or repository**, expressly identified as *evaluation*, and **do not accidentally mix it into training splits**.

### Dataset card must disclose

- Entirely synthetic industrial maintenance use case; not real-world fault records.
- Counts: **300** CPT paragraphs, **180** SFT examples, **120** DPO pairs.
- Fields and units: show one non-sensitive sample record from each stage and describe JSONL column names.
- Data generation and preprocessing method; provenance and redistribution permission.
- How records were split/grouped, which examples were excluded from evaluation and overlap risks.
- DPO meaning of `chosen` and `rejected`; uncertainty/hallucination caveats.
- Educational scope, prohibited safety-critical use, and AI-assisted rather than expert-validated evaluation.
- Appropriate license for **your original synthetic content**; do not automatically apply it to third-party material.

### How to create the Hugging Face Dataset repo (UI)

1. Sign in to [Hugging Face](https://huggingface.co/).
2. Open **New → Dataset** (or your profile's **New Dataset** action).
3. Choose `industrial-maintenance-post-training-synthetic` and choose **Public** only when you have checked the data's provenance; otherwise start **Private**.
4. Use **Files and versions → Add file** to upload the prepared JSONL files into `cpt/`, `sft/` and `dpo/`.
5. Create/complete the repo's `README.md` Dataset Card before making public.
6. Add the new HF dataset URL to the main GitHub README's Datasets section. Do **not** insert a guessed HF dataset URL until the repository exists.

**Official docs:** https://huggingface.co/docs/hub/en/datasets-adding and https://huggingface.co/docs/hub/en/datasets-cards

## Deployment and billing reminders

- Streamlit invokes RunPod's **normal Pod REST v2 API** to start and stop a known Pod ID.
- Start/stop routes use `POST /v2/pods/{pod_id}/action` with JSON `{"action":"start"}` or `{"action":"stop"}`.
- The GPU type is tied to the existing Pod config; the controller does **not** automatically select arbitrary GPU hardware.
- The Pod's GPU reservation is released on stop; future capacity isn't guaranteed.
- The Network Volume and its persistent Python venv stay available as long as the same volume is retained.
- Auto-stop is best-effort: confirm the Pod actually stops, and use an independent billing safeguard for wide public traffic.

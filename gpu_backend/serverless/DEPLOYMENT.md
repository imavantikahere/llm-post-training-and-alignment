# Serverless deployment guide — isolated from the working Pod deployment

**Purpose:** Deploy the existing Qwen2.5-1.5B Base/CPT/SFT/DPO LoRA inference engine on **RunPod Queue Serverless (Flex)**, without editing the existing `streamlit_app/app.py`, `gpu_backend/backend.py`, normal-Pod startup, or any notebooks. This guide and all Serverless-specific code were added as new files.

## What is and is not guaranteed

- Flex workers scale to zero and are not tied to restarting a specific Pod or host. The queue can allocate an available compatible worker within the endpoint's configured hardware and location.
- **GPU capacity is NOT guaranteed.** A worker can queue, start slowly, fail to load, or time out. Using your current **single network volume pins workers to its data center** and limits failover. Additional volumes in other data centers would need copies of the same model files to improve regional flexibility.
- A successful build is **not proof of working inference**. Only the real `/run` test and live Streamlit report confirm that.
- Auto-scaling to zero prevents continuous *worker* billing but does not prevent bills for startup time, execution time, configured idle timeout, container disk, or the Network Volume.
- **$20 is a desired budget, not an enforced RunPod hard limit.** Maximum workers = 1 limits simultaneous GPU instances, not total queued jobs or total spending. Because the current UI has no visitor password, restrict app access and check usage manually until you have a real durable quota.

## New files

| File | Use |
|---|---|
| `gpu_backend/serverless/Dockerfile.compiler` | Tested GPU image with GCC/G++ needed for Unsloth/Triton kernels |
| `gpu_backend/serverless/Dockerfile` | Original image; lacks the required C compiler |
| `gpu_backend/serverless/requirements.txt` | RunPod SDK and GPU inference dependencies |
| `gpu_backend/serverless/handler.py` | Serverless job handler using unmodified `gpu_backend/inference_engine.py` |
| `streamlit_app/serverless_client.py` | Submit async jobs and poll RunPod Serverless status |
| `streamlit_app/serverless_app.py` | Separate copy of the existing Streamlit UI wired to Serverless |

The old app remains deployed and unchanged. To use Serverless, deploy a **second Streamlit app** from `streamlit_app/serverless_app.py`, rather than changing the existing app's entrypoint.

## Step 1 — Stop the normal Pod (avoid double billing)

In RunPod → Pods, verify the existing Pod is **Stopped**. Do **not** terminate or delete the `qwen-model` Network Volume.

Open RunPod → Storage → Network Volumes and verify the **actual data center** and volume name. Select this same volume later. Do not guess the region; it is fixed at creation.

## Step 2 — Connect RunPod to your GitHub

1. Visit https://console.runpod.io/user/settings → Connections → GitHub → Connect.
2. Authorize access to **only** `imavantikahere/llm-post-training-and-alignment` when possible.
3. Go to https://console.runpod.io/serverless → **New Endpoint** → **Import Git Repository**.
4. Choose repository `imavantikahere/llm-post-training-and-alignment`, branch `main`.
5. Under **Dockerfile Path**, enter exactly `gpu_backend/serverless/Dockerfile.compiler` (the tested, compiler-enabled image).
6. Confirm build context resolves to repository root: Dockerfile uses `COPY gpu_backend/...` and should find the files. If the builder reports missing `COPY` paths, inspect build context settings; do not modify your working app to compensate.
7. Select endpoint type **Queue**, not Load Balancer.

RunPod's GitHub integration builds and publishes the container itself. You do not need Docker installed on your computer. It has a **30-minute Docker-build timeout**, so large PyTorch/Unsloth dependency installs could hit that limit. If so, stop and use a prebuilt container image through a registry rather than repeatedly incurring failed builds.

Official RunPod doc: https://docs.runpod.io/serverless/workers/github-integration

## Step 3 — Endpoint configuration (cost-focused)

- Endpoint name: `industrial-qwen-serverless`.
- Endpoint type: **Queue-based**.
- GPU: begin with **24 GB tier** (L4 / RTX A5000 / RTX 3090 if selectable). This is a proposed baseline for a 1.5B quantized model, not an already-verified memory fit. Enable more than one compatible GPU type where the console permits.
- **Active workers / min workers: 0**.
- **Max workers: 1**.
- **Idle timeout: 5 seconds** (or smallest allowed).
- **Execution timeout: 600 seconds** (10 min).
- **GPU count per worker: 1**.
- **Network Volume:** select your existing `qwen-model` volume. It will mount as **`/runpod-volume`**, not `/workspace`.
- **Container disk:** use default/recommended size for the builder/image; don't increase storage without verifying costs. Build image contains Python libraries but *not* model weights.
- **Environment variables:** none of the old normal-Pod management credentials are needed. The handler uses:
  - `PROJECT_ROOT=/runpod-volume/llm-post-training`
  - `HF_HOME=/runpod-volume/hf-cache`
  - `RUNPOD_VOLUME_MOUNT=/runpod-volume`

They are defaulted in the Dockerfile. No private HF token is required for your local adapters **when the Network Volume is correctly mounted and the Qwen base is cached**.

Your saved Pod layout was:
```text
/workspace/llm-post-training/models/stage1_non_instruction_adapter
/workspace/llm-post-training/models/stage2_domain_sft_adapter
/workspace/llm-post-training/models/stage3_dpo_adapter
/workspace/hf-cache
```
The Serverless worker sees the **same volume files** at:
```text
/runpod-volume/llm-post-training/models/stage1_non_instruction_adapter
/runpod-volume/llm-post-training/models/stage2_domain_sft_adapter
/runpod-volume/llm-post-training/models/stage3_dpo_adapter
/runpod-volume/hf-cache
```

**Important limitation:** The volume restricts workers to its data center, so this setup improves scheduling relative to the fixed stopped-Pod host but cannot offer global GPU failover.

RunPod doc: https://docs.runpod.io/storage/network-volumes

## Step 4 — Deploy and check the Build and Worker logs

Click **Deploy Endpoint**. Watch the endpoint's **Builds** tab until it says Completed (not merely Pending). Inspect logs if it fails.

Don't assume the first worker has successfully loaded Qwen just because the container image built. The first real job may trigger initial loading and reveal version/CUDA incompatibilities.

If build runs beyond the documented GitHub builder limit or fails, stop and inspect the log before attempting again.

## Step 5 — Submit ONE SFT test directly in RunPod

On your Serverless endpoint → **Requests** tab → test JSON, use:

```json
{
  "input": {
    "note": "During routine operation of cooling-water pump P-317, vibration rose from 2.1 mm/s to 5.8 mm/s over 25 minutes. Bearing temperature increased from 58 C to 76 C. Pump was placed on standby at 10:55; root cause remains unknown.",
    "stages": ["SFT"],
    "max_new_tokens": 160
  }
}
```

Run **once**. Look for `COMPLETED` with `output.outputs.SFT.response`; if it fails, inspect the worker logs. Do not run multiple expensive tests in parallel.

If you want to test with Python locally afterward (RunPod API key is stored in environment, not code), use `https://api.runpod.ai/v2/ENDPOINT_ID/run` and poll `https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID`. RunPod's queue API stores async results for 30 minutes after completion.

Official API doc: https://docs.runpod.io/serverless/endpoints/send-requests

## Step 6 — Deploy a *second* Streamlit Cloud app, leaving the old app untouched

1. Visit https://share.streamlit.io → **Create app** → **Deploy from GitHub**.
2. Repository: `imavantikahere/llm-post-training-and-alignment`.
3. Branch: `main`.
4. **Main file path:** `streamlit_app/serverless_app.py`.
5. Pick a **new app URL/name** to keep the normal-Pod app unaffected.
6. In the **new app's Secrets** (never GitHub), set:

```toml
SERVERLESS_ENDPOINT_ID = "YOUR_NEW_SERVERLESS_ENDPOINT_ID"
SERVERLESS_API_KEY = "YOUR_RUNPOD_API_KEY"
```

This app does not use the old `GPU_API_URL`, `RUNPOD_AUTO_START`, or `RUNPOD_POD_ID` settings. There is no demo password field in the current Serverless UI. Do not confuse `SERVERLESS_API_KEY` (RunPod API authorization) with the old `GPU_API_KEY` (FastAPI Bearer key).

7. Open the new Streamlit app, navigate to **Try a new note**, select **SFT only**, and press Generate.
8. Confirm a real report and latency/token metrics. If successful, try all four checkpoints once, with budget awareness.

**Public access warning:** The inference tab has no visitor password field. Anybody able to access the Streamlit app can submit billable GPU work. Max workers = 1 is NOT a spending cap. Restrict access or add durable authentication and rate limits before widely sharing the app.

## Step 7 — Confirm charges and stop testing

RunPod Serverless → endpoint → **Requests**, **Workers**, **Metrics** / **Billing**. Check that workers scale down to zero after idle timeout and review job charges.

As of November 2026, RunPod lists roughly **$0.75/hour** for 24 GB Serverless Flex GPUs, but verify your console's exact rate before deployment. At $0.75/hour, 10 minutes of billable GPU time is about $0.125. The **30 GB standard network volume** is about **$2.10/month** at $0.07/GB/month. Worker start/load time and additional container disk are also billable. The $20 budget is not automatically enforced.

**Do not leave the endpoint with active workers > 0.** Scale to zero and keep an eye on credits. If you want a stricter hard limit, set one through the billing controls if RunPod offers it for your account, or make the endpoint inaccessible/remove its secrets when you finish testing.

## Troubleshooting map

| Symptom | Likely cause / response |
|---|---|
| Build fails at COPY | GitHub build context/path mismatch; verify `gpu_backend/serverless/Dockerfile` and repo-root context |
| Dependency build exceeds 30 min | GitHub image-builder timeout; switch to a prebuilt image deployment |
| Worker says adapters missing | Existing network volume not attached or not mounted at `/runpod-volume` |
| Qwen loads fail | GPU/CUDA/Python/library compatibility or model cache issue; inspect worker logs |
| Job `IN_QUEUE` for a long time | No workers/GPU capacity, constrained volume region, or max-workers limit |
| Job `FAILED` | Inspect worker logs and input-validation error |
| Streamlit missing secrets | Configure the **new** Streamlit app's endpoint ID and API key Secrets, not the old app's |
| Worker keeps running | Set Active Workers=0 and short Idle Timeout; inspect workers and billing |

**Reference pricing:** https://www.runpod.io/pricing and https://docs.runpod.io/serverless/pricing

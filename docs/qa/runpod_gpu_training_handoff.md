# RunPod GPU Training Handoff

## Current Local State

- `BEAM_REQUIRE_GPU=true` is configured in `.env`.
- `BEAM_RVC_DEVICE=cuda:0` and `BEAM_RVC_IS_HALF=true` are configured in `.env`.
- `/ai-convert/health` now exposes `gpu`, `device`, and `requireGpu`.
- Local health check on this machine returns:
  - `gpu: false`
  - `device: cuda:0`
  - `requireGpu: true`
- Local conversion requests fail instead of falling back to CPU:
  - sync: `GPU is required but CUDA is not available`
  - async: job status becomes `failed` with the same error.

## Blocker

RunPod deployment and long training cannot start until SSH access is restored.

Add one of these public keys to the RunPod instance `~/.ssh/authorized_keys`, or provide a working SSH key/password and the RunPod host, port, and user:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPMIpC1NCK1K39DJAk0xExb9lrReN2xt91sOUbPWXUET anonymous@MacBook-Pro-7.local
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJwL9Yj7kSAIOOQybZTz2uAbWqUierIEHzYRU+k52Jzl runpod
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHI587yHK0q7V5IQz/c2hgT//CP4jVXMoXtA0I7cPBAK anonymous@tests-MacBook-Pro.local
```

The likely local private key candidates are:

- `~/.ssh/runpod_key`
- `~/.ssh/runpod_ed25519`
- `~/.ssh/id_ed25519`

## RunPod Tasks

1. Confirm SSH access.

```bash
ssh -i ~/.ssh/runpod_key -p <PORT> <USER>@<HOST>
```

2. Sync or pull the latest `beam-svc-server` changes on RunPod.

3. Confirm GPU availability.

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

4. Configure GPU-required runtime.

```bash
cat > .env <<'EOF'
BEAM_RVC_REPO=/path/to/Retrieval-based-Voice-Conversion-WebUI
BEAM_RVC_PYTHON=/path/to/Retrieval-based-Voice-Conversion-WebUI/.venv/bin/python
BEAM_DEMUCS_COMMAND=demucs
BEAM_FFMPEG_COMMAND=ffmpeg
BEAM_FFPROBE_COMMAND=ffprobe
BEAM_RVC_DEVICE=cuda:0
BEAM_RVC_IS_HALF=true
BEAM_RVC_F0_METHOD=rmvpe
BEAM_REQUIRE_GPU=true
EOF
```

5. Train or retrain `chris_martin` on RunPod GPU for 100-200 epochs.

Expected output paths for Beam registry:

```text
weights/rvc/chris_martin/model.pth
weights/rvc/chris_martin/added.index
```

If no usable index is produced, keep `indexPath: null` and `indexRatio: 0.0` in `model_registry/voices.json`.

6. Start the server on RunPod.

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8081
```

7. Verify health.

```bash
curl -sS http://127.0.0.1:8081/ai-convert/health
```

Required result:

```json
{
  "gpu": true,
  "device": "cuda:0",
  "requireGpu": true
}
```

8. Run 5-second Chris Martin QA.

```bash
BASE_URL=http://127.0.0.1:8081 \
TRIM_START=0 \
TRIM_DURATION=5 \
VOICE_ID=chris_martin \
VOICE_TYPE=singer \
OUT_FILE=/tmp/beam_chris_martin_5s.mp3 \
./scripts/smoke_test.sh /path/to/input.mp3
```

9. Inspect stage timings for the 5-second conversion.

For async conversions, poll the job endpoint and inspect `timings`:

```bash
curl -sS http://127.0.0.1:8081/ai-convert/jobs/<JOB_ID> | python3 -m json.tool
```

Expected timing keys include:

```text
upload_read
cache_lookup
input_write
normalize
trim
duration_probe
demucs
rmvpe
rvc_infer
remix
encode
output_read
cache_save
total
job_total
```

Use the largest stage to choose the optimization path:

- `demucs`: confirm Demucs is running with `-d cuda`, then warm the model after server start.
- `rmvpe` or `rvc_infer`: confirm RVC uses `cuda:0`, half precision, and GPU RMVPE.
- `normalize`, `trim`, `encode`: inspect ffmpeg path and input format; keep preview requests at 5-10 seconds.
- high first request only: treat as cold start and add startup warmup.

10. Run full voice matrix QA.

```bash
python scripts/beam_svc_matrix_qa.py \
  --base-url http://127.0.0.1:8081 \
  --manifest /path/to/qa_manifest.csv \
  --trim-duration 5 \
  --keep-going
```

The matrix QA `results.csv` includes `timings_json` so slow stages can be sorted across all voices.

11. Copy QA artifacts and trained model artifacts back to the expected deployment location.

At minimum preserve:

- `weights/rvc/chris_martin/model.pth`
- `weights/rvc/chris_martin/added.index` if usable
- `model_registry/voices.json`
- matrix QA `results.csv`
- matrix QA `summary.md`
- representative 5-second QA outputs

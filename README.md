# Beam SVC Server

Self-hosted singing voice conversion server for Beam.

## Stack
- FastAPI
- Uvicorn
- Python multipart uploads
- Redis/Celery optional later
- RVC v2 + Demucs + RMVPE planned

## Run
```bash
cd beam-svc-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8081
```

## Endpoints
- `GET /ai-convert/health`
- `GET /ai-convert/voices`
- `POST /ai-convert/voice-conversion`
- `GET /ai-convert/jobs/{job_id}`
- `GET /ai-convert/results/{job_id}`

## iOS preview-oriented fields
`POST /ai-convert/voice-conversion` also supports:
- `trim_start`
- `trim_duration`
- `return_job`

Use these for Beam Music iOS preview conversion so the app can switch voice on a short segment first, instead of converting a full track every time.

## Cache / async behavior
- sync response returns `X-Beam-Cache: HIT|MISS`
- async mode: `return_job=true`
- poll `GET /ai-convert/jobs/{job_id}`
- fetch final audio from `GET /ai-convert/results/{job_id}`

## Smoke test
```bash
BASE_URL=http://127.0.0.1:8081 scripts/smoke_test.sh /path/to/input.mp3
BASE_URL=http://127.0.0.1:8081 RETURN_JOB=true scripts/smoke_test.sh /path/to/input.mp3
```

## Current status
- API scaffold complete
- Voice registry loading complete
- Demucs/RMVPE/RVC/Remix service skeletons added
- Conversion pipeline wiring complete
- first voices wired through `model_registry/voices.json` (`dionn_v1_singing`, `freya_idol`)
- RMVPE service is still a stub and should be replaced later if needed

## Runtime notes
- `ffmpeg`, `ffprobe`, `demucs` must be installed on PATH
- RVC repo path is configured by `BEAM_RVC_REPO`
- current RVC adapter calls `tools/cmd/infer_cli.py`
- `voice.model.modelPath` / `indexPath` are resolved from `model_registry/voices.json`
- GPU server defaults: `BEAM_RVC_DEVICE=cuda:0`, `BEAM_RVC_IS_HALF=true`, `BEAM_RVC_F0_METHOD=rmvpe`
- remote setup guide: `docs/remote-svc-deployment.md`

## Startup log checklist
- uvicorn starts cleanly (`Application startup complete`)
- `GET /ai-convert/health` returns `ok=true`
- `GET /ai-convert/voices` returns `total_count >= 1`
- if `ok=false`, check doctor fields: `rvc_repo_exists`, `infer_cli_exists`, `rvc_python_exists`, `ffmpeg_exists`, `ffprobe_exists`, `demucs_exists`, `at_least_one_model_exists`

## Next implementation
- replace RMVPE stub with real extractor or rely on RVC internal RMVPE path
- verify RVC model path/index path conventions against your local repo
- add async queue for long jobs

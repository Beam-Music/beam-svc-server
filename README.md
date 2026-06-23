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

## Current status
- API scaffold complete
- Voice registry loading complete
- Demucs/RMVPE/RVC/Remix service skeletons added
- Conversion pipeline wiring complete
- `taylor_swift_singer` currently points to a local SZA RVC checkpoint for first end-to-end wiring
- RMVPE service is still a stub and should be replaced later if needed

## Runtime notes
- `ffmpeg`, `ffprobe`, `demucs` must be installed on PATH
- RVC repo path is configured by `BEAM_RVC_REPO`
- current RVC adapter calls `tools/cmd/infer_cli.py`
- `voice.model.modelPath` is resolved from `model_registry/voices.json`
- first working voice is configured with an absolute path to:
  - `/Users/anonymous/desktop/code/beamMusic/beam-voice-conversion/Retrieval-based-Voice-Conversion-WebUI/logs/sza/G_latest.pth`

## Next implementation
- install ffmpeg + demucs runtime
- replace RMVPE stub with real extractor or rely on RVC internal RMVPE path
- verify RVC model path/index path conventions against your local repo
- add async queue for long jobs

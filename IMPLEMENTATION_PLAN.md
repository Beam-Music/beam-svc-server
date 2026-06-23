# Beam SVC Server Implementation Plan

## Phase 1
- keep FastAPI sync endpoint working
- wire real registry metadata
- validate duration with ffprobe
- add temp dir cleanup/logging

## Phase 2
### Demucs service
Implement `app/services/demucs_service.py`
- input: wav/mp3 path
- output: `vocals.wav`, `instrumental.wav`
- command example:
```bash
demucs --two-stems=vocals -o /tmp/out input.wav
```

### RMVPE service
Implement `app/services/rmvpe_service.py`
- input: `vocals.wav`
- output: F0 contour

### RVC service
Implement `app/services/rvc_service.py`
- preload HuBERT/RMVPE/RVC model
- input: vocal stem + F0 + model meta
- output: converted vocal wav

### Remix service
Implement `app/services/remix_service.py`
- mix converted vocal with instrumental
- encode to mp3/wav/m4a via ffmpeg

## Phase 3
- async queue
- progress reporting
- model cache
- result cache
- object storage

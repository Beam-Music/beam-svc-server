# Model Layout

## Goal
Keep model placement predictable for Beam Music iOS voice switching.

## Registry-driven layout
`model_registry/voices.json` is the source of truth.

Example:
```json
{
  "voiceId": "taylor_swift_singer",
  "model": {
    "modelPath": "weights/rvc/taylor_swift_singer/model.pth",
    "indexPath": "weights/rvc/taylor_swift_singer/added.index"
  }
}
```

## Recommended directory
```text
beam-svc-server/
  weights/
    rvc/
      taylor_swift_singer/
        model.pth
        added.index
        meta.json
      pNInz6obpgDQGcFmaJgB/
        model.pth
        added.index
        meta.json
```

## Notes
- current RVC adapter passes:
  - `weight_root = parent(modelPath)`
  - `model_name = file name of modelPath`
- for current CLI compatibility, each voice folder should contain the target `.pth`
- `indexPath` is optional but recommended

## Suggested meta.json
```json
{
  "voiceId": "taylor_swift_singer",
  "engine": "rvc",
  "sampleRate": 48000,
  "f0Method": "rmvpe",
  "indexRatio": 0.75,
  "protect": 0.33,
  "filterRadius": 3,
  "mixRate": 0.9,
  "notes": "Beam iOS singer preset"
}
```

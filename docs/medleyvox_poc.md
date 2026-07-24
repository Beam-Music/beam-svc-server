# MedleyVox multi-singer PoC

This optional path separates the Demucs vocal stem into two anonymous lanes:
`singer_1` and `singer_2`. It supports overlapping assignments only when they
refer to different lanes. The lanes are not a real-world singer identity and
may swap over time; the client must let the user audition and map them.

## Model provenance

- Official code and dataset: `jeonchangbin49/MedleyVox` (the official repository
  states that it does not publish pretrained weights).
- Checkpoint used for this PoC: `Cyru5/MedleyVox`,
  `singing_librispeech_ft_iSRNet`, labelled CC BY 4.0 on its model card.

Download the two required files locally, then upload them to the Modal model
volume:

```bash
bash scripts/fetch_medleyvox_weights.sh /tmp/beam-medleyvox-weights
.venv/bin/modal volume put beam-svc-models \
  /tmp/beam-medleyvox-weights/ medleyvox/
```

The expected resulting path in Modal is:

```text
/opt/beam/weights/medleyvox/checkpoint/singing_librispeech_ft_iSRNet/
  vocals.json
  vocals.pth
```

## PoC acceptance test

Use rights-cleared recordings containing: alternating duet, overlapping duet,
same-pitch unison, and backing vocals. For each, listen for leak-through,
track swaps at chunk boundaries, intelligibility after RVC, and final mix
timing. Do not release it as a guaranteed singer-identity separation feature
until the listening evaluation passes.

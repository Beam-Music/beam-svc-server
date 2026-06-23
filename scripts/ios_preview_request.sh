#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8081}"
AUDIO_FILE="${1:-}"
VOICE_ID="${VOICE_ID:-taylor_swift_singer}"
TRIM_START="${TRIM_START:-0}"
TRIM_DURATION="${TRIM_DURATION:-20}"
OUT_FILE="${OUT_FILE:-/tmp/beam_ios_preview.mp3}"

if [[ -z "$AUDIO_FILE" ]]; then
  echo "usage: VOICE_ID=taylor_swift_singer TRIM_START=30 TRIM_DURATION=20 $0 /path/to/input.mp3"
  exit 1
fi

curl -f -X POST "$BASE_URL/ai-convert/voice-conversion" \
  -F "source_audio=@${AUDIO_FILE}" \
  -F "voiceId=${VOICE_ID}" \
  -F "voiceType=singer" \
  -F "language=en" \
  -F "preserve_melody=true" \
  -F "mix_with_instrumental=true" \
  -F "output_format=mp3" \
  -F "trim_start=${TRIM_START}" \
  -F "trim_duration=${TRIM_DURATION}" \
  --output "$OUT_FILE"

ls -lh "$OUT_FILE"
echo "saved: $OUT_FILE"

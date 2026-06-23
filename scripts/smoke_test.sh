#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8081}"
AUDIO_FILE="${1:-}"
VOICE_ID="${VOICE_ID:-taylor_swift_singer}"
OUT_FILE="${OUT_FILE:-/tmp/beam_svc_smoke.mp3}"

if [[ -z "$AUDIO_FILE" ]]; then
  echo "usage: BASE_URL=http://127.0.0.1:8081 $0 /path/to/input.mp3"
  exit 1
fi

curl -s "$BASE_URL/ai-convert/health" | python3 -m json.tool
curl -s "$BASE_URL/ai-convert/voices" | python3 -m json.tool

curl -f -X POST "$BASE_URL/ai-convert/voice-conversion" \
  -F "source_audio=@${AUDIO_FILE}" \
  -F "voiceId=${VOICE_ID}" \
  -F "voiceType=default" \
  -F "language=en" \
  -F "preserve_melody=true" \
  -F "mix_with_instrumental=true" \
  -F "output_format=mp3" \
  --output "$OUT_FILE"

ls -lh "$OUT_FILE"
echo "saved: $OUT_FILE"

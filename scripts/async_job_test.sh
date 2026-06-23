#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8081}"
AUDIO_FILE="${1:-}"
VOICE_ID="${VOICE_ID:-taylor_swift_singer}"

if [[ -z "$AUDIO_FILE" ]]; then
  echo "usage: $0 /path/to/input.mp3"
  exit 1
fi

JOB_JSON=$(curl -s -X POST "$BASE_URL/ai-convert/voice-conversion" \
  -F "source_audio=@${AUDIO_FILE}" \
  -F "voiceId=${VOICE_ID}" \
  -F "voiceType=singer" \
  -F "language=en" \
  -F "preserve_melody=true" \
  -F "mix_with_instrumental=true" \
  -F "output_format=mp3" \
  -F "trim_start=0" \
  -F "trim_duration=5" \
  -F "return_job=true")

echo "$JOB_JSON" | python3 -m json.tool
JOB_ID=$(echo "$JOB_JSON" | python3 -c 'import sys, json; print(json.load(sys.stdin)["jobId"])')

while true; do
  STATUS=$(curl -s "$BASE_URL/ai-convert/jobs/${JOB_ID}")
  echo "$STATUS" | python3 -m json.tool
  STATE=$(echo "$STATUS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["status"])')
  if [[ "$STATE" == "completed" ]]; then
    curl -f "$BASE_URL/ai-convert/results/${JOB_ID}" --output /tmp/beam_async_result.mp3
    ls -lh /tmp/beam_async_result.mp3
    break
  fi
  if [[ "$STATE" == "failed" ]]; then
    exit 1
  fi
  sleep 2
 done

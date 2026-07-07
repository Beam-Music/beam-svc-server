#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8081}"
AUDIO_FILE="${1:-}"
VOICE_ID="${VOICE_ID:-dionn_v1_singing}"
VOICE_TYPE="${VOICE_TYPE:-singer}"
LANGUAGE="${LANGUAGE:-en}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-mp3}"
TRIM_START="${TRIM_START:-}"
TRIM_DURATION="${TRIM_DURATION:-}"
RETURN_JOB="${RETURN_JOB:-false}"
OUT_FILE="${OUT_FILE:-/tmp/beam_svc_smoke.mp3}"
TMPDIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [[ -z "$AUDIO_FILE" ]]; then
  echo "usage: BASE_URL=http://127.0.0.1:8081 $0 /path/to/input.mp3"
  exit 1
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "missing file: $AUDIO_FILE"
  exit 1
fi

json_get() {
  python3 - "$1" "$2" <<'PY'
import json, sys
path = sys.argv[1]
key = sys.argv[2]
with open(path) as f:
    payload = json.load(f)
value = payload.get(key, "")
if isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)
PY
}

require_true() {
  local value="$1"
  local label="$2"
  if [[ "$value" != "true" ]]; then
    echo "$label failed: $value"
    exit 1
  fi
}

echo "== health =="
health_json="$TMPDIR/health.json"
curl -fsS "$BASE_URL/ai-convert/health" -o "$health_json"
python3 -m json.tool < "$health_json"
require_true "$(json_get "$health_json" ok)" "health.ok"

echo "== voices =="
voices_json="$TMPDIR/voices.json"
curl -fsS "$BASE_URL/ai-convert/voices" -o "$voices_json"
python3 -m json.tool < "$voices_json"
voices_total="$(json_get "$voices_json" total_count)"
if [[ -z "$voices_total" || "$voices_total" -lt 1 ]]; then
  echo "voices.total_count invalid: $voices_total"
  exit 1
fi

req=(
  -sS -X POST "$BASE_URL/ai-convert/voice-conversion"
  -F "source_audio=@${AUDIO_FILE}"
  -F "voiceId=${VOICE_ID}"
  -F "voiceType=${VOICE_TYPE}"
  -F "language=${LANGUAGE}"
  -F "output_format=${OUTPUT_FORMAT}"
)

if [[ -n "$TRIM_START" ]]; then req+=( -F "trim_start=${TRIM_START}" ); fi
if [[ -n "$TRIM_DURATION" ]]; then req+=( -F "trim_duration=${TRIM_DURATION}" ); fi
if [[ "$RETURN_JOB" == "true" ]]; then req+=( -F "return_job=true" ); fi

if [[ "$RETURN_JOB" == "true" ]]; then
  convert_json="$TMPDIR/convert.json"
  convert_headers="$TMPDIR/convert.headers"
  curl -fsS -D "$convert_headers" "${req[@]}" -o "$convert_json"
  python3 -m json.tool < "$convert_json"
  job_id="$(json_get "$convert_json" jobId)"
  if [[ -z "$job_id" ]]; then
    echo "jobId not found in response"
    exit 1
  fi

  echo "== poll job: $job_id =="
  for _ in $(seq 1 60); do
    status_json="$(curl -fsS "$BASE_URL/ai-convert/jobs/$job_id")"
    echo "$status_json" | python3 -m json.tool
    status="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("status",""))' "$status_json")"
    case "$status" in
      completed)
        echo "== download result =="
        curl -fsS "$BASE_URL/ai-convert/results/$job_id" --output "$OUT_FILE"
        ls -lh "$OUT_FILE"
        echo "saved: $OUT_FILE"
        exit 0
        ;;
      failed)
        error="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("error",""))' "$status_json")"
        echo "job failed: $error"
        exit 1
        ;;
      queued|processing)
        sleep 2
        ;;
      *)
        echo "unexpected job status: $status"
        exit 1
        ;;
    esac
  done

  echo "job timeout"
  exit 1
else
  convert_headers="$TMPDIR/convert.headers"
  curl -fsS -D "$convert_headers" "${req[@]}" --output "$OUT_FILE"
  cache_header="$(grep -i '^X-Beam-Cache:' "$convert_headers" | tail -1 | cut -d' ' -f2- | tr -d '\r')"
  if [[ -n "$cache_header" ]]; then
    echo "X-Beam-Cache: $cache_header"
  fi
  ls -lh "$OUT_FILE"
  if [[ ! -s "$OUT_FILE" ]]; then
    echo "output file is empty"
    exit 1
  fi
  echo "saved: $OUT_FILE"
fi

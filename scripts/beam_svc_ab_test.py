#!/usr/bin/env python3
"""
Run Beam SVC A/B conversion presets for selected voices.

Example:
  python3 scripts/beam_svc_ab_test.py \
    --base-url http://127.0.0.1:8081 \
    --audio /path/source.mp3 \
    --voices ariana_grande,freya_idol,the_weeknd,chris_martin \
    --trim-duration 15
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PRESETS: dict[str, list[dict[str, str]]] = {
    "ariana_grande": [
        {"pitch_shift": "2", "protect": "0.20", "index_ratio": "0.65", "mix_rate": "1.35"},
        {"pitch_shift": "3", "protect": "0.20", "index_ratio": "0.75", "mix_rate": "1.45"},
        {"pitch_shift": "4", "protect": "0.18", "index_ratio": "0.80", "mix_rate": "1.45"},
    ],
    "freya_idol": [
        {"pitch_shift": "1", "protect": "0.25", "index_ratio": "0.70", "mix_rate": "1.20"},
        {"pitch_shift": "2", "protect": "0.22", "index_ratio": "0.75", "mix_rate": "1.30"},
        {"pitch_shift": "3", "protect": "0.20", "index_ratio": "0.80", "mix_rate": "1.35"},
    ],
    "taylor_swift_singer": [
        {"pitch_shift": "2", "protect": "0.24", "index_ratio": "0.65", "mix_rate": "1.35"},
        {"pitch_shift": "3", "protect": "0.22", "index_ratio": "0.75", "mix_rate": "1.45"},
        {"pitch_shift": "4", "protect": "0.20", "index_ratio": "0.80", "mix_rate": "1.45"},
    ],
    "the_weeknd": [
        {"pitch_shift": "-5", "protect": "0.28", "index_ratio": "0.65", "mix_rate": "1.15"},
        {"pitch_shift": "-7", "protect": "0.25", "index_ratio": "0.75", "mix_rate": "1.25"},
        {"pitch_shift": "-9", "protect": "0.22", "index_ratio": "0.80", "mix_rate": "1.30"},
    ],
    "chris_martin": [
        {"pitch_shift": "-2", "protect": "0.30", "index_ratio": "0.65", "mix_rate": "1.10"},
        {"pitch_shift": "0", "protect": "0.25", "index_ratio": "0.75", "mix_rate": "1.20"},
        {"pitch_shift": "2", "protect": "0.22", "index_ratio": "0.80", "mix_rate": "1.25"},
    ],
    "drake": [
        {"pitch_shift": "-4", "protect": "0.28", "index_ratio": "0.65", "mix_rate": "1.15"},
        {"pitch_shift": "-5", "protect": "0.25", "index_ratio": "0.75", "mix_rate": "1.20"},
        {"pitch_shift": "-7", "protect": "0.22", "index_ratio": "0.80", "mix_rate": "1.25"},
    ],
}


@dataclass
class ABResult:
    voice_id: str
    voice_name: str
    preset: str
    status: str
    job_id: str
    output_path: str
    pitch_shift: str
    protect: str
    index_ratio: str
    mix_rate: str
    timings_json: str
    error: str
    review: str = ""
    notes: str = ""


def slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part)[:80] or "untitled"


def http_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def multipart_post(url: str, fields: dict[str, str], file_field: str, file_path: Path, timeout: int) -> dict[str, Any]:
    boundary = f"Boundary-{uuid.uuid4().hex}"
    body: list[bytes] = []

    for key, value in fields.items():
        body.append(f"--{boundary}\r\n".encode())
        body.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.append(str(value).encode())
        body.append(b"\r\n")

    content_type = mimetypes.guess_type(file_path.name)[0] or "audio/mpeg"
    body.append(f"--{boundary}\r\n".encode())
    body.append(f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode())
    body.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.append(file_path.read_bytes())
    body.append(b"\r\n")
    body.append(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        url,
        data=b"".join(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, output_path: Path, timeout: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        output_path.write_bytes(response.read())


def selected_voices(voices: list[dict[str, Any]], requested: str) -> list[dict[str, Any]]:
    wanted = {item.strip() for item in requested.split(",") if item.strip()}
    return [voice for voice in voices if not wanted or voice["voiceId"] in wanted]


def run_case(
    base_url: str,
    audio_path: Path,
    voice: dict[str, Any],
    preset_name: str,
    preset: dict[str, str],
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    fields = {
        "voiceId": voice["voiceId"],
        "voiceType": voice.get("voiceType") or "singer",
        "source_gender": args.source_gender,
        "language": "en",
        "preserve_melody": "true",
        "mix_with_instrumental": "true",
        "output_format": "mp3",
        "return_job": "true",
    }
    if not args.auto_pitch:
        fields.update(preset)
    else:
        fields.update({key: value for key, value in preset.items() if key != "pitch_shift"})
    if args.trim_duration > 0:
        fields["trim_start"] = str(args.trim_start)
        fields["trim_duration"] = str(args.trim_duration)

    response = multipart_post(
        f"{base_url}/ai-convert/voice-conversion",
        fields,
        "source_audio",
        audio_path,
        timeout=args.submit_timeout,
    )
    job_id = response["jobId"]
    deadline = time.time() + args.job_timeout
    last_status: dict[str, Any] = {}

    while time.time() < deadline:
        last_status = http_json(f"{base_url}/ai-convert/jobs/{job_id}", timeout=30)
        state = last_status.get("status")
        print(f"  {voice['voiceId']} {preset_name}: {state} {last_status.get('progress')}% {last_status.get('stage') or ''}".rstrip(), flush=True)
        if state == "completed":
            download(f"{base_url}/ai-convert/results/{job_id}", output_path, timeout=args.download_timeout)
            return job_id, json.dumps(last_status.get("timings") or {}, sort_keys=True)
        if state == "failed":
            raise RuntimeError(last_status.get("error") or json.dumps(last_status))
        time.sleep(args.poll_interval)

    raise TimeoutError(f"Timed out waiting for {job_id}: {last_status}")


def write_csv(path: Path, rows: list[ABResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ABResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Beam SVC voice preset A/B tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--voices", default="ariana_grande,freya_idol,taylor_swift_singer,the_weeknd,chris_martin,drake")
    parser.add_argument("--out-dir", default=f"/tmp/beam_svc_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--trim-start", type=float, default=0.0)
    parser.add_argument("--trim-duration", type=float, default=15.0, help="0 means full input.")
    parser.add_argument("--source-gender", choices=["unknown", "female", "male"], default="unknown")
    parser.add_argument("--auto-pitch", action="store_true", help="Do not send pitch_shift; let server pitchPolicy choose it.")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--submit-timeout", type=int, default=120)
    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--job-timeout", type=int, default=1200)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    audio_path = Path(args.audio).expanduser()
    if not audio_path.exists():
        raise SystemExit(f"Missing audio file: {audio_path}")

    out_dir = Path(args.out_dir).expanduser()
    output_dir = out_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    health = http_json(f"{base_url}/ai-convert/health")
    if not health.get("ok"):
        raise SystemExit(f"Health check failed: {health}")

    voices_payload = http_json(f"{base_url}/ai-convert/voices")
    voices = selected_voices(voices_payload.get("voices", []), args.voices)
    requested_ids = {item.strip() for item in args.voices.split(",") if item.strip()}
    selected_ids = {voice["voiceId"] for voice in voices}
    missing_ids = sorted(requested_ids - selected_ids)
    if missing_ids:
        raise SystemExit(f"Unknown voice IDs: {', '.join(missing_ids)}")
    voices_without_presets = sorted(
        voice["voiceId"] for voice in voices if not DEFAULT_PRESETS.get(voice["voiceId"])
    )
    if voices_without_presets:
        raise SystemExit(f"No A/B presets defined for: {', '.join(voices_without_presets)}")
    (out_dir / "voices.json").write_text(json.dumps(voices_payload, indent=2, ensure_ascii=False) + "\n")

    rows: list[ABResult] = []
    for voice in voices:
        presets = DEFAULT_PRESETS.get(voice["voiceId"], [])
        for index, preset in enumerate(presets, start=1):
            preset_name = f"p{index}"
            output_path = output_dir / f"{slug(voice['voiceId'])}_{preset_name}.mp3"
            status = "completed"
            error = ""
            job_id = ""
            timings_json = ""
            try:
                job_id, timings_json = run_case(base_url, audio_path, voice, preset_name, preset, output_path, args)
            except Exception as exc:
                status = "failed"
                error = str(exc).replace("\n", " ")[:1200]
                if not args.keep_going:
                    rows.append(
                        ABResult(
                            voice_id=voice["voiceId"],
                            voice_name=voice.get("name", ""),
                            preset=preset_name,
                            status=status,
                            job_id=job_id,
                            output_path="",
                            pitch_shift="auto" if args.auto_pitch else preset["pitch_shift"],
                            protect=preset["protect"],
                            index_ratio=preset["index_ratio"],
                            mix_rate=preset["mix_rate"],
                            timings_json=timings_json,
                            error=error,
                        )
                    )
                    write_csv(out_dir / "results.csv", rows)
                    raise

            rows.append(
                ABResult(
                    voice_id=voice["voiceId"],
                    voice_name=voice.get("name", ""),
                    preset=preset_name,
                    status=status,
                    job_id=job_id,
                    output_path=str(output_path) if output_path.exists() else "",
                    pitch_shift="auto" if args.auto_pitch else preset["pitch_shift"],
                    protect=preset["protect"],
                    index_ratio=preset["index_ratio"],
                    mix_rate=preset["mix_rate"],
                    timings_json=timings_json,
                    error=error,
                )
            )
            write_csv(out_dir / "results.csv", rows)

    (out_dir / "README.md").write_text(
        "# Beam SVC A/B Results\n\n"
        "Listen to each file under `outputs/`, then fill `review` and `notes` in `results.csv`.\n"
        "Recommended review values: `pass`, `too_quiet`, `pitch_wrong`, `robotic`, `distorted`, `weak_timbre`.\n",
        encoding="utf-8",
    )
    print(f"results: {out_dir / 'results.csv'}")
    print(f"outputs: {output_dir}")
    return 0 if all(row.status == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

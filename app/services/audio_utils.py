from pathlib import Path
import subprocess
import json
from app.settings import settings


def run_command(
    command: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def probe_duration_seconds(audio_path: Path) -> float:
    result = run_command([
        settings.ffprobe_command,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(audio_path),
    ])
    payload = json.loads(result.stdout)
    return float(payload["format"]["duration"])


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 48000) -> Path:
    run_command([
        settings.ffmpeg_command,
        "-y",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", "2",
        str(output_path),
    ])
    return output_path


def encode_audio(input_path: Path, output_path: Path, output_format: str) -> Path:
    codec_args = {
        "mp3": ["-codec:a", "libmp3lame", "-b:a", "192k"],
        "wav": ["-codec:a", "pcm_s16le"],
        "m4a": ["-codec:a", "aac", "-b:a", "192k"],
    }.get(output_format, ["-codec:a", "libmp3lame", "-b:a", "192k"])
    run_command([settings.ffmpeg_command, "-y", "-i", str(input_path), *codec_args, str(output_path)])
    return output_path

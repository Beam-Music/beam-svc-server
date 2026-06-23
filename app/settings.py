from pathlib import Path
import os


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


class Settings:
    project_dir = Path(__file__).resolve().parent.parent
    temp_dir = project_dir / "tmp"
    registry_path = project_dir / "model_registry" / "voices.json"

    rvc_repo = Path(os.getenv(
        "BEAM_RVC_REPO",
        "/Users/anonymous/desktop/code/beamMusic/beam-voice-conversion/Retrieval-based-Voice-Conversion-WebUI",
    ))
    rvc_python = os.getenv(
        "BEAM_RVC_PYTHON",
        "/Users/anonymous/desktop/code/beamMusic/beam-voice-conversion/Retrieval-based-Voice-Conversion-WebUI/.venv/bin/python",
    )
    demucs_command = os.getenv("BEAM_DEMUCS_COMMAND", "demucs")
    ffmpeg_command = os.getenv("BEAM_FFMPEG_COMMAND", "ffmpeg")
    ffprobe_command = os.getenv("BEAM_FFPROBE_COMMAND", "ffprobe")
    rvc_device = os.getenv("BEAM_RVC_DEVICE", "cpu")
    rvc_is_half = os.getenv("BEAM_RVC_IS_HALF", "false").lower() == "true"
    default_f0_method = os.getenv("BEAM_RVC_F0_METHOD", "rmvpe")


settings = Settings()

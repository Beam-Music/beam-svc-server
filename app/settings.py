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
    require_gpu = os.getenv("BEAM_REQUIRE_GPU", "false").lower() == "true"
    rvc_worker_enabled = os.getenv("BEAM_RVC_WORKER_ENABLED", "false").lower() == "true"
    max_conversion_seconds = float(os.getenv("BEAM_MAX_CONVERSION_SECONDS", "0") or "0")
    min_conversion_seconds = float(os.getenv("BEAM_MIN_CONVERSION_SECONDS", "0.1") or "0.1")
    medleyvox_enabled = os.getenv("BEAM_MEDLEYVOX_ENABLED", "false").lower() == "true"
    medleyvox_repo = Path(os.getenv("BEAM_MEDLEYVOX_REPO", "/opt/medleyvox"))
    medleyvox_model_dir = Path(os.getenv("BEAM_MEDLEYVOX_MODEL_DIR", "/opt/beam/weights/medleyvox"))
    medleyvox_experiment = os.getenv("BEAM_MEDLEYVOX_EXPERIMENT", "singing_librispeech_ft_iSRNet")
    medleyvox_python = os.getenv("BEAM_MEDLEYVOX_PYTHON", "python")


settings = Settings()

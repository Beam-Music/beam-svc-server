from pathlib import Path
import os
from app.models.schemas import VoiceInfo
from app.services.audio_utils import run_command
from app.settings import settings


class RVCService:
    def infer(
        self,
        vocals_wav: Path,
        f0_path: Path,
        voice: VoiceInfo,
        workdir: Path,
        pitch_shift: int = 0,
        index_ratio: float | None = None,
        protect: float | None = None,
        filter_radius: int | None = None,
    ) -> Path:
        output_path = workdir / "converted_vocals.wav"

        if voice.model is None:
            raise ValueError(f"Voice model metadata missing: {voice.voiceId}")

        model_path = Path(voice.model.modelPath)
        index_path = Path(voice.model.indexPath) if voice.model.indexPath else None

        resolved_model_path = model_path if model_path.is_absolute() else settings.project_dir / model_path
        resolved_index_path = index_path if (index_path and index_path.is_absolute()) else (settings.project_dir / index_path if index_path else None)

        if not resolved_model_path.exists():
            raise ValueError(f"RVC model not found: {resolved_model_path}")

        command = [
            settings.rvc_python,
            str(settings.rvc_repo / "tools" / "cmd" / "infer_cli.py"),
            "--input_path", str(vocals_wav),
            "--opt_path", str(output_path),
            "--model_name", resolved_model_path.name,
            "--device", settings.rvc_device,
            "--is_half", str(settings.rvc_is_half),
            "--f0method", settings.default_f0_method,
            "--f0up_key", str(pitch_shift),
            "--index_rate", str(index_ratio if index_ratio is not None else voice.model.indexRatio),
            "--protect", str(protect if protect is not None else voice.model.protect),
            "--filter_radius", str(filter_radius if filter_radius is not None else voice.model.filterRadius),
            "--resample_sr", str(voice.model.sampleRate),
            "--rms_mix_rate", str(voice.model.mixRate),
        ]

        if resolved_index_path and resolved_index_path.exists():
            command.extend(["--index_path", str(resolved_index_path)])

        env = os.environ.copy()
        env["weight_root"] = str(resolved_model_path.parent)
        numba_cache_dir = settings.temp_dir / "numba_cache"
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))

        run_command(command, cwd=settings.rvc_repo, env=env)
        return output_path

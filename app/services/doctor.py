from pathlib import Path
import json
from shutil import which
from app.settings import settings


class RuntimeDoctor:
    def _resolve_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else settings.project_dir / path

    def check(self) -> dict:
        registry_exists = Path(settings.registry_path).exists()
        at_least_one_model_exists = False
        at_least_one_index_exists = False
        voice_count = 0
        missing_models: list[str] = []
        missing_indexes: list[str] = []

        if registry_exists:
            payload = json.loads(Path(settings.registry_path).read_text())
            voices = payload.get("voices", [])
            voice_count = len(voices)
            for voice in voices:
                model = voice.get("model", {})
                model_path = self._resolve_path(model.get("modelPath"))
                index_path = self._resolve_path(model.get("indexPath"))

                if model_path is not None and model_path.exists():
                    at_least_one_model_exists = True
                elif model_path is not None:
                    missing_models.append(str(model_path))

                if index_path is not None and index_path.exists():
                    at_least_one_index_exists = True
                elif index_path is not None:
                    missing_indexes.append(str(index_path))

        return {
            "rvc_repo_exists": settings.rvc_repo.exists(),
            "infer_cli_exists": (settings.rvc_repo / "tools" / "cmd" / "infer_cli.py").exists(),
            "rvc_python_exists": Path(settings.rvc_python).exists(),
            "ffmpeg_exists": which(settings.ffmpeg_command) is not None,
            "ffprobe_exists": which(settings.ffprobe_command) is not None,
            "demucs_exists": which(settings.demucs_command) is not None,
            "registry_exists": registry_exists,
            "voice_count": voice_count,
            "at_least_one_model_exists": at_least_one_model_exists,
            "at_least_one_index_exists": at_least_one_index_exists,
            "missing_models": missing_models,
            "missing_indexes": missing_indexes,
        }

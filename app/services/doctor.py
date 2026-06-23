from pathlib import Path
import json
from app.settings import settings


class RuntimeDoctor:
    def check(self) -> dict:
        registry_exists = Path(settings.registry_path).exists()
        at_least_one_model_exists = False

        if registry_exists:
            payload = json.loads(Path(settings.registry_path).read_text())
            for voice in payload.get("voices", []):
                model_path = voice.get("model", {}).get("modelPath")
                if model_path and Path(model_path).exists():
                    at_least_one_model_exists = True
                    break

        return {
            "rvc_repo_exists": settings.rvc_repo.exists(),
            "infer_cli_exists": (settings.rvc_repo / "tools" / "cmd" / "infer_cli.py").exists(),
            "registry_exists": registry_exists,
            "at_least_one_model_exists": at_least_one_model_exists,
        }

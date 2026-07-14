from pathlib import Path
from app.services.audio_utils import run_command
from app.settings import settings


class DemucsService:
    def separate(self, input_wav: Path, workdir: Path) -> tuple[Path, Path]:
        output_root = workdir / "demucs"
        command = [
            settings.demucs_command,
            "--two-stems=vocals",
            "-o", str(output_root),
            str(input_wav),
        ]
        if settings.rvc_device.lower().startswith("cuda"):
            command[1:1] = ["-d", "cuda"]
        run_command(command)

        model_dir = output_root / "htdemucs" / input_wav.stem
        vocals = model_dir / "vocals.wav"
        instrumental = model_dir / "no_vocals.wav"
        return vocals, instrumental

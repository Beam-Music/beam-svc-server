from pathlib import Path
from app.services.audio_utils import run_command
from app.settings import settings


class RemixService:
    def remix(self, converted_vocals: Path, instrumental: Path, output_wav: Path, mix_rate: float = 1.0) -> Path:
        vocal_gain = max(0.0, min(mix_rate, 2.0))
        run_command([
            settings.ffmpeg_command,
            "-y",
            "-i", str(converted_vocals),
            "-i", str(instrumental),
            "-filter_complex", f"[0:a]volume={vocal_gain}[v];[1:a][v]amix=inputs=2:duration=longest:normalize=0",
            str(output_wav),
        ])
        return output_wav

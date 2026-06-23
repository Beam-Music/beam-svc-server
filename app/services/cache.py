from pathlib import Path
import hashlib
from app.settings import settings


class ConversionCache:
    def __init__(self):
        self.cache_dir = settings.project_dir / "tmp" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        audio_bytes: bytes,
        *,
        voice_id: str,
        voice_type: str | None,
        language: str,
        preserve_melody: bool,
        mix_with_instrumental: bool,
        output_format: str,
        trim_start: float | None,
        trim_duration: float | None,
        pitch_shift: int,
        index_ratio: float | None,
        protect: float | None,
        filter_radius: int | None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(audio_bytes)
        digest.update(str({
            "voice_id": voice_id,
            "voice_type": voice_type,
            "language": language,
            "preserve_melody": preserve_melody,
            "mix_with_instrumental": mix_with_instrumental,
            "output_format": output_format,
            "trim_start": trim_start,
            "trim_duration": trim_duration,
            "pitch_shift": pitch_shift,
            "index_ratio": index_ratio,
            "protect": protect,
            "filter_radius": filter_radius,
        }).encode("utf-8"))
        return digest.hexdigest()

    def get_path(self, cache_key: str, output_format: str) -> Path:
        return self.cache_dir / f"{cache_key}.{output_format}"

    def load(self, cache_key: str, output_format: str) -> bytes | None:
        path = self.get_path(cache_key, output_format)
        if path.exists():
            return path.read_bytes()
        return None

    def save(self, cache_key: str, output_format: str, data: bytes) -> Path:
        path = self.get_path(cache_key, output_format)
        path.write_bytes(data)
        return path

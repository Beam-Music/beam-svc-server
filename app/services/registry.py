import json
from pathlib import Path
from app.models.schemas import VoiceInfo, VoiceListResponse


class VoiceRegistry:
    def __init__(self, registry_path: str = "model_registry/voices.json"):
        self.registry_path = Path(registry_path)
        self._voices: list[VoiceInfo] = []
        self.reload()

    def reload(self) -> None:
        if not self.registry_path.exists():
            self._voices = []
            return
        payload = json.loads(self.registry_path.read_text())
        self._voices = [VoiceInfo(**voice) for voice in payload.get("voices", [])]

    def list_voices(self) -> VoiceListResponse:
        return VoiceListResponse(voices=self._voices, total_count=len(self._voices))

    def get_voice(self, voice_id: str) -> VoiceInfo | None:
        return next((voice for voice in self._voices if voice.voiceId == voice_id), None)

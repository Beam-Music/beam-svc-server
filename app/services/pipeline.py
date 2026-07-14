from dataclasses import dataclass
from pathlib import Path
from app.services.audio_utils import convert_to_wav, encode_audio, probe_duration_seconds, run_command
from app.services.demucs_service import DemucsService
from app.settings import settings
from app.services.registry import VoiceRegistry
from app.services.remix_service import RemixService
from app.services.rmvpe_service import RMVPEService
from app.services.rvc_service import RVCService


@dataclass
class ConversionOptions:
    voice_id: str
    voice_type: str | None
    language: str
    preserve_melody: bool
    mix_with_instrumental: bool
    output_format: str
    trim_start: float | None = None
    trim_duration: float | None = None
    pitch_shift: int = 0
    index_ratio: float | None = None
    protect: float | None = None
    filter_radius: int | None = None
    is_async: bool = False


class BeamSVCPipeline:
    def __init__(self):
        self.registry = VoiceRegistry()
        self.demucs = DemucsService()
        self.rmvpe = RMVPEService()
        self.rvc = RVCService()
        self.remix = RemixService()

    def health(self) -> dict:
        has_models = any(
            voice.model is not None and (settings.project_dir / voice.model.modelPath).exists()
            for voice in self.registry.list_voices().voices
        )
        try:
            import torch
            gpu_ok = torch.cuda.is_available()
        except ImportError:
            gpu_ok = False
        return {
            "gpu": gpu_ok,
            "modelsLoaded": has_models,
        }

    def convert(self, input_path: Path, options: ConversionOptions) -> bytes:
        voice = self.registry.get_voice(options.voice_id)
        if voice is None:
            raise ValueError(f"Unknown voiceId: {options.voice_id}")

        workdir = input_path.parent
        normalized_wav = convert_to_wav(input_path, workdir / "normalized.wav")
        if options.trim_start is not None or options.trim_duration is not None:
            normalized_wav = self._trim_audio(
                normalized_wav,
                workdir / "trimmed.wav",
                start=options.trim_start or 0.0,
                duration=options.trim_duration,
            )
        duration = probe_duration_seconds(normalized_wav)
        if not options.is_async and duration > 60:
            raise ValueError("Sync conversion currently supports audio up to 60 seconds")

        vocals_wav, instrumental_wav = self.demucs.separate(normalized_wav, workdir)
        f0_path = self.rmvpe.extract_f0(vocals_wav, workdir)
        converted_vocals = self.rvc.infer(
            vocals_wav=vocals_wav,
            f0_path=f0_path,
            voice=voice,
            workdir=workdir,
            pitch_shift=options.pitch_shift,
            index_ratio=options.index_ratio,
            protect=options.protect,
            filter_radius=options.filter_radius,
        )

        final_wav = workdir / "final_mix.wav"
        if options.mix_with_instrumental:
            mix_rate = voice.model.mixRate if voice.model else 1.0
            final_wav = self.remix.remix(converted_vocals, instrumental_wav, final_wav, mix_rate=mix_rate)
        else:
            final_wav.write_bytes(converted_vocals.read_bytes())

        output_path = workdir / f"output.{options.output_format}"
        encode_audio(final_wav, output_path, options.output_format)
        return output_path.read_bytes()

    def _trim_audio(self, input_wav: Path, output_wav: Path, start: float, duration: float | None) -> Path:
        command = [settings.ffmpeg_command, "-y", "-ss", str(start), "-i", str(input_wav)]
        if duration is not None:
            command.extend(["-t", str(duration)])
        command.append(str(output_wav))
        run_command(command)
        return output_wav

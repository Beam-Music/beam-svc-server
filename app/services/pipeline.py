from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from app.services.audio_utils import convert_to_wav, encode_audio, probe_duration_seconds, run_command
from app.services.demucs_service import DemucsService
from app.settings import settings
from app.services.registry import VoiceRegistry
from app.services.remix_service import RemixService
from app.services.rmvpe_service import RMVPEService
from app.services.rvc_service import RVCService


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


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
    timings: dict[str, float] = field(default_factory=dict)
    stage_callback: Callable[[str, float, dict[str, float]], None] | None = None


class BeamSVCPipeline:
    rvc_chunk_seconds = 4.5

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
        gpu_ok = _cuda_available()
        return {
            "gpu": gpu_ok,
            "device": settings.rvc_device,
            "requireGpu": settings.require_gpu,
            "modelsLoaded": has_models,
        }

    def convert(self, input_path: Path, options: ConversionOptions) -> bytes:
        total_start = perf_counter()
        if settings.require_gpu and not _cuda_available():
            raise RuntimeError("GPU is required but CUDA is not available")

        voice = self.registry.get_voice(options.voice_id)
        if voice is None:
            raise ValueError(f"Unknown voiceId: {options.voice_id}")

        workdir = input_path.parent
        normalized_wav = self._time_stage(
            options,
            "normalize",
            lambda: convert_to_wav(input_path, workdir / "normalized.wav"),
        )
        if options.trim_start is not None or options.trim_duration is not None:
            normalized_wav = self._time_stage(
                options,
                "trim",
                lambda: self._trim_audio(
                    normalized_wav,
                    workdir / "trimmed.wav",
                    start=options.trim_start or 0.0,
                    duration=options.trim_duration,
                ),
            )
        duration = self._time_stage(options, "duration_probe", lambda: probe_duration_seconds(normalized_wav))
        if not options.is_async and duration > 60:
            raise ValueError("Sync conversion currently supports audio up to 60 seconds")

        final_wav = workdir / "final_mix.wav"
        if duration > self.rvc_chunk_seconds:
            final_wav = self._time_stage(
                options,
                "chunked_convert",
                lambda: self._convert_input_chunked(
                    input_wav=normalized_wav,
                    voice=voice,
                    workdir=workdir,
                    output_wav=final_wav,
                    options=options,
                ),
            )
        else:
            vocals_wav, instrumental_wav = self._time_stage(
                options,
                "demucs",
                lambda: self.demucs.separate(normalized_wav, workdir),
            )
            f0_path = self._time_stage(options, "rmvpe", lambda: self.rmvpe.extract_f0(vocals_wav, workdir))
            converted_vocals = self._time_stage(
                options,
                "rvc_infer",
                lambda: self._infer_vocals_chunked(
                    vocals_wav=vocals_wav,
                    f0_path=f0_path,
                    voice=voice,
                    workdir=workdir,
                    options=options,
                ),
            )
            if options.mix_with_instrumental:
                mix_rate = voice.model.mixRate if voice.model else 1.0
                final_wav = self._time_stage(
                    options,
                    "remix",
                    lambda: self.remix.remix(converted_vocals, instrumental_wav, final_wav, mix_rate=mix_rate),
                )
            else:
                self._time_stage(options, "copy_vocals", lambda: final_wav.write_bytes(converted_vocals.read_bytes()))

        output_path = workdir / f"output.{options.output_format}"
        self._time_stage(options, "encode", lambda: encode_audio(final_wav, output_path, options.output_format))
        output = self._time_stage(options, "output_read", output_path.read_bytes)
        self._record_stage(options, "total", perf_counter() - total_start)
        return output

    def _time_stage(self, options: ConversionOptions, stage: str, action):
        start = perf_counter()
        if options.stage_callback is not None:
            options.stage_callback(stage, 0.0, dict(options.timings))
        result = action()
        self._record_stage(options, stage, perf_counter() - start)
        return result

    def _record_stage(self, options: ConversionOptions, stage: str, elapsed: float) -> None:
        rounded = round(elapsed, 3)
        options.timings[stage] = rounded
        if options.stage_callback is not None:
            options.stage_callback(stage, rounded, dict(options.timings))

    def _trim_audio(self, input_wav: Path, output_wav: Path, start: float, duration: float | None) -> Path:
        command = [settings.ffmpeg_command, "-y", "-ss", str(start), "-i", str(input_wav)]
        if duration is not None:
            command.extend(["-t", str(duration)])
        command.append(str(output_wav))
        run_command(command)
        return output_wav

    def _convert_input_chunked(
        self,
        input_wav: Path,
        voice,
        workdir: Path,
        output_wav: Path,
        options: ConversionOptions,
    ) -> Path:
        chunk_root = workdir / "conversion_chunks"
        input_chunk_dir = chunk_root / "input"
        processed_chunk_dir = chunk_root / "processed"
        input_chunks = self._split_audio(input_wav, input_chunk_dir, self.rvc_chunk_seconds)
        processed_chunks: list[Path] = []
        total_chunks = len(input_chunks)

        for index, chunk_path in enumerate(input_chunks, start=1):
            chunk_workdir = processed_chunk_dir / f"chunk_{index:05d}"
            chunk_workdir.mkdir(parents=True, exist_ok=True)
            self._notify_stage(options, f"chunk_{index}_of_{total_chunks}_demucs")
            vocals_wav, instrumental_wav = self.demucs.separate(chunk_path, chunk_workdir)
            self._notify_stage(options, f"chunk_{index}_of_{total_chunks}_rmvpe")
            f0_path = self.rmvpe.extract_f0(vocals_wav, chunk_workdir)
            self._notify_stage(options, f"chunk_{index}_of_{total_chunks}_rvc")
            converted_vocals = self.rvc.infer(
                vocals_wav=vocals_wav,
                f0_path=f0_path,
                voice=voice,
                workdir=chunk_workdir,
                pitch_shift=options.pitch_shift,
                index_ratio=options.index_ratio,
                protect=options.protect,
                filter_radius=options.filter_radius,
            )

            final_chunk = chunk_workdir / "final_chunk.wav"
            if options.mix_with_instrumental:
                self._notify_stage(options, f"chunk_{index}_of_{total_chunks}_remix")
                mix_rate = voice.model.mixRate if voice.model else 1.0
                final_chunk = self.remix.remix(
                    converted_vocals,
                    instrumental_wav,
                    final_chunk,
                    mix_rate=mix_rate,
                )
            else:
                final_chunk.write_bytes(converted_vocals.read_bytes())
            processed_chunks.append(final_chunk)

        return self._concat_audio(processed_chunks, output_wav, chunk_root / "processed_chunks.txt")

    def _notify_stage(self, options: ConversionOptions, stage: str) -> None:
        if options.stage_callback is not None:
            options.stage_callback(stage, 0.0, dict(options.timings))

    def _split_audio(self, input_wav: Path, output_dir: Path, chunk_seconds: float) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_pattern = output_dir / "chunk_%05d.wav"
        run_command([
            settings.ffmpeg_command,
            "-y",
            "-i", str(input_wav),
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1",
            "-ar", "48000",
            "-ac", "2",
            str(chunk_pattern),
        ])
        chunks = sorted(output_dir.glob("chunk_*.wav"))
        if not chunks:
            raise RuntimeError("Audio chunking produced no input chunks")
        return chunks

    def _concat_audio(self, chunks: list[Path], output_path: Path, concat_list: Path) -> Path:
        concat_list.parent.mkdir(parents=True, exist_ok=True)
        concat_list.write_text(
            "".join(f"file '{chunk.as_posix()}'\n" for chunk in chunks),
            encoding="utf-8",
        )
        run_command([
            settings.ffmpeg_command,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-ar", "48000",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(output_path),
        ])
        return output_path

    def _infer_vocals_chunked(
        self,
        vocals_wav: Path,
        f0_path: Path,
        voice,
        workdir: Path,
        options: ConversionOptions,
    ) -> Path:
        duration = probe_duration_seconds(vocals_wav)
        if duration <= self.rvc_chunk_seconds:
            return self.rvc.infer(
                vocals_wav=vocals_wav,
                f0_path=f0_path,
                voice=voice,
                workdir=workdir,
                pitch_shift=options.pitch_shift,
                index_ratio=options.index_ratio,
                protect=options.protect,
                filter_radius=options.filter_radius,
            )

        chunk_root = workdir / "rvc_chunks"
        input_chunk_dir = chunk_root / "input"
        converted_chunk_dir = chunk_root / "converted"
        converted_chunk_dir.mkdir(parents=True, exist_ok=True)

        input_chunks = self._split_audio(vocals_wav, input_chunk_dir, self.rvc_chunk_seconds)
        converted_chunks: list[Path] = []
        total_chunks = len(input_chunks)
        for index, chunk_path in enumerate(input_chunks, start=1):
            self._notify_stage(options, f"rvc_chunk_{index}_of_{total_chunks}")
            chunk_workdir = converted_chunk_dir / f"chunk_{index:05d}"
            chunk_workdir.mkdir(parents=True, exist_ok=True)
            converted = self.rvc.infer(
                vocals_wav=chunk_path,
                f0_path=f0_path,
                voice=voice,
                workdir=chunk_workdir,
                pitch_shift=options.pitch_shift,
                index_ratio=options.index_ratio,
                protect=options.protect,
                filter_radius=options.filter_radius,
            )
            converted_chunks.append(converted)

        return self._concat_audio(
            converted_chunks,
            workdir / "converted_vocals.wav",
            chunk_root / "converted_chunks.txt",
        )

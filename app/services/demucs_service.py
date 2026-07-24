from pathlib import Path
from threading import Lock
from app.services.audio_utils import probe_duration_seconds, run_command
from app.settings import settings


class DemucsService:
    def __init__(self):
        self._model = None
        self._model_lock = Lock()

    def warmup(self) -> None:
        self._load_model()

    def separate(self, input_wav: Path, workdir: Path) -> tuple[Path, Path]:
        if settings.require_gpu and settings.rvc_device.lower().startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("GPU is required for Demucs but CUDA is not available")
            except ImportError as error:
                raise RuntimeError("GPU is required for Demucs but torch is not installed") from error

        if settings.rvc_device.lower().startswith("cuda"):
            return self._separate_in_process(input_wav, workdir)
        return self._separate_subprocess(input_wav, workdir)

    def _load_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model

            import torch
            from demucs.pretrained import get_model

            model = get_model("htdemucs")
            model.to("cuda" if settings.rvc_device.lower().startswith("cuda") else "cpu")
            model.eval()
            self._model = model
            # Touch CUDA once during startup/warmup instead of during the request path.
            if settings.rvc_device.lower().startswith("cuda"):
                torch.cuda.synchronize()
            return self._model

    def _separate_in_process(self, input_wav: Path, workdir: Path) -> tuple[Path, Path]:
        import torch
        import torchaudio as ta
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, convert_audio, save_audio

        model = self._load_model()
        output_root = workdir / "demucs" / "htdemucs" / input_wav.stem
        output_root.mkdir(parents=True, exist_ok=True)
        vocals = output_root / "vocals.wav"
        instrumental = output_root / "no_vocals.wav"

        try:
            wav = AudioFile(input_wav).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
        except Exception:
            wav, sample_rate = ta.load(str(input_wav))
            wav = convert_audio(wav, sample_rate, model.samplerate, model.audio_channels)

        ref = wav.mean(0)
        wav = wav - ref.mean()
        wav = wav / ref.std()
        duration = probe_duration_seconds(input_wav)
        # `htdemucs` is trained on short segments (about 7.8 seconds).  Passing
        # a longer waveform to apply_model with split=False raises a ValueError,
        # which was happening for 30-second multi-vocal renders.  Use the model
        # segment when available so this remains correct if the Demucs model is
        # changed later.
        segment_seconds = getattr(model, "segment", None) or 7.8
        split = duration > float(segment_seconds)
        with torch.no_grad():
            sources = apply_model(
                model,
                wav[None],
                device="cuda" if settings.rvc_device.lower().startswith("cuda") else "cpu",
                shifts=1,
                split=split,
                overlap=0.25,
                progress=False,
                num_workers=0,
            )[0]
        sources = sources * ref.std()
        sources = sources + ref.mean()

        source_list = list(sources)
        vocal_index = model.sources.index("vocals")
        vocal_stem = source_list.pop(vocal_index)
        other_stem = torch.zeros_like(source_list[0])
        for source in source_list:
            other_stem += source

        save_audio(vocal_stem.cpu(), vocals, model.samplerate, clip="rescale", bits_per_sample=16)
        save_audio(other_stem.cpu(), instrumental, model.samplerate, clip="rescale", bits_per_sample=16)
        return vocals, instrumental

    def _separate_subprocess(self, input_wav: Path, workdir: Path) -> tuple[Path, Path]:
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

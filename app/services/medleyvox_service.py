import os
from pathlib import Path

from app.services.audio_utils import convert_to_wav, run_command
from app.settings import settings


class MedleyVoxService:
    """Two-singer separation adapter for the public MedleyVox reproduction."""

    source_track_ids = ("singer_1", "singer_2")

    def available(self) -> bool:
        checkpoint = settings.medleyvox_model_dir / "checkpoint" / settings.medleyvox_experiment
        return (
            settings.medleyvox_enabled
            and settings.medleyvox_repo.is_dir()
            and (checkpoint / "vocals.json").is_file()
            and (checkpoint / "vocals.pth").is_file()
        )

    def separate(self, vocals_wav: Path, workdir: Path) -> dict[str, Path]:
        if not self.available():
            raise RuntimeError("MedleyVox multi-singer separation is not configured")

        input_dir = workdir / "medleyvox" / "input"
        output_dir = workdir / "medleyvox" / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        stereo_input = convert_to_wav(vocals_wav, input_dir / "vocals_stereo.wav", sample_rate=24000)
        mono_input = input_dir / "vocals_mono.wav"
        run_command([
            settings.ffmpeg_command, "-y", "-i", str(stereo_input),
            "-ac", "1", "-ar", "24000", str(mono_input),
        ])
        stereo_input.unlink(missing_ok=True)

        run_command([
            settings.medleyvox_python, "-m", "svs.inference",
            "--target", "vocals",
            "--exp_name", settings.medleyvox_experiment,
            "--model_dir", str(settings.medleyvox_model_dir),
            "--inference_data_dir", str(input_dir),
            "--results_save_dir", str(output_dir),
            "--use_overlapadd", "ola",
            "--use_gpu", "True",
        ], cwd=settings.medleyvox_repo, env={
            **os.environ,
            "PYTHONPATH": str(settings.medleyvox_repo),
        })

        # The checkpoint can override `mix_consistent_out`, which changes the
        # directory suffix (`_ola` vs `_ola_inconsistent`).  Discover the
        # emitted pair instead of assuming one particular suffix.
        lane_1 = next(iter(sorted(output_dir.rglob("vocals_mono_output*_1.wav"))), None)
        if lane_1 is None:
            raise RuntimeError("MedleyVox completed without both singer stems")
        lane_2 = lane_1.with_name(lane_1.name.removesuffix("_1.wav") + "_2.wav")
        if not lane_2.is_file():
            raise RuntimeError("MedleyVox completed without both singer stems")
        return {"singer_1": lane_1, "singer_2": lane_2}

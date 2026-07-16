from pathlib import Path
import os
import json
import subprocess
from threading import Lock
from app.models.schemas import VoiceInfo
from app.services.audio_utils import run_command
from app.settings import settings


class RVCService:
    def __init__(self):
        self._worker: subprocess.Popen | None = None
        self._worker_lock = Lock()
        self._stderr_handle = None

    def warmup(self) -> None:
        if settings.rvc_worker_enabled and settings.rvc_device.lower().startswith("cuda"):
            self._ensure_worker()

    def infer(
        self,
        vocals_wav: Path,
        f0_path: Path,
        voice: VoiceInfo,
        workdir: Path,
        pitch_shift: int = 0,
        index_ratio: float | None = None,
        protect: float | None = None,
        filter_radius: int | None = None,
        mix_rate: float | None = None,
    ) -> Path:
        output_path = workdir / "converted_vocals.wav"

        if voice.model is None:
            raise ValueError(f"Voice model metadata missing: {voice.voiceId}")

        model_path = Path(voice.model.modelPath)
        index_path = Path(voice.model.indexPath) if voice.model.indexPath else None

        resolved_model_path = model_path if model_path.is_absolute() else settings.project_dir / model_path
        resolved_index_path = index_path if (index_path and index_path.is_absolute()) else (settings.project_dir / index_path if index_path else None)

        if not resolved_model_path.exists():
            raise ValueError(f"RVC model not found: {resolved_model_path}")

        if settings.rvc_worker_enabled and settings.rvc_device.lower().startswith("cuda"):
            return self._infer_with_worker(
                vocals_wav=vocals_wav,
                output_path=output_path,
                voice=voice,
                resolved_model_path=resolved_model_path,
                resolved_index_path=resolved_index_path,
                pitch_shift=pitch_shift,
                index_ratio=index_ratio,
                protect=protect,
                filter_radius=filter_radius,
                mix_rate=mix_rate,
            )

        command = [
            settings.rvc_python,
            str(settings.rvc_repo / "tools" / "cmd" / "infer_cli.py"),
            "--input_path", str(vocals_wav),
            "--opt_path", str(output_path),
            "--model_name", resolved_model_path.name,
            "--device", settings.rvc_device,
            "--is_half", str(settings.rvc_is_half),
            "--f0method", settings.default_f0_method,
            "--f0up_key", str(pitch_shift),
            "--index_rate", str(index_ratio if index_ratio is not None else voice.model.indexRatio),
            "--protect", str(protect if protect is not None else voice.model.protect),
            "--filter_radius", str(filter_radius if filter_radius is not None else voice.model.filterRadius),
            "--resample_sr", str(voice.model.sampleRate),
            "--rms_mix_rate", str(mix_rate if mix_rate is not None else voice.model.mixRate),
        ]

        if resolved_index_path and resolved_index_path.exists():
            command.extend(["--index_path", str(resolved_index_path)])

        env = os.environ.copy()
        env["weight_root"] = str(resolved_model_path.parent)
        numba_cache_dir = settings.temp_dir / "numba_cache"
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))

        run_command(command, cwd=settings.rvc_repo, env=env)
        return output_path

    def _ensure_worker(self) -> subprocess.Popen:
        with self._worker_lock:
            if self._worker is not None and self._worker.poll() is None:
                return self._worker

            env = os.environ.copy()
            env["BEAM_RVC_REPO"] = str(settings.rvc_repo)
            env["BEAM_RVC_DEVICE"] = settings.rvc_device
            env["BEAM_RVC_IS_HALF"] = str(settings.rvc_is_half).lower()
            numba_cache_dir = settings.temp_dir / "numba_cache"
            numba_cache_dir.mkdir(parents=True, exist_ok=True)
            env.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
            log_path = settings.temp_dir / "rvc_worker.log"
            self._stderr_handle = log_path.open("a", encoding="utf-8")
            worker_path = settings.project_dir / "scripts" / "rvc_worker.py"
            self._worker = subprocess.Popen(
                [settings.rvc_python, str(worker_path)],
                cwd=settings.rvc_repo,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_handle,
                text=True,
                bufsize=1,
            )
            ready_line = self._worker.stdout.readline() if self._worker.stdout else ""
            if not ready_line:
                raise RuntimeError("RVC worker failed to start")
            ready = json.loads(ready_line)
            if not ready.get("ready"):
                raise RuntimeError(f"RVC worker did not become ready: {ready}")
            return self._worker

    def _infer_with_worker(
        self,
        vocals_wav: Path,
        output_path: Path,
        voice: VoiceInfo,
        resolved_model_path: Path,
        resolved_index_path: Path | None,
        pitch_shift: int,
        index_ratio: float | None,
        protect: float | None,
        filter_radius: int | None,
        mix_rate: float | None,
    ) -> Path:
        worker = self._ensure_worker()
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("RVC worker pipes are not available")

        request = {
            "inputPath": str(vocals_wav),
            "outputPath": str(output_path),
            "modelName": resolved_model_path.name,
            "weightRoot": str(resolved_model_path.parent),
            "indexPath": str(resolved_index_path) if resolved_index_path and resolved_index_path.exists() else "",
            "f0Method": settings.default_f0_method,
            "pitchShift": pitch_shift,
            "indexRate": index_ratio if index_ratio is not None else voice.model.indexRatio,
            "protect": protect if protect is not None else voice.model.protect,
            "filterRadius": filter_radius if filter_radius is not None else voice.model.filterRadius,
            "resampleSr": voice.model.sampleRate,
            "rmsMixRate": mix_rate if mix_rate is not None else voice.model.mixRate,
        }

        with self._worker_lock:
            worker.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            worker.stdin.flush()
            response_line = worker.stdout.readline()
        if not response_line:
            self._worker = None
            raise RuntimeError("RVC worker exited without a response")
        response = json.loads(response_line)
        if not response.get("ok"):
            detail = response.get("error") or str(response)
            traceback_text = response.get("traceback")
            if traceback_text:
                detail = f"{detail}\n{traceback_text}"
            raise RuntimeError(detail)
        return output_path

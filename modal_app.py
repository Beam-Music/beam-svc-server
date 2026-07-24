import json
import os
from pathlib import Path

import modal


APP_NAME = "beam-svc-t4-demo"
ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/opt/beam"
MODEL_VOLUME_NAME = "beam-svc-models"
MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_DEMO_SECONDS = 15.0
RVC_COMMIT = "3b4a546cede4a1ea9f70e5fbd235a0f2bb83626c"
HUBERT_SHA256 = "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96"
RMVPE_SHA256 = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"

ALLOWED_TYPES = {
    "application/octet-stream",
    "audio/aac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-m4a",
    "audio/x-wav",
}
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
}

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
read_only_models = models_volume.with_mount_options(read_only=True)


def _add_beam_source(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir(ROOT / "app", f"{REMOTE_ROOT}/app", copy=True)
        .add_local_file(
            ROOT / "model_registry" / "voices.json",
            f"{REMOTE_ROOT}/model_registry/voices.json",
            copy=True,
        )
        .add_local_file(ROOT / "modal_app.py", f"{REMOTE_ROOT}/modal_app.py", copy=True)
        .workdir(REMOTE_ROOT)
    )


gateway_image = _add_beam_source(
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi==0.116.1",
        "uvicorn[standard]==0.35.0",
        "python-multipart==0.0.20",
        "pydantic==2.11.7",
    )
    .env({"PYTHONPATH": REMOTE_ROOT})
)

gpu_environment = {
    "PYTHONPATH": f"{REMOTE_ROOT}:/opt/rvc:/opt/medleyvox",
    "BEAM_RVC_REPO": "/opt/rvc",
    "BEAM_RVC_PYTHON": "/usr/local/bin/python",
    "BEAM_DEMUCS_COMMAND": "demucs",
    "BEAM_FFMPEG_COMMAND": "ffmpeg",
    "BEAM_FFPROBE_COMMAND": "ffprobe",
    "BEAM_RVC_DEVICE": "cuda:0",
    "BEAM_RVC_IS_HALF": "true",
    "BEAM_RVC_F0_METHOD": "rmvpe",
    "BEAM_REQUIRE_GPU": "true",
    "BEAM_RVC_WORKER_ENABLED": "true",
    "BEAM_MEDLEYVOX_ENABLED": "true",
    "BEAM_MEDLEYVOX_REPO": "/opt/medleyvox",
    "BEAM_MEDLEYVOX_MODEL_DIR": f"{REMOTE_ROOT}/weights/medleyvox",
    "BEAM_MEDLEYVOX_PYTHON": "/usr/local/bin/python",
    "NUMBA_CACHE_DIR": f"{REMOTE_ROOT}/tmp/numba_cache",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}

gpu_image = _add_beam_source(
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libsndfile1",
    )
    .pip_install(
        "torch==2.5.1+cu121",
        "torchaudio==2.5.1+cu121",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install_from_requirements(str(ROOT / "requirements.modal-gpu.txt"))
    # Asteroid declares torch<2.0, but MedleyVox inference only uses its
    # model/filterbank/overlap-add modules. Keep the server's CUDA 12.1
    # PyTorch 2.5 stack intact instead of letting pip downgrade it.
    .pip_install(
        "asteroid==0.6.1",
        "asteroid-filterbanks==0.4.0",
        extra_options="--no-deps",
    )
    # Install the remaining MedleyVox inference dependencies normally.  Only
    # Asteroid needs --no-deps because of its obsolete torch upper bound.
    .pip_install(
        "future==1.0.0",
        "huggingface-hub==0.26.5",
        "matplotlib==3.9.4",
        "pillow==11.1.0",
        "praat-parselmouth==0.4.5",
        "pyloudnorm==0.1.1",
        "webrtcvad==2.0.10",
    )
    .run_commands(
        "git clone https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI.git /opt/rvc",
        f"cd /opt/rvc && git checkout {RVC_COMMIT}",
        "git clone --depth 1 https://github.com/jeonchangbin49/MedleyVox.git /opt/medleyvox",
        # The public MedleyVox fork exports four stale class names which no
        # longer exist in base_models.py.  They are unused by this checkpoint;
        # remove only those imports before the build-time smoke check.
        "sed -i '/BaseEncoderMaskerDecoder_output_no_maksed/d; /BaseEncoderMaskerDecoder_output_no_maksed_tf/d; /BaseEncoderMaskerDecoder_output_no_maksed_residual/d; /BaseEncoderMaskerDecoder_output_source2_residual/d; /from .discriminator import (/,/)/d' /opt/medleyvox/svs/models/__init__.py",
        "PYTHONPATH=/opt/medleyvox python -c 'import svs.inference'",
        "mkdir -p /opt/rvc/assets/hubert /opt/rvc/assets/rmvpe",
        "curl -L --fail --retry 3 https://huggingface.co/fumiama/RVC-Pretrained-Models/resolve/main/hubert/hubert_base.pt -o /opt/rvc/assets/hubert/hubert_base.pt",
        f"echo '{HUBERT_SHA256}  /opt/rvc/assets/hubert/hubert_base.pt' | sha256sum -c -",
        "curl -L --fail --retry 3 https://huggingface.co/fumiama/RVC-Pretrained-Models/resolve/main/rmvpe/rmvpe.pt -o /opt/rvc/assets/rmvpe/rmvpe.pt",
        f"echo '{RMVPE_SHA256}  /opt/rvc/assets/rmvpe/rmvpe.pt' | sha256sum -c -",
        "python -c \"from demucs.pretrained import get_model; get_model('htdemucs')\"",
    )
    .add_local_file(
        ROOT / "scripts" / "rvc_worker.py",
        f"{REMOTE_ROOT}/scripts/rvc_worker.py",
        copy=True,
    )
    .env(gpu_environment)
)


def _normalized_gender(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"female", "woman", "f"}:
        return "female"
    if normalized in {"male", "man", "m"}:
        return "male"
    return "unknown"


def _effective_pitch_shift(voice, requested: int | None, source_gender: str | None) -> int:
    default = voice.model.defaultPitchShift if voice.model else 0
    if requested is not None and requested != 0:
        return requested

    policy = voice.pitchPolicy or {}
    gender = _normalized_gender(source_gender)
    if gender in policy:
        return policy[gender]
    if "unknown" in policy:
        return policy["unknown"]
    if requested is None:
        return default
    if requested == 0 and default != 0:
        return default
    return requested


@app.cls(
    image=gpu_image,
    gpu="T4",
    cpu=4,
    memory=16384,
    volumes={f"{REMOTE_ROOT}/weights": read_only_models},
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=300,
    timeout=900,
    startup_timeout=900,
    include_source=False,
)
@modal.concurrent(max_inputs=1)
class BeamConverter:
    @modal.enter()
    def startup(self) -> None:
        os.chdir(REMOTE_ROOT)
        from app.api.convert import pipeline
        from app.api.multi_vocal import pipeline as multi_vocal_pipeline

        Path(f"{REMOTE_ROOT}/tmp/numba_cache").mkdir(parents=True, exist_ok=True)
        pipeline.rvc_chunk_seconds = 6.0
        pipeline.demucs.warmup()
        pipeline.rvc.warmup()
        # Reuse the already-warmed GPU services for multi-vocal work.
        multi_vocal_pipeline.single = pipeline

    @modal.method()
    def convert(self, payload: bytes, filename: str, options: dict) -> dict:
        import tempfile
        import uuid

        os.chdir(REMOTE_ROOT)
        from app.api.convert import pipeline
        from app.services.pipeline import ConversionOptions
        from app.services.registry import VoiceRegistry

        registry = VoiceRegistry()
        voice = registry.get_voice(options["voice_id"])
        if voice is None:
            raise ValueError(f"Unknown voiceId: {options['voice_id']}")

        pitch_shift = _effective_pitch_shift(
            voice,
            options.get("pitch_shift"),
            options.get("source_gender"),
        )
        with tempfile.TemporaryDirectory(prefix="beam_modal_") as workdir:
            suffix = Path(filename).suffix or ".bin"
            input_path = Path(workdir) / f"{uuid.uuid4().hex}{suffix}"
            input_path.write_bytes(payload)
            pipeline_options = ConversionOptions(
                voice_id=options["voice_id"],
                voice_type=options.get("voice_type"),
                language=options.get("language", "en"),
                preserve_melody=options.get("preserve_melody", True),
                mix_with_instrumental=options.get("mix_with_instrumental", True),
                output_format=options["output_format"],
                trim_start=options["trim_start"],
                trim_duration=options["trim_duration"],
                pitch_shift=pitch_shift,
                index_ratio=options.get("index_ratio"),
                protect=options.get("protect"),
                filter_radius=options.get("filter_radius"),
                mix_rate=options.get("mix_rate"),
                is_async=True,
            )
            audio = pipeline.convert(input_path, pipeline_options)

        return {
            "audio": audio,
            "output_format": options["output_format"],
            "media_type": MEDIA_TYPES[options["output_format"]],
            "timings": pipeline_options.timings,
            "voice_id": options["voice_id"],
            "trim_duration": options["trim_duration"],
        }

    @modal.method()
    def analyze_vocals(self, payload: bytes, filename: str) -> dict:
        import tempfile
        import uuid

        os.chdir(REMOTE_ROOT)
        from app.api.multi_vocal import pipeline

        with tempfile.TemporaryDirectory(prefix="beam_modal_analysis_") as workdir:
            suffix = Path(filename).suffix or ".bin"
            input_path = Path(workdir) / f"{uuid.uuid4().hex}{suffix}"
            input_path.write_bytes(payload)
            duration = pipeline.analyze(input_path, Path(workdir))

        return {
            "analysisVersion": "manual-sequential-v1",
            "durationSeconds": round(duration, 3),
            "candidates": [{
                "trackId": "vocal_candidate_1",
                "label": "Separated vocal stem",
                "requiresManualSegmentation": True,
                "supportsOverlaps": False,
            }],
            "limitations": [
                "This version does not identify singers automatically.",
                "Create non-overlapping time ranges for sequential duet sections.",
                "Simultaneous vocals and backing-vocal separation are not supported.",
            ],
        }

    @modal.method()
    def convert_multi(self, payload: bytes, filename: str, raw_assignments: list[dict], options: dict) -> dict:
        import tempfile
        import uuid

        os.chdir(REMOTE_ROOT)
        from app.api.convert import pipeline as single_pipeline
        from app.api.multi_vocal import pipeline as multi_vocal_pipeline
        from app.models.schemas import VocalAssignment
        from app.services.multi_vocal_pipeline import MultiConversionOptions

        assignments = [VocalAssignment.model_validate(item) for item in raw_assignments]
        # Avoid loading a second Demucs/RVC model on the one available GPU.
        multi_vocal_pipeline.single = single_pipeline
        with tempfile.TemporaryDirectory(prefix="beam_modal_multi_") as workdir:
            suffix = Path(filename).suffix or ".bin"
            input_path = Path(workdir) / f"{uuid.uuid4().hex}{suffix}"
            input_path.write_bytes(payload)
            audio = multi_vocal_pipeline.convert(
                input_path,
                assignments,
                MultiConversionOptions(
                    output_format=options["output_format"],
                    mix_with_instrumental=options["mix_with_instrumental"],
                    preserve_unassigned_vocals=options["preserve_unassigned_vocals"],
                ),
            )

        return {
            "audio": audio,
            "output_format": options["output_format"],
            "media_type": MEDIA_TYPES[options["output_format"]],
        }


converter = BeamConverter()


@app.function(
    image=gateway_image,
    cpu=1,
    memory=1024,
    volumes={f"{REMOTE_ROOT}/weights": read_only_models},
    min_containers=0,
    max_containers=2,
    buffer_containers=0,
    scaledown_window=60,
    timeout=300,
    include_source=False,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def gateway():
    os.chdir(REMOTE_ROOT)

    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response

    from app.services.registry import VoiceRegistry

    api = FastAPI(title="Beam SVC Modal Gateway", version="0.2.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    registry = VoiceRegistry()

    async def read_call(job_id: str):
        try:
            call = modal.FunctionCall.from_id(job_id)
            return await call.get.aio(timeout=0), "completed"
        except (TimeoutError, modal.exception.TimeoutError):
            return None, "processing"
        except (modal.exception.NotFoundError, modal.exception.OutputExpiredError) as error:
            raise HTTPException(
                status_code=404,
                detail={"error": "job_not_found", "message": str(error)},
            ) from error
        except (
            modal.exception.FunctionTimeoutError,
            modal.exception.RemoteError,
            modal.exception.ExecutionError,
        ) as error:
            return str(error), "failed"

    @api.get("/ai-convert/health")
    async def health():
        voices = registry.list_voices().voices
        models_loaded = all(
            voice.model is not None
            and (Path(REMOTE_ROOT) / voice.model.modelPath).exists()
            for voice in voices
        )
        return {
            "ok": models_loaded,
            "service": "beam-svc-modal",
            "version": "0.2.0",
            "gpu": True,
            "device": "cuda:0",
            "requireGpu": True,
            "modelsLoaded": models_loaded,
            "maxDemoSeconds": MAX_DEMO_SECONDS,
        }

    @api.get("/ai-convert/voices")
    async def voices():
        return registry.list_voices().model_dump()

    @api.post("/ai-convert/voice-conversion")
    async def voice_conversion(
        request: Request,
        source_audio: UploadFile = File(...),
        voiceId: str = Form(...),
        voiceType: str | None = Form(None),
        source_gender: str | None = Form(None),
        language: str = Form("en"),
        preserve_melody: bool = Form(True),
        mix_with_instrumental: bool = Form(True),
        output_format: str = Form("mp3"),
        trim_start: float | None = Form(None),
        trim_duration: float | None = Form(None),
        pitch_shift: int | None = Form(None),
        index_ratio: float | None = Form(None),
        protect: float | None = Form(None),
        filter_radius: int | None = Form(None),
        mix_rate: float | None = Form(None),
        return_job: bool = Form(False),
    ):
        del request
        if registry.get_voice(voiceId) is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "voice_not_found", "message": f"Unknown voiceId: {voiceId}"},
            )
        if output_format not in MEDIA_TYPES:
            raise HTTPException(
                status_code=400,
                detail={"error": "unsupported_output_format", "message": output_format},
            )
        if source_audio.content_type and source_audio.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unsupported_media_type",
                    "message": f"Unsupported content type: {source_audio.content_type}",
                },
            )

        payload = await source_audio.read(MAX_FILE_SIZE + 1)
        if not payload:
            raise HTTPException(
                status_code=400,
                detail={"error": "empty_audio", "message": "source_audio is empty"},
            )
        if len(payload) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={"error": "file_too_large", "message": "Max 25MB"},
            )

        effective_duration = min(
            max(float(trim_duration or MAX_DEMO_SECONDS), 0.1),
            MAX_DEMO_SECONDS,
        )
        options = {
            "voice_id": voiceId,
            "voice_type": voiceType,
            "source_gender": source_gender,
            "language": language,
            "preserve_melody": preserve_melody,
            "mix_with_instrumental": mix_with_instrumental,
            "output_format": output_format,
            "trim_start": max(float(trim_start or 0.0), 0.0),
            "trim_duration": effective_duration,
            "pitch_shift": pitch_shift,
            "index_ratio": index_ratio,
            "protect": protect,
            "filter_radius": filter_radius,
            "mix_rate": mix_rate,
        }
        filename = source_audio.filename or "input.mp3"

        if return_job:
            call = await converter.convert.spawn.aio(payload, filename, options)
            return {"jobId": call.object_id, "status": "queued"}

        result = await converter.convert.remote.aio(payload, filename, options)
        return Response(
            content=result["audio"],
            media_type=result["media_type"],
            headers={"X-Beam-Demo-Max-Seconds": str(MAX_DEMO_SECONDS)},
        )

    @api.post("/ai-convert/vocal-analysis")
    async def vocal_analysis(source_audio: UploadFile = File(...)):
        import tempfile
        import uuid

        from app.services.audio_utils import probe_duration_seconds

        if source_audio.content_type and source_audio.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail={"error": "unsupported_media_type", "message": f"Unsupported content type: {source_audio.content_type}"},
            )
        payload = await source_audio.read(MAX_FILE_SIZE + 1)
        if not payload:
            raise HTTPException(status_code=400, detail={"error": "empty_audio", "message": "source_audio is empty"})
        if len(payload) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail={"error": "file_too_large", "message": "Max 25MB"})
        with tempfile.TemporaryDirectory(prefix="beam_gateway_analysis_") as workdir:
            suffix = Path(source_audio.filename or "input.mp3").suffix or ".bin"
            input_path = Path(workdir) / f"{uuid.uuid4().hex}{suffix}"
            input_path.write_bytes(payload)
            duration = probe_duration_seconds(input_path)
        medleyvox_checkpoint = (
            Path(REMOTE_ROOT) / "weights" / "medleyvox" / "checkpoint" / "singing_librispeech_ft_iSRNet"
        )
        medleyvox_ready = (medleyvox_checkpoint / "vocals.json").is_file() and (medleyvox_checkpoint / "vocals.pth").is_file()
        if medleyvox_ready:
            candidates = [
                {"trackId": "singer_1", "label": "Singer lane 1", "requiresManualSegmentation": False, "supportsOverlaps": True},
                {"trackId": "singer_2", "label": "Singer lane 2", "requiresManualSegmentation": False, "supportsOverlaps": True},
            ]
            limitations = [
                "Singer lanes are anonymous estimates and can contain leak-through or swap between chunks.",
                "Review each lane before assigning a target voice.",
            ]
            analysis_version = "medleyvox-two-singer-poc-v1"
        else:
            candidates = [{
                "trackId": "vocal_candidate_1",
                "label": "Separated vocal stem",
                "requiresManualSegmentation": True,
                "supportsOverlaps": False,
            }]
            limitations = [
                "This version does not identify singers automatically.",
                "Create non-overlapping time ranges for sequential duet sections.",
                "Simultaneous vocals and backing-vocal separation are not supported.",
            ]
            analysis_version = "manual-sequential-v1"
        return {
            "analysisVersion": analysis_version,
            "durationSeconds": round(duration, 3),
            "candidates": candidates,
            "limitations": limitations,
        }

    @api.post("/ai-convert/multi-voice-conversion")
    async def multi_voice_conversion(
        source_audio: UploadFile = File(...),
        vocalAssignments: str = Form(...),
        output_format: str = Form("mp3"),
        mix_with_instrumental: bool = Form(True),
        preserve_unassigned_vocals: bool = Form(False),
    ):
        if output_format not in MEDIA_TYPES:
            raise HTTPException(status_code=400, detail={"error": "unsupported_output_format", "message": output_format})
        if source_audio.content_type and source_audio.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail={"error": "unsupported_media_type", "message": f"Unsupported content type: {source_audio.content_type}"},
            )
        try:
            assignments = json.loads(vocalAssignments)
            if not isinstance(assignments, list):
                raise ValueError("vocalAssignments must be a JSON array")
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(status_code=422, detail={"error": "invalid_vocal_assignments", "message": str(error)}) from error
        payload = await source_audio.read(MAX_FILE_SIZE + 1)
        if not payload:
            raise HTTPException(status_code=400, detail={"error": "empty_audio", "message": "source_audio is empty"})
        if len(payload) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail={"error": "file_too_large", "message": "Max 25MB"})
        try:
            result = await converter.convert_multi.remote.aio(
                payload,
                source_audio.filename or "input.mp3",
                assignments,
                {
                    "output_format": output_format,
                    "mix_with_instrumental": mix_with_instrumental,
                    "preserve_unassigned_vocals": preserve_unassigned_vocals,
                },
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail={"error": "multi_vocal_conversion_failed", "message": str(error)},
            ) from error
        return Response(content=result["audio"], media_type=result["media_type"])

    @api.get("/ai-convert/jobs/{job_id}")
    async def get_job(request: Request, job_id: str):
        result, status = await read_call(job_id)
        if status == "processing":
            return {
                "jobId": job_id,
                "status": "processing",
                "progress": 50,
                "stage": "gpu_conversion",
                "timings": None,
                "resultUrl": None,
                "resultPath": None,
                "error": None,
            }
        if status == "failed":
            return {
                "jobId": job_id,
                "status": "failed",
                "progress": 100,
                "stage": "failed",
                "timings": None,
                "resultUrl": None,
                "resultPath": None,
                "error": result,
            }
        result_url = str(request.url_for("get_result", job_id=job_id))
        return {
            "jobId": job_id,
            "status": "completed",
            "progress": 100,
            "stage": "completed",
            "timings": result["timings"],
            "resultUrl": result_url,
            "resultPath": None,
            "error": None,
        }

    @api.get("/ai-convert/results/{job_id}", name="get_result")
    async def get_result(job_id: str):
        result, status = await read_call(job_id)
        if status == "processing":
            raise HTTPException(
                status_code=409,
                detail={"error": "result_not_ready", "message": "Conversion is still running"},
            )
        if status == "failed":
            raise HTTPException(
                status_code=500,
                detail={"error": "conversion_failed", "message": result},
            )
        return Response(
            content=result["audio"],
            media_type=result["media_type"],
            headers={"Cache-Control": "private, max-age=3600"},
        )

    return api

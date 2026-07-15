from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.pipeline import BeamSVCPipeline
from app.services.doctor import RuntimeDoctor

router = APIRouter()
pipeline = BeamSVCPipeline()
doctor = RuntimeDoctor()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = pipeline.health()
    checks = doctor.check()
    ok = all(
        [
            checks["rvc_repo_exists"],
            checks["infer_cli_exists"],
            checks["rvc_python_exists"],
            checks["ffmpeg_exists"],
            checks["ffprobe_exists"],
            checks["demucs_exists"],
            checks["registry_exists"],
            checks["at_least_one_model_exists"],
        ]
    )
    return HealthResponse(
        ok=ok,
        service="beam-svc",
        version="0.1.0",
        gpu=status["gpu"],
        device=status["device"],
        requireGpu=status["requireGpu"],
        modelsLoaded=status["modelsLoaded"],
    )

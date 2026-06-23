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
    return HealthResponse(
        ok=checks["rvc_repo_exists"] and checks["infer_cli_exists"] and checks["registry_exists"] and checks["at_least_one_model_exists"],
        service="beam-svc",
        version="0.1.0",
        gpu=status["gpu"],
        modelsLoaded=status["modelsLoaded"],
    )

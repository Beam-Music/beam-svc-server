import json
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError
try:
    from pydantic import TypeAdapter
except ImportError:  # Local developer images may still carry Pydantic v1.
    TypeAdapter = None

from app.api.convert import ALLOWED_TYPES, MAX_FILE_SIZE, MEDIA_TYPES, conversion_lock
from app.models.schemas import VocalAnalysisResponse, VocalAssignment, VocalCandidate
from app.services.multi_vocal_pipeline import MultiConversionOptions, MultiVocalPipeline

router = APIRouter()
pipeline = MultiVocalPipeline()
assignment_adapter = TypeAdapter(list[VocalAssignment]) if TypeAdapter else None


def parse_assignments(raw: str) -> list[VocalAssignment]:
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("vocalAssignments must be a JSON array")
    if assignment_adapter is not None:
        return assignment_adapter.validate_python(payload)
    return [VocalAssignment.parse_obj(item) for item in payload]


async def _read_upload(source_audio: UploadFile) -> bytes:
    if source_audio.content_type and source_audio.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail={"error": "unsupported_media_type", "message": f"Unsupported content type: {source_audio.content_type}"})
    payload = await source_audio.read(MAX_FILE_SIZE + 1)
    if not payload:
        raise HTTPException(status_code=400, detail={"error": "empty_audio", "message": "source_audio is empty"})
    if len(payload) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail={"error": "file_too_large", "message": "Max 25MB"})
    return payload


@router.post("/vocal-analysis", response_model=VocalAnalysisResponse)
async def vocal_analysis(source_audio: UploadFile = File(...)) -> VocalAnalysisResponse:
    payload = await _read_upload(source_audio)
    with tempfile.TemporaryDirectory(prefix="beam_svc_analysis_") as workdir:
        input_path = Path(workdir) / f"{uuid.uuid4().hex}{Path(source_audio.filename or 'input.mp3').suffix or '.bin'}"
        input_path.write_bytes(payload)
        duration = pipeline.analyze(input_path, Path(workdir))
    if pipeline.medleyvox.available():
        candidates = [
            VocalCandidate(trackId="singer_1", label="Singer lane 1", requiresManualSegmentation=False, supportsOverlaps=True),
            VocalCandidate(trackId="singer_2", label="Singer lane 2", requiresManualSegmentation=False, supportsOverlaps=True),
        ]
        limitations = [
            "Singer lanes are anonymous estimates and can contain leak-through or swap between chunks.",
            "Review each lane before assigning a target voice.",
        ]
        analysis_version = "medleyvox-two-singer-poc-v1"
    else:
        candidates = [VocalCandidate(trackId="vocal_candidate_1", label="Separated vocal stem", requiresManualSegmentation=True, supportsOverlaps=False)]
        limitations = ["This version does not identify singers automatically.", "Create non-overlapping time ranges for sequential duet sections.", "Simultaneous vocals and backing-vocal separation are not supported."]
        analysis_version = "manual-sequential-v1"
    return VocalAnalysisResponse(
        analysisVersion=analysis_version,
        durationSeconds=round(duration, 3),
        candidates=candidates,
        limitations=limitations,
    )


@router.post("/multi-voice-conversion")
async def multi_voice_conversion(
    request: Request,
    source_audio: UploadFile = File(...),
    vocalAssignments: str = Form(...),
    output_format: str = Form("mp3"),
    mix_with_instrumental: bool = Form(True),
    preserve_unassigned_vocals: bool = Form(False),
):
    if output_format not in MEDIA_TYPES:
        raise HTTPException(status_code=400, detail={"error": "unsupported_output_format", "message": output_format})
    try:
        assignments = parse_assignments(vocalAssignments)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail={"error": "invalid_vocal_assignments", "message": str(error)}) from error
    payload = await _read_upload(source_audio)
    with tempfile.TemporaryDirectory(prefix="beam_svc_multi_") as workdir:
        input_path = Path(workdir) / f"{uuid.uuid4().hex}{Path(source_audio.filename or 'input.mp3').suffix or '.bin'}"
        input_path.write_bytes(payload)
        with conversion_lock:
            result = pipeline.convert(
                input_path,
                assignments,
                MultiConversionOptions(
                    output_format=output_format,
                    mix_with_instrumental=mix_with_instrumental,
                    preserve_unassigned_vocals=preserve_unassigned_vocals,
                ),
            )
    return Response(content=result, media_type=MEDIA_TYPES[output_format])

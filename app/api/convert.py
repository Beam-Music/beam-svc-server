from pathlib import Path
import tempfile
import uuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from app.models.schemas import ConversionAcceptedResponse, JobStatusResponse
from app.services.cache import ConversionCache
from app.services.jobs import job_store
from app.services.pipeline import BeamSVCPipeline, ConversionOptions
from app.services.registry import VoiceRegistry

router = APIRouter()
registry = VoiceRegistry()
pipeline = BeamSVCPipeline()
cache = ConversionCache()

ALLOWED_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/m4a", "application/octet-stream"}
MAX_FILE_SIZE = 25 * 1024 * 1024


def run_conversion_job(
    job_id: str,
    payload: bytes,
    filename: str,
    voice_id: str,
    voice_type: str | None,
    language: str,
    preserve_melody: str,
    mix_with_instrumental: str,
    output_format: str,
    trim_start: float | None,
    trim_duration: float | None,
    pitch_shift: int,
    index_ratio: float | None,
    protect: float | None,
    filter_radius: int | None,
    cache_key: str,
    result_url: str,
) -> None:
    job_store.update(job_id, status="processing", progress=5, stage="preparing")
    try:
        with tempfile.TemporaryDirectory(prefix="beam_svc_job_") as workdir:
            suffix = Path(filename).suffix or ".bin"
            input_path = Path(workdir) / f"{uuid.uuid4().hex}{suffix}"
            input_path.write_bytes(payload)
            job_store.update(job_id, progress=20, stage="converting")
            result = pipeline.convert(
                input_path,
                ConversionOptions(
                    voice_id=voice_id,
                    voice_type=voice_type,
                    language=language,
                    preserve_melody=preserve_melody.lower() == "true",
                    mix_with_instrumental=mix_with_instrumental.lower() == "true",
                    output_format=output_format,
                    trim_start=trim_start,
                    trim_duration=trim_duration,
                    pitch_shift=pitch_shift,
                    index_ratio=index_ratio,
                    protect=protect,
                    filter_radius=filter_radius,
                ),
            )
        saved_path = cache.save(cache_key, output_format, result)
        job_store.update(job_id, status="completed", progress=100, stage="completed", resultUrl=result_url, resultPath=str(saved_path), error=None)
    except Exception as error:
        job_store.update(job_id, status="failed", progress=100, stage="failed", error=str(error))


@router.post("/voice-conversion")
async def voice_conversion(
    request: Request,
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(...),
    voiceId: str = Form(...),
    voiceType: str | None = Form(None),
    language: str = Form("en"),
    preserve_melody: str = Form("true"),
    mix_with_instrumental: str = Form("true"),
    output_format: str = Form("mp3"),
    trim_start: float | None = Form(None),
    trim_duration: float | None = Form(None),
    pitch_shift: int = Form(0),
    index_ratio: float | None = Form(None),
    protect: float | None = Form(None),
    filter_radius: int | None = Form(None),
    return_job: bool = Form(False),
):
    voice = registry.get_voice(voiceId)
    if voice is None:
        raise HTTPException(status_code=404, detail={"error": "voice_not_found", "message": f"Unknown voiceId: {voiceId}"})

    if source_audio.content_type and source_audio.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail={"error": "unsupported_media_type", "message": f"Unsupported content type: {source_audio.content_type}"})

    payload = await source_audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail={"error": "empty_audio", "message": "source_audio is empty"})
    if len(payload) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail={"error": "file_too_large", "message": "Max 25MB"})

    cache_key = cache.make_key(
        payload,
        voice_id=voiceId,
        voice_type=voiceType,
        language=language,
        preserve_melody=preserve_melody.lower() == "true",
        mix_with_instrumental=mix_with_instrumental.lower() == "true",
        output_format=output_format,
        trim_start=trim_start,
        trim_duration=trim_duration,
        pitch_shift=pitch_shift,
        index_ratio=index_ratio,
        protect=protect,
        filter_radius=filter_radius,
    )

    cached = cache.load(cache_key, output_format)
    if cached is not None and not return_job:
        media_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "m4a": "audio/mp4",
        }.get(output_format, "audio/mpeg")
        return Response(content=cached, media_type=media_type, headers={"X-Beam-Cache": "HIT"})

    if return_job:
        job_id = f"svc_{uuid.uuid4().hex}"
        result_url = str(request.url_for("get_result", job_id=job_id))
        job_store.save(JobStatusResponse(jobId=job_id, status="queued", progress=0, stage="queued", resultUrl=result_url, resultPath=None, error=None))
        background_tasks.add_task(
            run_conversion_job,
            job_id,
            payload,
            source_audio.filename or "input.mp3",
            voiceId,
            voiceType,
            language,
            preserve_melody,
            mix_with_instrumental,
            output_format,
            trim_start,
            trim_duration,
            pitch_shift,
            index_ratio,
            protect,
            filter_radius,
            cache_key,
            result_url,
        )
        return ConversionAcceptedResponse(jobId=job_id, status="queued")

    with tempfile.TemporaryDirectory(prefix="beam_svc_") as workdir:
        suffix = Path(source_audio.filename or "input.mp3").suffix or ".bin"
        input_path = Path(workdir) / f"{uuid.uuid4().hex}{suffix}"
        input_path.write_bytes(payload)

        try:
            result = pipeline.convert(
                input_path,
                ConversionOptions(
                    voice_id=voiceId,
                    voice_type=voiceType,
                    language=language,
                    preserve_melody=preserve_melody.lower() == "true",
                    mix_with_instrumental=mix_with_instrumental.lower() == "true",
                    output_format=output_format,
                    trim_start=trim_start,
                    trim_duration=trim_duration,
                    pitch_shift=pitch_shift,
                    index_ratio=index_ratio,
                    protect=protect,
                    filter_radius=filter_radius,
                ),
            )
        except ValueError as error:
            return JSONResponse(status_code=400, content={"error": "conversion_validation_failed", "message": str(error)})
        except RuntimeError as error:
            return JSONResponse(status_code=500, content={"error": "conversion_runtime_failed", "message": str(error)})

    cache.save(cache_key, output_format, result)

    media_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
    }.get(output_format, "audio/mpeg")
    return Response(content=result, media_type=media_type, headers={"X-Beam-Cache": "MISS"})

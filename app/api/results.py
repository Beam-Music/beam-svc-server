from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.services.jobs import job_store

router = APIRouter()


@router.get("/results/{job_id}", name="get_result")
def get_result(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "job_not_found", "message": f"Unknown jobId: {job_id}"})
    if not job.resultPath:
        raise HTTPException(status_code=404, detail={"error": "result_not_ready", "message": f"Result not ready for jobId: {job_id}"})

    path = Path(job.resultPath)
    if not path.exists():
        raise HTTPException(status_code=404, detail={"error": "result_missing", "message": f"Result file missing for jobId: {job_id}"})

    media_type = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)

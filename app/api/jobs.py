from fastapi import APIRouter, HTTPException
from app.models.schemas import JobStatusResponse
from app.services.jobs import job_store

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "job_not_found", "message": f"Unknown jobId: {job_id}"})
    return job

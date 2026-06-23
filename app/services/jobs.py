from app.models.schemas import JobStatusResponse


class JobStore:
    def __init__(self):
        self._jobs: dict[str, JobStatusResponse] = {}

    def get(self, job_id: str) -> JobStatusResponse | None:
        return self._jobs.get(job_id)

    def save(self, job: JobStatusResponse) -> None:
        self._jobs[job.jobId] = job

    def update(self, job_id: str, **changes) -> JobStatusResponse | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        updated = job.model_copy(update=changes)
        self._jobs[job_id] = updated
        return updated


job_store = JobStore()

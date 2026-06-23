from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from app.api.health import router as health_router
from app.api.voices import router as voices_router
from app.api.convert import router as convert_router
from app.api.jobs import router as jobs_router
from app.api.results import router as results_router

app = FastAPI(title="Beam SVC Server", version="0.1.0")
app.include_router(health_router, prefix="/ai-convert", tags=["health"])
app.include_router(voices_router, prefix="/ai-convert", tags=["voices"])
app.include_router(convert_router, prefix="/ai-convert", tags=["convert"])
app.include_router(jobs_router, prefix="/ai-convert", tags=["jobs"])
app.include_router(results_router, prefix="/ai-convert", tags=["results"])


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "message": str(exc.detail)})

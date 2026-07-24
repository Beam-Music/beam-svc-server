import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from app.api.health import router as health_router
from app.api.voices import router as voices_router
from app.api.convert import router as convert_router
from app.api.multi_vocal import router as multi_vocal_router
from app.api.jobs import router as jobs_router
from app.api.results import router as results_router

app = FastAPI(title="Beam SVC Server", version="0.1.0")
app.include_router(health_router, prefix="/ai-convert", tags=["health"])
app.include_router(voices_router, prefix="/ai-convert", tags=["voices"])
app.include_router(convert_router, prefix="/ai-convert", tags=["convert"])
app.include_router(multi_vocal_router, prefix="/ai-convert", tags=["multi-vocal"])
app.include_router(jobs_router, prefix="/ai-convert", tags=["jobs"])
app.include_router(results_router, prefix="/ai-convert", tags=["results"])
logger = logging.getLogger(__name__)

@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "message": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled request error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Internal server error"},
    )

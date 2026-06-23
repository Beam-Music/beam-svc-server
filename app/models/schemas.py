from pydantic import BaseModel, Field
from typing import Optional, Literal


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str
    gpu: bool
    modelsLoaded: bool


class VoiceModelMeta(BaseModel):
    engine: str
    sampleRate: int
    f0: bool
    indexRatio: float
    protect: float
    filterRadius: int
    mixRate: float
    modelPath: str
    indexPath: Optional[str] = None


class VoiceInfo(BaseModel):
    voiceId: str
    name: str
    category: str
    description: Optional[str] = None
    preview_url: Optional[str] = None
    language: list[str] = Field(default_factory=list)
    voiceType: Optional[str] = None
    model: Optional[VoiceModelMeta] = None


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]
    total_count: int


class JobStatusResponse(BaseModel):
    jobId: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = 0
    stage: Optional[str] = None
    resultUrl: Optional[str] = None
    resultPath: Optional[str] = None
    error: Optional[str] = None


class ConversionAcceptedResponse(BaseModel):
    jobId: str
    status: Literal["queued"]

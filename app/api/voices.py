from fastapi import APIRouter
from app.models.schemas import VoiceListResponse
from app.services.registry import VoiceRegistry

router = APIRouter()
registry = VoiceRegistry()


@router.get("/voices", response_model=VoiceListResponse)
def list_voices() -> VoiceListResponse:
    return registry.list_voices()

from fastapi import APIRouter

from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {"status": "ok", "message": "Model info endpoint"}


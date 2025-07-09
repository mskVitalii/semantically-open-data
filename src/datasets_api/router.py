from fastapi import APIRouter

from src.datasets_api.datasets_controller import router as datasets_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(datasets_router)

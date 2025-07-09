from fastapi import APIRouter, Depends, HTTPException

from .datasets_dto import DatasetSearchRequest, DatasetSearchResponse
from ..domain.services.dataset_service import DatasetService, get_dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/search", response_model=DatasetSearchResponse)
async def search_datasets(
    request: DatasetSearchRequest,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetSearchResponse:
    """
    Dataset search

    Supported parameters:
    - query: full-text search by name and description
    - tags: filter by tags
    - limit: number of results (1–100)
    - offset: pagination offset
    """
    try:
        return await service.search_datasets(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

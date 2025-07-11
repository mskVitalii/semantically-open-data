from ...datasets.bootstrap import bootstrap_data
from ...datasets.datasets_metadata import DatasetMetadataWithContent
from ...datasets_api.datasets_dto import (
    DatasetSearchRequest,
    DatasetSearchResponse,
    DatasetResponse,
)
from ...infrastructure.config import USE_GRPC
from ...infrastructure.logger import get_logger
from ...vector_search.vector_db import VectorDB


logger = get_logger(__name__)


class DatasetService:
    """Service for working with datasets"""

    vector_db = VectorDB(USE_GRPC)

    async def search_datasets(
        self, request: DatasetSearchRequest
    ) -> DatasetSearchResponse:
        """Search for datasets"""
        # Convert DTO to domain object
        # criteria = SearchCriteria(
        #     query=request.query,
        #     tags=request.tags,
        #     limit=request.limit,
        #     offset=request.offset,
        # )

        # TODO: extend request
        datasets = self.vector_db.search(request.query, None)

        metadatas: list[DatasetResponse] = []
        for dataset in datasets:
            metadatas.append(
                DatasetResponse(
                    metadata=DatasetMetadataWithContent(**dataset.payload),
                    score=dataset.score,
                ),
            )

        # logger.info(f"\nFound {len(results)} results:")
        # for i, result in enumerate(results, 1):
        #     logger.info(f"\n{i}. Score: {result.score:.4f}")
        #     logger.info(f"   Title: {result.payload['title']}")
        #     logger.info(f"   City: {result.payload['city']}")
        #     logger.info(f"   Organization: {result.payload['organization']}")
        #     if result.payload.get("description"):
        #         desc = (
        #             result.payload["description"][:200] + "..."
        #             if len(result.payload["description"]) > 200
        #             else result.payload["description"]
        #         )
        #         logger.info(f"   Description: {desc}")

        return DatasetSearchResponse(
            datasets=metadatas,
            total=len(metadatas),
            limit=request.limit,
            offset=request.offset,
        )

    async def bootstrap_datasets(self) -> bool:
        await self.vector_db.remove_collection()
        await bootstrap_data()
        return True


def get_dataset_service() -> DatasetService:
    return DatasetService()

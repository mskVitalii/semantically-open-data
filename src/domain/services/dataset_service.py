# src/datasets_api/service.py
from fastapi import Depends
from numpy import ndarray

from src.datasets.bootstrap import bootstrap_data
from src.datasets.datasets_metadata import DatasetMetadataWithContent
from src.domain.repositories.dataset_repository import (
    DatasetRepository,
    get_dataset_repository,
)
from src.datasets_api.datasets_dto import (
    DatasetSearchRequest,
    DatasetSearchResponse,
    DatasetResponse,
)
from src.infrastructure.logger import get_prefixed_logger
from src.vector_search.embedder import embed
from src.vector_search.vector_db import VectorDB, get_vector_db

logger = get_prefixed_logger(__name__, "DATASET_SERVICE")


class DatasetService:
    """Service for working with datasets"""

    def __init__(self, vector_db: VectorDB, repository: DatasetRepository):
        self.vector_db = vector_db
        self.repository = repository

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

        # Generate query embedding
        embedding = await embed(request.query)
        datasets = await self.vector_db.search(embedding)

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

    async def search_datasets_with_embeddings(
        self, embeddings: ndarray
    ) -> list[DatasetResponse]:
        """Search for datasets"""
        datasets = await self.vector_db.search(embeddings)

        logger.info(datasets)
        return [
            DatasetResponse(
                metadata=DatasetMetadataWithContent(
                    **{k: v for k, v in dataset.payload.items() if k != "content"}
                ),
                score=dataset.score,
            )
            for dataset in datasets
        ]

    async def bootstrap_datasets(self) -> bool:
        """Bootstrap datasets - clear and reload all data"""
        try:
            # Clear MongoDB
            deleted_count = await self.repository.delete_all()
            logger.info(f"Deleted {deleted_count} datasets from MongoDB")

            # Clear vector DB
            await self.vector_db.remove_collection()
            await self.vector_db.setup_collection()

            # Bootstrap data (this should populate both MongoDB and vector DB)
            await bootstrap_data()

            # Create indexes for better performance
            await self.repository.create_indexes()

            # Get statistics
            stats = await self.repository.get_statistics()
            logger.info(f"Bootstrap completed. Stats: {stats}")

            return True
        except Exception as e:
            logger.error(f"bootstrap_datasets error: {e}")
            return False


# Dependency injection
async def get_dataset_service(
    vector_db: VectorDB = Depends(get_vector_db),
    repository: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetService:
    """Get DatasetService instance with dependencies"""
    return DatasetService(vector_db, repository)

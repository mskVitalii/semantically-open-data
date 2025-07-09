from ...datasets_api.dataset_dto import (
    DatasetSearchRequest,
    DatasetSearchResponse,
    DatasetResponse,
    SearchCriteria,
)


class DatasetService:
    """Service for working with datasets"""

    async def search_datasets(
        self, request: DatasetSearchRequest
    ) -> DatasetSearchResponse:
        """Search for datasets"""
        # Convert DTO to domain object
        criteria = SearchCriteria(
            query=request.query,
            tags=request.tags,
            limit=request.limit,
            offset=request.offset,
        )

        # Call repository
        datasets = await self._dataset_repository.search(criteria)

        # Convert result to DTO
        dataset_responses = [
            DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                tags=dataset.tags,
                metadata=dataset.metadata,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
            )
            for dataset in datasets
        ]

        return DatasetSearchResponse(
            datasets=dataset_responses,
            total=len(
                dataset_responses
            ),  # In a real app, this should be a separate query
            limit=request.limit,
            offset=request.offset,
        )


def get_dataset_service() -> DatasetService:
    return DatasetService()

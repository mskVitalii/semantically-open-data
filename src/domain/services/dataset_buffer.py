import asyncio

from src.datasets.datasets_metadata import DatasetMetadataWithContent
from src.domain.repositories.dataset_repository import DatasetRepository
from src.infrastructure.logger import get_prefixed_logger
from src.utils.buffer_abc import AsyncBuffer

logger = get_prefixed_logger(__name__, "DATASET_DB_BUFFER")

# TODO:
#  1. Store the metadata (with the collection name for dataset)
#  2. Create the collection
#  3. Save the actual dataset to the collection


class DatasetDBBuffer(AsyncBuffer[DatasetMetadataWithContent]):
    """Buffer for batching dataset storing"""

    def __init__(self, repository: DatasetRepository, buffer_size: int = 150):
        """
        Initialize the buffer

        Args:
            repository: The MongoDBManager instance to use for store
            buffer_size: Maximum number of records to hold before auto-flushing
        """
        self.repository = repository
        self.buffer_size = buffer_size
        self._buffer: list[DatasetMetadataWithContent] = []
        self._lock = asyncio.Lock()
        self._total_stored = 0

    # region Buffer logic

    async def add(self, dataset: DatasetMetadataWithContent) -> None:
        """
        Add a single dataset to the buffer

        Args:
            dataset: Dataset to add to the buffer
        """
        async with self._lock:
            self._buffer.append(dataset)
            logger.debug(f"Added dataset to buffer. Current size: {len(self._buffer)}")

            if len(self._buffer) >= self.buffer_size:
                await self._flush_internal()

    async def flush(self) -> int:
        """
        Manually flush the buffer

        Returns:
            Number of datasets indexed
        """
        async with self._lock:
            return await self._flush_internal()

    async def clear(self) -> None:
        """Clear the buffer without indexing"""
        async with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            logger.info(f"Cleared {count} datasets from buffer without indexing")

    @property
    async def size(self) -> int:
        """Current number of items in the buffer"""
        async with self._lock:
            return len(self._buffer)

    @property
    def total_indexed(self) -> int:
        return self._total_stored

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - always flush remaining data"""
        try:
            await self.flush()
        except Exception as e:
            logger.error(f"Error flushing buffer on exit: {e}")
            if exc_type is None:  # Only raise if there wasn't already an exception
                raise

    # endregion

    # region Data handle

    async def _flush_internal(self) -> int:
        """
        Internal flush method (must be called with lock held)

        Returns:
            Number of datasets indexed
        """
        if not self._buffer:
            logger.debug("Buffer is empty, nothing to flush")
            return 0

        # Get the datasets to index
        meta_to_index = self._buffer[:]

        try:
            # Index the datasets
            logger.info(f"Flushing {len(meta_to_index)} datasets from buffer")
            await self.repository.batch_insert(
                [meta.to_dict() for meta in meta_to_index],
                batch_size=self.buffer_size,
            )

            # Clear the buffer only after successful indexing
            self._buffer.clear()
            self._total_stored += len(meta_to_index)

            logger.info(
                f"Successfully flushed {len(meta_to_index)} datasets. Total indexed: {self._total_stored}"
            )
            return len(meta_to_index)

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            # Buffer is not cleared on error, so data is not lost
            raise

    # endregion

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
        need_flush = False
        async with self._lock:
            self._buffer.append(dataset)
            logger.debug(f"Added dataset to buffer. Current size: {len(self._buffer)}")
            need_flush = len(self._buffer) >= self.buffer_size

        if need_flush:
            await self._flush_internal()

    async def flush(self) -> int:
        """
        Manually flush the buffer

        Returns:
            Number of datasets indexed
        """
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
        Internal flush method

        Returns:
            Number of datasets indexed
        """
        async with self._lock:
            if not self._buffer:
                logger.debug("Buffer is empty, nothing to flush")
                return 0
            data_to_store = self._buffer[:]
            self._buffer.clear()
        data_count = len(data_to_store)

        try:
            # Index the datasets
            logger.info(f"Flushing {len(data_to_store)} datasets from buffer")

            async def insert_and_cleanup():
                try:
                    await self.repository.batch_insert(
                        [meta.to_dict() for meta in data_to_store],
                        batch_size=self.buffer_size,
                    )
                    async with self._lock:
                        self._total_stored += data_count
                    logger.info(
                        f"Successfully flushed {data_count} datasets. Total indexed: {self._total_stored}"
                    )

                except Exception as _e:
                    logger.error(f"Background insert failed: {_e}")

            asyncio.create_task(insert_and_cleanup())
            return data_count

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            raise

    # endregion

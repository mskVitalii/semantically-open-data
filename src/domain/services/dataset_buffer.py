import asyncio

from src.datasets.datasets_metadata import Dataset
from src.domain.repositories.dataset_repository import DatasetRepository
from src.infrastructure.logger import get_prefixed_logger
from src.utils.buffer_abc import AsyncBuffer

logger = get_prefixed_logger(__name__, "DATASET_DB_BUFFER")


class DatasetDBBuffer(AsyncBuffer[Dataset]):
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
        self._buffer: list[Dataset] = []
        self._lock = asyncio.Lock()
        self._total_stored = 0

    # region Buffer logic

    async def add(self, dataset: Dataset) -> None:
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
        Internal flush method with semaphore for parallel processing control

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
                    # Create semaphore to limit concurrent operations
                    # Adjust max_concurrent based on your system's capacity
                    max_concurrent = min(
                        10, len(data_to_store)
                    )  # Max 10 parallel tasks
                    # semaphore = asyncio.Semaphore(max_concurrent)

                    async def index_single_dataset(dataset: Dataset) -> int:
                        # async with semaphore:
                        try:
                            return await self.repository.index_dataset(
                                dataset,
                                self.buffer_size,
                            )
                        except Exception as e:
                            logger.error(f"Failed to index dataset, {e}", exc_info=True)
                            return 0

                    # Process all datasets concurrently with semaphore control
                    results = await asyncio.gather(
                        *[index_single_dataset(dataset) for dataset in data_to_store],
                        return_exceptions=False,
                    )

                    # Count successful indexing operations
                    successful_count = sum(1 for r in results if r > 0)

                    # Update total stored count
                    async with self._lock:
                        self._total_stored += successful_count

                    if successful_count < data_count:
                        logger.warning(
                            f"Partially flushed: {successful_count}/{data_count} datasets indexed"
                        )
                    else:
                        logger.info(
                            f"Successfully flushed {successful_count} datasets. Total indexed: {self._total_stored}"
                        )

                except Exception as _e:
                    logger.error(f"Background insert failed: {_e}")

            asyncio.create_task(insert_and_cleanup())
            return data_count

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            raise

    # endregion

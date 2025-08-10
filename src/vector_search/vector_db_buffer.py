import asyncio
from typing import List
import logging

from src.datasets.datasets_metadata import DatasetMetadataWithContent
from src.vector_search.vector_db import VectorDB

logger = logging.getLogger(__name__)


# TODO: Make this buffer common and many handlers?
class VectorDBBuffer:
    """Buffer for batching dataset indexing operations"""

    def __init__(
        self, vector_db: VectorDB, buffer_size: int = 150, auto_flush: bool = True
    ):
        """
        Initialize the buffer

        Args:
            vector_db: The AsyncVectorDB instance to use for indexing
            buffer_size: Maximum number of records to hold before auto-flushing
            auto_flush: Whether to automatically flush when buffer is full
        """
        self.vector_db = vector_db
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        self._buffer: list[DatasetMetadataWithContent] = []
        self._lock = asyncio.Lock()  # Async lock for thread safety
        self._total_indexed = 0

    async def add(self, dataset: DatasetMetadataWithContent) -> None:
        """
        Add a single dataset to the buffer

        Args:
            dataset: Dataset to add to the buffer
        """
        async with self._lock:
            self._buffer.append(dataset)
            logger.debug(f"Added dataset to buffer. Current size: {len(self._buffer)}")

            if self.auto_flush and len(self._buffer) >= self.buffer_size:
                await self._flush_internal()

    async def add_batch(self, datasets: List[DatasetMetadataWithContent]) -> None:
        """
        Add multiple datasets to the buffer

        Args:
            datasets: List of datasets to add
        """
        async with self._lock:
            self._buffer.extend(datasets)
            logger.debug(
                f"Added {len(datasets)} datasets to buffer. Current size: {len(self._buffer)}"
            )

            if self.auto_flush and len(self._buffer) >= self.buffer_size:
                await self._flush_internal()

    async def flush(self) -> int:
        """
        Manually flush the buffer

        Returns:
            Number of datasets indexed
        """
        async with self._lock:
            return await self._flush_internal()

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
        datasets_to_index = self._buffer[:]

        try:
            # Index the datasets
            logger.info(f"Flushing {len(datasets_to_index)} datasets from buffer")
            await self.vector_db.index_datasets(
                datasets_to_index, batch_size=self.buffer_size
            )

            # Clear the buffer only after successful indexing
            self._buffer.clear()
            self._total_indexed += len(datasets_to_index)

            logger.info(
                f"Successfully flushed {len(datasets_to_index)} datasets. Total indexed: {self._total_indexed}"
            )
            return len(datasets_to_index)

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            # Buffer is not cleared on error, so data is not lost
            raise

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

    @property
    async def size(self) -> int:
        """Current number of items in the buffer"""
        async with self._lock:
            return len(self._buffer)

    @property
    def total_indexed(self) -> int:
        """Total number of datasets indexed through this buffer"""
        return self._total_indexed

    async def clear(self) -> None:
        """Clear the buffer without indexing"""
        async with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            logger.info(f"Cleared {count} datasets from buffer without indexing")


# Example usage:
# async def main():
#     # Initialize AsyncVectorDB
#     from src.vector_search.vector_db import get_vector_db
#
#     vector_db_main = await get_vector_db(use_grpc=True)
#
#     try:
#         # Create buffer with auto-flush at 100 records
#         buffer = VectorDBBuffer(vector_db_main, buffer_size=100, auto_flush=True)
#
#         # Example 1: Add datasets one by one
#         for i in range(250):
#             dataset_main = DatasetMetadataWithContent(
#                 id=f"dataset_{i}",
#                 title=f"Dataset {i}",
#                 description=f"Description for dataset {i}",
#                 content=f"Content for dataset {i}",
#                 city="Chemnitz",
#                 state="Saxony",
#                 country="Germany",
#             )
#             await buffer.add(dataset_main)  # Will auto-flush at 100 and 200
#
#         # Flush remaining datasets
#         await buffer.flush()  # Will flush the remaining 50
#
#         # Example 2: Using async context manager
#         async with VectorDBBuffer(vector_db_main, buffer_size=50) as buffer2:
#             datasets_main = [
#                 DatasetMetadataWithContent(
#                     id=f"batch_{i}",
#                     title=f"Batch Dataset {i}",
#                     description=f"Batch description {i}",
#                     content=f"Batch content {i}",
#                     city="Chemnitz",
#                 )
#                 for i in range(30)
#             ]
#             await buffer2.add_batch(datasets_main)
#             # Buffer will be automatically flushed when exiting the context
#
#         print(f"Total datasets indexed: {buffer.total_indexed}")
#
#     finally:
#         # Clean up
#         await vector_db_main.qdrant.close()
#
#
# # Example 3: Concurrent operations
# async def concurrent_example():
#     from src.vector_search.vector_db import get_vector_db
#
#     vector_db = await get_vector_db(use_grpc=True)
#
#     try:
#         async with VectorDBBuffer(vector_db, buffer_size=100) as buffer:
#             # Create multiple tasks that add data concurrently
#             async def add_datasets(start_idx: int, count: int):
#                 for i in range(start_idx, start_idx + count):
#                     dataset = DatasetMetadataWithContent(
#                         id=f"concurrent_{i}",
#                         title=f"Concurrent Dataset {i}",
#                         description=f"Description {i}",
#                         content=f"Content {i}",
#                         city="Dresden",
#                     )
#                     await buffer.add(dataset)
#
#             # Run multiple concurrent tasks
#             tasks = [
#                 add_datasets(0, 50),
#                 add_datasets(50, 50),
#                 add_datasets(100, 50),
#             ]
#             await asyncio.gather(*tasks)
#
#         print(f"Concurrently indexed: {buffer.total_indexed} datasets")
#
#     finally:
#         await vector_db.qdrant.close()
#
#
# if __name__ == "__main__":
#     # Run the async main function
#     asyncio.run(main())
#
#     # Or run the concurrent example
#     # asyncio.run(concurrent_example())

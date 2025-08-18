from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set, Dict, Any, Optional
import asyncio
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
import logging

from src.domain.repositories.dataset_repository import get_dataset_repository
from src.domain.services.dataset_buffer import DatasetDBBuffer
from src.infrastructure.mongo_db import get_mongo_database
from src.vector_search.vector_db import get_vector_db
from src.vector_search.vector_db_buffer import VectorDBBuffer

logger = logging.getLogger(__name__)


class BaseDataDownloader(ABC):
    """Abstract base class for async data downloaders"""

    # region INIT
    def __init__(
        self,
        output_dir: str = "data",
        max_workers: int = 20,
        delay: float = 0.05,
        is_embeddings: bool = False,
        is_store: bool = False,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 1,
    ):
        """
        Initialize base downloader

        Args:
            output_dir: Directory to save data
            max_workers: Number of parallel workers
            delay: Delay between requests in seconds
            is_embeddings: Whether to generate embeddings
            is_store: Whether to save datasets to DB or not
            connection_limit: Total connection pool size
            connection_limit_per_host: Per-host connection limit
            batch_size: Size of dataset batches to process
            max_retries: Maximum retry attempts for failed requests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.max_workers = max_workers
        self.delay = delay
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_embeddings = is_embeddings
        self.is_store = is_store

        # Connection configuration
        self.connection_limit = connection_limit
        self.connection_limit_per_host = connection_limit_per_host

        # TODO: Statistics
        # self.stats = {
        #     "datasets_found": 0,
        #     "datasets_processed": 0,
        #     "files_downloaded": 0,
        #     "errors": 0,
        #     "failed_datasets": set(),
        #     "start_time": datetime.now(),
        #     "cache_hits": 0,
        #     "retries": 0,
        # }
        self.stats_lock = asyncio.Lock()

        # Cache for metadata to avoid redundant API calls
        self.cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()

        # Track failed URLs for retry optimization
        self.failed_urls: Set[str] = set()
        self.failed_urls_lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry with optimized session"""
        # Create connector with connection pooling
        connector = TCPConnector(
            limit=self.connection_limit,
            limit_per_host=self.connection_limit_per_host,
            ttl_dns_cache=300,  # DNS cache for 5 minutes
            enable_cleanup_closed=True,
            force_close=True,
        )

        # Optimized timeout settings
        timeout = ClientTimeout(
            total=60,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=30,  # Socket read timeout
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "Data Downloader (Python/aiohttp)",
                "Accept-Encoding": "gzip, deflate",  # Enable compression
            },
        )

        # Call child class initialization if needed
        """Initialize Dresden-specific resources"""
        if self.is_embeddings:
            vector_db = await get_vector_db(use_grpc=True)
            self.vector_db_buffer = VectorDBBuffer(vector_db)

        if self.is_store:
            database = await get_mongo_database()
            dataset_db = await get_dataset_repository(database=database)
            self.dataset_db_buffer = DatasetDBBuffer(dataset_db)

        await self._initialize_resources()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Flush embeddings buffer if it exists
        if self.vector_db_buffer:
            try:
                await self.vector_db_buffer.flush()
            except Exception as e:
                logger.error(f"Error flushing VECTOR buffer: {e}")

        if self.dataset_db_buffer:
            try:
                await self.dataset_db_buffer.flush()
            except Exception as e:
                logger.error(f"Error flushing MONGO buffer: {e}")

        # Close session
        if self.session:
            await self.session.close()

        # Clean up child class resources first
        await self._cleanup_resources()

    async def _initialize_resources(self):
        """
        Hook for child classes to initialize their specific resources.
        Called during __aenter__.
        """
        pass

    async def _cleanup_resources(self):
        """
        Hook for child classes to clean up their specific resources.
        Called during __aexit__.
        """
        pass

    # endregion

    @abstractmethod
    async def process_all_datasets(self):
        """
        Abstract method to process all datasets.
        Must be implemented by child classes.
        """
        pass

    # region CACHE
    async def add_to_cache(self, key: str, value: Any):
        """
        Thread-safe cache addition

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self.cache_lock:
            self.cache[key] = value

    async def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Thread-safe cache retrieval

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        async with self.cache_lock:
            return self.cache.get(key)

    # endregion

    # region STATS
    # TODO: add STATS

    # async def update_stats(self, key: str, value: Any = 1, operation: str = "add"):
    #     """
    #     Thread-safe statistics update
    #
    #     Args:
    #         key: Statistics key to update
    #         value: Value to add/set
    #         operation: 'add' to increment, 'set' to replace
    #     """
    #     async with self.stats_lock:
    #         if operation == "add":
    #             if key in self.stats:
    #                 if isinstance(self.stats[key], (int, float)):
    #                     self.stats[key] += value
    #                 elif isinstance(self.stats[key], set):
    #                     self.stats[key].add(value)
    #         elif operation == "set":
    #             self.stats[key] = value

    # endregion

    # region FAILED URLS
    async def mark_url_failed(self, url: str):
        """
        Mark URL as failed for retry tracking

        Args:
            url: Failed URL
        """
        async with self.failed_urls_lock:
            self.failed_urls.add(url)

    async def is_url_failed(self, url: str) -> bool:
        """
        Check if URL has previously failed

        Args:
            url: URL to check

        Returns:
            True if URL has failed before
        """
        async with self.failed_urls_lock:
            return url in self.failed_urls

    # endregion

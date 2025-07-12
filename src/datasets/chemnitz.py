import asyncio
import io
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import aiohttp
import aiofiles
from aiohttp import ClientTimeout, TCPConnector

from src.datasets.datasets_metadata import (
    DatasetMetadataWithContent,
    DatasetJSONEncoder,
)
from src.utils.datasets_utils import sanitize_filename, safe_delete
from src.infrastructure.logger import get_logger
from src.utils.embeddings_utils import extract_data_content
from src.vector_search.vector_db import get_vector_db, VectorDB
from src.vector_search.vector_db_buffer import VectorDBBuffer

if TYPE_CHECKING:
    from _typeshed import SupportsWrite  # noqa: F401

logger = get_logger(__name__)


class ChemnitzDataDownloader:
    """Optimized async class for downloading Chemnitz open data"""

    def __init__(
        self,
        csv_file_path: str,
        output_dir: str = "chemnitz",
        max_workers: int = 20,
        delay: float = 0.05,
        is_embeddings: bool = False,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 1,
    ):
        """
        Initialize optimized downloader

        Args:
            csv_file_path: Path to CSV file with dataset metadata
            output_dir: Directory to save data
            max_workers: Number of parallel workers
            delay: Delay between requests in seconds
            is_embeddings: Whether to generate embeddings
            connection_limit: Total connection pool size
            connection_limit_per_host: Per-host connection limit
            batch_size: Size of dataset batches to process
            max_retries: Maximum retry attempts for failed requests
        """
        self.csv_file_path = csv_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.delay = delay
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.session = None

        # Connection configuration
        self.connection_limit = connection_limit
        self.connection_limit_per_host = connection_limit_per_host

        # Statistics
        self.stats = {
            "datasets_found": 0,
            "datasets_processed": 0,
            "files_downloaded": 0,
            "layers_downloaded": 0,
            "errors": 0,
            "failed_datasets": set(),
            "start_time": datetime.now(),
            "cache_hits": 0,
            "retries": 0,
        }
        self.stats_lock = asyncio.Lock()
        self.index_lock = asyncio.Lock()

        # Cache for service info to avoid redundant API calls
        self.service_cache = {}
        self.cache_lock = asyncio.Lock()

        # Track failed URLs for retry optimization
        self.failed_urls: Set[str] = set()
        self.failed_urls_lock = asyncio.Lock()

        self.is_embeddings = is_embeddings
        # Async VectorDB will be initialized in __aenter__
        self.vector_db: VectorDB | None = None
        self.vector_db_buffer: VectorDBBuffer | None = None

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
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Encoding": "gzip, deflate",  # Enable compression
            },
        )

        # Initialize async VectorDB if embeddings are enabled
        if self.is_embeddings:
            self.vector_db = await get_vector_db(use_grpc=True)
            self.vector_db_buffer = VectorDBBuffer(
                self.vector_db, buffer_size=100, auto_flush=True
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Flush embeddings buffer if it exists
        if self.vector_db_buffer:
            try:
                await self.vector_db_buffer.flush()
            except Exception as e:
                logger.error(f"Error flushing index buffer: {e}")

        if self.session:
            await self.session.close()

    async def load_datasets_metadata_from_csv(self) -> List[Dict[str, str]]:
        """Load dataset metadata from CSV file asynchronously"""
        datasets = []
        async with aiofiles.open(self.csv_file_path, "r", encoding="utf-8") as file:
            content = await file.read()

        # Parse CSV content
        csv_file = io.StringIO(content)
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row.get("url") and row.get("url").strip():
                datasets.append(
                    {
                        "title": row.get("title", "").strip(),
                        "url": row.get("url").strip(),
                        "type": row.get("type", "").strip(),
                        "description": row.get("description", "").strip(),
                    }
                )

        self.stats["datasets_found"] = len(datasets)
        return datasets

    async def update_stats(self, field: str, increment: int = 1):
        """Thread-safe statistics update"""
        async with self.stats_lock:
            if field == "failed_datasets":
                # Special handling for set
                return
            self.stats[field] += increment

    async def get_service_info(self, service_url: str) -> Optional[dict]:
        """Get service info with caching and retry logic"""
        # Check cache first
        async with self.cache_lock:
            if service_url in self.service_cache:
                await self.update_stats("cache_hits")
                return self.service_cache[service_url]

        # Check if URL previously failed
        async with self.failed_urls_lock:
            if service_url in self.failed_urls:
                return None

        info_url = f"{service_url}?f=json"

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(info_url) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Cache successful result
                    async with self.cache_lock:
                        self.service_cache[service_url] = data

                    return data

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(f"Error getting service info from {service_url}: {e}")
                    async with self.failed_urls_lock:
                        self.failed_urls.add(service_url)
                    return None
        return None

    async def download_layer_data(
        self,
        service_url: str,
        layer_id: int,
        layer_name: str,
        dataset_dir: Path,
    ) -> bool:
        """Download data for a single layer with optimized retry logic"""
        formats_to_try = [
            ("geojson", "json"),
            ("csv", "csv"),
        ]

        for format_name, file_ext in formats_to_try:
            query_url = f"{service_url}/{layer_id}/query"
            params = {
                "where": "1=1",
                "outFields": "*",
                "f": "geojson" if format_name == "geojson" else format_name,
                "returnGeometry": "true",
            }

            # Check if already downloaded
            file_name = f"{layer_name}.{file_ext}"
            file_path = dataset_dir / file_name
            if file_path.exists() and file_path.stat().st_size > 0:
                logger.debug(f"\t\tLayer already downloaded: {file_name}")
                return True

            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(query_url, params=params) as response:
                        if response.status == 200:
                            dataset_dir.mkdir(exist_ok=True)

                            # Create temporary file for atomic write
                            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

                            if format_name == "geojson":
                                try:
                                    data = await response.json()
                                    features = data.get("features", [])

                                    # Use aiofiles for async write
                                    async with aiofiles.open(
                                        temp_path, "w", encoding="utf-8"
                                    ) as f:
                                        await f.write(
                                            json.dumps(
                                                features, ensure_ascii=False, indent=2
                                            )
                                        )

                                    # Atomic rename
                                    temp_path.rename(file_path)

                                    logger.debug(f"\t\t✓ Saved as {file_name}")
                                    await self.update_stats("layers_downloaded")
                                    return True

                                except json.JSONDecodeError:
                                    if temp_path.exists():
                                        temp_path.unlink()
                                    continue
                            else:
                                content = await response.read()
                                async with aiofiles.open(temp_path, "wb") as f:
                                    await f.write(content)

                                # Atomic rename
                                temp_path.rename(file_path)

                                logger.debug(f"\t\t✓ Saved as {file_name}")
                                await self.update_stats("layers_downloaded")
                                return True

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await self.update_stats("retries")
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(
                            f"\t\tError downloading layer {layer_name} with format {format_name}: {e}"
                        )
                        continue

        logger.error(f"\t\t⚠ Couldn't download layer {layer_name}")
        return False

    async def download_feature_service_data(
        self,
        service_url: str,
        title: str,
        description: str = "",
    ) -> bool:
        """Download all data from a feature service with optimized concurrency"""
        await asyncio.sleep(self.delay)  # Minimal delay to respect server

        try:
            # Get service info
            service_info = await self.get_service_info(service_url)
            if not service_info:
                return False

            # Prepare folder
            safe_title = sanitize_filename(title)
            dataset_dir = self.output_dir / safe_title

            # Skip if already processed
            metadata_file = dataset_dir / "metadata.json"
            if metadata_file.exists():
                logger.debug(f"\tDataset already processed: {title}")
                await self.update_stats("datasets_processed")
                return True

            # Prepare metadata
            package_meta = DatasetMetadataWithContent(
                id=service_info.get("serviceItemId"),
                title=title,
                description=description,
                city="Chemnitz",
                state="Saxony",
                country="Germany",
            )

            # Get all features
            layers = service_info.get("layers", [])
            tables = service_info.get("tables", [])
            all_features = layers + tables

            if not all_features:
                logger.debug(f"\tNo layers to download in {title}")
                await self.update_stats("datasets_processed")
                return True

            logger.debug(f"\tProcessing dataset: {title} ({len(all_features)} layers)")

            # Download layers concurrently with limited concurrency
            layer_semaphore = asyncio.Semaphore(5)  # Limit concurrent layer downloads

            async def download_with_semaphore(feature):
                async with layer_semaphore:
                    layer_id = feature.get("id", 0)
                    layer_name = feature.get("name", f"layer_{layer_id}")
                    return await self.download_layer_data(
                        service_url, layer_id, layer_name, dataset_dir
                    )

            # Create download tasks
            download_tasks = [
                download_with_semaphore(feature) for feature in all_features
            ]

            # Wait for all downloads
            results = await asyncio.gather(*download_tasks, return_exceptions=True)

            # Count successful downloads
            success_count = sum(
                1
                for result in results
                if result is True and not isinstance(result, Exception)
            )

            if success_count > 0:
                logger.debug(
                    f"\tDownloaded {success_count}/{len(all_features)} layers for: {title}"
                )

                # Save metadata
                async with aiofiles.open(metadata_file, "w", encoding="utf-8") as f:
                    await f.write(
                        json.dumps(
                            package_meta,
                            indent=2,
                            ensure_ascii=False,
                            cls=DatasetJSONEncoder,
                        )
                    )

                await self.update_stats("files_downloaded")

                if self.is_embeddings and self.vector_db_buffer:
                    package_meta.content = await extract_data_content(dataset_dir)
                    async with self.index_lock:
                        await self.vector_db_buffer.add(package_meta)
            else:
                # Clean up empty dataset
                safe_delete(dataset_dir, logger)

            await self.update_stats("datasets_processed")
            return True

        except Exception as e:
            logger.error(f"\tError processing dataset {title}: {e}")
            async with self.stats_lock:
                self.stats["failed_datasets"].add(title)
            await self.update_stats("errors")
            return False

    async def process_dataset(self, metadata: Dict[str, str]) -> bool:
        """Process a single dataset"""
        title = metadata["title"]
        url = metadata["url"]
        dataset_type = metadata["type"]
        description = metadata.get("description", "")

        logger.debug(f"Processing: {title}")
        logger.debug(f"\tURL: {url}")

        try:
            if "Feature Service" == dataset_type:
                return await self.download_feature_service_data(url, title, description)
            else:
                logger.debug(f"\t⚠ Unknown type {dataset_type} for {title}")
                return False

        except Exception as e:
            logger.error(f"\t❌ Error processing {title}: {e}")
            async with self.stats_lock:
                self.stats["failed_datasets"].add(title)
            await self.update_stats("errors")
            return False

    async def print_progress(self):
        """Enhanced progress reporting"""
        async with self.stats_lock:
            processed = self.stats["datasets_processed"]
            total = self.stats["datasets_found"]
            files = self.stats["files_downloaded"]
            layers = self.stats["layers_downloaded"]
            errors = self.stats["errors"]
            cache_hits = self.stats["cache_hits"]
            retries = self.stats["retries"]

            if total > 0:
                percentage = (processed / total) * 100
                elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0

                logger.debug(
                    f"Progress: {processed}/{total} ({percentage:.1f}%) - "
                    f"Files: {files} - Layers: {layers} - Errors: {errors} - "
                    f"Cache hits: {cache_hits} - Retries: {retries} - "
                    f"Rate: {rate:.1f} datasets/s - ETA: {eta:.0f}s"
                )

    async def download_all_datasets(self):
        """Download all datasets with optimized batching and concurrency"""
        logger.info("[CHEMNITZ] Starting optimized Chemnitz Open Data download")

        # Load dataset metadata
        metadatas = await self.load_datasets_metadata_from_csv()
        if not metadatas:
            logger.error("No datasets found in CSV file")
            return

        logger.info(
            f"[CHEMNITZ] Found {len(metadatas)} datasets for download with {self.max_workers} workers"
        )
        logger.debug(f"Saving datasets to folder: {self.output_dir.absolute()}")
        logger.debug("-" * 50)

        # Progress reporting task
        async def progress_reporter():
            while True:
                await asyncio.sleep(5)  # Report every 5 seconds
                async with self.stats_lock:
                    if self.stats["datasets_processed"] >= self.stats["datasets_found"]:
                        break
                await self.print_progress()

        progress_task = asyncio.create_task(progress_reporter())

        # Process in batches to avoid overwhelming memory
        for i in range(0, len(metadatas), self.batch_size):
            batch = metadatas[i : i + self.batch_size]

            # Create semaphore for this batch
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(metadata: Dict[str, str]):
                async with semaphore:
                    return await self.process_dataset(metadata)

            # Process batch
            tasks = [process_with_semaphore(metadata) for metadata in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for metadata, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Exception in task for {metadata['title']}: {result}")
                    await self.update_stats("errors")

            logger.info(
                "[CHEMNITZ] "
                f"Completed batch {i // self.batch_size + 1}/"
                f"{(len(metadatas) + self.batch_size - 1) // self.batch_size}"
            )

        # Cancel progress reporter
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        if self.is_embeddings:
            await self.vector_db_buffer.flush()

        # Final statistics
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]

        logger.debug("-" * 50)
        logger.debug("=" * 60)
        logger.debug("DOWNLOAD STATISTICS")
        logger.debug("=" * 60)
        logger.debug(f"Datasets found: {self.stats['datasets_found']}")
        logger.debug(f"Datasets processed: {self.stats['datasets_processed']}")
        logger.debug(f"Files downloaded: {self.stats['files_downloaded']}")
        logger.debug(f"Layers downloaded: {self.stats['layers_downloaded']}")
        logger.debug(f"Errors: {self.stats['errors']}")
        logger.debug(f"Failed datasets: {len(self.stats['failed_datasets'])}")
        logger.debug(f"Cache hits: {self.stats['cache_hits']}")
        logger.debug(f"Retries: {self.stats['retries']}")
        logger.debug(f"Execution time: {duration}")
        logger.debug(
            f"Average time per dataset: {duration / max(1, self.stats['datasets_processed'])}"
        )
        logger.debug(f"Data saved to: {self.output_dir.absolute()}")


async def async_main():
    """Async main function with optimized settings"""
    csv_file = "open_data_portal_stadt_chemnitz.csv"
    if not Path(csv_file).exists():
        logger.error(f"❌ File {csv_file} not found!")
        logger.error("Make sure that CSV with datasets links is in the same folder.")
        return 1

    import argparse

    parser = argparse.ArgumentParser(description="Download Chemnitz open data")
    parser.add_argument(
        "--output",
        "-o",
        default="./chemnitz",
        help="Output directory for downloaded datasets (default: ./chemnitz)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=20,
        help="Number of parallel workers (default: 20)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.05,
        help="Delay between requests in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--connection-limit",
        type=int,
        default=100,
        help="Total connection pool size (default: 100)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retry attempts for failed requests (default: 1)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        async with ChemnitzDataDownloader(
            csv_file,
            output_dir=args.output,
            max_workers=args.max_workers,
            delay=args.delay,
            is_embeddings=True,
            connection_limit=args.connection_limit,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
        ) as downloader:
            await downloader.download_all_datasets()
        return 0

    except KeyboardInterrupt:
        logger.warning("⚠️ Download interrupted by user")
        return 130  # Standard for Ctrl+C

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Synchronous entry point"""
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())

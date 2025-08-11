import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

import aiohttp
import aiofiles
from aiohttp import ClientTimeout, TCPConnector

from src.datasets.datasets_metadata import (
    DatasetJSONEncoder,
    DatasetMetadataWithContent,
)
from src.domain.repositories.dataset_repository import get_dataset_repository
from src.domain.services.dataset_buffer import DatasetDBBuffer
from src.infrastructure.logger import get_prefixed_logger
from src.infrastructure.mongo_db import get_mongo_database
from src.utils.datasets_utils import sanitize_filename, safe_delete
from src.utils.embeddings_utils import extract_data_content
from src.utils.file import save_file_with_task
from src.vector_search.vector_db import get_vector_db
from src.vector_search.vector_db_buffer import VectorDBBuffer

logger = get_prefixed_logger(__name__, "DRESDEN")


class DresdenOpenDataDownloader:
    """Optimized async class for downloading Dresden open data"""

    def __init__(
        self,
        output_dir: str = "dresden",
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
        Initialize optimized downloader

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
        self.base_url = "https://register.opendata.sachsen.de"
        self.search_endpoint = f"{self.base_url}/store/search"
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
            "errors": 0,
            "failed_datasets": set(),
            "start_time": datetime.now(),
            "cache_hits": 0,
            "retries": 0,
        }
        self.stats_lock = asyncio.Lock()

        # Cache for metadata to avoid redundant API calls
        self.metadata_cache = {}
        self.cache_lock = asyncio.Lock()

        # Track failed URLs for retry optimization
        self.failed_urls: Set[str] = set()
        self.failed_urls_lock = asyncio.Lock()

        # Async VectorDB & Dataset will be initialized in __aenter__
        self.is_embeddings = is_embeddings
        self.is_store = is_store
        self.vector_db_buffer: VectorDBBuffer | None = None
        self.dataset_db_buffer: DatasetDBBuffer | None = None

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
                "User-Agent": "Dresden OpenData Downloader (Python/aiohttp)",
                "Accept-Encoding": "gzip, deflate",  # Enable compression
            },
        )

        if self.is_embeddings:
            vector_db = await get_vector_db(use_grpc=True)
            self.vector_db_buffer = VectorDBBuffer(vector_db)

        if self.is_store:
            database = await get_mongo_database()
            dataset_db = await get_dataset_repository(database=database)
            self.dataset_db_buffer: DatasetDBBuffer | None = DatasetDBBuffer(dataset_db)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

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

    async def update_stats(self, field: str, increment: int = 1):
        """Thread-safe statistics update"""
        async with self.stats_lock:
            if field == "failed_datasets":
                # Special handling for set
                return
            self.stats[field] += increment

    async def search_dresden_datasets(self, limit: int = 100, offset: int = 0) -> Dict:
        """
        Search Dresden datasets via API with retry logic

        Args:
            limit: Number of results per request (max 100)
            offset: Offset for pagination

        Returns:
            API response with datasets
        """
        params = {
            "type": "solr",
            "query": "rdfType:http\\://www.w3.org/ns/dcat#Dataset AND public:true AND resource:*dresden*",
            "limit": min(limit, 100),
            "offset": offset,
            "sort": "modified desc",
        }

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Searching datasets: offset={offset}, limit={limit}")
                async with self.session.get(
                    self.search_endpoint, params=params
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(f"Error searching datasets: {e}")
                    return {}
        return {}

    async def get_dataset_metadata(
        self, context_id: str, entry_id: str
    ) -> Optional[Dict]:
        """
        Get metadata for specific dataset with caching

        Args:
            context_id: Context ID
            entry_id: Entry ID

        Returns:
            Dataset metadata
        """
        cache_key = f"{context_id}/{entry_id}"

        # Check cache first
        async with self.cache_lock:
            if cache_key in self.metadata_cache:
                await self.update_stats("cache_hits")
                return self.metadata_cache[cache_key]

        # Try different URL variants to get metadata
        urls_to_try = [
            f"{self.base_url}/store/{context_id}/metadata/{entry_id}",
            f"{self.base_url}/store/{context_id}/entry/{entry_id}",
            f"{self.base_url}/store/{context_id}/resource/{entry_id}",
        ]

        for url in urls_to_try:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        metadata = await response.json()
                        # Cache successful result
                        async with self.cache_lock:
                            self.metadata_cache[cache_key] = metadata
                        return metadata
            except Exception:
                continue

        logger.warning(f"Could not get additional metadata for {context_id}/{entry_id}")
        return None

    async def extract_download_urls(
        self, dataset_metadata: Dict, dataset_uri: str
    ) -> List[Dict]:
        """
        Extract download URLs from metadata using direct content.csv approach

        Args:
            dataset_metadata: Dataset metadata
            dataset_uri: URI of the dataset

        Returns:
            List of dictionaries with download file information
        """
        downloads = []

        if not dataset_uri:
            logger.warning("No dataset URI provided")
            return downloads

        # Formats to try in priority order
        format_attempts = [
            {"suffix": "/content.json", "format": "application/json", "ext": ".json"},
            {
                "suffix": "/content.json",
                "format": "application/json",
                "ext": ".geojson",
            },
            {"suffix": "/content.csv", "format": "text/csv", "ext": ".csv"},
            # {"suffix": "/content.xml", "format": "application/xml", "ext": ".xml"},
            # {
            #     "suffix": "/content.xlsx",
            #     "format": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            #     "ext": ".xlsx",
            # },
        ]

        # Check each format concurrently
        check_tasks = []
        for format_info in format_attempts:
            url = f"{dataset_uri}{format_info['suffix']}"
            task = self._check_url_availability(url, format_info)
            check_tasks.append(task)

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Add successful results
        for result in results:
            if isinstance(result, dict) and result:
                downloads.append(result)

        # Fallback: try to extract from distribution metadata if no direct files found
        if not downloads:
            logger.debug("No direct content files found, trying distribution metadata")
            downloads = self.extract_from_distributions(dataset_metadata)

        logger.debug(f"Found {len(downloads)} files for download")
        return downloads

    async def _check_url_availability(
        self, url: str, format_info: Dict
    ) -> Optional[Dict]:
        """Check if a URL is available and return download info"""
        try:
            logger.debug(f"Checking availability: {url}")
            async with self.session.head(
                url, timeout=ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return {
                        "url": url,
                        "title": "content",
                        "format": format_info["format"],
                        "extension": format_info["ext"],
                    }
        except Exception:
            pass
        return None

    @staticmethod
    def extract_from_distributions(metadata: Dict) -> List[Dict]:
        """
        Fallback method to extract download URLs from distribution metadata

        Args:
            metadata: Dataset metadata

        Returns:
            List of download information
        """
        downloads = []

        if not metadata:
            return downloads

        logger.debug(f"Extracting URLs from metadata with {len(metadata)} entries")

        # Search for distributions in metadata
        for subject, predicates in metadata.items():
            logger.debug(f"Checking subject: {subject}")

            # Look for dcat:distribution
            distributions = predicates.get("http://www.w3.org/ns/dcat#distribution", [])
            logger.debug(f"Found distributions: {len(distributions)}")

            for dist in distributions:
                if dist.get("type") == "uri":
                    dist_uri = dist.get("value")
                    logger.debug(f"Processing distribution: {dist_uri}")

                    # Get distribution information
                    dist_info = metadata.get(dist_uri, {})

                    # Look for download URL
                    download_url = None
                    access_urls = dist_info.get(
                        "http://www.w3.org/ns/dcat#downloadURL", []
                    )
                    if not access_urls:
                        access_urls = dist_info.get(
                            "http://www.w3.org/ns/dcat#accessURL", []
                        )

                    if access_urls and access_urls[0].get("type") == "uri":
                        download_url = access_urls[0].get("value")
                        logger.debug(f"Found download URL: {download_url}")

                    # Look for file format
                    format_info = dist_info.get("http://purl.org/dc/terms/format", [])
                    media_type = dist_info.get(
                        "http://www.w3.org/ns/dcat#mediaType", []
                    )

                    file_format = None
                    if format_info and format_info[0].get("value"):
                        file_format = format_info[0]["value"]
                    elif media_type and media_type[0].get("value"):
                        file_format = media_type[0]["value"]

                    # Look for title
                    title = dist_info.get("http://purl.org/dc/terms/title", [])
                    file_title = (
                        title[0].get("value", "untitled") if title else "untitled"
                    )

                    # Determine file extension
                    extension = ""
                    if file_format:
                        format_lower = file_format.lower()
                        if "csv" in format_lower:
                            extension = ".csv"
                        elif "json" in format_lower:
                            extension = ".json"
                        # elif "xml" in format_lower:
                        #     extension = ".xml"
                        # elif "xlsx" in format_lower or "excel" in format_lower:
                        #     extension = ".xlsx"

                    if download_url:
                        downloads.append(
                            {
                                "url": download_url,
                                "title": file_title,
                                "format": file_format,
                                "extension": extension,
                                "distribution_uri": dist_uri,
                            }
                        )
                        logger.debug(f"Added download file: {file_title}")

        return downloads

    async def download_file(self, url: str, filename: str, dataset_dir: Path) -> bool:
        """
        Download file with optimized async streaming

        Args:
            url: File URL
            filename: Filename to save
            dataset_dir: Dataset directory

        Returns:
            True if file successfully downloaded
        """
        filepath = dataset_dir / filename

        # Check if file already exists
        if filepath.exists() and filepath.stat().st_size > 0:
            logger.debug(f"File already exists: {filepath}")
            return True

        # Check if URL previously failed
        async with self.failed_urls_lock:
            if url in self.failed_urls:
                return False

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Downloading: {url}")
                async with self.session.get(url) as response:
                    response.raise_for_status()

                    # Check content type and length
                    content_type = response.headers.get("content-type", "").lower()
                    content_length = response.headers.get("content-length")

                    if (
                        content_length
                        and int(content_length) < 100
                        and "html" in content_type
                    ):
                        logger.warning(f"Response appears to be an error page: {url}")
                        async with self.failed_urls_lock:
                            self.failed_urls.add(url)
                        return False

                    # Create temporary file for atomic write
                    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")

                    # Download with streaming
                    async with aiofiles.open(temp_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(
                            65536
                        ):  # 64KB chunks
                            await f.write(chunk)

                    # Verify file is not empty
                    if temp_path.stat().st_size == 0:
                        logger.warning(f"Downloaded file is empty: {filepath}")
                        temp_path.unlink()
                        async with self.failed_urls_lock:
                            self.failed_urls.add(url)
                        return False

                    # Atomic rename
                    temp_path.rename(filepath)

                    logger.debug(
                        f"File saved: {filepath} ({filepath.stat().st_size} bytes)"
                    )
                    await self.update_stats("files_downloaded")
                    return True

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)
                else:
                    logger.error(f"Error downloading {url}: {e}")
                    await self.update_stats("errors")
                    async with self.failed_urls_lock:
                        self.failed_urls.add(url)
                    return False
        return False

    async def process_dataset(self, dataset_info: Dict) -> bool:
        """Process a single dataset with optimized async operations"""
        await asyncio.sleep(self.delay)  # Minimal delay to respect server

        context_id = dataset_info.get("contextId")
        entry_id = dataset_info.get("entryId")

        if not context_id or not entry_id:
            logger.warning("Missing contextId or entryId")
            return False

        # Check if already processed
        safe_key = f"{context_id}_{entry_id}"
        existing_dirs = [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith(safe_key)
        ]
        if existing_dirs:
            logger.debug(f"Dataset already processed: {safe_key}")
            await self.update_stats("datasets_processed")
            return True

        # Use metadata from dataset_info
        dataset_metadata = dataset_info.get("metadata", {})

        if not dataset_metadata:
            logger.warning(f"Missing metadata for dataset {context_id}/{entry_id}")
            return False

        # Extract dataset information
        title = "Unknown Dataset"
        dataset_uri = None
        keywords = []
        description = None

        # Find dataset URI, title, keywords, and description
        for uri, predicates in dataset_metadata.items():
            if (
                uri.startswith("http")
                and "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in predicates
            ):
                rdf_types = predicates[
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                ]
                if any(
                    t.get("value") == "http://www.w3.org/ns/dcat#Dataset"
                    for t in rdf_types
                ):
                    dataset_uri = uri

                    # Extract title
                    title_info = predicates.get("http://purl.org/dc/terms/title", [])
                    if title_info:
                        title = title_info[0].get("value", title)

                    # Extract keywords as an array
                    keyword_info = predicates.get(
                        "http://www.w3.org/ns/dcat#keyword", []
                    )
                    keywords = [
                        kw.get("value") for kw in keyword_info if kw.get("value")
                    ]

                    # Extract description
                    description_info = predicates.get(
                        "http://purl.org/dc/terms/description", []
                    )
                    if description_info:
                        description = description_info[0].get("value")

                    break

        if not dataset_uri:
            logger.warning(f"Dataset URI not found in metadata {context_id}/{entry_id}")
            return False

        # Directory creation
        safe_title = sanitize_filename(title)
        dataset_dir = self.output_dir / f"{context_id}_{entry_id}_{safe_title}"
        dataset_dir.mkdir(exist_ok=True)

        # Prepare metadata
        package_meta = DatasetMetadataWithContent(
            id=f"{context_id}/{entry_id}",
            url=dataset_uri,
            title=title,
            description=description,
            tags=keywords,
            city="Dresden",
            state="Saxony",
            country="Germany",
        )

        logger.debug(f"Processing dataset: {title}")
        logger.debug(f"Dataset URI: {dataset_uri}")

        # Extract download links
        downloads = await self.extract_download_urls(dataset_metadata, dataset_uri)

        if not downloads:
            logger.debug(f"No download files found for dataset: {title}")
            # Still save metadata even if no downloads
            metadata_file = dataset_dir / "metadata.json"
            content = json.dumps(
                package_meta,
                indent=2,
                ensure_ascii=False,
                cls=DatasetJSONEncoder,
            )
            save_file_with_task(metadata_file, content)
            await self.update_stats("datasets_processed")
            return True

        # Sort downloads to prioritize JSON files
        json_downloads = [d for d in downloads if d.get("extension") == ".json"]
        other_downloads = [d for d in downloads if d.get("extension") != ".json"]
        sorted_downloads = json_downloads + other_downloads

        # Download files with limited concurrency
        download_semaphore = asyncio.Semaphore(
            3
        )  # Limit concurrent downloads per dataset

        async def download_with_semaphore(download_info):
            async with download_semaphore:
                url = download_info["url"]
                file_title = download_info.get("title", "file")
                extension = download_info.get("extension", "")
                filename = sanitize_filename(f"{file_title}{extension}")
                return await self.download_file(url, filename, dataset_dir)

        # Try to download files
        success = False
        for download_info in sorted_downloads:
            if await download_with_semaphore(download_info):
                success = True
                logger.debug(
                    f"Successfully downloaded {download_info.get('extension')} file"
                )
                break  # Stop after first successful download

        if success:
            # Save metadata
            metadata_file = dataset_dir / "metadata.json"
            content = json.dumps(
                package_meta,
                indent=2,
                ensure_ascii=False,
                cls=DatasetJSONEncoder,
            )
            save_file_with_task(metadata_file, content)

            package_meta.content = await extract_data_content(dataset_dir)

            if self.is_embeddings and self.vector_db_buffer:
                await self.vector_db_buffer.add(package_meta)

            if self.is_store and self.dataset_db_buffer:
                await self.dataset_db_buffer.add(package_meta)
        else:
            # Clean up empty dataset
            safe_delete(dataset_dir, logger)

        await self.update_stats("datasets_processed")
        return success

    async def collect_all_datasets(self) -> List[Dict]:
        """Collect all datasets from the API"""
        all_datasets = []
        offset = 0
        limit = 100

        while True:
            # Search for datasets
            search_result = await self.search_dresden_datasets(
                limit=limit, offset=offset
            )

            if not search_result or "resource" not in search_result:
                logger.warning("Empty response from API or invalid format")
                break

            children = search_result["resource"].get("children", [])
            total_results = search_result.get("results", 0)

            if not children:
                logger.debug("No more datasets found")
                break

            if offset == 0:
                self.stats["datasets_found"] = total_results
                logger.debug(f"Total datasets found: {total_results}")

            all_datasets.extend(children)

            # Move to next page
            offset += limit
            logger.debug(f"Collected datasets: {len(all_datasets)} of {total_results}")

            # If we got fewer results than requested, this is the last page
            if len(children) < limit:
                break

        return all_datasets

    async def print_progress(self):
        """Enhanced progress reporting"""
        async with self.stats_lock:
            processed = self.stats["datasets_processed"]
            total = self.stats["datasets_found"]
            files = self.stats["files_downloaded"]
            errors = self.stats["errors"]
            cache_hits = self.stats["cache_hits"]
            retries = self.stats["retries"]

            if total > 0:
                percentage = (processed / total) * 100
                elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {processed}/{total} ({percentage:.1f}%)\t"
                    f"Files: {files}\tErrors: {errors}\t"
                    f"Cache hits: {cache_hits}\tRetries: {retries}\t"
                    f"Rate: {rate:.1f} datasets/s\tETA: {eta:.0f}s"
                )

    async def download_all_datasets(self):
        """Download all datasets with optimized async processing"""
        logger.info("Starting optimized Dresden Open Data download")

        # First, collect all datasets
        all_datasets = await self.collect_all_datasets()

        if not all_datasets:
            logger.error("No datasets found")
            return

        logger.debug(
            f"Starting download of {len(all_datasets)} datasets with {self.max_workers} workers"
        )

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
        for i in range(0, len(all_datasets), self.batch_size):
            batch = all_datasets[i : i + self.batch_size]

            # Create semaphore for this batch
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(dataset: Dict):
                async with semaphore:
                    try:
                        return await self.process_dataset(dataset)
                    except Exception as e:
                        logger.error(f"Error processing dataset: {e}")
                        await self.update_stats("errors")
                        return False

            # Process batch
            tasks = [process_with_semaphore(dataset) for dataset in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.debug(
                f"Completed batch {i // self.batch_size + 1}/"
                f"{(len(all_datasets) + self.batch_size - 1) // self.batch_size}"
            )

        # Cancel progress reporter
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        if self.is_embeddings:
            await self.vector_db_buffer.flush()
        if self.is_store:
            await self.dataset_db_buffer.flush()

        logger.info("🎉 Download completed!")

        # Final statistics
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]

        logger.debug("=" * 60)
        logger.debug("DOWNLOAD STATISTICS")
        logger.debug("=" * 60)
        logger.debug(f"Datasets found: {self.stats['datasets_found']}")
        logger.debug(f"Datasets processed: {self.stats['datasets_processed']}")
        logger.debug(f"Files downloaded: {self.stats['files_downloaded']}")
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
    import argparse

    parser = argparse.ArgumentParser(description="Download Dresden open data")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dresden",
        help="Directory to save data (default: dresden)",
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
        async with DresdenOpenDataDownloader(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            delay=args.delay,
            batch_size=args.batch_size,
            connection_limit=args.connection_limit,
            max_retries=args.max_retries,
            is_embeddings=True,
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

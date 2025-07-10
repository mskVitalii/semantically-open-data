import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
import aiofiles
from playwright.async_api import async_playwright

from src.datasets.datasets_metadata import (
    DatasetJSONEncoder,
    DatasetMetadataWithContent,
)
from src.infrastructure.logger import get_logger
from src.utils.datasets_utils import (
    allowed_extensions,
    allowed_formats,
    safe_delete,
    sanitize_filename,
    skip_formats,
)
from src.utils.embeddings_utils import extract_data_content
from src.vector_search.vector_db import VectorDB
from src.vector_search.vector_db_buffer import VectorDBBuffer

if TYPE_CHECKING:
    from _typeshed import SupportsWrite  # noqa: F401

logger = get_logger(__name__)


class BerlinOpenDataDownloader:
    """Async class for downloading Berlin open data with parallel processing"""

    def __init__(
        self,
        output_dir: str = "berlin",
        max_workers: int = 5,
        delay: float = 0.2,
        is_embeddings: bool = False,
    ):
        """
        Initialize downloader

        Args:
            output_dir: Directory to save data
            max_workers: Number of parallel workers
            delay: Delay between requests in seconds
        """
        self.base_url = "https://datenregister.berlin.de"
        self.api_url = f"{self.base_url}/api/3/action"
        self.output_dir = Path(__file__).parent / Path(output_dir)
        self.max_workers = max_workers
        self.delay = delay
        self.session = None  # Will be created in async context

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe statistics (asyncio.Lock for async)
        self.stats = {
            "datasets_found": 0,
            "datasets_processed": 0,
            "files_downloaded": 0,
            "errors": 0,
            "failed_datasets": set(),  # Track failed datasets for cleanup
            "start_time": datetime.now(),
        }
        self.stats_lock = asyncio.Lock()
        self.index_lock = asyncio.Lock()
        self.is_embeddings = is_embeddings
        if self.is_embeddings:
            vector_db = VectorDB(use_grpc=True)
            self.index_buffer = VectorDBBuffer(
                vector_db, buffer_size=100, auto_flush=True
            )

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": "Berlin OpenData Downloader (Python/aiohttp)"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def update_stats(self, field: str, increment: int = 1):
        """Thread-safe statistics update"""
        async with self.stats_lock:
            self.stats[field] += increment

    async def get_all_packages(self) -> List[str]:
        """
        Get list of all package names from CKAN API

        Returns:
            List of package names
        """
        try:
            logger.info("Fetching list of all datasets...")
            async with self.session.get(f"{self.api_url}/package_list") as response:
                response.raise_for_status()
                data = await response.json()

                if data.get("success"):
                    packages = data.get("result", [])
                    self.stats["datasets_found"] = len(packages)
                    logger.info(f"Found {len(packages)} datasets")
                    return packages
                else:
                    logger.error(f"API error: {data.get('error')}")
                    return []

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching package list: {e}")
            return []

    async def get_package_details(self, package_name: str) -> Optional[Dict]:
        """
        Get detailed information about a package

        Args:
            package_name: Name of the package

        Returns:
            Package details or None
        """
        try:
            async with self.session.get(
                f"{self.api_url}/package_show", params={"id": package_name}
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if data.get("success"):
                    return data.get("result")
                else:
                    logger.warning(f"Package not found or error: {package_name}")
                    return None

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching package details for {package_name}: {e}")
            return None

    @staticmethod
    def get_file_extension(url: str, format_hint: str = None) -> str:
        """
        Determine file extension from URL or format hint

        Args:
            url: Resource URL
            format_hint: Format hint from metadata

        Returns:
            File extension with dot
        """
        # First try to get extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        if path.endswith(".csv"):
            return ".csv"
        elif path.endswith(".json"):
            return ".json"
        elif path.endswith(".xml"):
            return ".xml"
        elif path.endswith(".xlsx"):
            return ".xlsx"
        elif path.endswith(".xls"):
            return ".xls"
        elif path.endswith(".pdf"):
            return ".pdf"
        elif path.endswith(".txt"):
            return ".txt"

        # Try format hint
        if format_hint:
            format_lower = format_hint.lower()
            if "csv" in format_lower:
                return ".csv"
            elif "json" in format_lower:
                return ".json"
            elif "xml" in format_lower:
                return ".xml"
            elif "xlsx" in format_lower or "excel" in format_lower:
                return ".xlsx"
            elif "pdf" in format_lower:
                return ".pdf"

        return ""

    @staticmethod
    def should_skip_resource(resource: Dict) -> bool:
        """
        Check if resource should be skipped based on format

        Args:
            resource: Resource dictionary

        Returns:
            True if resource should be skipped
        """
        url = resource.get("url", "").lower()
        format_hint = resource.get("format", "").lower()

        # Skip Atom feeds
        if "atom" in format_hint or url.endswith(".atom"):
            return True

        # Skip WMS/WFS services
        if any(service in url for service in ["wms", "wfs", "wmts"]):
            return True

        if format_hint in skip_formats:
            return True

        # Check if format is explicitly allowed
        if format_hint and any(fmt in format_hint for fmt in allowed_formats):
            return False

        # Check if URL has allowed extension
        if any(url.endswith(ext) for ext in allowed_extensions):
            return False

        # Skip everything else
        return True

    async def download_file(self, url: str, filepath: Path) -> bool:
        """
        Download file from URL

        Args:
            url: File URL
            filepath: Path to save file

        Returns:
            True if file successfully downloaded
        """
        # Check if file already exists
        if filepath.exists():
            logger.debug(f"File already exists: {filepath.name}")
            return True

        try:
            logger.debug(f"Downloading: {url}")

            if url.lower().endswith("/") or url.lower().startswith(
                "https://www.statistik-berlin-brandenburg.de"
            ):
                # Playwright for pages without extension
                logger.info(
                    f"URL without file extension detected, using Playwright: {url}"
                )
                return await self.download_file_playwright(url, filepath)

            async with self.session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if "html" in content_type and response.headers.get("content-length"):
                    content_length = int(response.headers.get("content-length", 0))
                    if content_length < 1000:  # Likely an error page
                        logger.warning(
                            f"Response appears to be HTML error page, skipping: {url}"
                        )
                        return False

                async with aiofiles.open(filepath, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            await f.write(chunk)

            # Check if downloaded file is not empty
            if filepath.stat().st_size == 0:
                logger.warning(f"Downloaded file is empty, removing: {filepath}")
                filepath.unlink()
                return False

            logger.debug(
                f"Downloaded: {filepath.name} ({filepath.stat().st_size} bytes)"
            )
            await self.update_stats("files_downloaded")
            return True

        except aiohttp.ClientError as e:
            logger.error(f"Error downloading {url}: {e}")
            await self.update_stats("errors")
            return False

    @staticmethod
    async def download_file_playwright(url: str, filepath: Path) -> bool:
        """
        Download file using Playwright for JavaScript-rendered pages
        """
        if filepath.exists():
            logger.debug(f"File already exists: {filepath.name}")
            return True

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                accept_downloads=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            page = await context.new_page()

            try:
                async with page.expect_download() as download_info:
                    await page.goto(url)
                    # Waiting max 5 seconds
                    download = await download_info.value

                await download.save_as(filepath)

                logger.debug(f"Downloaded via Playwright: {filepath.name}")
                await browser.close()
                return True

            except Exception as e:
                logger.error(f"Playwright error: {e}")
                await browser.close()
                return False

    async def process_dataset(self, package_name: str) -> bool:
        """
        Process and download one dataset

        Args:
            package_name: Name of the package to process

        Returns:
            True if dataset successfully processed
        """
        # Add delay to avoid overwhelming the server
        await asyncio.sleep(self.delay)

        # =============== METADATA
        try:
            package = await self.get_package_details(package_name)
            if not package:
                await self.update_stats("errors")
                return False

            meta_id = package.get("id")
            title = package.get("title", package_name)
            resources = package.get("resources", [])

            # Create dataset directory
            safe_title = sanitize_filename(title)
            dataset_dir = self.output_dir / f"{package_name}_{safe_title}"
            dataset_dir.mkdir(exist_ok=True)

            if not resources:
                logger.debug(f"No resources found for dataset: {title}")
                await self.update_stats("datasets_processed")
                return True
        except Exception as e:
            logger.error(
                f"Error downloading METADATA of the dataset {package_name}: {e}"
            )
            async with self.stats_lock:
                self.stats["failed_datasets"].add(package_name)

            await self.update_stats("errors")
            return False

        logger.debug(f"Processing dataset: {title} ({len(resources)} resources)")

        # =============== DATASET Process resources
        try:
            success_count = 0
            for i, resource in enumerate(resources):
                url = resource.get("url")
                if not url:
                    continue

                # Skip non-CSV/JSON resources
                if self.should_skip_resource(resource):
                    logger.debug(
                        f"Skipping resource {i}: {resource.get('format', 'unknown')} format"
                    )
                    continue

                # Determine filename
                resource_name = resource.get("name", f"resource_{i}")
                resource_format = resource.get("format", "")
                extension = self.get_file_extension(url, resource_format)

                filename = sanitize_filename(f"{resource_name}_{i}{extension}")
                filepath = dataset_dir / filename

                if await self.download_file(url, filepath):
                    success_count += 1
                    break

            if success_count > 0:
                logger.info(f"Downloaded {success_count} files for: {title}")
                package_meta = DatasetMetadataWithContent(
                    id=meta_id,
                    title=title,
                    groups=[group.get("title") for group in package.get("groups", [])],
                    organization=package.get("organization", {}).get("title"),
                    tags=[tag.get("name") for tag in package.get("tags", [])],
                    description=package.get("notes"),
                    city="Berlin",
                    state="Saxony",
                    country="Germany",
                )

                async with aiofiles.open(
                    dataset_dir / "metadata.json", "w", encoding="utf-8"
                ) as f:
                    await f.write(
                        json.dumps(
                            package_meta,
                            indent=2,
                            ensure_ascii=False,
                            cls=DatasetJSONEncoder,
                        )
                    )

                if self.is_embeddings:
                    package_meta.content = extract_data_content(dataset_dir)
                    async with self.index_lock:
                        self.index_buffer.add(package_meta)

            else:
                # not suitable dataset
                safe_delete(dataset_dir, logger)

            await self.update_stats("datasets_processed")
            return True
        except Exception as e:
            logger.error(f"Error processing DATASET {package_name}: {e}")
            # Mark as failed and clean up
            async with self.stats_lock:
                self.stats["failed_datasets"].add(package_name)

            safe_delete(dataset_dir, logger)
            logger.debug(f"Cleaned up failed dataset directory: {dataset_dir}")

            await self.update_stats("errors")
            return False

    async def print_progress(self):
        """Print current progress"""
        async with self.stats_lock:
            processed = self.stats["datasets_processed"]
            total = self.stats["datasets_found"]
            files = self.stats["files_downloaded"]
            errors = self.stats["errors"]

            if total > 0:
                percentage = (processed / total) * 100
                logger.info(
                    f"Progress: {processed}/{total} ({percentage:.1f}%) - Files: {files} - Errors: {errors}"
                )

    async def download_all_datasets(self):
        """Download all Berlin datasets using parallel processing"""
        logger.info("Starting Berlin Open Data download with async parallel processing")

        # Get list of all packages
        packages = await self.get_all_packages()
        if not packages:
            logger.error("No packages found or error fetching package list")
            return

        logger.info(
            f"Starting download of {len(packages)} datasets with {self.max_workers} workers"
        )

        # Progress reporting task
        async def progress_reporter():
            while True:
                await asyncio.sleep(10)  # Report every 10 seconds
                async with self.stats_lock:
                    if self.stats["datasets_processed"] >= self.stats["datasets_found"]:
                        break
                await self.print_progress()

        # Create progress reporter task
        progress_task = asyncio.create_task(progress_reporter())

        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(package_name: str):
            async with semaphore:
                return await self.process_dataset(package_name)

        # Process datasets concurrently
        tasks = [process_with_semaphore(package) for package in packages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        for package, result in zip(packages, results):
            if isinstance(result, Exception):
                logger.error(f"Exception in task for {package}: {result}")
                await self.update_stats("errors")

        # Cancel progress reporter
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        if self.is_embeddings:
            self.index_buffer.flush()

        # Final statistics
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]

        logger.info("=" * 60)
        logger.info("DOWNLOAD STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Datasets found: {self.stats['datasets_found']}")
        logger.info(f"Datasets processed: {self.stats['datasets_processed']}")
        logger.info(f"Files downloaded: {self.stats['files_downloaded']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Failed datasets cleaned up: {len(self.stats['failed_datasets'])}")
        logger.info(f"Execution time: {duration}")
        logger.info(
            f"Average time per dataset: {duration / max(1, self.stats['datasets_processed'])}"
        )
        logger.info(f"Data saved to: {self.output_dir.absolute()}")


async def main():
    """Main async function"""
    import argparse

    parser = argparse.ArgumentParser(description="Download Berlin open data")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="berlin",
        help="Directory to save data (default: berlin)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.2,
        help="Delay between requests in seconds (default: 0.2)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        async with BerlinOpenDataDownloader(
            output_dir=args.output_dir, max_workers=args.max_workers, delay=args.delay
        ) as downloader:
            await downloader.download_all_datasets()

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

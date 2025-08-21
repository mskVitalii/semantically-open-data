import asyncio
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import aiohttp
import aiofiles
import pandas as pd
from playwright.async_api import async_playwright, ViewportSize, Error

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithContent,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import (
    safe_delete,
    sanitize_filename,
    skip_formats,
)
from src.utils.file import save_file_with_task


class Berlin(BaseDataDownloader):
    """Optimized async class for downloading Berlin open data"""

    # region INIT

    def __init__(
        self,
        output_dir: str = "berlin",
        max_workers: int = 128,
        delay: float = 0.05,
        is_file_system: bool = True,
        is_embeddings: bool = False,
        is_store: bool = False,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
    ):
        """
        Initialize optimized downloader

        Args:
            output_dir: Directory to save data
            max_workers: Number of parallel workers
            delay: Delay between requests in seconds
            is_file_system: Whether to save datasets to filesystem
            is_embeddings: Whether to generate embeddings
            is_store: Whether to save datasets to DB or not
            connection_limit: Total connection pool size
            connection_limit_per_host: Per-host connection limit
            batch_size: Size of package batches to process
        """
        super().__init__(
            output_dir=output_dir,
            max_workers=max_workers,
            delay=delay,
            is_file_system=is_file_system,
            is_embeddings=is_embeddings,
            is_store=is_store,
            connection_limit=connection_limit,
            connection_limit_per_host=connection_limit_per_host,
            batch_size=batch_size,
        )
        self.base_url = "https://datenregister.berlin.de"
        self.api_url = f"{self.base_url}/api/3/action"
        self.logger = get_prefixed_logger(__name__, "BERLIN")
        self.stats["playwright_downloads"] = 0

        # Track domains that require Playwright
        self.playwright_domains: Set[str] = set()
        self.playwright_lock = asyncio.Lock()

        # Playwright browser instance
        self.browser = None
        self.browser_lock = asyncio.Lock()

    async def _cleanup_resources(self):
        if self.browser:
            await self.browser.close()

    # endregion

    # region STATS
    async def get_additional_metrics(self) -> list[str]:
        return ["playwright_downloads"]

    # endregion

    # region PLAYWRIGHT
    async def get_or_create_browser(self):
        """Get or create a shared Playwright browser instance"""
        async with self.browser_lock:
            if not self.browser:
                p = await async_playwright().start()
                self.browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                    ],
                )
            return self.browser

    async def download_file_playwright(self, url: str, filepath: Path) -> bool:
        """Optimized Playwright download with browser reuse"""
        if filepath.exists() and filepath.stat().st_size > 0:
            self.logger.debug(f"File already exists: {filepath.name}")
            return True

        browser = await self.get_or_create_browser()
        context = await browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport=ViewportSize(1920, 1080),
            java_script_enabled=True,
        )

        try:
            page = await context.new_page()

            # Set longer timeout for complex pages
            page.set_default_timeout(10000)

            async with page.expect_download(timeout=10000) as download_info:
                await page.goto(url, wait_until="networkidle", timeout=10000)
                # Wait a bit for JS to initialize
                await page.wait_for_timeout(2000)

                # Try to find and click download buttons
                download_selectors = [
                    "a[download]",
                    'button:has-text("Download")',
                    'a:has-text("Download")',
                    'button:has-text("Herunterladen")',
                    'a:has-text("Herunterladen")',
                    ".download-button",
                    '[class*="download"]',
                ]

                for selector in download_selectors:
                    try:
                        if await page.locator(selector).first.is_visible():
                            await page.locator(selector).first.click()
                            break
                    # noqa: F821
                    except (TimeoutError, Error):
                        continue

                download = await download_info.value

            await download.save_as(filepath)
            await context.close()

            self.logger.debug(f"Downloaded via Playwright: {filepath.name}")
            await self.update_stats("playwright_downloads")
            return True

        except Exception as e:
            self.logger.error(f"Playwright error for {url}: {e}")
            await context.close()
            return False

    # endregion

    # region LOGIC STEPS

    @staticmethod
    def get_file_extension(url: str, format_hint: str = None) -> str:
        """Optimized file extension detection"""
        # Quick lookup table
        extension_map = {
            ".csv": ".csv",
            ".json": ".json",
            ".xml": ".xml",
            ".xlsx": ".xlsx",
            ".xls": ".xls",
            ".pdf": ".pdf",
            ".txt": ".txt",
        }

        url_lower = url.lower()
        for ext in extension_map:
            if url_lower.endswith(ext):
                return ext

        # Format hint lookup
        if format_hint:
            format_lower = format_hint.lower()
            format_extension_map = {
                "csv": ".csv",
                "json": ".json",
                "xml": ".xml",
                "xlsx": ".xlsx",
                "excel": ".xlsx",
                "xls": ".xls",
                "pdf": ".pdf",
                "txt": ".txt",
                "text": ".txt",
            }

            for fmt, ext in format_extension_map.items():
                if fmt in format_lower:
                    return ext

        return ""

    @staticmethod
    def should_skip_resource(resource: Dict) -> bool:
        """Optimized resource skip check"""
        url = resource.get("url", "").lower()
        format_hint = resource.get("format", "").lower()

        # Check for allowed formats
        allowed_indicators = ["csv", "json", "geojson"]
        if any(
            indicator in format_hint or url.endswith(f".{indicator}")
            for indicator in allowed_indicators
        ):
            return False

        # Quick checks
        skip_indicators = ["atom", "wms", "wfs", "wmts", "sparql", "api"]
        if any(
            indicator in url or indicator in format_hint
            for indicator in skip_indicators
        ):
            return True

        # Check against skip formats
        if format_hint in skip_formats:
            return True

        return True

    # 5.
    async def download_file(
        self, url: str, filepath: Path
    ) -> tuple[bool, list[dict] | None]:
        """Optimized file download with streaming"""
        # Check if file already exists
        if self.is_file_system and filepath.exists() and filepath.stat().st_size > 0:
            self.logger.debug(f"File already exists: {filepath.name}")
            return True, None

        try:
            self.logger.debug(f"Downloading: {url}")
            async with self.session.get(url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if not self.is_file_system:
                    if "csv" in content_type:
                        content = await response.read()
                        df = pd.read_csv(io.BytesIO(content))
                        features = df.to_dict("records")
                        await self.update_stats("files_downloaded")
                        return True, features
                    else:
                        try:
                            data = await response.json()
                            features = data.get("features", [])
                            await self.update_stats("files_downloaded")
                            return True, features
                        except json.JSONDecodeError as e:
                            self.logger.debug(
                                f"JSON in memory download failed for {url}: {e}"
                            )
                            return False, None
                else:
                    # Create temporary file for atomic write
                    temp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")

                    content_type = response.headers.get("content-type", "").lower()
                    if response.status != 200:
                        return False, None

                    if "html" in content_type:
                        # self.logger.info(f"HTML in content_type: {content_type}")
                        # async with self.playwright_lock:
                        #     self.playwright_domains.add(domain)
                        # return await self.download_file_playwright(url, filepath)
                        return False, None
                    else:
                        # Normal download with larger buffer
                        async with aiofiles.open(temp_filepath, "wb") as f:
                            async for chunk in response.content.iter_chunked(65536):
                                await f.write(chunk)

                    # Verify file is not empty
                    if temp_filepath.stat().st_size == 0:
                        self.logger.warning(f"Downloaded file is empty: {filepath}")
                        temp_filepath.unlink()
                        return False, None

                    # Atomic rename
                    temp_filepath.rename(filepath)

                    self.logger.debug(
                        f"Downloaded: {filepath.name} ({filepath.stat().st_size} bytes)"
                    )
                    await self.update_stats("files_downloaded")
                    return True, None

        except Exception as e:
            self.logger.debug(f"Regular download failed for {url}: {e}")
            return False, None

    # 4.
    async def get_package_details_by_api(self, package_name: str) -> Optional[Dict]:
        """Get package details with caching"""
        cached_service_info = await self.get_from_cache(package_name)
        if cached_service_info is not None:
            return cached_service_info

        try:
            async with self.session.get(
                f"{self.api_url}/package_show", params={"id": package_name}
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if data.get("success"):
                    result = data.get("result")
                    await self.add_to_cache(package_name, result)
                    return result
                else:
                    self.logger.warning(f"Package not found or error: {package_name}")
                    return None

        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching package details for {package_name}: {e}")
            return None

    # 3.
    async def process_dataset(self, package_name: str) -> bool:
        """Process dataset with optimized resource handling"""
        # Minimal delay to respect server
        await asyncio.sleep(self.delay)

        try:
            package = await self.get_package_details_by_api(package_name)
            if not package:
                await self.update_stats("errors")
                return False

            meta_id = package.get("id")
            title = package.get("title", package_name)
            resources = package.get("resources", [])

            # Create dataset directory
            safe_title = sanitize_filename(title)
            dataset_dir = self.output_dir / f"{package_name}_{safe_title}"

            # Skip if already processed successfully
            metadata_file = dataset_dir / "metadata.json"
            if self.is_file_system:
                if metadata_file.exists():
                    self.logger.debug(f"Dataset already processed: {title}")
                    await self.update_stats("datasets_processed")
                    return True
                dataset_dir.mkdir(exist_ok=True)

            if not resources:
                self.logger.debug(f"No resources found for dataset: {title}")
                await self.update_stats("datasets_processed")
                await self.update_stats("failed_datasets")
                return True

            self.logger.debug(
                f"Processing dataset: {title} ({len(resources)} resources)"
            )

            # Filter and prioritize resources
            valid_resources = []
            for i, resource in enumerate(resources):
                if not resource.get("url") or self.should_skip_resource(resource):
                    continue

                # Prioritize by format
                format_priority = {
                    "json": 1,
                    "geojson": 2,
                    "csv": 3,
                }
                format_str = resource.get("format", "").lower()
                priority = min(
                    format_priority.get(fmt, 999)
                    for fmt in format_priority
                    if fmt in format_str
                )
                if priority == 999:
                    priority = 7  # Unknown format

                valid_resources.append((priority, i, resource))

            # Sort by priority
            valid_resources.sort(key=lambda x: x[0])

            # Download resources concurrently (but limit to avoid overwhelming)
            success_count = 0
            download_tasks = []

            for priority, i, resource in valid_resources:
                url = resource.get("url")
                resource_name = resource.get("name", f"resource_{i}")
                resource_format = resource.get("format", "")
                extension = self.get_file_extension(url, resource_format)

                filename = sanitize_filename(f"{resource_name}_{i}{extension}")
                filepath = dataset_dir / filename

                task = self.download_file(url, filepath)
                download_tasks.append(task)

            # Wait for downloads
            if download_tasks:
                results = await asyncio.gather(*download_tasks, return_exceptions=True)
                success_count = sum(1 for r in results if r[0] is True)

            if success_count > 0:
                self.logger.debug(f"Downloaded {success_count} files for: {title}")

                # Save metadata
                package_meta = DatasetMetadataWithContent(
                    id=meta_id,
                    title=title,
                    groups=[group.get("title") for group in package.get("groups", [])],
                    organization=package.get("organization", {}).get("title"),
                    tags=[tag.get("name") for tag in package.get("tags", [])],
                    description=package.get("notes"),
                    city="Berlin",
                    state="Berlin",
                    country="Germany",
                )

                # TODO: add the dataset to buffer

                if self.is_file_system:
                    save_file_with_task(metadata_file, package_meta.to_json())

                # package_meta.fields = await extract_data_content(dataset_dir)

                if self.is_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

                if self.is_store and self.dataset_db_buffer:
                    await self.dataset_db_buffer.add(package_meta)
            else:
                # Clean up empty dataset
                if self.is_file_system:
                    safe_delete(dataset_dir, self.logger)

            await self.update_stats("datasets_processed")
            return True

        except Exception as e:
            self.logger.error(f"Error processing dataset {package_name}: {e}")
            await self.update_stats("failed_datasets", package_name)
            await self.update_stats("errors")
            return False

    # 2.
    async def get_all_packages_by_api(self) -> list[str]:
        """Get list of all package names with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug("Fetching list of all datasets...")
                async with self.session.get(f"{self.api_url}/package_list") as response:
                    response.raise_for_status()
                    data = await response.json()

                    if data.get("success"):
                        packages = data.get("result", [])
                        await self.update_stats("datasets_found", len(packages))
                        self.logger.info(f"Found {len(packages)} datasets")
                        return packages
                    else:
                        self.logger.error(f"API error: {data.get('error')}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)  # Exponential backoff
                        continue

            except aiohttp.ClientError as e:
                self.logger.error(
                    f"Error fetching package list (attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        return []

    # 1.
    async def process_all_datasets(self):
        """Optimized download with batching and better concurrency"""
        self.logger.info("Starting optimized Berlin Open Data download")

        # Get list of all packages
        packages = await self.get_all_packages_by_api()
        if not packages:
            self.logger.error("No packages found or error fetching package list")
            return

        self.logger.debug(
            f"Starting download of {len(packages)} datasets with {self.max_workers} workers"
        )

        # Progress reporting task
        progress_task = asyncio.create_task(self.progress_reporter())

        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(package_name: str):
            async with semaphore:
                return await self.process_dataset(package_name)

        # Process in batches to avoid overwhelming memory
        for i in range(0, len(packages), self.batch_size):
            batch = packages[i : i + self.batch_size]
            tasks = [process_with_semaphore(package) for package in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for package, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Exception in task for {package}: {result}")
                    await self.update_stats("errors")

            self.logger.debug(
                f"Completed batch {i // self.batch_size + 1}/{(len(packages) + self.batch_size - 1) // self.batch_size}"
            )

        # Cancel progress reporter
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        # Final flush of embeddings buffer
        if self.vector_db_buffer:
            await self.vector_db_buffer.flush()

        # STATS
        self.logger.info("🎉 Download completed!")
        await self.print_final_report()

    # endregion


# region MAIN
async def main():
    """Main async function with optimized settings"""
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        async with Berlin(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            delay=args.delay,
            is_embeddings=True,
            connection_limit=args.connection_limit,
            batch_size=args.batch_size,
        ) as downloader:
            await downloader.process_all_datasets()

    except KeyboardInterrupt:
        print("Download interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
# endregion

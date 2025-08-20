import asyncio
import io
import json
import logging
import sys
import os
from typing import Dict, Optional
from urllib.parse import urlparse, unquote

import aiofiles
import pandas as pd

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithContent,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import sanitize_filename, safe_delete
from src.utils.file import save_file_with_task


class Leipzig(BaseDataDownloader):
    """Optimized async class for downloading Leipzig CSV/JSON data"""

    # region INIT

    def __init__(
        self,
        output_dir: str = "leipzig",
        max_workers: int = 128,
        delay: float = 0.05,
        is_file_system: bool = True,
        is_embeddings: bool = False,
        is_store: bool = False,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 3,
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
            batch_size: Size of dataset batches to process
            max_retries: Maximum retry attempts for failed requests
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
            max_retries=max_retries,
        )
        self.base_url = "https://opendata.leipzig.de"
        self.api_url = f"{self.base_url}/api/3/action"
        self.logger = get_prefixed_logger(__name__, "LEIPZIG")

    # endregion

    # region LOGIC STEPS

    @staticmethod
    def get_file_extension(url: str, format_hint: str) -> str:
        """Determine file extension from URL or format hint"""
        format_extensions = {"csv": ".csv", "json": ".json", "geojson": ".geojson"}

        if format_hint in format_extensions:
            return format_extensions[format_hint]

        parsed = urlparse(url)
        path = unquote(parsed.path)
        if "." in os.path.basename(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in [".csv", ".json", ".geojson"]:
                return ext

        return ".data"

    # 7.
    async def download_resource(
        self, resource: Dict, safe_title: str
    ) -> tuple[bool, list[dict]] | bool:
        """Download a single resource with retry logic"""
        try:
            url = resource.get("url")
            resource_name = resource.get("name", resource.get("id", "unnamed"))
            resource_format = resource.get("format", "").lower()

            # Update format-specific counters
            self.logger.debug(
                f"\t📄 Downloading: {resource_name} ({resource_format.upper()})"
            )

            # Download with retry
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url) as response:
                        response.raise_for_status()
                        dataset_format = self.get_file_extension(url, resource_format)
                        if not self.is_file_system:
                            if dataset_format == ".csv":
                                content = await response.read()
                                df = pd.read_csv(io.BytesIO(content))
                                features = df.to_dict("records")
                                await self.update_stats("layers_downloaded")
                                return True, features
                            else:
                                try:
                                    data = await response.json()
                                    features = data.get("features", [])
                                    await self.update_stats("layers_downloaded")
                                    return True, features
                                except json.JSONDecodeError:
                                    continue

                        else:
                            # Determine filename
                            safe_name = sanitize_filename(resource_name)
                            filename = f"{safe_name}{dataset_format}"

                            # region Avoid duplication
                            counter = 1
                            original_filename = filename
                            dataset_dir = self.output_dir / safe_title
                            while (dataset_dir / filename).exists():
                                name, ext = os.path.splitext(original_filename)
                                filename = f"{name}_{counter}{ext}"
                                counter += 1
                            # endregion

                            file_path = dataset_dir / filename

                            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

                            # Download with streaming
                            async with aiofiles.open(temp_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(
                                    65536
                                ):  # 64KB chunks
                                    await f.write(chunk)

                            # Verify file is not empty
                            if temp_path.stat().st_size == 0:
                                self.logger.warning(
                                    f"Downloaded file is empty: {filename}"
                                )
                                temp_path.unlink()
                                await self.mark_url_failed(url)
                                return False

                            # Atomic rename
                            temp_path.rename(file_path)
                            file_size = file_path.stat().st_size
                            self.logger.debug(f"\t✅ {filename} ({file_size:,} bytes)")
                        return True

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await self.update_stats("retries")
                        await asyncio.sleep(2**attempt)
                    else:
                        self.logger.error(f"\t❌ Error downloading {url}: {e}")
                        await self.mark_url_failed(url)
                        return False

            return False
        except Exception as e:
            self.logger.error(f"\t❌ Unexpected error: {e}")
            return False

    # 6.
    async def download_package(self, metadata: Dict) -> bool:
        """Download all resources for a package"""
        try:
            package_data = metadata["package_data"]
            target_resources = metadata["target_resources"]

            package_title = package_data.get("title", metadata["package_id"])
            organization = package_data.get("organization", {}).get("title", "Unknown")

            self.logger.debug(f"\n📦 Processing: {package_title}")
            self.logger.debug(f"\t🏢 Organization: {organization}")
            self.logger.debug(f"\t📊 Target resources: {len(target_resources)}")

            # Skip if already processed
            safe_title = sanitize_filename(package_title)
            if self.is_file_system:
                dataset_dir = self.output_dir / safe_title
                metadata_file = dataset_dir / "metadata.json"
                if metadata_file.exists():
                    self.logger.debug(f"\tDataset already processed: {safe_title}")
                    await self.update_stats("datasets_processed")
                    return True

            # Prepare metadata
            package_meta = DatasetMetadataWithContent(
                id=package_data.get("id"),
                title=package_title,
                organization=organization,
                author=package_data.get("author"),
                description=package_data.get("notes"),
                metadata_created=package_data.get("metadata_created"),
                metadata_modified=package_data.get("metadata_modified"),
                tags=[tag.get("name") for tag in package_data.get("tags", [])],
                groups=[group.get("title") for group in package_data.get("groups", [])],
                url=f"{self.base_url}/dataset/{package_data.get('name')}",
                city="Leipzig",
                state="Saxony",
                country="Germany",
            )

            # Sort resources - prioritize JSON
            json_resources = [
                r for r in target_resources if r.get("format", "").lower() == "json"
            ]
            other_resources = [
                r for r in target_resources if r.get("format", "").lower() != "json"
            ]
            sorted_resources = json_resources + other_resources

            download_semaphore = asyncio.Semaphore(self.max_workers)

            async def download_with_semaphore(_resource, _safe_title):
                url = _resource.get("url")
                if not url:
                    return False
                if await self.is_url_failed(url):
                    return False

                async with download_semaphore:
                    return await self.download_resource(_resource, _safe_title)

            # Try to download a resource
            success = False
            dataset = []
            for resource in sorted_resources:
                success, dataset = await download_with_semaphore(resource, safe_title)
                if success:
                    await self.update_stats("files_downloaded")
                    break  # Stop after first successful download
                else:
                    await self.update_stats("errors")

            if success:
                # TODO: add the dataset to buffer
                # Save metadata
                if self.is_file_system:
                    dataset_dir = self.output_dir / safe_title
                    metadata_file = dataset_dir / "metadata.json"
                    save_file_with_task(metadata_file, package_meta.to_json())

                if self.is_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

                if self.is_store and self.dataset_db_buffer:
                    await self.dataset_db_buffer.add(package_meta)
            else:
                # Clean up empty dataset
                if self.is_file_system:
                    dataset_dir = self.output_dir / safe_title
                    safe_delete(dataset_dir, self.logger)

            return success

        except Exception as e:
            self.logger.error(f"\t❌ Error processing package: {e}")
            await self.update_stats("errors")
            return False

    # 5.
    async def get_package_details_by_api(self, package_id: str) -> Optional[Dict]:
        """Get package details with caching and retry logic"""
        await asyncio.sleep(self.delay)  # Rate limiting

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(
                    f"{self.api_url}/package_show", params={"id": package_id}
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if data["success"]:
                        result = data["result"]
                        return result
                    return None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)
                else:
                    self.logger.warning(
                        f"Failed to get package details for {package_id}: {e}"
                    )
                    return None
        return None

    # 4.
    async def analyze_package(self, package_id: str) -> Optional[Dict]:
        """Analyze a package and return metadata if it has target formats"""
        package_data = await self.get_package_details_by_api(package_id)
        if not package_data:
            return None

        # Check for target formats
        target_resources = []
        for resource in package_data.get("resources", []):
            resource_format = resource.get("format", "").lower()
            if resource_format in {"csv", "json", "geojson"}:
                target_resources.append(resource)

        if target_resources:
            await self.update_stats("datasets_found", len(target_resources))

            return {
                "package_id": package_id,
                "package_data": package_data,
                "target_resources": target_resources,
            }

        return None

    # 3.
    async def get_package_list_by_api(self) -> list[str]:
        """Get list of all package IDs with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.logger.debug("🔍 Fetching package list...")
                async with self.session.get(f"{self.api_url}/package_list") as response:
                    response.raise_for_status()
                    data = await response.json()

                    if not data["success"]:
                        self.logger.error(f"API error: {data.get('error')}")
                        return []

                    packages = data["result"]
                    self.logger.info(f"📊 Total packages found: {len(packages)}")
                    return packages

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"❌ Error fetching package list: {e}")
                    return []
        return []

    # 2.
    async def get_packages_with_target_formats(self) -> list[dict]:
        """Get all packages that contain CSV/JSON/GeoJSON resources"""
        try:
            # Get all package IDs
            all_packages = await self.get_package_list_by_api()
            if not all_packages:
                return []

            self.logger.debug("🔎 Analyzing packages for CSV/JSON/GeoJSON resources...")

            # Process packages in batches
            target_packages = []

            semaphore = asyncio.Semaphore(self.max_workers)

            async def analyze_with_semaphore(package_id: str):
                async with semaphore:
                    return await self.analyze_package(package_id)

            for i in range(0, len(all_packages), self.batch_size):
                batch = all_packages[i : i + self.batch_size]

                # Analyze batch
                tasks = [analyze_with_semaphore(pkg_id) for pkg_id in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect valid results
                for result in results:
                    if isinstance(result, dict) and result:
                        target_packages.append(result)

                # Progress update
                checked = min(i + self.batch_size, len(all_packages))
                self.logger.debug(
                    f"\tChecked {checked}/{len(all_packages)} packages..."
                )

            datasets_found = await self.get_stat_value("datasets_found")
            self.logger.info(
                f"✅ Found {len(target_packages)} CSV/JSON/GeoJSON packages with {datasets_found} target resources"
            )
            return target_packages

        except Exception as e:
            self.logger.error(f"❌ Error searching datasets: {e}")
            return []

    # 1.
    async def process_all_datasets(self):
        """Main download method"""
        self.logger.info("Start Leipzig Open Data Downloader")

        # Get target packages
        target_packages = await self.get_packages_with_target_formats()

        if not target_packages:
            self.logger.error("❌ No packages found with CSV/JSON data")
            return

        # Apply limit for testing
        self.logger.debug(
            f"\n🚀 Starting download of {len(target_packages)} packages..."
        )
        self.logger.debug("-" * 50)

        # Process packages in batches
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(metadata: Dict):
            async with semaphore:
                return await self.download_package(metadata)

        for i in range(0, len(target_packages), self.batch_size):
            batch = target_packages[i : i + self.batch_size]

            # Process batch
            tasks = [process_with_semaphore(metadata) for metadata in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Progress update
            processed = min(i + self.batch_size, len(target_packages))
            await self.print_progress(processed, len(target_packages))

        if self.is_embeddings:
            await self.vector_db_buffer.flush()
        if self.is_store:
            await self.dataset_db_buffer.flush()

        # STATS
        self.logger.info("🎉 Download completed!")
        await self.print_final_report()

    # endregion


# region MAIN
async def async_main():
    """Async main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Leipzig open data (CSV/JSON)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./leipzig",
        help="Output directory for downloaded datasets (default: ./leipzig)",
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
        async with Leipzig(
            output_dir=args.output,
            max_workers=args.max_workers,
            delay=args.delay,
            batch_size=args.batch_size,
            connection_limit=args.connection_limit,
            is_embeddings=True,
        ) as downloader:
            await downloader.process_all_datasets()
        return 0

    except KeyboardInterrupt:
        print("⚠️ Download interrupted by user")
        return 130  # Standard for Ctrl+C

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Synchronous entry point"""
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
# endregion

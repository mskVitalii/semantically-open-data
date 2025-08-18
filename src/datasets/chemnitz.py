import asyncio
import io
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithContent,
)
from src.utils.datasets_utils import sanitize_filename, safe_delete
from src.infrastructure.logger import get_prefixed_logger
from src.utils.embeddings_utils import extract_data_content
from src.utils.file import save_file_with_task


class Chemnitz(BaseDataDownloader):
    """Class for downloading Chemnitz open data"""

    # region INIT

    def __init__(
        self,
        csv_file_path: str,
        output_dir: str = "chemnitz",
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
            csv_file_path: Path to CSV file with dataset metadata
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
        super().__init__(
            output_dir,
            max_workers,
            delay,
            is_embeddings,
            is_store,
            connection_limit,
            connection_limit_per_host,
            batch_size,
            max_retries,
        )
        self.csv_file_path = csv_file_path
        self.stats["layers_downloaded"] = 0
        self.logger = get_prefixed_logger(__name__, "CHEMNITZ")

    # endregion

    # region STATS
    async def get_additional_metrics(self) -> list[str]:
        return ["layers_downloaded"]

    # endregion

    # region LOGIC STEPS
    # 6.
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
                self.logger.debug(f"\t\tLayer already downloaded: {file_name}")
                return True

            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(query_url, params=params) as response:
                        if response.status == 200:
                            dataset_dir.mkdir(exist_ok=True)

                            if format_name == "geojson":
                                try:
                                    data = await response.json()
                                    features = data.get("features", [])

                                    # Use aiofiles for async write
                                    content = json.dumps(
                                        features, ensure_ascii=False, indent=2
                                    )
                                    save_file_with_task(file_path, content)

                                    self.logger.debug(f"\t\t✓ Saved as {file_name}")
                                    await self.update_stats("layers_downloaded")
                                    return True

                                except json.JSONDecodeError:
                                    if file_path.exists():
                                        file_path.unlink()
                                    continue
                            else:
                                content = await response.read()
                                save_file_with_task(file_path, content, binary=True)

                                self.logger.debug(f"\t\t✓ Saved as {file_name}")
                                await self.update_stats("layers_downloaded")
                                return True

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await self.update_stats("retries")
                        await asyncio.sleep(2**attempt)
                    else:
                        self.logger.error(
                            f"\t\tError downloading layer {layer_name} with format {format_name}: {e}"
                        )
                        continue

        self.logger.error(f"\t\t⚠ Couldn't download layer {layer_name}")
        return False

    # 5.
    async def get_service_info(self, service_url: str) -> Optional[dict]:
        """Get service info with caching and retry logic"""
        # Check cache first
        async with self.cache_lock:
            if service_url in self.cache:
                await self.update_stats("cache_hits")
                return self.cache[service_url]

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
                        self.cache[service_url] = data

                    return data

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(
                        f"Error getting service info from {service_url}: {e}"
                    )
                    async with self.failed_urls_lock:
                        self.failed_urls.add(service_url)
                    return None
        return None

    # 4.
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
                self.logger.debug(f"\tDataset already processed: {title}")
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
                self.logger.debug(f"\tNo layers to download in {title}")
                await self.update_stats("datasets_processed")
                return True

            self.logger.debug(
                f"\tProcessing dataset: {title} ({len(all_features)} layers)"
            )

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
                self.logger.debug(
                    f"\tDownloaded {success_count}/{len(all_features)} layers for: {title}"
                )

                # Save metadata
                content = package_meta.to_json()
                save_file_with_task(metadata_file, content)

                await self.update_stats("files_downloaded")

                package_meta.fields = await extract_data_content(dataset_dir)

                if self.is_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

                if self.is_store and self.dataset_db_buffer:
                    await self.dataset_db_buffer.add(package_meta)
            else:
                # Clean up empty dataset
                safe_delete(dataset_dir, self.logger)

            await self.update_stats("datasets_processed")
            return True

        except Exception as e:
            self.logger.error(f"\tError processing dataset {title}: {e}")
            async with self.stats_lock:
                self.stats["failed_datasets"].add(title)
            await self.update_stats("errors")
            return False

    # 3.
    async def process_dataset(self, metadata: Dict[str, str]) -> bool:
        """Process a single dataset"""
        title = metadata["title"]
        url = metadata["url"]
        dataset_type = metadata["type"]
        description = metadata.get("description", "")

        self.logger.debug(f"Processing: {title}")
        self.logger.debug(f"\tURL: {url}")

        try:
            if "Feature Service" == dataset_type:
                return await self.download_feature_service_data(url, title, description)
            else:
                self.logger.debug(f"\t⚠ Unknown type {dataset_type} for {title}")
                return False

        except Exception as e:
            self.logger.error(f"\t❌ Error processing {title}: {e}")
            async with self.stats_lock:
                self.stats["failed_datasets"].add(title)
            await self.update_stats("errors")
            return False

    # 2.
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

    # 1.
    async def process_all_datasets(self):
        """Download all datasets with optimized batching and concurrency"""
        self.logger.info("Starting optimized Chemnitz Open Data download")

        # region Load datasets metadatas
        metadatas = await self.load_datasets_metadata_from_csv()
        if not metadatas:
            self.logger.error("No datasets found in CSV file")
            return

        self.logger.info(
            f"Found {len(metadatas)} datasets for download with {self.max_workers} workers"
        )
        self.logger.debug(f"Saving datasets to folder: {self.output_dir.absolute()}")
        self.logger.debug("-" * 50)
        # endregion

        progress_task = asyncio.create_task(self.progress_reporter())

        # Process in batches to avoid overwhelming memory
        for i in range(0, len(metadatas), self.batch_size):
            batch = metadatas[i : i + self.batch_size]

            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(metadata: Dict[str, str]):
                async with semaphore:
                    return await self.process_dataset(metadata)

            tasks = [process_with_semaphore(metadata) for metadata in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for metadata, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"Exception in task for {metadata['title']}: {result}"
                    )
                    await self.update_stats("errors")

            self.logger.info(
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
        if self.is_store:
            await self.dataset_db_buffer.flush()

        # STATS
        self.logger.info("🎉 Download completed!")
        await self.print_final_report()

    # endregion


# region MAIN
async def async_main():
    """Async main function with optimized settings"""
    csv_file = "open_data_portal_stadt_chemnitz.csv"
    if not Path(csv_file).exists():
        print(f"❌ File {csv_file} not found!")
        print("Make sure that CSV with datasets links is in the same folder.")
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
        async with Chemnitz(
            csv_file,
            output_dir=args.output,
            max_workers=args.max_workers,
            delay=args.delay,
            is_embeddings=True,
            is_store=True,
            connection_limit=args.connection_limit,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
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

import asyncio
import io
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

import aiofiles
import pandas as pd
from aiohttp import (
    ClientTimeout,
    ClientError,
    ClientConnectionError,
    ClientResponseError,
)

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithContent,
    Dataset,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import sanitize_title, safe_delete
from src.utils.file import save_file_with_task


class Dresden(BaseDataDownloader):
    """Optimized async class for downloading Dresden open data"""

    # region INIT
    def __init__(
        self,
        output_dir: str = "dresden",
        max_workers: int = 20,
        delay: float = 0.05,
        is_file_system: bool = True,
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
        self.base_url = "https://register.opendata.sachsen.de"
        self.search_endpoint = f"{self.base_url}/store/search"
        self.logger = get_prefixed_logger(__name__, "DRESDEN")

    # endregion

    # region LOGIC STEPS

    # 8.
    async def download_file(
        self, url: str, filename: str, dataset_dir: Path
    ) -> tuple[bool, list[dict]] | bool:
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
        if self.is_file_system and filepath.exists() and filepath.stat().st_size > 0:
            self.logger.debug(f"File already exists: {filepath}")
            return True

        # Check if URL previously failed
        if await self.is_url_failed(url):
            return False

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Downloading: {url}")
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
                        self.logger.warning(
                            f"Response appears to be an error page: {url}"
                        )
                        await self.mark_url_failed(url)
                        return False

                    if not self.is_file_system:
                        if filepath.suffix == ".csv":
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
                            except json.JSONDecodeError:
                                continue
                    else:
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
                            self.logger.warning(f"Downloaded file is empty: {filepath}")
                            temp_path.unlink()
                            await self.mark_url_failed(url)
                            return False

                        # Atomic rename
                        temp_path.rename(filepath)

                        self.logger.debug(
                            f"File saved: {filepath} ({filepath.stat().st_size} bytes)"
                        )
                        await self.update_stats("files_downloaded")
                        return True

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)
                else:
                    self.logger.error(f"Error downloading {url}: {e}")
                    await self.update_stats("errors")
                    await self.mark_url_failed(url)
                    return False
        return False

    # 7.
    def extract_from_distributions(self, metadata: Dict) -> List[Dict]:
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

        self.logger.debug(f"Extracting URLs from metadata with {len(metadata)} entries")

        # Search for distributions in metadata
        for subject, predicates in metadata.items():
            self.logger.debug(f"Checking subject: {subject}")

            # Look for dcat:distribution
            distributions = predicates.get("http://www.w3.org/ns/dcat#distribution", [])
            self.logger.debug(f"Found distributions: {len(distributions)}")

            for dist in distributions:
                if dist.get("type") == "uri":
                    dist_uri = dist.get("value")
                    self.logger.debug(f"Processing distribution: {dist_uri}")

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
                        self.logger.debug(f"Found download URL: {download_url}")

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
                        self.logger.debug(f"Added download file: {file_title}")

        return downloads

    # 6.
    async def check_url_availability_by_api(
        self, url: str, format_info: Dict
    ) -> Optional[Dict]:
        """Check if a URL is available and return download info"""
        try:
            self.logger.debug(f"Checking availability: {url}")
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
        except (
            ClientError,
            ClientConnectionError,
            ClientResponseError,
            asyncio.TimeoutError,
        ):
            pass
        return None

    # 5.
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
            self.logger.warning("No dataset URI provided")
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
            task = self.check_url_availability_by_api(url, format_info)
            check_tasks.append(task)

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Add successful results
        for result in results:
            if isinstance(result, dict) and result:
                downloads.append(result)

        # Fallback: try to extract from distribution metadata if no direct files found
        if not downloads:
            self.logger.debug(
                "No direct content files found, trying distribution metadata"
            )
            downloads = self.extract_from_distributions(dataset_metadata)

        self.logger.debug(f"Found {len(downloads)} files for download")
        return downloads

    # 4.
    async def process_dataset(self, dataset_info: Dict) -> bool:
        """Process a single dataset with optimized async operations"""
        await asyncio.sleep(self.delay)  # Minimal delay to respect server

        context_id = dataset_info.get("contextId")
        entry_id = dataset_info.get("entryId")

        if not context_id or not entry_id:
            self.logger.warning("Missing contextId or entryId")
            return False

        # Check if already processed
        safe_key = f"{context_id}_{entry_id}"
        if self.is_file_system:
            existing_dirs = [
                d
                for d in self.output_dir.iterdir()
                if d.is_dir() and d.name.startswith(safe_key)
            ]
            if existing_dirs:
                self.logger.debug(f"Dataset already processed: {safe_key}")
                await self.update_stats("datasets_processed")
                return True

        # Use metadata from dataset_info
        dataset_metadata = dataset_info.get("metadata", {})
        if not dataset_metadata:
            self.logger.warning(f"Missing metadata for dataset {context_id}/{entry_id}")
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
            self.logger.warning(
                f"Dataset URI not found in metadata {context_id}/{entry_id}"
            )
            return False

        # Directory creation
        safe_title = sanitize_title(title)
        dataset_dir = self.output_dir / f"{context_id}_{entry_id}_{safe_title}"
        if self.is_file_system:
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

        self.logger.debug(f"Processing dataset: {title}")
        self.logger.debug(f"Dataset URI: {dataset_uri}")

        # Extract download links
        downloads = await self.extract_download_urls(dataset_metadata, dataset_uri)
        if not downloads:
            self.logger.debug(f"No download files found for dataset: {title}")
            await self.update_stats("datasets_processed")
            await self.update_stats("failed_datasets")
            return False

        # Sort downloads to prioritize JSON files
        json_downloads = [d for d in downloads if d.get("extension") == ".json"]
        other_downloads = [d for d in downloads if d.get("extension") != ".json"]
        sorted_downloads = json_downloads + other_downloads

        # Download files with limited concurrency
        download_semaphore = asyncio.Semaphore(self.max_workers)

        async def download_with_semaphore(_download_info):
            async with download_semaphore:
                url = _download_info["url"]
                file_title = _download_info.get("title", "file")
                extension = _download_info.get("extension", "")
                filename = sanitize_title(f"{file_title}{extension}")
                return await self.download_file(url, filename, dataset_dir)

        # Try to download files
        success = False
        data = []

        for download_info in sorted_downloads:
            success, data = await download_with_semaphore(download_info)
            if success:
                self.logger.debug(
                    f"Successfully downloaded {download_info.get('extension')} file"
                )
                break  # Stop after first successful download

        if success:
            # Save metadata
            if self.is_file_system:
                metadata_file = dataset_dir / "metadata.json"
                content = package_meta.to_json()
                save_file_with_task(metadata_file, content)

            # package_meta.fields = await extract_data_content(dataset_dir)

            if self.is_embeddings and self.vector_db_buffer:
                await self.vector_db_buffer.add(package_meta)

            if self.is_store and self.dataset_db_buffer:
                dataset = Dataset(metadata=package_meta, data=data)
                await self.dataset_db_buffer.add(dataset)
        else:
            # Clean up empty dataset
            if self.is_file_system:
                safe_delete(dataset_dir, self.logger)

        await self.update_stats("datasets_processed")
        return success

    # 3.
    async def search_datasets_by_api(self, limit: int = 100, offset: int = 0) -> Dict:
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
            "limit": limit,
            "offset": offset,
            "sort": "modified desc",
        }

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Searching datasets: offset={offset}, limit={limit}")
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
                    self.logger.error(f"Error searching datasets: {e}")
                    return {}
        return {}

    # 2.
    async def collect_datasets(self) -> List[Dict]:
        """Collect all datasets from the API"""
        all_datasets = []
        offset = 0
        limit = 100

        while True:
            # Search for datasets
            search_result = await self.search_datasets_by_api(
                limit=limit, offset=offset
            )

            if not search_result or "resource" not in search_result:
                self.logger.warning("Empty response from API or invalid format")
                break

            children = search_result["resource"].get("children", [])
            total_results = search_result.get("results", 0)

            if not children:
                self.logger.debug("No more datasets found")
                break

            if offset == 0:
                await self.update_stats("datasets_found", total_results, "set")
                self.logger.debug(f"Total datasets found: {total_results}")

            all_datasets.extend(children)

            # Move to next page
            offset += limit
            self.logger.debug(
                f"Collected datasets: {len(all_datasets)} of {total_results}"
            )

            # If we got fewer results than requested, this is the last page
            if len(children) < limit:
                break

        return all_datasets

    # 1.
    async def process_all_datasets(self):
        """Download all datasets with optimized async processing"""
        self.logger.info("Starting optimized Dresden Open Data download")

        # First, collect all datasets
        all_datasets = await self.collect_datasets()

        if not all_datasets:
            self.logger.error("No datasets found")
            return

        self.logger.debug(
            f"Starting download of {len(all_datasets)} datasets with {self.max_workers} workers"
        )

        progress_task = asyncio.create_task(self.progress_reporter())

        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(dataset: Dict):
            async with semaphore:
                try:
                    return await self.process_dataset(dataset)
                except Exception as e:
                    self.logger.error(f"Error processing dataset: {e}")
                    await self.update_stats("errors")
                    return False

        for i in range(0, len(all_datasets), self.batch_size):
            batch = all_datasets[i : i + self.batch_size]
            tasks = [process_with_semaphore(dataset) for dataset in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

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
        async with Dresden(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            delay=args.delay,
            is_embeddings=True,
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

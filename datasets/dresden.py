import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from datasets.utils import sanitize_filename


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)-10s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/dresden_opendata_download.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class DresdenOpenDataDownloader:
    def __init__(
        self, output_dir: str = "dresden", delay: float = 0.1, max_workers: int = 10
    ):
        self.base_url = "https://register.opendata.sachsen.de"
        self.search_endpoint = f"{self.base_url}/store/search"
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_workers = max_workers

        # Create thread-safe session factory
        self.session_factory = self._create_session_factory()

        # Thread-safe locks
        self.stats_lock = Lock()
        self.dir_lock = Lock()

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Download statistics
        self.stats = {
            "datasets_found": 0,
            "datasets_downloaded": 0,
            "files_downloaded": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

    def _create_session_factory(self):
        """Create session factory with retry strategy"""

        def create_session():
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=self.max_workers,
                pool_maxsize=self.max_workers * 2,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            session.headers.update(
                {"User-Agent": "Dresden OpenData Downloader (Python/requests)"}
            )
            return session

        return create_session

    def search_dresden_datasets(self, limit: int = 100, offset: int = 0) -> Dict:
        """
        Search Dresden datasets via API

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

        session = self.session_factory()
        try:
            logger.info(f"Searching datasets: offset={offset}, limit={limit}")
            response = session.get(self.search_endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error searching datasets: {e}")
            return {}
        finally:
            session.close()

    def get_dataset_metadata(self, context_id: str, entry_id: str) -> Optional[Dict]:
        """
        Get metadata for specific dataset

        Args:
            context_id: Context ID
            entry_id: Entry ID

        Returns:
            Dataset metadata
        """
        session = self.session_factory()
        try:
            # Try different URL variants to get metadata
            urls_to_try = [
                f"{self.base_url}/store/{context_id}/metadata/{entry_id}",
                f"{self.base_url}/store/{context_id}/entry/{entry_id}",
                f"{self.base_url}/store/{context_id}/resource/{entry_id}",
            ]

            for url in urls_to_try:
                try:
                    response = session.get(url)
                    if response.status_code == 200:
                        return response.json()
                except requests.RequestException:
                    continue

            logger.warning(
                f"Could not get additional metadata for {context_id}/{entry_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error getting metadata {context_id}/{entry_id}: {e}")
            return None
        finally:
            session.close()

    def extract_download_urls(
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

        session = self.session_factory()
        try:
            # Simple approach: try dataset_uri + "/content.csv"
            csv_url = f"{dataset_uri}/content.csv"

            # Check if the CSV file exists by making a HEAD request
            try:
                logger.debug(f"Checking CSV availability: {csv_url}")
                response = session.head(csv_url, timeout=10)

                if response.status_code == 200:
                    downloads.append(
                        {
                            "url": csv_url,
                            "title": "content",
                            "format": "text/csv",
                            "extension": ".csv",
                        }
                    )
                    logger.debug(f"Found CSV file: {csv_url}")
                else:
                    logger.debug(
                        f"CSV not available (status {response.status_code}): {csv_url}"
                    )
            except requests.RequestException as e:
                logger.debug(f"Error checking CSV availability: {e}")

            # Also try other common formats
            for format_info in [
                {
                    "suffix": "/content.json",
                    "format": "application/json",
                    "ext": ".json",
                },
                {"suffix": "/content.xml", "format": "application/xml", "ext": ".xml"},
                {
                    "suffix": "/content.xlsx",
                    "format": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "ext": ".xlsx",
                },
            ]:
                try:
                    format_url = f"{dataset_uri}{format_info['suffix']}"
                    response = session.head(format_url, timeout=10)

                    if response.status_code == 200:
                        downloads.append(
                            {
                                "url": format_url,
                                "title": "content",
                                "format": format_info["format"],
                                "extension": format_info["ext"],
                            }
                        )
                        logger.debug(f"Found {format_info['ext']} file: {format_url}")
                except requests.RequestException:
                    continue

            # Fallback: try to extract from distribution metadata if no direct files found
            if not downloads:
                logger.debug(
                    "No direct content files found, trying distribution metadata"
                )
                downloads = self.extract_from_distributions(dataset_metadata)

            logger.info(f"Found {len(downloads)} files for download")
            return downloads
        finally:
            session.close()

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
                    logger.debug(f"Distribution info: {list(dist_info.keys())}")

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
                        if "csv" in file_format.lower():
                            extension = ".csv"
                        elif "json" in file_format.lower():
                            extension = ".json"
                        elif "xml" in file_format.lower():
                            extension = ".xml"
                        elif (
                            "xlsx" in file_format.lower()
                            or "excel" in file_format.lower()
                        ):
                            extension = ".xlsx"

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

    def download_file(self, url: str, filename: str, dataset_dir: Path) -> bool:
        """
        Download file

        Args:
            url: File URL
            filename: Filename to save
            dataset_dir: Dataset directory

        Returns:
            True if file successfully downloaded
        """
        filepath = dataset_dir / filename

        # Check if file already exists
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return True

        session = self.session_factory()
        try:
            logger.info(f"Downloading: {url}")
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Check if response contains actual data (not just an error page)
            content_type = response.headers.get("content-type", "").lower()
            content_length = response.headers.get("content-length")

            if content_length and int(content_length) < 100 and "html" in content_type:
                logger.warning(f"Response appears to be an error page, skipping: {url}")
                return False

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Check if downloaded file is not empty
            if filepath.stat().st_size == 0:
                logger.warning(f"Downloaded file is empty, removing: {filepath}")
                filepath.unlink()
                return False

            logger.info(f"File saved: {filepath} ({filepath.stat().st_size} bytes)")

            with self.stats_lock:
                self.stats["files_downloaded"] += 1

            return True

        except requests.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            with self.stats_lock:
                self.stats["errors"] += 1
            return False
        finally:
            session.close()

    def process_dataset(self, dataset_info: Dict) -> bool:
        """Process a single dataset"""
        context_id = dataset_info.get("contextId")
        entry_id = dataset_info.get("entryId")

        if not context_id or not entry_id:
            logger.warning("Missing contextId or entryId")
            return False

        # Use metadata from dataset_info as it already contains all needed information
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

        # Directory creation with lock
        safe_title = sanitize_filename(title)
        dataset_dir = self.output_dir / f"{context_id}_{entry_id}_{safe_title}"

        with self.dir_lock:
            dataset_dir.mkdir(exist_ok=True)

        # METADATA
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_uri": dataset_uri,
                    "title": title,
                    "description": description,
                    "keywords": keywords,
                    "city": "Dresden",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Processing dataset: {title}")
        logger.info(f"Dataset URI: {dataset_uri}")

        # Extract download links using the new approach
        downloads = self.extract_download_urls(dataset_metadata, dataset_uri)

        if not downloads:
            logger.warning(f"No download files found for dataset: {title}")
            return True

        # Sort downloads to prioritize JSON files
        json_downloads = [d for d in downloads if d.get("extension") == ".json"]
        other_downloads = [d for d in downloads if d.get("extension") != ".json"]

        # Process JSON files first, then others if needed
        sorted_downloads = json_downloads + other_downloads

        success = False
        for i, download in enumerate(sorted_downloads):
            url = download["url"]
            file_title = download.get("title", f"file_{i}")
            extension = download.get("extension", "")
            filename = sanitize_filename(f"{file_title}{extension}")

            if self.download_file(url, filename, dataset_dir):
                success = True
                logger.info(
                    f"Successfully downloaded {extension} file, skipping others"
                )
                break

            # Small delay between files from same dataset
            time.sleep(self.delay)

        with self.stats_lock:
            if success:
                self.stats["datasets_downloaded"] += 1

        return success

    def download_all_datasets_parallel(self):
        """Download all datasets using parallel processing"""
        logger.info("Starting parallel search and download of Dresden datasets")

        # First, collect all datasets
        all_datasets = []
        offset = 0
        limit = 100

        while True:
            # Search for datasets
            search_result = self.search_dresden_datasets(limit=limit, offset=offset)

            if not search_result or "resource" not in search_result:
                logger.warning("Empty response from API or invalid format")
                break

            children = search_result["resource"].get("children", [])
            total_results = search_result.get("results", 0)

            if not children:
                logger.info("No more datasets found")
                break

            if offset == 0:
                self.stats["datasets_found"] = total_results
                logger.info(f"Total datasets found: {total_results}")

            all_datasets.extend(children)

            # Move to next page
            offset += limit
            logger.info(f"Collected datasets: {len(all_datasets)} of {total_results}")

            # If we got fewer results than requested, this is the last page
            if len(children) < limit:
                break

        # Process datasets in parallel
        logger.info(
            f"Starting parallel processing of {len(all_datasets)} datasets with {self.max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_dataset = {
                executor.submit(self.process_dataset, dataset): dataset
                for dataset in all_datasets
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_dataset):
                # dataset = future_to_dataset[future]
                completed += 1

                try:
                    success = future.result()
                    if success:
                        logger.info(
                            f"Successfully processed dataset ({completed}/{len(all_datasets)})"
                        )
                    else:
                        logger.warning(
                            f"Failed to process dataset ({completed}/{len(all_datasets)})"
                        )
                except Exception as e:
                    logger.error(f"Error processing dataset: {e}")
                    with self.stats_lock:
                        self.stats["errors"] += 1

                # Log progress every 10 datasets
                if completed % 10 == 0:
                    logger.info(
                        f"Progress: {completed}/{len(all_datasets)} datasets processed"
                    )

        # Print statistics
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]

        logger.info("=" * 50)
        logger.info("DOWNLOAD STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Datasets found: {self.stats['datasets_found']}")
        logger.info(f"Datasets processed: {self.stats['datasets_downloaded']}")
        logger.info(f"Files downloaded: {self.stats['files_downloaded']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Execution time: {duration}")
        logger.info(f"Data saved to: {self.output_dir.absolute()}")

    def download_all_datasets(self):
        """Wrapper method that calls the parallel version"""
        self.download_all_datasets_parallel()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Dresden open data")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dresden",
        help="Directory to save data (default: dresden)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.1,
        help="Delay between requests in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        downloader = DresdenOpenDataDownloader(
            output_dir=args.output_dir, delay=args.delay, max_workers=args.workers
        )
        downloader.download_all_datasets()

    except KeyboardInterrupt:
        logger.warning("⚠️ Download interrupted by user")
        return 130  # Standard for Ctrl+C

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()

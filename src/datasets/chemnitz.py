import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from src.datasets.datasets_metadata import (
    DatasetMetadataWithContent,
    DatasetJSONEncoder,
)
from src.utils.datasets_utils import sanitize_filename, safe_delete
from src.infrastructure.logger import get_logger
from src.utils.embeddings_utils import extract_data_content
from src.vector_search.vector_db import VectorDB
from src.vector_search.vector_db_buffer import VectorDBBuffer

if TYPE_CHECKING:
    from _typeshed import SupportsWrite  # noqa: F401

logger = get_logger(__name__)


class ChemnitzDataDownloader:
    def __init__(
        self, csv_file_path, output_dir="chemnitz", is_embeddings: bool = False
    ):
        self.csv_file_path = csv_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.is_embeddings = is_embeddings
        if self.is_embeddings:
            vector_db = VectorDB(use_grpc=True)
            self.index_buffer = VectorDBBuffer(
                vector_db, buffer_size=100, auto_flush=True
            )

    def load_datasets_metadata_from_csv(self):
        datasets: list[dict[str, str]] = []
        with open(self.csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
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
        return datasets

    def get_service_info(self, service_url):
        try:
            info_url = f"{service_url}?f=json"
            response = self.session.get(info_url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting value from service {service_url}: {e}")
            return None

    def download_feature_service_data(self, service_url, title):
        try:
            # META
            service_info = self.get_service_info(service_url)
            if not service_info:
                return False

            # Folder
            safe_title = sanitize_filename(title)
            dataset_dir = self.output_dir / safe_title

            # Save META
            # TODO: save to MongoDB buffer
            package_meta = DatasetMetadataWithContent(
                id=service_info.get("serviceItemId"),
                title=title,
                city="Chemnitz",
                state="Saxony",
                country="Germany",
            )

            # DATASET
            layers = service_info.get("layers", [])
            tables = service_info.get("tables", [])

            all_features = layers + tables

            if not all_features:
                logger.debug(f"\tNo layers to download in {title}")
                return True

            is_downloaded_anything = False

            for feature in all_features:
                layer_id = feature.get("id", 0)
                layer_name = feature.get("name", f"layer_{layer_id}")

                logger.debug(f"\tDownloading layer: {layer_name}")

                # Пробуем разные форматы
                formats_to_try = [
                    ("geojson", "json"),
                    ("csv", "csv"),
                ]

                layer_downloaded = False

                for format_name, file_ext in formats_to_try:
                    try:
                        # URL для запроса данных
                        query_url = f"{service_url}/{layer_id}/query"

                        params = {
                            "where": "1=1",
                            "outFields": "*",
                            "f": "geojson" if format_name == "geojson" else format_name,
                            "returnGeometry": "true",
                        }

                        response = self.session.get(
                            query_url, params=params, timeout=60
                        )

                        if response.status_code == 200:
                            dataset_dir.mkdir(exist_ok=True)
                            file_name = f"{layer_name}.{file_ext}"
                            file_path = dataset_dir / file_name

                            if format_name == "geojson":
                                try:
                                    data = response.json()
                                    features = data.get("features", [])
                                    with open(file_path, "w", encoding="utf-8") as f:
                                        json.dump(
                                            features, f, ensure_ascii=False, indent=2
                                        )
                                    logger.debug(f"\t\t✓ Saved as {file_name}")
                                    layer_downloaded = True
                                    is_downloaded_anything = True
                                    break
                                except json.JSONDecodeError:
                                    continue
                            else:
                                with open(file_path, "wb") as f:
                                    f.write(response.content)
                                logger.debug(f"\t\t✓ Saved as {file_name}")
                                is_downloaded_anything = True
                                layer_downloaded = True
                                break

                    except Exception as e:
                        logger.error(f"\t\tError with format {format_name}: {e}")
                        continue

                if not layer_downloaded:
                    logger.error(f"\t\t⚠ Couldn't download layer {layer_name}")

            if not is_downloaded_anything:
                safe_delete(dataset_dir, logger)
            else:
                with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:  # type: SupportsWrite[str]
                    json.dump(
                        package_meta,
                        f,
                        ensure_ascii=False,
                        indent=2,
                        cls=DatasetJSONEncoder,
                    )
                if self.is_embeddings:
                    package_meta.content = extract_data_content(dataset_dir)
                    self.index_buffer.add(package_meta)

            return True

        except Exception as e:
            logger.error(f"\tDownloading error {title}: {e}")
            return False

    def download_all_datasets(self):
        metadatas = self.load_datasets_metadata_from_csv()

        logger.info(f"Found {len(metadatas)} DATASETS for download")
        logger.debug(f"Saving DATASETS to folder: {self.output_dir.absolute()}")
        logger.debug("-" * 50)

        success_count = 0

        for i, meta in enumerate(metadatas, 1):
            title = meta["title"]
            url = meta["url"]
            dataset_type = meta["type"]
            logger.debug(f"[{i}/{len(metadatas)}] {title}")
            logger.debug(f"\tURL: {url}")

            try:
                if "Feature Service" == dataset_type:
                    if self.download_feature_service_data(url, title):
                        success_count += 1
                else:
                    logger.warning(f"\t⚠ Unknown type {dataset_type}")

            except Exception as e:
                logger.error(f"\t❌ Error: {e}")

            time.sleep(1)

        if self.is_embeddings:
            self.index_buffer.flush()
        logger.debug("-" * 50)
        logger.info(
            f"Completed! Successfully processed: {success_count}/{len(metadatas)} datasets"
        )
        logger.debug(f"DATASETS in folder: {self.output_dir.absolute()}")


def main():
    csv_file = "open_data_portal_stadt_chemnitz.csv"
    if not os.path.exists(csv_file):
        logger.error(f"❌ File {csv_file} not found!")
        logger.error("Make sure that CSV with DATASETS links in the same folder.")
        return None

    import argparse

    parser = argparse.ArgumentParser(description="Download Chemnitz open data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument(
        "--output",
        "-o",
        default="./chemnitz",
        help="Output directory for downloaded datasets (default: ./chemnitz)",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        downloader = ChemnitzDataDownloader(
            csv_file, output_dir=args.output, is_embeddings=True
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

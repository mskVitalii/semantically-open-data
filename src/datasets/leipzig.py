import logging
import os
import json
import time
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote
from datetime import datetime

from src.datasets.datasets_metadata import DatasetMetadata
from src.utils.datasets_utils import sanitize_filename, safe_delete
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class LeipzigCSVJSONDownloader:
    def __init__(self, output_dir="leipzig"):
        self.base_url = "https://opendata.leipzig.de"
        self.api_url = f"{self.base_url}/api/3/action"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Leipzig-CSV-JSON-Downloader/1.0",
                "Accept": "application/json",
            }
        )

        # Filters for the formats
        self.target_formats = {"csv", "json", "geojson"}

        # Stats
        self.stats = {
            "total_packages_checked": 0,
            "packages_with_target_formats": 0,
            "total_target_resources": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "csv_count": 0,
            "json_count": 0,
            "geojson_count": 0,
        }

    def get_packages_with_target_formats(self):
        try:
            logger.debug("🔍 Search packages in CSV and JSON...")

            response = self.session.get(f"{self.api_url}/package_list")
            response.raise_for_status()

            data = response.json()
            if not data["success"]:
                logger.error(f"Error API: {data.get('error')}")
                return []

            all_packages = data["result"]
            logger.info(f"📊 Total packages: {len(all_packages)}")

            target_packages = []

            logger.debug("🔎 Analyse each package...")
            for i, package_id in enumerate(all_packages, 1):
                if i % 50 == 0:
                    logger.info(f"\tChecked {i}/{len(all_packages)} packages...")

                package_data = self.get_package_details(package_id)
                if not package_data:
                    continue

                self.stats["total_packages_checked"] += 1

                # Check if there are target formats
                has_target_format = False
                target_resources = []

                for resource in package_data.get("resources", []):
                    resource_format = resource.get("format", "").lower()
                    if resource_format in self.target_formats:
                        has_target_format = True
                        target_resources.append(resource)

                if has_target_format:
                    target_packages.append(
                        {
                            "package_id": package_id,
                            "package_data": package_data,
                            "target_resources": target_resources,
                        }
                    )
                    self.stats["packages_with_target_formats"] += 1
                    self.stats["total_target_resources"] += len(target_resources)

                time.sleep(0.1)

            logger.info(f"✅ Found {len(target_packages)} packages with CSV/JSON")
            logger.info(
                f"📋 Total target resources: {self.stats['total_target_resources']}"
            )

            return target_packages

        except Exception as e:
            logger.error(f"❌ Error searching datasets: {e}")
            return []

    def get_package_details(self, package_id):
        """Gets detailed information about a package"""
        try:
            response = self.session.get(
                f"{self.api_url}/package_show", params={"id": package_id}, timeout=10
            )
            response.raise_for_status()

            data = response.json()
            if data["success"]:
                return data["result"]
            return None

        except (requests.RequestException, requests.Timeout, ValueError) as e:
            logger.warning(f"Failed to get package details for {package_id}: {e}")
            return None

    @staticmethod
    def get_file_extension(url, format_hint):
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

    def download_resource(self, resource, dataset_dir):
        try:
            url = resource.get("url")
            if not url:
                return False

            resource_name = resource.get("name", resource.get("id", "unnamed"))
            resource_format = resource.get("format", "").lower()

            if resource_format == "csv":
                self.stats["csv_count"] += 1
            elif resource_format == "json":
                self.stats["json_count"] += 1
            elif resource_format == "geojson":
                self.stats["geojson_count"] += 1

            logger.debug(f"\t📄 {resource_name} ({resource_format.upper()})")

            file_extension = self.get_file_extension(url, resource_format)
            safe_name = sanitize_filename(resource_name)
            filename = f"{safe_name}{file_extension}"

            # Not duplication
            counter = 1
            original_filename = filename
            while (dataset_dir / filename).exists():
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                counter += 1

            file_path = dataset_dir / filename

            # Download
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = file_path.stat().st_size
            logger.debug(f"\t✅ {filename} ({file_size:,} byte)")
            return True

        except Exception as e:
            logger.error(f"\t❌ Error: {e}")
            return False

    def download_package(self, metadata):
        try:
            package_data = metadata["package_data"]
            target_resources = metadata["target_resources"]

            package_title = package_data.get("title", metadata["package_id"])
            organization = package_data.get("organization", {}).get("title", "Unknown")

            logger.debug(f"\n📦 {package_title}")
            logger.debug(f"\t🏢 {organization}")
            logger.debug(f"\t📊 {len(target_resources)} target resources")

            # Folder
            safe_title = sanitize_filename(package_title)
            dataset_dir = self.output_dir / safe_title
            dataset_dir.mkdir(exist_ok=True)

            # Save META
            package_meta = DatasetMetadata(
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

            # Download DATASET
            json_resources = [d for d in target_resources if d.get("format") == "JSON"]
            other_resources = [d for d in target_resources if d.get("format") != "JSON"]
            sorted_target_resources = json_resources + other_resources

            success_count = 0
            for resource in sorted_target_resources:
                if self.download_resource(resource, dataset_dir):
                    success_count += 1
                    self.stats["successful_downloads"] += 1
                    break
                else:
                    self.stats["failed_downloads"] += 1

                time.sleep(0.5)

            if success_count == 0:
                safe_delete(dataset_dir, logger)
            else:
                with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(package_meta, f, ensure_ascii=False, indent=2)

            return success_count > 0

        except Exception as e:
            logger.error(f"\t❌ Error: {e}")
            return False

    def create_summary_report(self):
        text_report = f"""
Leipzig Open Data CSV & JSON Download Summary
============================================

Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Filter: CSV and JSON formats only

📊 Statistics:
- Total packages checked: {self.stats["total_packages_checked"]}
- Packages with CSV/JSON: {self.stats["packages_with_target_formats"]}
- Target resources found: {self.stats["total_target_resources"]}
- Successfully downloaded: {self.stats["successful_downloads"]}
- Download errors: {self.stats["failed_downloads"]}

📄 By format:
- CSV files: {self.stats["csv_count"]}
- JSON files: {self.stats["json_count"]}
- GeoJSON files: {self.stats["geojson_count"]}

📁 Data saved to: {self.output_dir.absolute()}

💡 Each dataset contains:
- Data files (CSV/JSON/GeoJSON)
- metadata.json - dataset metadata
- *.meta.json - metadata for each file
"""
        logger.info(text_report)

    def download_csv_json_only(self, limit=None):
        logger.info("🎯 Leipzig CSV & JSON Data Downloader")
        logger.info(f"📁 Folder: {self.output_dir.absolute()}")
        logger.info("=" * 50)

        # Target packages
        target_packages = self.get_packages_with_target_formats()

        if not target_packages:
            logger.error("❌ Not found packages with CSV/JSON data")
            return

        # Restrictions for the testing
        if limit:
            target_packages = target_packages[:limit]
            logger.info(f"⚠\tTesting mode: handling {limit} packages")

        logger.debug(f"\n🚀 Starting downloading {len(target_packages)}...")
        logger.debug("-" * 50)

        # Download each package
        for i, metadata in enumerate(target_packages, 1):
            logger.info(
                f"\n[{i}/{len(target_packages)}] Progress: {i / len(target_packages) * 100:.1f}%"
            )
            self.download_package(metadata)
            time.sleep(1)

        # Итоговый отчет
        logger.info("\n" + "=" * 50)
        logger.info("🎉 Downloading completed!")
        self.create_summary_report()


def main():
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
        downloader = LeipzigCSVJSONDownloader()
        # downloader.download_csv_json_only(limit=3)
        downloader.download_csv_json_only()
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

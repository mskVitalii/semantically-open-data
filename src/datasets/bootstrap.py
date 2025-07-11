import asyncio
from pathlib import Path

from src.datasets.berlin import BerlinOpenDataDownloader
from src.datasets.chemnitz import ChemnitzDataDownloader
from src.datasets.dresden import DresdenOpenDataDownloader
from src.datasets.leipzig import LeipzigCSVJSONDownloader
from src.infrastructure.logger import get_logger
from src.utils.datasets_utils import safe_delete

logger = get_logger(__name__)


async def download_berlin():
    """Download Berlin datasets."""
    safe_delete(Path("berlin"), logger)

    async with BerlinOpenDataDownloader(
        output_dir="berlin",
        max_workers=20,
        delay=0.05,
        batch_size=50,
        connection_limit=100,
        is_embeddings=True,
    ) as downloader:
        await downloader.download_all_datasets()
    logger.info("✅ Berlin download completed")


async def download_chemnitz():
    """Download Chemnitz datasets."""
    safe_delete(Path("chemnitz"), logger)

    csv_file = "open_data_portal_stadt_chemnitz.csv"
    if not Path(csv_file).exists():
        logger.error(f"❌ File {csv_file} not found!")
        return

    async with ChemnitzDataDownloader(
        csv_file,
        output_dir="chemnitz",
        max_workers=20,
        delay=0.05,
        batch_size=50,
        connection_limit=100,
        is_embeddings=True,
        max_retries=1,
    ) as downloader:
        await downloader.download_all_datasets()
    logger.info("✅ Chemnitz download completed")


async def download_leipzig():
    """Download Leipzig datasets."""
    safe_delete(Path("leipzig"), logger)

    async with LeipzigCSVJSONDownloader(
        output_dir="leipzig",
        max_workers=20,
        delay=0.05,
        batch_size=50,
        connection_limit=100,
        is_embeddings=True,
    ) as downloader:
        await downloader.download_csv_json_only(limit=None)
    logger.info("✅ Leipzig download completed")


async def download_dresden():
    """Download Dresden datasets."""
    safe_delete(Path("dresden"), logger)

    async with DresdenOpenDataDownloader(
        output_dir="dresden",
        max_workers=20,
        delay=0.05,
        batch_size=50,
        connection_limit=100,
        max_retries=1,
        is_embeddings=True,
    ) as downloader:
        await downloader.download_all_datasets()
    logger.info("✅ Dresden download completed")


async def bootstrap_data():
    """Run all city downloads in parallel."""
    logger.info("🚀 Starting parallel download for all cities...")

    # Create tasks for each city
    tasks = [
        download_berlin(),
        download_chemnitz(),
        download_leipzig(),
        download_dresden(),
    ]

    # Run all tasks concurrently
    try:
        await asyncio.gather(*tasks)
        logger.info("✅ All downloads completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error during parallel download: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(bootstrap_data())

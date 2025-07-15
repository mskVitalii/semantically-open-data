import logging
import shutil
from logging import Logger
from pathlib import Path


def safe_delete(path: Path, logger: Logger | logging.LoggerAdapter):
    if path.exists():
        if path.is_file():
            path.unlink()
            logger.debug(f"Deleted file: {path}")
        elif path.is_dir():
            shutil.rmtree(path)
            logger.debug(f"Deleted folder: {path}")
    else:
        logger.debug(f"Path doesn't exist: {path}")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?* '
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Limit length
    if len(filename) > 200:
        filename = filename[:200]

    return filename


skip_formats = [
    "atom",
    "wms",
    "wfs",
    "wmts",
    "api",
    "html",
    "htm",
    "xlsx",
    "xls",
    "zip",
]

allowed_formats = ["csv", "json", "txt"]
allowed_extensions = [".csv", ".json", ".xlsx", ".xls", ".txt"]

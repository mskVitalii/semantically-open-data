import shutil
from logging import Logger
from pathlib import Path


def safe_delete(path: Path, logger: Logger):
    if path.exists():
        if path.is_file():
            path.unlink()
            logger.debug(f"Удалён файл: {path}")
        elif path.is_dir():
            shutil.rmtree(path)
            logger.debug(f"Deleted folder: {path}")
    else:
        logger.debug(f"Путь не существует: {path}")


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


skip_formats = ["atom", "wms", "wfs", "wmts", "api", "html", "htm"]

allowed_formats = ["csv", "json", "xlsx", "xls", "txt"]
allowed_extensions = [".csv", ".json", ".xlsx", ".xls", ".txt"]

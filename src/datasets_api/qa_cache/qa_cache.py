import json
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiofiles.os

from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import sanitize_title

logger = get_prefixed_logger("QA_CACHE")

CACHE_DIR = Path(__file__).parent / "qa_cache"


async def check_qa_cache(query: str, step: int) -> Optional[dict]:
    """
    Check if cached data exists for the given query and step.

    Args:
        query: The query string to check cache for
        step: The step number in the process

    Returns:
        Tuple of (exists: bool, data: dict | None)
    """
    filename = sanitize_title(f"{query}_{step}")
    filepath = CACHE_DIR / f"{filename}.json"

    try:
        if await aiofiles.os.path.exists(filepath):
            async with aiofiles.open(filepath, mode="r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                return data
        else:
            return None
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Cache read error for {filepath}: {e}")
        return None


async def set_qa_cache(query: str, step: int, value: Any) -> bool:
    """
    Save data to cache as JSON for the given query and step.

    Args:
        query: The query string to cache data for
        step: The step number in the process
        value: The data to cache (must be JSON serializable)

    Returns:
        bool: True if successfully cached, False otherwise
    """
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    filename = sanitize_title(f"{query}_{step}")
    filepath = CACHE_DIR / f"{filename}.json"

    try:
        # Serialize to JSON and write asynchronously
        json_content = json.dumps(value, indent=2, ensure_ascii=False)

        async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
            await f.write(json_content)

        return True
    except (TypeError, IOError) as e:
        # Handle JSON serialization errors or file write errors
        logger.info(f"Cache write error for {filepath}: {e}")
        return False

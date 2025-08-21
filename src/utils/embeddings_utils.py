#!/usr/bin/env python3
"""
Data Extraction Utilities (Async Version)
Functions to extract schemas, examples, and convert data to searchable strings
"""

import json
from functools import partial

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
import gzip
import bz2
import lzma
import tarfile
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor()


# =============================================================================
# COMPREHENSIVE TEXT EXTRACTION (MAIN FUNCTION)
# =============================================================================

# region MAIN


def format_metadata_text(metadata: dict) -> str:
    text_parts = [
        f"{key}: {value}"
        for key, value in metadata.items()
        if isinstance(value, str) and value.strip()
    ]
    result = " ".join(text_parts)
    return result


# endregion

# =============================================================================
# DATA EXAMPLE EXTRACTION FUNCTIONS
# =============================================================================

# region DATA


# 2.1
async def get_csv_example(
    file_path: Union[str, Path], row_index: int = 0
) -> Dict[str, Any]:
    """
    Extract one example row from CSV file

    Args:
        file_path: Path to CSV file
        row_index: Index of row to extract (default: first row)

    Returns:
        Dictionary mapping column names to values

    Raises:
        FileNotFoundError: If file doesn't exist
        IndexError: If row_index is out of bounds
        Exception: If CSV cannot be read
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # List of encodings to try
    encodings = [
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "latin-1",
        "iso-8859-1",
        "cp1252",
    ]

    for encoding in encodings:
        try:
            # Read CSV in thread pool (pandas is not async)
            df = await asyncio.get_event_loop().run_in_executor(
                executor,
                partial(pd.read_csv, file_path, nrows=row_index + 1, encoding=encoding),
            )
            if df.empty or len(df) <= row_index:
                logger.warning(f"Row index {row_index} not found in CSV {file_path}")
                raise IndexError(f"Row index {row_index} not found in CSV")

            # Get the specified row as dictionary
            row_data = df.iloc[row_index].to_dict()

            # Clean up NaN values
            cleaned_data = {}
            for key, value in row_data.items():
                if pd.notna(value):
                    cleaned_data[key] = value
                else:
                    cleaned_data[key] = None

            logger.debug(
                f"Extracted CSV example from {file_path.name} using {encoding} encoding: {len(cleaned_data)} fields"
            )
            return cleaned_data

        except UnicodeDecodeError:
            logger.debug(f"Failed to decode {file_path.name} with {encoding}")
            continue
        except Exception as e:
            # If it's not an encoding error, raise it
            logger.error(f"Error extracting CSV example from {file_path}: {e}")
            raise

    # If we've tried all encodings and none worked
    logger.error(
        f"Could not decode {file_path} with any of the attempted encodings: {encodings}"
    )
    raise ValueError("Unable to decode CSV file with any of the attempted encodings")


# 3.1
async def get_json_example(
    file_path: Union[str, Path], object_index: int = 0
) -> Dict[str, Any]:
    """
    Extract one example object from JSON file

    Args:
        file_path: Path to JSON file
        object_index: Index of object to extract (for JSON arrays)

    Returns:
        Dictionary representing the example object

    Raises:
        FileNotFoundError: If file doesn't exist
        IndexError: If object_index is out of bounds
        Exception: If JSON cannot be parsed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"JSON file not found: {file_path}")
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        # Check file type by reading magic bytes
        async with aiofiles.open(file_path, "rb") as f:
            magic_bytes = await f.read(4)

        # Handle compressed files in thread pool
        if magic_bytes[:2] == b"\x1f\x8b":
            # Could be gzip or tar.gz
            data = await asyncio.get_event_loop().run_in_executor(
                executor, _handle_gzip_file, file_path
            )
        elif magic_bytes[:3] == b"BZh" or file_path.suffix == ".bz2":
            data = await asyncio.get_event_loop().run_in_executor(
                executor, _handle_bz2_file, file_path
            )
        elif magic_bytes[:6] == b"\xfd7zXZ\x00" or file_path.suffix in [".xz", ".lzma"]:
            data = await asyncio.get_event_loop().run_in_executor(
                executor, _handle_lzma_file, file_path
            )
        else:
            # Not compressed - try different encodings
            data = await _read_json_with_encodings(file_path)

        # Process the loaded JSON data
        if isinstance(data, list):
            if not data or len(data) <= object_index:
                logger.warning(
                    f"Object index {object_index} not found in JSON array {file_path}"
                )
                raise IndexError(f"Object index {object_index} not found in JSON array")
            result = data[object_index]

        elif isinstance(data, dict):
            if object_index > 0:
                logger.warning(
                    f"JSON contains single object, cannot access index {object_index}"
                )
                raise IndexError(
                    f"JSON contains single object, cannot access index {object_index}"
                )
            result = data

        else:
            logger.error(f"JSON contains primitive value, not an object: {type(data)}")
            raise ValueError(
                f"JSON contains primitive value, not an object: {type(data)}"
            )

        logger.debug(
            f"Extracted JSON example from {file_path.name}: {len(result) if isinstance(result, dict) else 'non-dict'}"
        )
        return result

    except Exception as e:
        logger.error(f"Error extracting JSON example from {file_path}: {e}")
        raise


# 3.1.1
def _handle_gzip_file(file_path: Path) -> Any:
    """Handle gzip or tar.gz files synchronously"""
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            # It's a tar.gz file
            logger.debug(f"Detected tar.gz archive: {file_path.name}")

            # Find JSON files in the archive
            json_members = [
                m for m in tar.getmembers() if m.name.endswith(".json") and m.isfile()
            ]

            if not json_members:
                raise ValueError(
                    f"No JSON files found in tar.gz archive {file_path.name}"
                )

            # Use the first JSON file found
            json_member = json_members[0]
            logger.debug(f"Extracting {json_member.name} from archive")

            # Extract and read the JSON file
            json_file = tar.extractfile(json_member)
            if json_file is None:
                raise ValueError(f"Could not extract {json_member.name} from archive")

            content = json_file.read().decode("utf-8")
            return json.loads(content)

    except tarfile.ReadError:
        # Not a tar file, treat as regular gzip
        logger.debug(f"Not a tar.gz, treating as gzipped JSON: {file_path.name}")
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)


# 3.1.2
def _handle_bz2_file(file_path: Path) -> Any:
    """Handle bzip2 compressed files"""
    logger.debug(f"Detected bzip2 compressed file: {file_path.name}")
    with bz2.open(file_path, "rt", encoding="utf-8") as f:
        return json.load(f)


# 3.1.3
def _handle_lzma_file(file_path: Path) -> Any:
    """Handle xz/lzma compressed files"""
    logger.debug(f"Detected xz/lzma compressed file: {file_path.name}")
    with lzma.open(file_path, "rt", encoding="utf-8") as f:
        return json.load(f)


# 3.1.4
async def _read_json_with_encodings(file_path: Path) -> Any:
    """Try reading JSON file with different encodings"""
    encodings = [
        "utf-8",
        "utf-16-le",  # UTF-16 Little Endian without BOM
        "utf-16-be",  # UTF-16 Big Endian without BOM
        "utf-16",  # UTF-16 with BOM
        "latin-1",
        "iso-8859-1",
        "cp1252",
    ]

    data = None
    last_error = None

    for encoding in encodings:
        try:
            async with aiofiles.open(file_path, "r", encoding=encoding) as f:
                content = await f.read()
                if not content.strip():
                    raise ValueError(f"File {file_path.name} is empty")
                data = json.loads(content)
            logger.debug(
                f"Successfully loaded {file_path.name} with {encoding} encoding"
            )
            break
        except UnicodeDecodeError as e:
            last_error = e
            logger.debug(f"Failed to decode {file_path.name} with {encoding}: {str(e)}")
            continue
        except json.JSONDecodeError as e:
            last_error = e
            logger.debug(
                f"Failed to parse JSON from {file_path.name} with {encoding}: {str(e)}"
            )
            continue
        except Exception as e:
            logger.error(
                f"Unexpected error reading {file_path.name} with {encoding}: {str(e)}"
            )
            raise

    if data is None:
        error_msg = f"Could not load JSON from {file_path.name} with any encoding. "
        if last_error:
            error_msg += f"Last error: {type(last_error).__name__}: {str(last_error)}"
        raise ValueError(error_msg)

    return data


# endregion

# =============================================================================
# JSON TO STRING CONVERSION FUNCTIONS (These remain synchronous as they're CPU-bound)
# =============================================================================

# region SEARCH


# 2|3 -> 4
def json_to_searchable_string(
    data: Dict[str, Any],
    max_length: int = 500,
    max_depth: int = 2,
) -> str:
    """
    Convert JSON object to a searchable string representation

    Args:
        data: JSON object (dictionary)
        max_length: Maximum length of output string
        max_depth: Maximum depth for nested objects

    Returns:
        Formatted string representation
    """
    if not isinstance(data, dict):
        result = str(data)[:max_length]
        logger.debug(f"Converted non-dict to string: {len(result)} chars")
        return result

    try:
        result = _json_to_key_value_string(data, max_length, max_depth)
        logger.debug(f"Converted JSON to string: {len(result)} chars")
        return result

    except Exception as e:
        logger.error(f"Error converting JSON to string: {e}")
        return str(data)[:max_length]


# 5
def _json_to_key_value_string(
    data: Dict[str, Any], max_length: int, max_depth: int, current_depth: int = 0
) -> str:
    """Convert JSON to 'key: value; key: value' format"""
    pairs = []

    for key, value in data.items():
        if len("; ".join(pairs)) > max_length:
            break

        # Format the value based on its type
        if isinstance(value, dict) and current_depth < max_depth:
            nested_str = _json_to_key_value_string(
                value, max_length // 2, max_depth, current_depth + 1
            )
            formatted_value = f"[{nested_str}]"
        elif isinstance(value, list):
            if value and isinstance(value[0], dict) and current_depth < max_depth:
                nested_str = _json_to_key_value_string(
                    value[0], max_length // 4, max_depth, current_depth + 1
                )
                formatted_value = f"[{nested_str}, ...]"
            else:
                formatted_value = f"[{', '.join(str(v) for v in value[:3])}{'...' if len(value) > 3 else ''}]"
        else:
            formatted_value = str(value)

        # Limit individual value length
        if len(formatted_value) > 50:
            formatted_value = formatted_value[:47] + "..."

        pairs.append(f"{key}: {formatted_value}")

    result = "; ".join(pairs)
    return result[:max_length]


# endregion

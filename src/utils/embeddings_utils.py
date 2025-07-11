#!/usr/bin/env python3
"""
Data Extraction Utilities (Async Version)
Functions to extract schemas, examples, and convert data to searchable strings
"""

import json
from functools import partial

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
import logging
import re
import gzip
import bz2
import lzma
import tarfile
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor()


# =============================================================================
# COMPREHENSIVE TEXT EXTRACTION (MAIN FUNCTION)
# =============================================================================


async def extract_comprehensive_text(dataset_path: Path) -> str:
    """
    Extract searchable text from all available sources

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Combined searchable text from metadata, content, and schema
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return ""

    # 1. Metadata (always available)
    metadata_text = await extract_metadata_text(dataset_path / "metadata.json")

    # 2. Data content (CSV/JSON files)
    content_text = await extract_data_content(dataset_path)

    combined_text = f"""METADATA: {metadata_text}
CONTENT: {content_text}""".strip()

    return combined_text


async def extract_metadata_text(metadata_file: Path) -> str:
    if not metadata_file.exists():
        logger.warning(f"Metadata file not found: {metadata_file}")
        return ""

    try:
        async with aiofiles.open(metadata_file, "r", encoding="utf-8") as f:
            content = await f.read()
            metadata = json.loads(content)
        return format_metadata_text(metadata)

    except Exception as e:
        logger.error(f"Error extracting metadata from {metadata_file}: {e}")
        return ""


def format_metadata_text(metadata: dict) -> str:
    text_parts = [
        f"{key}: {value}"
        for key, value in metadata.items()
        if isinstance(value, str) and value.strip()
    ]
    result = " ".join(text_parts)
    logger.debug(f"Extracted metadata text: {len(result)} chars")
    return result


async def extract_data_content(dataset_path: Path) -> str:
    """
    Extract example data content from CSV/JSON files

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Formatted example data text
    """
    content_parts = []
    tasks = []

    # Process CSV files
    csv_files = list(dataset_path.glob("*.csv"))
    for csv_file in csv_files:
        task = process_csv_file(csv_file)
        tasks.append(task)

    # Process JSON files (skip metadata.json)
    json_files = [f for f in dataset_path.glob("*.json") if f.name != "metadata.json"]
    for json_file in json_files:
        task = process_json_file(json_file)
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, str) and result:
            content_parts.append(result)
        elif isinstance(result, Exception):
            logger.warning(f"Task failed with error: {result}")

    result = " ".join(content_parts)
    logger.debug(
        f"Extracted data content: {len(content_parts)} files, {len(result)} chars"
    )
    return result


async def process_csv_file(csv_file: Path) -> str:
    """Process a single CSV file asynchronously"""
    try:
        example = await get_csv_example(csv_file)
        if example:
            formatted = json_to_searchable_string(
                example, format_style="key_value", max_length=200
            )
            logger.debug(f"Extracted CSV content from {csv_file.name}")
            return f"CSV_DATA: {formatted}"
    except Exception as e:
        logger.warning(f"Failed to extract CSV content from {csv_file}: {e}")
    return ""


async def process_json_file(json_file: Path) -> str:
    """Process a single JSON file asynchronously"""
    try:
        example = await get_json_example(json_file)
        if example:
            formatted = json_to_searchable_string(
                example, format_style="key_value", max_length=200
            )
            logger.debug(f"Extracted JSON content from {json_file.name}")
            return f"JSON_DATA: {formatted}"
    except Exception as e:
        logger.warning(f"Failed to extract JSON content from {json_file}: {e}")
    return ""


# =============================================================================
# SCHEMA EXTRACTION FUNCTIONS
# =============================================================================


async def get_csv_schema(file_path: Union[str, Path]) -> List[str]:
    """
    Extract column headers (schema) from CSV file

    Args:
        file_path: Path to CSV file

    Returns:
        List of column names

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If CSV cannot be read
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        # Read CSV header in thread pool (pandas is not async)
        df = await asyncio.get_event_loop().run_in_executor(
            executor, partial(pd.read_csv, file_path, nrows=0)
        )
        fields = df.columns.tolist()
        logger.debug(f"Extracted {len(fields)} fields from CSV: {file_path.name}")
        return fields

    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file is empty: {file_path}")
        return []

    except Exception as e:
        logger.error(f"Error reading CSV schema from {file_path}: {e}")
        raise


async def get_json_schema(file_path: Union[str, Path], max_depth: int = 3) -> List[str]:
    """
    Extract field names (schema) from JSON file

    Args:
        file_path: Path to JSON file
        max_depth: Maximum depth to explore for nested objects

    Returns:
        List of field names (flattened for nested objects)

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If JSON cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"JSON file not found: {file_path}")
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)

        fields = _extract_json_fields(data, max_depth=max_depth)
        logger.debug(f"Extracted {len(fields)} fields from JSON: {file_path.name}")
        return fields

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Error reading JSON schema from {file_path}: {e}")
        raise


def _extract_json_fields(
    data: Any, prefix: str = "", max_depth: int = 3, current_depth: int = 0
) -> List[str]:
    """
    Recursively extract field names from JSON data structure

    Args:
        data: JSON data (dict, list, or primitive)
        prefix: Current field path prefix
        max_depth: Maximum recursion depth
        current_depth: Current recursion level

    Returns:
        List of field names (with dots for nested fields)
    """
    fields = []

    if current_depth >= max_depth:
        return fields

    if isinstance(data, dict):
        for key, value in data.items():
            field_name = f"{prefix}.{key}" if prefix else key
            fields.append(field_name)

            # Recurse into nested objects
            if isinstance(value, (dict, list)) and current_depth < max_depth:
                nested_fields = _extract_json_fields(
                    value, field_name, max_depth, current_depth + 1
                )
                fields.extend(nested_fields)

    elif isinstance(data, list) and data:
        # For arrays, analyze the first element
        if isinstance(data[0], dict):
            nested_fields = _extract_json_fields(
                data[0], prefix, max_depth, current_depth
            )
            fields.extend(nested_fields)

    return fields


# =============================================================================
# DATA EXAMPLE EXTRACTION FUNCTIONS
# =============================================================================


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


def _handle_bz2_file(file_path: Path) -> Any:
    """Handle bzip2 compressed files"""
    logger.debug(f"Detected bzip2 compressed file: {file_path.name}")
    with bz2.open(file_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _handle_lzma_file(file_path: Path) -> Any:
    """Handle xz/lzma compressed files"""
    logger.debug(f"Detected xz/lzma compressed file: {file_path.name}")
    with lzma.open(file_path, "rt", encoding="utf-8") as f:
        return json.load(f)


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


# =============================================================================
# JSON TO STRING CONVERSION FUNCTIONS (These remain synchronous as they're CPU-bound)
# =============================================================================


def json_to_searchable_string(
    data: Dict[str, Any],
    format_style: str = "key_value",
    max_length: int = 500,
    max_depth: int = 2,
) -> str:
    """
    Convert JSON object to a searchable string representation

    Args:
        data: JSON object (dictionary)
        format_style: Output format ('key_value', 'natural', 'compact')
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
        if format_style == "key_value":
            result = _json_to_key_value_string(data, max_length, max_depth)
        elif format_style == "natural":
            result = _json_to_natural_string(data, max_length)
        elif format_style == "compact":
            result = _json_to_compact_string(data, max_length)
        else:
            logger.warning(f"Unknown format_style: {format_style}, using key_value")
            result = _json_to_key_value_string(data, max_length, max_depth)

        logger.debug(f"Converted JSON to {format_style} string: {len(result)} chars")
        return result

    except Exception as e:
        logger.error(f"Error converting JSON to string: {e}")
        return str(data)[:max_length]


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


def _json_to_natural_string(data: Dict[str, Any], max_length: int) -> str:
    """Convert JSON to natural language description"""
    parts = []

    for key, value in data.items():
        if len(" ".join(parts)) > max_length:
            break

        # Convert key to readable format
        readable_key = re.sub(r"[_-]", " ", key)
        readable_key = re.sub(r"([a-z])([A-Z])", r"\1 \2", readable_key)

        if isinstance(value, (str, int, float)):
            parts.append(f"{readable_key} is {value}")
        elif isinstance(value, list):
            if value:
                parts.append(
                    f"{readable_key} includes {', '.join(str(v) for v in value[:3])}"
                )
        elif isinstance(value, dict):
            parts.append(f"{readable_key} contains {len(value)} fields")
        else:
            parts.append(f"{readable_key} {value}")

    result = ". ".join(parts)
    return result[:max_length]


def _json_to_compact_string(data: Dict[str, Any], max_length: int) -> str:
    """Convert JSON to compact string (similar to JSON but more readable)"""
    try:
        compact_json = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

        # Make it more readable
        compact_json = re.sub(r'[{}"]', "", compact_json)  # Remove braces and quotes
        compact_json = re.sub(r":", ": ", compact_json)  # Add space after colons
        compact_json = re.sub(r",", ", ", compact_json)  # Add space after commas

        return compact_json[:max_length]

    except Exception as e:
        logger.warning(f"Error in compact JSON conversion: {e}")
        # Fallback to string representation
        return str(data)[:max_length]


def flatten_json_to_string(
    data: Dict[str, Any],
    separator: str = " ",
    include_keys: bool = True,
    max_items: int = 50,
) -> str:
    """
    Flatten JSON to a simple string by extracting all values

    Args:
        data: JSON object
        separator: String to join values
        include_keys: Whether to include key names
        max_items: Maximum number of items to include

    Returns:
        Flattened string representation
    """
    items = []

    def _extract_values(obj, current_items):
        if len(current_items) >= max_items:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                if len(current_items) >= max_items:
                    break

                if include_keys:
                    # Convert key to readable format
                    readable_key = re.sub(r"[_-]", " ", key)
                    current_items.append(readable_key)

                if isinstance(value, (str, int, float)):
                    current_items.append(str(value))
                elif isinstance(value, (dict, list)):
                    _extract_values(value, current_items)

        elif isinstance(obj, list):
            for item in obj:
                if len(current_items) >= max_items:
                    break
                if isinstance(item, (str, int, float)):
                    current_items.append(str(item))
                elif isinstance(item, (dict, list)):
                    _extract_values(item, current_items)

    _extract_values(data, items)
    result = separator.join(items)
    logger.debug(f"Flattened JSON to string: {len(items)} items, {len(result)} chars")
    return result


# =============================================================================
# MANUAL TESTING
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data extraction utilities")
    parser.add_argument("directory", help="Directory path to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Testing directory: {args.directory}")

    # Run async tests
    async def main():
        dir_path = Path(args.directory)
        text_for_embeddings = await extract_comprehensive_text(dir_path)
        logger.info(f"""\n\n\n{text_for_embeddings}\n\n\n""")

    asyncio.run(main())

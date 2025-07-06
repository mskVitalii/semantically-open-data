#!/usr/bin/env python3
"""
Data Extraction Utilities
Functions to extract schemas, examples, and convert data to searchable strings
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
import logging
import re
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# COMPREHENSIVE TEXT EXTRACTION (MAIN FUNCTION)
# =============================================================================


def extract_comprehensive_text(dataset_path: Path) -> str:
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
    metadata_text = extract_metadata_text(dataset_path / "metadata.json")

    # 2. Data content (CSV/JSON files)
    content_text = extract_data_content(dataset_path)

    combined_text = f"""METADATA: {metadata_text}
CONTENT: {content_text}""".strip()

    return combined_text


def extract_metadata_text(metadata_file: Path) -> str:
    """
    Extract searchable text from metadata.json

    Args:
        metadata_file: Path to metadata.json file

    Returns:
        Formatted metadata text
    """
    if not metadata_file.exists():
        logger.warning(f"Metadata file not found: {metadata_file}")
        return ""

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        text_parts = []

        # Extract other string fields
        for key, value in metadata.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value}")

        result = " ".join(text_parts)
        logger.debug(f"Extracted metadata text: {len(result)} chars")
        return result

    except Exception as e:
        logger.error(f"Error extracting metadata from {metadata_file}: {e}")
        return ""


def extract_data_content(dataset_path: Path) -> str:
    """
    Extract example data content from CSV/JSON files

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Formatted example data text
    """
    content_parts = []

    # Process CSV files
    csv_files = list(dataset_path.glob("*.csv"))
    for csv_file in csv_files:
        try:
            example = get_csv_example(csv_file)
            if example:
                formatted = json_to_searchable_string(
                    example, format_style="key_value", max_length=200
                )
                content_parts.append(f"CSV_DATA: {formatted}")
                logger.debug(f"Extracted CSV content from {csv_file.name}")
        except Exception as e:
            logger.warning(f"Failed to extract CSV content from {csv_file}: {e}")

    # Process JSON files (skip metadata.json)
    json_files = [f for f in dataset_path.glob("*.json") if f.name != "metadata.json"]
    for json_file in json_files:
        try:
            example = get_json_example(json_file)
            if example:
                formatted = json_to_searchable_string(
                    example, format_style="key_value", max_length=200
                )
                content_parts.append(f"JSON_DATA: {formatted}")
                logger.debug(f"Extracted JSON content from {json_file.name}")
        except Exception as e:
            logger.warning(f"Failed to extract JSON content from {json_file}: {e}")

    result = " ".join(content_parts)
    logger.debug(
        f"Extracted data content: {len(content_parts)} files, {len(result)} chars"
    )
    return result


# =============================================================================
# SCHEMA EXTRACTION FUNCTIONS
# =============================================================================


def get_csv_schema(file_path: Union[str, Path]) -> List[str]:
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
        # Read only the header (0 rows of data)
        df = pd.read_csv(file_path, nrows=0)
        fields = df.columns.tolist()
        logger.debug(f"Extracted {len(fields)} fields from CSV: {file_path.name}")
        return fields

    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file is empty: {file_path}")
        return []

    except Exception as e:
        logger.error(f"Error reading CSV schema from {file_path}: {e}")
        raise


def get_json_schema(file_path: Union[str, Path], max_depth: int = 3) -> List[str]:
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
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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


def get_csv_example(file_path: Union[str, Path], row_index: int = 0) -> Dict[str, Any]:
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

    try:
        # Read minimal data (just the row we need + headers)
        df = pd.read_csv(file_path, nrows=row_index + 1)

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
            f"Extracted CSV example from {file_path.name}: {len(cleaned_data)} fields"
        )
        return cleaned_data

    except Exception as e:
        logger.error(f"Error extracting CSV example from {file_path}: {e}")
        raise


def get_json_example(
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
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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


def get_data_examples_from_directory(
    directory_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract example data from all CSV and JSON files in a directory

    Args:
        directory_path: Path to directory containing data files

    Returns:
        Dictionary mapping file names to their example data
    """
    directory = Path(directory_path)
    examples = {}

    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Process CSV files
    csv_count = 0
    for csv_file in directory.glob("*.csv"):
        try:
            example = get_csv_example(csv_file)
            examples[csv_file.name] = example
            csv_count += 1
        except Exception as e:
            logger.warning(f"Failed to extract example from {csv_file}: {e}")

    # Process JSON files (skip metadata.json)
    json_count = 0
    for json_file in directory.glob("*.json"):
        if json_file.name == "metadata.json":
            continue

        try:
            example = get_json_example(json_file)
            examples[json_file.name] = example
            json_count += 1
        except Exception as e:
            logger.warning(f"Failed to extract example from {json_file}: {e}")

    logger.info(
        f"Extracted examples from {csv_count} CSV and {json_count} JSON files in {directory.name}"
    )
    return examples


# =============================================================================
# JSON TO STRING CONVERSION FUNCTIONS
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
# UTILITY FUNCTIONS FOR TESTING
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data extraction utilities")
    parser.add_argument("directory", help="Directory path to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Testing directory: {args.directory}")

    # Run tests
    dir_path = Path(args.directory)

    text_for_embeddings = extract_comprehensive_text(dir_path)
    logger.info(f"""\n\n\n{text_for_embeddings}\n\n\n""")

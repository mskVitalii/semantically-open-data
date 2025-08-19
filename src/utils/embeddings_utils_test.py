import json
import logging
import shutil
import tempfile
from pathlib import Path
import asyncio

import pandas as pd
import pytest

# Import the module to test (assuming it's named text_extraction.py)
from src.utils.embeddings_utils import (
    extract_data_content,
    get_csv_example,
    get_json_example,
    json_to_searchable_string,
    _json_to_key_value_string,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_metadata():
    """Sample metadata.json content"""
    return {
        "title": "Test Dataset",
        "description": "A test dataset for unit testing",
        "author": "Test Author",
        "version": "1.0.0",
        "tags": ["test", "sample", "data"],
    }


@pytest.fixture
def sample_csv_data():
    """Sample CSV data"""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Paris"],
        }
    )


@pytest.fixture
def sample_json_data():
    """Sample JSON data"""
    return [
        {"id": 1, "product": "Widget", "price": 10.99, "in_stock": True},
        {"id": 2, "product": "Gadget", "price": 25.50, "in_stock": False},
        {"id": 3, "product": "Doohickey", "price": 5.00, "in_stock": True},
    ]


@pytest.fixture
def nested_json_data():
    """Sample nested JSON data"""
    return {
        "company": "Test Corp",
        "employees": [
            {
                "name": "John Doe",
                "department": {"name": "Engineering", "location": "Building A"},
                "skills": ["Python", "JavaScript"],
            }
        ],
        "metadata": {"created": "2024-01-01", "version": 2},
    }


# =============================================================================
# TEST DATA CONTENT EXTRACTION
# =============================================================================


@pytest.mark.asyncio
async def test_extract_data_content_csv(temp_dir, sample_csv_data):
    """Test data content extraction from CSV files"""
    csv_file = temp_dir / "data.csv"
    sample_csv_data.to_csv(csv_file, index=False)

    result = await extract_data_content(temp_dir)

    assert "CSV_DATA:" in result
    assert "Alice" in result or "Bob" in result  # At least one name should be present


@pytest.mark.asyncio
async def test_extract_data_content_json(temp_dir, sample_json_data):
    """Test data content extraction from JSON files"""
    json_file = temp_dir / "data.json"
    with open(json_file, "w") as f:
        json.dump(sample_json_data, f)

    result = await extract_data_content(temp_dir)

    assert "JSON_DATA:" in result
    assert "Widget" in result or "Gadget" in result


@pytest.mark.asyncio
async def test_extract_data_content_skip_metadata(temp_dir, sample_metadata):
    """Test that metadata.json is skipped in data content extraction"""
    metadata_file = temp_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(sample_metadata, f)

    result = await extract_data_content(temp_dir)

    # Should not contain metadata content as data
    assert "JSON_DATA:" not in result or "Test Dataset" not in result


# =============================================================================
# TEST EXAMPLE EXTRACTION
# =============================================================================


@pytest.mark.asyncio
async def test_get_csv_example_success(temp_dir, sample_csv_data):
    """Test successful CSV example extraction"""
    csv_file = temp_dir / "data.csv"
    sample_csv_data.to_csv(csv_file, index=False)

    example = await get_csv_example(csv_file, row_index=0)

    assert example["id"] == 1
    assert example["name"] == "Alice"
    assert example["age"] == 25
    assert example["city"] == "New York"


@pytest.mark.asyncio
async def test_get_csv_example_with_nan(temp_dir):
    """Test CSV example extraction with NaN values"""
    df = pd.DataFrame({"id": [1], "name": ["Test"], "optional": [pd.NA]})
    csv_file = temp_dir / "data.csv"
    df.to_csv(csv_file, index=False)

    example = await get_csv_example(csv_file)

    assert example["id"] == 1
    assert example["name"] == "Test"
    assert example["optional"] is None


@pytest.mark.asyncio
async def test_get_csv_example_index_out_of_bounds(temp_dir, sample_csv_data):
    """Test CSV example extraction + out of bounds index"""
    csv_file = temp_dir / "data.csv"
    sample_csv_data.to_csv(csv_file, index=False)

    with pytest.raises(IndexError):
        await get_csv_example(csv_file, row_index=10)


@pytest.mark.asyncio
async def test_get_json_example_array(temp_dir, sample_json_data):
    """Test JSON example extraction from array"""
    json_file = temp_dir / "data.json"
    with open(json_file, "w") as f:
        json.dump(sample_json_data, f)

    example = await get_json_example(json_file, object_index=1)

    assert example["id"] == 2
    assert example["product"] == "Gadget"
    assert example["price"] == 25.50


@pytest.mark.asyncio
async def test_get_json_example_object(temp_dir, nested_json_data):
    """Test JSON example extraction from single object"""
    json_file = temp_dir / "data.json"
    with open(json_file, "w") as f:
        json.dump(nested_json_data, f)

    example = await get_json_example(json_file)

    assert example["company"] == "Test Corp"
    assert "employees" in example


@pytest.mark.asyncio
async def test_get_json_example_primitive_value(temp_dir):
    """Test JSON example extraction from primitive value"""
    json_file = temp_dir / "data.json"
    with open(json_file, "w") as f:
        json.dump("just a string", f)

    with pytest.raises(ValueError):
        await get_json_example(json_file)


# =============================================================================
# TEST JSON TO STRING CONVERSION
# =============================================================================


def test_json_to_searchable_string_key_value():
    """Test JSON to key-value string conversion"""
    data = {"name": "Test", "age": 30, "city": "New York"}

    result = json_to_searchable_string(data)

    assert "name: Test" in result
    assert "age: 30" in result
    assert "city: New York" in result


def test_json_to_searchable_string_compact():
    """Test JSON to compact string conversion"""
    data = {"id": 1, "name": "Test"}

    result = json_to_searchable_string(data)

    assert "id: 1" in result
    assert "name: Test" in result
    assert "{" not in result  # Braces should be removed


def test_json_to_searchable_string_max_length():
    """Test JSON string conversion respects max_length"""
    data = {"very_long_key": "x" * 1000}

    result = json_to_searchable_string(data, max_length=50)

    assert len(result) <= 50


def test_json_to_searchable_string_nested():
    """Test JSON string conversion with nested objects"""
    data = {"user": {"name": "John", "address": {"city": "NYC"}}}

    result = json_to_searchable_string(data, max_depth=2)

    assert "user:" in result
    assert "[name: John" in result


def test_json_to_key_value_string_with_list():
    """Test key-value conversion with lists"""
    data = {"tags": ["python", "testing", "code"], "numbers": [1, 2, 3, 4, 5]}

    result = _json_to_key_value_string(data, max_length=500, max_depth=2)

    assert "tags: [python, testing, code]" in result
    assert "numbers: [1, 2, 3...]" in result


# =============================================================================
# TEST ERROR HANDLING AND EDGE CASES
# =============================================================================


def test_logging_output(caplog):
    """Test that logging works correctly"""
    with caplog.at_level(logging.DEBUG):
        data = {"test": "value"}
        json_to_searchable_string(data)

        assert "Converted JSON to" in caplog.text


@pytest.mark.asyncio
async def test_unicode_handling(temp_dir):
    """Test handling of Unicode characters"""
    data = {"name": "José", "city": "São Paulo", "emoji": "😊"}
    json_file = temp_dir / "unicode.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    example = await get_json_example(json_file)

    assert example["name"] == "José"
    assert example["city"] == "São Paulo"
    assert example["emoji"] == "😊"


@pytest.mark.asyncio
async def test_large_csv_performance(temp_dir):
    """Test performance with larger CSV files"""
    # Create a CSV with 1000 rows
    large_df = pd.DataFrame(
        {"id": range(1000), "value": [f"value_{i}" for i in range(1000)]}
    )
    csv_file = temp_dir / "large.csv"
    large_df.to_csv(csv_file, index=False)

    # Should only read the first row
    example = await get_csv_example(csv_file)

    assert example["id"] == 0
    assert example["value"] == "value_0"


@pytest.mark.asyncio
async def test_concurrent_file_access(temp_dir, sample_csv_data):
    """Test handling of concurrent file access"""
    csv_file = temp_dir / "data.csv"
    sample_csv_data.to_csv(csv_file, index=False)

    # Simulate multiple concurrent reads
    tasks = []
    for i in range(3):
        task = get_csv_example(csv_file, row_index=i)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert results[0]["name"] == "Alice"
    assert results[1]["name"] == "Bob"
    assert results[2]["name"] == "Charlie"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

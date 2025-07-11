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
    extract_comprehensive_text,
    extract_metadata_text,
    extract_data_content,
    get_csv_schema,
    get_json_schema,
    get_csv_example,
    get_json_example,
    json_to_searchable_string,
    flatten_json_to_string,
    _extract_json_fields,
    _json_to_key_value_string,
    _json_to_natural_string,
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
# TEST COMPREHENSIVE TEXT EXTRACTION
# =============================================================================


@pytest.mark.asyncio
async def test_extract_comprehensive_text_success(
    temp_dir, sample_metadata, sample_csv_data
):
    """Test successful comprehensive text extraction"""
    # Setup
    metadata_file = temp_dir / "metadata.json"
    csv_file = temp_dir / "data.csv"

    with open(metadata_file, "w") as f:
        json.dump(sample_metadata, f)

    sample_csv_data.to_csv(csv_file, index=False)

    # Test
    result = await extract_comprehensive_text(temp_dir)

    # Assert
    assert "METADATA:" in result
    assert "CONTENT:" in result
    assert "Test Dataset" in result
    assert "CSV_DATA:" in result


@pytest.mark.asyncio
async def test_extract_comprehensive_text_missing_path():
    """Test comprehensive text extraction with non-existent path"""
    result = await extract_comprehensive_text(Path("/non/existent/path"))
    assert result == ""


# =============================================================================
# TEST METADATA EXTRACTION
# =============================================================================


@pytest.mark.asyncio
async def test_extract_metadata_text_success(temp_dir, sample_metadata):
    """Test successful metadata extraction"""
    metadata_file = temp_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(sample_metadata, f)

    result = await extract_metadata_text(metadata_file)

    assert "title: Test Dataset" in result
    assert "description: A test dataset for unit testing" in result
    assert "author: Test Author" in result


@pytest.mark.asyncio
async def test_extract_metadata_text_missing_file():
    """Test metadata extraction with missing file"""
    result = await extract_metadata_text(Path("/non/existent/metadata.json"))
    assert result == ""


@pytest.mark.asyncio
async def test_extract_metadata_text_invalid_json(temp_dir):
    """Test metadata extraction with invalid JSON"""
    metadata_file = temp_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        f.write("invalid json content")

    result = await extract_metadata_text(metadata_file)
    assert result == ""


@pytest.mark.asyncio
async def test_extract_metadata_text_empty_values(temp_dir):
    """Test metadata extraction with empty string values"""
    metadata = {
        "title": "Test",
        "description": "",
        "author": "   ",
        "valid": "Valid content",
    }
    metadata_file = temp_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    result = await extract_metadata_text(metadata_file)

    assert "title: Test" in result
    assert "valid: Valid content" in result
    assert "description:" not in result
    assert "author:" not in result


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
# TEST SCHEMA EXTRACTION
# =============================================================================


@pytest.mark.asyncio
async def test_get_csv_schema_success(temp_dir, sample_csv_data):
    """Test successful CSV schema extraction"""
    csv_file = temp_dir / "data.csv"
    sample_csv_data.to_csv(csv_file, index=False)

    schema = await get_csv_schema(csv_file)

    assert schema == ["id", "name", "age", "city"]


@pytest.mark.asyncio
async def test_get_csv_schema_empty_file(temp_dir):
    """Test CSV schema extraction from empty file"""
    csv_file = temp_dir / "empty.csv"
    csv_file.touch()

    schema = await get_csv_schema(csv_file)

    assert schema == []


@pytest.mark.asyncio
async def test_get_csv_schema_missing_file():
    """Test CSV schema extraction with missing file"""
    with pytest.raises(FileNotFoundError):
        await get_csv_schema(Path("/non/existent/file.csv"))


@pytest.mark.asyncio
async def test_get_json_schema_simple(temp_dir, sample_json_data):
    """Test JSON schema extraction from simple data"""
    json_file = temp_dir / "data.json"
    with open(json_file, "w") as f:
        json.dump(sample_json_data[0], f)

    schema = await get_json_schema(json_file)

    assert "id" in schema
    assert "product" in schema
    assert "price" in schema
    assert "in_stock" in schema


@pytest.mark.asyncio
async def test_get_json_schema_nested(temp_dir, nested_json_data):
    """Test JSON schema extraction from nested data"""
    json_file = temp_dir / "nested.json"
    with open(json_file, "w") as f:
        json.dump(nested_json_data, f)

    schema = await get_json_schema(json_file, max_depth=3)

    assert "company" in schema
    assert "employees" in schema
    assert "employees.name" in schema
    assert "employees.department" in schema
    assert "employees.department.name" in schema
    assert "metadata.created" in schema


@pytest.mark.asyncio
async def test_get_json_schema_max_depth(temp_dir, nested_json_data):
    """Test JSON schema extraction respects max_depth"""
    json_file = temp_dir / "nested.json"
    with open(json_file, "w") as f:
        json.dump(nested_json_data, f)

    schema = await get_json_schema(json_file, max_depth=1)

    assert "company" in schema
    assert "employees" in schema
    assert "employees.name" not in schema  # Should not go deeper than max_depth


def test_extract_json_fields_array():
    """Test _extract_json_fields with arrays"""
    data = {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}

    fields = _extract_json_fields(data, max_depth=2)

    assert "items" in fields
    assert "items.id" in fields
    assert "items.name" in fields


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

    result = json_to_searchable_string(data, format_style="key_value")

    assert "name: Test" in result
    assert "age: 30" in result
    assert "city: New York" in result


def test_json_to_searchable_string_natural():
    """Test JSON to natural language string conversion"""
    data = {"user_name": "John", "total_orders": 5}

    result = json_to_searchable_string(data, format_style="natural")

    assert "user name is John" in result
    assert "total orders is 5" in result


def test_json_to_searchable_string_compact():
    """Test JSON to compact string conversion"""
    data = {"id": 1, "name": "Test"}

    result = json_to_searchable_string(data, format_style="compact")

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

    result = json_to_searchable_string(data, format_style="key_value", max_depth=2)

    assert "user:" in result
    assert "[name: John" in result


def test_json_to_searchable_string_non_dict():
    """Test JSON string conversion with non-dict input"""
    result = json_to_searchable_string([1, 2, 3], max_length=10)

    assert result == "[1, 2, 3]"


def test_json_to_key_value_string_with_list():
    """Test key-value conversion with lists"""
    data = {"tags": ["python", "testing", "code"], "numbers": [1, 2, 3, 4, 5]}

    result = _json_to_key_value_string(data, max_length=500, max_depth=2)

    assert "tags: [python, testing, code]" in result
    assert "numbers: [1, 2, 3...]" in result


def test_json_to_natural_string_with_list():
    """Test natural language conversion with lists"""
    data = {"skills": ["Python", "JavaScript"]}

    result = _json_to_natural_string(data, max_length=500)

    assert "skills includes Python, JavaScript" in result


def test_flatten_json_to_string():
    """Test flattening JSON to string"""
    data = {"name": "Test", "details": {"age": 30, "city": "NYC"}, "tags": ["a", "b"]}

    result = flatten_json_to_string(data, separator=" ", include_keys=True)

    assert "name" in result
    assert "Test" in result
    assert "details" in result
    assert "30" in result
    assert "NYC" in result


def test_flatten_json_to_string_without_keys():
    """Test flattening JSON without including keys"""
    data = {"name": "Test", "age": 30}

    result = flatten_json_to_string(data, include_keys=False)

    assert "name" not in result
    assert "Test" in result
    assert "30" in result


def test_flatten_json_to_string_max_items():
    """Test flattening JSON respects max_items"""
    data = {f"key{i}": f"value{i}" for i in range(100)}

    result = flatten_json_to_string(data, max_items=10)

    items = result.split(" ")
    assert len(items) <= 10


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

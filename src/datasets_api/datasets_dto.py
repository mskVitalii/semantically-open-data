import json

from pydantic import BaseModel, Field
from typing import Optional, List
from dataclasses import dataclass

from src.datasets.datasets_metadata import (
    DatasetMetadataWithContent,
    DatasetJSONEncoder,
)


class DatasetSearchRequest(BaseModel):
    """DTO for dataset search request"""

    query: Optional[str] = Field(None, description="Search query")
    tags: Optional[List[str]] = Field(None, description="List of tags for filtering")
    limit: int = Field(10, ge=1, le=100, description="Number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class DatasetResponse(BaseModel):
    """DTO for dataset information response"""

    score: float
    metadata: DatasetMetadataWithContent

    class Config:
        from_attributes = True

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(
            self,
            indent=2,
            ensure_ascii=False,
            cls=DatasetJSONEncoder,
        )


class DatasetSearchResponse(BaseModel):
    """DTO for dataset search response"""

    datasets: List[DatasetResponse]
    total: int
    limit: int
    offset: int


@dataclass
class SearchCriteria:
    """Search criteria for datasets"""

    query: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 10
    offset: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# class QAResponse(BaseModel):

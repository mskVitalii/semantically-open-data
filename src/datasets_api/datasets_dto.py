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

    model_config = {"from_attributes": True}

    def to_json(self) -> str:
        data = self.model_dump()
        if isinstance(self.metadata, DatasetMetadataWithContent):
            data["metadata"] = self.metadata.to_json()
        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            cls=DatasetJSONEncoder,
        )

    def to_dict(self):
        return self.model_dump()


class DatasetSearchResponse(BaseModel):
    """DTO for dataset search response"""

    datasets: list[DatasetResponse]
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
if __name__ == "__main__":
    resp = DatasetResponse(
        score=0.95, metadata=DatasetMetadataWithContent(id="123", title="Demo dataset")
    )

    print("model_dump():", resp.model_dump())
    print("json.dumps():", json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
    print("pydantic .model_dump_json():", resp.model_dump_json(indent=2))

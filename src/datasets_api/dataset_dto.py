from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass


class DatasetSearchRequest(BaseModel):
    """DTO for dataset search request"""

    query: Optional[str] = Field(None, description="Search query")
    tags: Optional[List[str]] = Field(None, description="List of tags for filtering")
    limit: int = Field(10, ge=1, le=100, description="Number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class DatasetResponse(BaseModel):
    """DTO for dataset information response"""

    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


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

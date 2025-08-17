import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional

from src.utils.embeddings_utils import format_metadata_text


@dataclass
class DatasetMetadata:
    """Simple dataset metadata structure"""

    id: str
    title: str
    description: Optional[str] = None
    organization: Optional[str] = None
    metadata_created: Optional[str] = None
    metadata_modified: Optional[str] = None
    city: Optional[str] = None  # Chemnitz
    state: Optional[str] = None  # Saxony
    country: Optional[str] = None  # Germany
    tags: Optional[list[str]] = None
    groups: Optional[list[str]] = None
    url: Optional[str] = None
    author: Optional[Any] = None

    def to_searchable_text(self) -> str:
        """Combine title and description for embedding"""
        payload = {
            "title": self.title,
            "description": self.description,
            "organization": self.organization,
            "metadata_created": self.metadata_created,
            "metadata_modified": self.metadata_modified,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "tags": self.tags,
            "groups": self.groups,
        }
        return format_metadata_text(payload)

        # return f"{self.title}\n{self.description}" if self.description else self.title

    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "organization": self.organization,
            "metadata_created": self.metadata_created,
            "metadata_modified": self.metadata_modified,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "tags": self.tags,
            "groups": self.groups,
            "url": self.url,
            "author": self.author,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class DatasetMetadataWithContent(DatasetMetadata):
    """Dataset metadata with additional content field"""

    fields: Optional[str] = None

    def to_searchable_text(self) -> str:
        """Combine title, description, and content for embedding"""
        base_text = super().to_searchable_text()

        # If you want to include content in the searchable text
        if self.fields:
            return f"{base_text}\n{self.fields}"
        return base_text

    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload including content"""
        payload = super().to_payload()
        payload["content"] = self.fields
        return payload

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(
            self,
            indent=2,
            ensure_ascii=False,
            cls=DatasetJSONEncoder,
        )


class DatasetJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles DatasetMetadata objects and datetime"""

    def default(self, obj):
        if isinstance(obj, (DatasetMetadata, DatasetMetadataWithContent)):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DatasetMetadata:
    """Simple dataset metadata structure"""

    id: str
    title: str
    description: str
    city: str
    organization: str
    metadata_created: str
    metadata_modified: str
    # TODO: add more fields

    def to_searchable_text(self) -> str:
        """Combine title and description for embedding"""
        return f"{self.title}\n{self.description}" if self.description else self.title

    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "city": self.city,
            "organization": self.organization,
            "metadata_created": self.metadata_created,
            "metadata_modified": self.metadata_modified,
        }

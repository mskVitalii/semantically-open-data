from dataclasses import dataclass
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


# if __name__ == "__main__":
#     sample_dataset = DatasetMetadata(
#         id="4db3895e-92a9-4bb7-bb33-f792178d331f",
#         title="Landtagswahl 2024: Wahlbezirksergebnisse",
#         description="Der vorliegende Datensatz präsentiert die Ergebnisse der Wahl zum Sächsischen Landtag 2024 in den Leipziger Wahlbezirken.",
#         organization="Amt für Statistik und Wahlen",
#         metadata_created="2024-09-18T09:50:49.841530",
#         metadata_modified="2024-11-15T14:04:14.392074",
#         city="Leipzig",
#         state="Saxony",
#         country="Germany",
#     )
#     print(sample_dataset.to_searchable_text())

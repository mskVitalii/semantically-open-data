import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)
import numpy as np
import logging
import sys

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "datasets_metadata"
EMBEDDING_DIM = 512  # Reduced dimension for performance
ENV = os.getenv("ENV", "development")

# Setup logging based on environment
log_handlers = [logging.StreamHandler(sys.stdout)]
if ENV != "production":
    # Only add file handler in development
    os.makedirs("./logs", exist_ok=True)
    log_handlers.append(logging.FileHandler("./logs/embeddings_and_qdrant.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)


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


class LocalJinaEmbedder:
    """Local Jina embeddings using sentence-transformers"""

    def __init__(
        self, model_name: str = "jinaai/jina-embeddings-v3", dimensions: int = 512
    ):
        logger.info(f"Loading model {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.dimensions = dimensions

        # Set device
        if torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def embed(self, text: str) -> np.ndarray:
        """Embed single text"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Truncate to desired dimensions
        return embedding[: self.dimensions]

    def embed_batch(self, texts: List[str], batch_size: int = 4) -> List[np.ndarray]:
        """Embed multiple texts with small batch size for M1"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        # Truncate each embedding
        return [emb[: self.dimensions] for emb in embeddings]


class RAGSystem:
    """Simple RAG system for MVP"""

    def __init__(self):
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False
        )

        # Initialize embedder (using v3 for better performance on limited hardware)
        self.embedder = LocalJinaEmbedder(
            model_name="jinaai/jina-embeddings-v3", dimensions=EMBEDDING_DIM
        )

        # Setup collection
        self._setup_collection()

    def _setup_collection(self):
        """Create Qdrant collection if not exists"""
        collections = self.qdrant.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            logger.info(f"Creating collection {COLLECTION_NAME}")
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )

            # Create indexes for filtering
            for field in ["city", "organization"]:
                self.qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info("Collection created with indexes")
        else:
            logger.info(f"Collection {COLLECTION_NAME} already exists")

    def index_datasets(self, datasets: List[DatasetMetadata]):
        """Index multiple datasets"""
        logger.info(f"Indexing {len(datasets)} datasets...")

        # Prepare texts for embedding
        texts = [ds.to_searchable_text() for ds in datasets]

        # Generate embeddings in batches
        embeddings = self.embedder.embed_batch(texts, batch_size=4)

        # Prepare points for Qdrant
        points = [
            PointStruct(id=idx, vector=embedding.tolist(), payload=dataset.to_payload())
            for idx, (dataset, embedding) in enumerate(zip(datasets, embeddings))
        ]

        # Upload to Qdrant
        self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Successfully indexed {len(datasets)} datasets")

    def search(self, query: str, city_filter: str = None, limit: int = 5):
        """Search for datasets"""
        logger.info(f"\nSearching for: '{query}'")
        if city_filter:
            logger.info(f"Filtering by city: {city_filter}")

        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Build filter if city specified
        search_filter = None
        if city_filter:
            search_filter = Filter(
                must=[FieldCondition(key="city", match=MatchValue(value=city_filter))]
            )

        # Search
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

        logger.info(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. Score: {result.score:.4f}")
            logger.info(f"   Title: {result.payload['title']}")
            logger.info(f"   City: {result.payload['city']}")
            logger.info(f"   Organization: {result.payload['organization']}")
            if result.payload.get("description"):
                desc = (
                    result.payload["description"][:200] + "..."
                    if len(result.payload["description"]) > 200
                    else result.payload["description"]
                )
                logger.info(f"   Description: {desc}")

        return results


def main():
    """Main MVP function"""
    logger.info("Starting RAG MVP System...")

    # Wait for Qdrant to be ready
    time.sleep(5)

    # Initialize system
    rag = RAGSystem()

    # Sample data - your actual metadata
    sample_datasets = [
        DatasetMetadata(
            id="4db3895e-92a9-4bb7-bb33-f792178d331f",
            title="Landtagswahl 2024: Wahlbezirksergebnisse",
            description="Der vorliegende Datensatz präsentiert die Ergebnisse der Wahl zum Sächsischen Landtag 2024 in den Leipziger Wahlbezirken.",
            city="Leipzig",
            organization="Amt für Statistik und Wahlen",
            metadata_created="2024-09-18T09:50:49.841530",
            metadata_modified="2024-11-15T14:04:14.392074",
        ),
        DatasetMetadata(
            id="bd1e8543af3e43eeac6b7423abcb424b",
            title="Bodenrichtwerte_1998",
            description="Bodenrichtwerte für Chemnitz aus dem Jahr 1998",
            city="Chemnitz",
            organization="Stadtverwaltung Chemnitz",
            metadata_created="1998-01-01T00:00:00",
            metadata_modified="2024-01-01T00:00:00",
        ),
        DatasetMetadata(
            id="d0d8ba5f-40b1-4c4e-84de-3e9e91035add",
            title="Einwohnerinnen und Einwohner in den Ortsteilen Berlins am 31.12.2011",
            description="Einwohnerinnen und Einwohner in den Ortsteilen Berlins am 31.12.2011",
            city="Berlin",
            organization="Amt für Statistik Berlin-Brandenburg",
            metadata_created="2012-01-15T00:00:00",
            metadata_modified="2024-01-01T00:00:00",
        ),
        DatasetMetadata(
            id="dataset_4",
            title="Einwohner - Wanderungen - bezogen auf Dresdner Basiswohnung",
            description="Wanderungsbewegungen und Migrationsdaten für Dresden",
            city="Dresden",
            organization="Landeshauptstadt Dresden",
            metadata_created="2018-01-01T00:00:00",
            metadata_modified="2024-01-01T00:00:00",
        ),
        DatasetMetadata(
            id="dataset_5",
            title="Kommunalwahl 2024: Ergebnisse Leipzig",
            description="Detaillierte Ergebnisse der Kommunalwahl 2024 in Leipzig nach Wahlbezirken",
            city="Leipzig",
            organization="Amt für Statistik und Wahlen",
            metadata_created="2024-06-01T00:00:00",
            metadata_modified="2024-06-15T00:00:00",
        ),
    ]

    # Index the data
    rag.index_datasets(sample_datasets)

    logger.info("\n" + "=" * 50)
    logger.info("DATA INDEXED SUCCESSFULLY!")
    logger.info("=" * 50)

    # Example queries
    queries = [
        ("Какая явка была на выборах в Лейпциге?", "Leipzig"),
        ("election results", None),
        ("Bevölkerung Berlin", "Berlin"),
        ("Wahlergebnisse 2024", None),
        ("migration data Dresden", "Dresden"),
    ]

    # Run searches
    for query, city_filter in queries:
        logger.info("\n" + "-" * 50)
        rag.search(query, city_filter)
        time.sleep(1)  # Small delay between queries

    logger.info("\n" + "=" * 50)
    logger.info("MVP DEMO COMPLETE!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

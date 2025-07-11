import time
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryRequest, ScoredPoint
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)
import logging

from src.datasets.datasets_metadata import DatasetMetadataWithContent
from src.infrastructure.config import (
    USE_GRPC,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    EMBEDDING_DIM,
    QDRANT_COLLECTION_NAME,
)
from src.vector_search.embedder import LocalJinaEmbedder


logger = logging.getLogger(__name__)


class VectorDB:
    """Vector DB system with gRPC support"""

    def __init__(self, use_grpc: bool = True):
        """Initialize with gRPC or HTTP client"""
        # Use environment variable if not explicitly set
        self.use_grpc = use_grpc if use_grpc is not None else USE_GRPC

        # Initialize Qdrant client
        if self.use_grpc:
            logger.info(
                f"Connecting to Qdrant via gRPC at {QDRANT_HOST}:{QDRANT_GRPC_PORT}"
            )
            self.qdrant = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_GRPC_PORT,
                grpc_port=QDRANT_GRPC_PORT,
                prefer_grpc=True,
                timeout=30,
            )
        else:
            logger.info(
                f"Connecting to Qdrant via HTTP at {QDRANT_HOST}:{QDRANT_HTTP_PORT}"
            )
            self.qdrant = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_HTTP_PORT,
                prefer_grpc=False,
            )

        # Wait for Qdrant to be ready
        self._wait_for_qdrant()

        # Initialize embedder
        self.embedder = LocalJinaEmbedder(
            model_name="jinaai/jina-embeddings-v4", dimensions=EMBEDDING_DIM
        )

        # Setup collection
        self._setup_collection()

    def _wait_for_qdrant(self, max_retries: int = 10, retry_delay: int = 2):
        """Wait for Qdrant to be ready"""
        for i in range(max_retries):
            try:
                self.qdrant.get_collections()
                logger.info(
                    f"Qdrant is ready! (Using {'gRPC' if self.use_grpc else 'HTTP'})"
                )
                return
            except Exception as e:
                if i < max_retries - 1:
                    logger.info(f"Waiting for Qdrant... ({i + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Qdrant failed to become ready: {e}")

    def _setup_collection(self):
        """Create Qdrant collection if not exists"""
        collections = self.qdrant.get_collections().collections
        if not any(c.name == QDRANT_COLLECTION_NAME for c in collections):
            logger.info(f"Creating collection {QDRANT_COLLECTION_NAME}")
            self.qdrant.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )

            # Create indexes for filtering
            for field in ["city", "organization"]:
                self.qdrant.create_payload_index(
                    collection_name=QDRANT_COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info("Collection created with indexes")
        else:
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} already exists")

    def index_datasets(
        self, datasets: List[DatasetMetadataWithContent], batch_size: int = 100
    ):
        """Index multiple datasets with batching for better performance"""
        logger.info(f"Indexing {len(datasets)} datasets...")

        # Prepare texts for embedding
        texts = [ds.to_searchable_text() for ds in datasets]

        # Generate vector_search in batches
        embeddings = self.embedder.embed_batch(texts, batch_size=4)

        # Prepare points for Qdrant
        points = [
            PointStruct(id=idx, vector=embedding.tolist(), payload=dataset.to_payload())
            for idx, (dataset, embedding) in enumerate(zip(datasets, embeddings))
        ]

        # Upload to Qdrant in batches for better performance
        total_points = len(points)
        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            self.qdrant.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=batch,
                wait=True,  # Ensure consistency
            )
            if len(datasets) > batch_size:
                logger.info(
                    f"Uploaded batch {i // batch_size + 1}/{(total_points + batch_size - 1) // batch_size}"
                )

        logger.info(f"Successfully indexed {len(datasets)} datasets")

    def search(
        self, query: str, city_filter: Optional[str] = None, limit: int = 5
    ) -> list[ScoredPoint]:
        """Search for datasets using query_points method"""
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

        # Use query_points (works with both gRPC and HTTP)
        query_result = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding.tolist(),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

        # Extract points from the result
        results = query_result.points
        return results

    def batch_search(
        self, queries: List[str], city_filter: Optional[str] = None, limit: int = 5
    ):
        """Batch search - especially efficient with gRPC"""
        logger.info(f"\nBatch searching for {len(queries)} queries")

        # Generate vector_search for all queries
        query_embeddings = [self.embedder.embed(q) for q in queries]

        # Build filter
        search_filter = None
        if city_filter:
            search_filter = Filter(
                must=[FieldCondition(key="city", match=MatchValue(value=city_filter))]
            )

        # Batch query - very efficient with gRPC
        batch_results = self.qdrant.query_batch_points(
            collection_name=QDRANT_COLLECTION_NAME,
            requests=[
                QueryRequest(
                    query=emb.tolist(),
                    filter=search_filter,
                    limit=limit,
                    with_payload=True,
                )
                for emb in query_embeddings
            ],
        )

        # Process results
        all_results = []
        for i, (query, result) in enumerate(zip(queries, batch_results)):
            logger.info(f"\nQuery {i + 1}: '{query}'")
            logger.info(f"Found {len(result.points)} results")
            all_results.append(result.points)

        return all_results

    def get_stats(self):
        """Get collection statistics"""
        info = self.qdrant.get_collection(QDRANT_COLLECTION_NAME)
        logger.info("\nCollection stats:")
        logger.info(f"  Vectors count: {info.vectors_count}")
        logger.info(f"  Points count: {info.points_count}")
        logger.info(f"  Indexed vectors: {info.indexed_vectors_count}")
        logger.info(f"  Protocol: {'gRPC' if self.use_grpc else 'HTTP'}")
        return info

    async def remove_collection(
        self, collection_name: str = QDRANT_COLLECTION_NAME
    ) -> bool:
        """
        Remove a collection from Qdrant.

        Args:
            collection_name: Name of the collection to remove.
                            Defaults to QDRANT_COLLECTION_NAME if not provided.

        Returns:
            bool: True if collection was removed successfully, False otherwise.
        """
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False

            # Delete the collection
            logger.info(f"Removing collection '{collection_name}'...")
            self.qdrant.delete_collection(collection_name=collection_name)

            logger.info(f"✅ Collection '{collection_name}' removed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to remove collection '{collection_name}': {e}")
            raise

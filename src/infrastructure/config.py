import os

# Get all the env variables & optional validation

ENV = os.getenv("ENV", "development")

USE_GRPC = os.getenv("USE_GRPC", "true").lower() == "true"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", 6333))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
COLLECTION_NAME = "datasets_metadata"
EMBEDDING_DIM = 512

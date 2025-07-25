import os

# Get all the env variables & optional validation

ENV = os.getenv("ENV", "development")
IS_DOCKER = os.getenv("IS_DOCKER", "false") == "true"
USE_GRPC = os.getenv("USE_GRPC", "true").lower() == "true"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", 6333))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
QDRANT_COLLECTION_NAME = "datasets_metadata"

EMBEDDING_DIM = 1024

EMBEDDER_HOST = os.getenv("EMBEDDER_HOST", "localhost")
EMBEDDER_PORT = int(os.getenv("EMBEDDER_PORT", 8080))
EMBEDDER_URL = f"http://{EMBEDDER_HOST}:{EMBEDDER_PORT}"

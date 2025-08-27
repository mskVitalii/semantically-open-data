import os
from dotenv import load_dotenv

load_dotenv()
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

MONGO_INITDB_DATABASE = os.getenv("MONGO_INITDB_DATABASE", "db")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USER = os.getenv("MONGO_USER", "appuser")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "")
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@mongodb:{MONGO_PORT}/{MONGO_INITDB_DATABASE}?authSource={MONGO_INITDB_DATABASE}",
)

LLM_HOST = os.getenv("LLM_HOST", "localhost")
LLM_PORT = int(os.getenv("LLM_PORT", 11434))
LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}"

os.environ["GRPC_VERBOSITY"] = "NONE"

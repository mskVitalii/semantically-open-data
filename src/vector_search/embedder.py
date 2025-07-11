from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.infrastructure.config import IS_DOCKER, EMBEDDING_DIM
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class LocalJinaEmbedder:
    """Local Jina vector_search using sentence-transformers"""

    def __init__(
        self, model_name: str = "jinaai/jina-embeddings-v4", dimensions: int = 512
    ):
        # Set device
        if torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Loading model {model_name} on {self.device}...")
        self.model = SentenceTransformer(
            model_name,
            cache_folder="app/cache" if IS_DOCKER else "../../cache",
            trust_remote_code=True,
            device=self.device,
            revision="1e94d7f53488267e2a5a07a2656d0c943a8c3710",
            model_kwargs={"default_task": "retrieval"},
        )
        self.dimensions = dimensions

        logger.info(f"Model {model_name} loaded")

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
            show_progress_bar=False,
        )
        # Truncate each embedding
        return [emb[: self.dimensions] for emb in embeddings]

    def __del__(self):
        logger.warning("LocalJinaEmbedder instance is being deleted".upper())


_embedder_instance = None


def get_embedder() -> LocalJinaEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = LocalJinaEmbedder(
            model_name="jinaai/jina-embeddings-v4", dimensions=EMBEDDING_DIM
        )
    return _embedder_instance

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class LocalJinaEmbedder:
    """Local Jina vector_search using sentence-transformers"""

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

import httpx
import numpy as np

from src.infrastructure.config import EMBEDDER_URL
from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__, "EMBEDDER")


async def embed(text: str) -> np.ndarray:
    """Embed single text"""
    res = await embed_batch([text])
    return res[0]


async def embed_batch(texts: list[str]) -> list[np.ndarray]:
    logger.info(f"working with texts {len(texts)}")
    logger.info(f"URL: {EMBEDDER_URL}")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=EMBEDDER_URL + "/embed",
            json={"texts": texts},
            timeout=len(texts) * 5,
        )
    response.raise_for_status()
    data = response.json()["embeddings"]
    return [np.array(vec, dtype=np.float32) for vec in data]

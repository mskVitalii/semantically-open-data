# main.py or dependencies.py
from src.vector_search.vector_db import VectorDB, create_vector_db

vector_db: VectorDB | None = None


async def get_vector_db() -> VectorDB:
    global vector_db
    if vector_db is None:
        vector_db = await create_vector_db()
    return vector_db

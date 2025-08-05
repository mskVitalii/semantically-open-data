from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional, Annotated

from src.infrastructure.config import MONGODB_URI, MONGO_INITDB_DATABASE
from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__, "MONGO")


class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    database: Optional[AsyncIOMotorDatabase] = None


db = MongoDB()


async def connect_to_mongo():
    """Create connection to MongoDB"""
    db.client = AsyncIOMotorClient(MONGODB_URI)
    db.database = db.client[MONGO_INITDB_DATABASE]

    # Проверка подключения
    await db.client.admin.command("ping")
    logger.info("Connected to MongoDB")


async def close_mongo_connection():
    """Close connection to MongoDB"""
    if db.client:
        db.client.close()
        logger.info("Disconnected from MongoDB")


def get_mongo_database() -> AsyncIOMotorDatabase:
    """Get DB instance"""
    return db.database


async def get_mongo_client() -> AsyncIOMotorClient:
    """Dependency для получения клиента MongoDB"""
    if db.client is None:
        raise RuntimeError("MongoDB client is not initialized")
    return db.client


MongoDBDep = Annotated[AsyncIOMotorDatabase, Depends(get_mongo_database)]
MongoClientDep = Annotated[AsyncIOMotorClient, Depends(get_mongo_client)]

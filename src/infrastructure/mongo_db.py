# src/infrastructure/mongo_db.py
from typing import Optional, Annotated
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from fastapi import Depends
from pymongo.errors import PyMongoError

from src.infrastructure.config import MONGODB_URI, MONGO_INITDB_DATABASE
from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__, "MONGO")


class MongoDBManager:
    """Manager for handling MongoDB connection"""

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        is_testing: bool = False,
    ):
        self.mongodb_uri = mongo_uri or MONGODB_URI
        self.database_name = database_name or MONGO_INITDB_DATABASE
        self.is_testing = is_testing
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None

    @property
    def client(self) -> AsyncIOMotorClient:
        """Get MongoDB client"""
        if self._client is None:
            raise RuntimeError("MongoDB client is not initialized")
        return self._client

    @property
    def database(self) -> AsyncIOMotorDatabase:
        """Get MongoDB database"""
        if self._database is None:
            raise RuntimeError("MongoDB database is not initialized")
        return self._database

    async def connect(self):
        """Create connection to MongoDB"""
        if self.is_testing:
            # Use mongomock for tests
            from mongomock_motor import AsyncMongoMockClient

            self._client = AsyncMongoMockClient()
        else:
            self._client = AsyncIOMotorClient(self.mongodb_uri)

        self._database = self._client[self.database_name]

        # Check connection (only for real DB)
        if not self.is_testing:
            await self._client.admin.command("ping")
            logger.info(f"Connected to MongoDB: {self.database_name}")

    async def disconnect(self):
        """Close connection to MongoDB"""
        if self._client:
            self._client.close()
            logger.info("Disconnected from MongoDB")
            self._client = None
            self._database = None

    @asynccontextmanager
    async def lifespan(self):
        """Context manager for lifecycle management"""
        await self.connect()
        try:
            yield
        finally:
            await self.disconnect()

    async def ping(self) -> bool:
        """Check MongoDB availability"""
        try:
            if self.is_testing:
                return True  # mongomock is always available
            await self.client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    async def get_database_stats(self) -> dict:
        """Get database statistics"""
        if self.is_testing:
            # Return mock data for tests
            return {
                "collections": 0,
                "objects": 0,
                "dataSize": 0,
                "storageSize": 0,
            }
        return await self.database.command("dbStats")


# Singleton instance for production use
_mongodb_manager: Optional[MongoDBManager] = None


def get_mongodb_manager() -> MongoDBManager:
    """Get MongoDB manager instance"""
    global _mongodb_manager
    if _mongodb_manager is None:
        _mongodb_manager = MongoDBManager()
    return _mongodb_manager


def set_mongodb_manager(manager: MongoDBManager):
    """Set custom manager (for tests)"""
    global _mongodb_manager
    _mongodb_manager = manager


# FastAPI dependencies
async def get_mongo_database() -> AsyncIOMotorDatabase:
    """Dependency to get database"""
    manager = get_mongodb_manager()
    return manager.database


async def get_mongo_client() -> AsyncIOMotorClient:
    """Dependency to get client"""
    manager = get_mongodb_manager()
    return manager.client


MongoDBDep = Annotated[AsyncIOMotorDatabase, Depends(get_mongo_database)]
MongoClientDep = Annotated[AsyncIOMotorClient, Depends(get_mongo_client)]

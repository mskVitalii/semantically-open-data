# src/datasets/repository.py
from typing import Optional, Any
from datetime import datetime, UTC

from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ReturnDocument
from bson import ObjectId
from pymongo.errors import PyMongoError

from src.datasets.datasets_metadata import Dataset
from src.infrastructure.logger import get_prefixed_logger
from src.infrastructure.mongo_db import MongoDBDep
from src.utils.datasets_utils import sanitize_title

logger = get_prefixed_logger(__name__, "DATASET_REPOSITORY")


class DatasetRepository:
    """Repository for datasets in MongoDB"""

    META_COLLECTION_NAME = "metadata"

    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.meta_collection = database[self.META_COLLECTION_NAME]

    async def create_indexes(self):
        """Create indexes for better performance"""
        # Index for text search
        await self.meta_collection.create_index(
            [("title", "text"), ("description", "text")]
        )
        # Compound index for common queries
        # await self.meta_collection.create_index([("city", 1), ("organization", 1)])
        logger.info("Indexes created successfully")

    @staticmethod
    async def insert_one(
        dataset: dict[str, Any], collection: AsyncIOMotorCollection
    ) -> str:
        """Insert single dataset"""
        dataset["created_at"] = datetime.now(UTC)
        dataset["updated_at"] = datetime.now(UTC)
        result = await collection.insert_one(dataset)
        logger.debug(f"Inserted dataset with id: {result.inserted_id}")
        return str(result.inserted_id)

    @staticmethod
    async def insert_many(
        datasets: list[dict[str, Any]], collection: AsyncIOMotorCollection
    ) -> list[str]:
        """Insert multiple datasets"""
        for dataset in datasets:
            dataset["created_at"] = datetime.now(UTC)
            dataset["updated_at"] = datetime.now(UTC)

        result = await collection.insert_many(datasets)
        logger.debug(f"Inserted {len(result.inserted_ids)} datasets")
        return [str(inserted_id) for inserted_id in result.inserted_ids]

    @staticmethod
    async def find_by_id(
        dataset_id: str, collection: AsyncIOMotorCollection
    ) -> Optional[dict[str, Any]]:
        """Find dataset by ID"""
        try:
            document = await collection.find_one({"_id": ObjectId(dataset_id)})
            if document:
                document = dict(document)
                document["_id"] = str(document["_id"])
            return document
        except Exception as e:
            logger.error(f"Error finding dataset by id {dataset_id}: {e}")
            return None

    @staticmethod
    async def find_by_external_id(
        external_id: str, collection: AsyncIOMotorCollection
    ) -> Optional[dict[str, Any]]:
        """Find dataset by external ID (from original source)"""
        document = await collection.find_one({"external_id": external_id})
        if document:
            document = dict(document)
            document["_id"] = str(document["_id"])
        return document

    @staticmethod
    async def find_all(
        collection: AsyncIOMotorCollection,
        skip: int = 0,
        limit: int = 100,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Find all datasets with pagination"""
        filter_dict = filter_dict or {}
        cursor = collection.find(filter_dict).skip(skip).limit(limit)
        documents = await cursor.to_list(length=limit)

        for doc in documents:
            doc["_id"] = str(doc["_id"])

        return documents

    async def search_text(
        self, query: str, skip: int = 0, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Full text search in title and description"""
        cursor = (
            self.meta_collection.find(
                {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}
            )
            .sort([("score", {"$meta": "textScore"})])
            .skip(skip)
            .limit(limit)
        )

        documents = await cursor.to_list(length=limit)
        for doc in documents:
            doc["_id"] = str(doc["_id"])

        return documents

    async def update_one(
        self, dataset_id: str, update_data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update single dataset"""
        try:
            update_data = dict(update_data)  # ensure it's mutable
            update_data["updated_at"] = datetime.now(UTC)
            result = await self.meta_collection.find_one_and_update(
                {"_id": ObjectId(dataset_id)},
                {"$set": update_data},
                return_document=ReturnDocument.AFTER,
            )
            if result:
                result = dict(result)
                result["_id"] = str(result["_id"])
            return result
        except (PyMongoError, Exception) as e:
            logger.error(f"Error updating dataset {dataset_id}: {e}")
            return None

    async def delete_one(self, dataset_id: str) -> bool:
        """Delete single dataset"""
        try:
            result = await self.meta_collection.delete_one(
                {"_id": ObjectId(dataset_id)}
            )
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {e}")
            return False

    async def delete_all(self) -> int:
        """Delete all datasets (use with caution!)"""
        result = await self.meta_collection.delete_many({})
        logger.warning(f"Deleted {result.deleted_count} datasets")
        return result.deleted_count

    async def count(self, filter_dict: Optional[dict[str, Any]] = None) -> int:
        """Count datasets"""
        filter_dict = filter_dict or {}
        return await self.meta_collection.count_documents(filter_dict)

    async def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists"""
        try:
            count = await self.meta_collection.count_documents(
                {"_id": ObjectId(dataset_id)}, limit=1
            )
            return count > 0
        except (InvalidId, PyMongoError):
            return False

    async def exists_by_external_id(self, external_id: str) -> bool:
        """Check if dataset exists by external ID"""
        count = await self.meta_collection.count_documents(
            {"external_id": external_id}, limit=1
        )
        return count > 0

    async def get_distinct_values(self, field: str) -> list[Any]:
        """Get distinct values for a field (e.g., all unique cities)"""
        return await self.meta_collection.distinct(field)

    async def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run aggregation pipeline"""
        cursor = self.meta_collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_statistics(self) -> dict[str, Any]:
        """Get collection statistics"""
        total = await self.count()
        cities = await self.get_distinct_values("city")
        organizations = await self.get_distinct_values("organization")

        # Get counts by city
        city_stats = await self.aggregate(
            [
                {"$group": {"_id": "$city", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
        )

        return {
            "total_datasets": total,
            "unique_cities": len(cities),
            "unique_organizations": len(organizations),
            "datasets_by_city": city_stats,
            "cities": cities,
            "organizations": organizations,
        }

    async def index_dataset(self, dataset: Dataset, batch_size: int = 1000):
        if not isinstance(dataset, Dataset):
            raise Exception("dataset is not Dataset type. WTF?")

        if len(dataset.data) == 0:
            return 0

        # region index metadata
        meta = dataset.metadata.to_dict()
        safe_title = sanitize_title(dataset.metadata.title)
        existing_meta = await self.meta_collection.find_one(
            {"title": dataset.metadata.title}
        )
        if existing_meta:
            meta_id = str(existing_meta["_id"])
        else:
            now = datetime.now(UTC)
            meta["created_at"] = now
            meta["updated_at"] = now
            meta_id = await self.insert_one(meta, self.meta_collection)
        # endregion

        # region index data
        dataset_collection_name = safe_title + "_" + meta_id
        inserted = 1
        for i in range(0, len(dataset.data), batch_size):
            try:
                batch = dataset.data[i : i + batch_size]
                now = datetime.now(UTC)
                for ds in batch:
                    ds["created_at"] = now
                    ds["updated_at"] = now
                result = await self.db[dataset_collection_name].insert_many(batch)
                inserted += len(result.inserted_ids)
            except Exception as e:
                logger.error(f"Error on batch buffer: {e}, {i}, {type(dataset)}")
        logger.info(f"Inserted 1 meta + {inserted} datasets in batches of {batch_size}")
        return inserted
        # endregion


# Dependency injection function
async def get_dataset_repository(database: MongoDBDep) -> DatasetRepository:
    """Get DatasetRepository instance"""
    return DatasetRepository(database)

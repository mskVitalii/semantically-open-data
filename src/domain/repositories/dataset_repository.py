# src/datasets/repository.py
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC

from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument
from bson import ObjectId
from pymongo.errors import PyMongoError

from src.infrastructure.logger import get_prefixed_logger
from src.infrastructure.mongo_db import MongoDBDep

logger = get_prefixed_logger(__name__, "DATASET_REPOSITORY")


class DatasetRepository:
    """Repository for datasets in MongoDB"""

    # TODO: change this class to create collection for each dataset (add field collection everywhere)
    COLLECTION_NAME = "metadata"

    def __init__(self, database: AsyncIOMotorDatabase):
        self.database = database
        self.collection = database[self.COLLECTION_NAME]

    async def create_indexes(self):
        """Create indexes for better performance"""
        # Index for text search
        await self.collection.create_index([("title", "text"), ("description", "text")])
        # Index for filtering
        await self.collection.create_index("city")
        await self.collection.create_index("organization")
        await self.collection.create_index("tags")
        await self.collection.create_index("created_at")
        # Compound index for common queries
        await self.collection.create_index([("city", 1), ("organization", 1)])
        logger.info("Indexes created successfully")

    async def insert_one(self, dataset: Dict[str, Any]) -> str:
        """Insert single dataset"""
        dataset["created_at"] = datetime.now(UTC)
        dataset["updated_at"] = datetime.now(UTC)
        result = await self.collection.insert_one(dataset)
        logger.debug(f"Inserted dataset with id: {result.inserted_id}")
        return str(result.inserted_id)

    async def insert_many(self, datasets: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple datasets"""
        for dataset in datasets:
            dataset["created_at"] = datetime.now(UTC)
            dataset["updated_at"] = datetime.now(UTC)

        result = await self.collection.insert_many(datasets)
        logger.info(f"Inserted {len(result.inserted_ids)} datasets")
        return [str(inserted_id) for inserted_id in result.inserted_ids]

    async def find_by_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Find dataset by ID"""
        try:
            document = await self.collection.find_one({"_id": ObjectId(dataset_id)})
            if document:
                document = dict(document)
                document["_id"] = str(document["_id"])
            return document
        except Exception as e:
            logger.error(f"Error finding dataset by id {dataset_id}: {e}")
            return None

    async def find_by_external_id(self, external_id: str) -> Optional[Dict[str, Any]]:
        """Find dataset by external ID (from original source)"""
        document = await self.collection.find_one({"external_id": external_id})
        if document:
            document = dict(document)
            document["_id"] = str(document["_id"])
        return document

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Find all datasets with pagination"""
        filter_dict = filter_dict or {}
        cursor = self.collection.find(filter_dict).skip(skip).limit(limit)
        documents = await cursor.to_list(length=limit)

        for doc in documents:
            doc["_id"] = str(doc["_id"])

        return documents

    async def find_by_city(
        self, city: str, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find datasets by city"""
        return await self.find_all(skip, limit, {"city": city})

    async def find_by_organization(
        self, organization: str, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find datasets by organization"""
        return await self.find_all(skip, limit, {"organization": organization})

    async def find_by_tags(
        self, tags: List[str], skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find datasets by tags (any of the tags)"""
        return await self.find_all(skip, limit, {"tags": {"$in": tags}})

    async def search_text(
        self, query: str, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Full text search in title and description"""
        cursor = (
            self.collection.find(
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
    ) -> Optional[Dict[str, Any]]:
        """Update single dataset"""
        try:
            update_data = dict(update_data)  # ensure it's mutable
            update_data["updated_at"] = datetime.now(UTC)
            result = await self.collection.find_one_and_update(
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
            result = await self.collection.delete_one({"_id": ObjectId(dataset_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {e}")
            return False

    async def delete_all(self) -> int:
        """Delete all datasets (use with caution!)"""
        result = await self.collection.delete_many({})
        logger.warning(f"Deleted {result.deleted_count} datasets")
        return result.deleted_count

    async def count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """Count datasets"""
        filter_dict = filter_dict or {}
        return await self.collection.count_documents(filter_dict)

    async def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists"""
        try:
            count = await self.collection.count_documents(
                {"_id": ObjectId(dataset_id)}, limit=1
            )
            return count > 0
        except (InvalidId, PyMongoError):
            return False

    async def exists_by_external_id(self, external_id: str) -> bool:
        """Check if dataset exists by external ID"""
        count = await self.collection.count_documents(
            {"external_id": external_id}, limit=1
        )
        return count > 0

    async def get_distinct_values(self, field: str) -> List[Any]:
        """Get distinct values for a field (e.g., all unique cities)"""
        return await self.collection.distinct(field)

    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation pipeline"""
        cursor = self.collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_statistics(self) -> Dict[str, Any]:
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

    async def batch_insert(
        self, datasets: List[Dict[str, Any]], batch_size: int = 1000
    ) -> List[str]:
        inserted_ids = []
        for i in range(0, len(datasets), batch_size):
            batch = datasets[i : i + batch_size]
            now = datetime.now(UTC)
            for dataset in batch:
                dataset["created_at"] = now
                dataset["updated_at"] = now
            result = await self.collection.insert_many(batch)
            inserted_ids.extend(str(_id) for _id in result.inserted_ids)
        logger.info(f"Inserted {len(inserted_ids)} datasets in batches of {batch_size}")
        return inserted_ids


# Dependency injection function
async def get_dataset_repository(database: MongoDBDep) -> DatasetRepository:
    """Get DatasetRepository instance"""
    return DatasetRepository(database)

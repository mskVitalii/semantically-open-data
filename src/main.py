from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette import status
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.datasets_api.router import v1_router
from src.infrastructure.config import MONGO_INITDB_DATABASE
from src.infrastructure.logger import get_logger
from src.infrastructure.mongo_db import (
    get_mongodb_manager,
    MongoClientDep,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting application...")

    manager = get_mongodb_manager()
    await manager.connect()

    yield

    await manager.disconnect()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Semantic Open Data API",
    description="API to semantically search datasets. Responses to the questions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)


@app.get("/")
async def root():
    return {"message": "Semantic Open Data API is running"}


@app.get("/health")
async def health_check(client: MongoClientDep):
    try:
        manager = get_mongodb_manager()
        is_healthy = await manager.ping()

        if is_healthy:
            return {
                "status": "healthy",
                "database": "connected",
                "database_name": MONGO_INITDB_DATABASE,
            }
        else:
            raise Exception("MongoDB ping failed")

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": "error", "error": str(e)},
        )


@app.get("/health/detailed")
async def detailed_health_check(client: MongoClientDep):
    """Детальная проверка здоровья с метриками"""
    health_status = {"status": "healthy", "checks": {}}

    try:
        manager = get_mongodb_manager()

        is_connected = await manager.ping()
        if not is_connected:
            raise Exception("MongoDB is not responding")

        if not manager.is_testing:
            server_info = await client.server_info()
            version = server_info.get("version")
        else:
            version = "mock"

        db_stats = await manager.get_database_stats()

        health_status["checks"]["mongodb"] = {
            "status": "up",
            "version": version,
            "database": MONGO_INITDB_DATABASE,
            "collections": db_stats.get("collections"),
            "objects": db_stats.get("objects"),
            "dataSize": db_stats.get("dataSize"),
        }

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["mongodb"] = {"status": "down", "error": str(e)}
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=health_status
        )

    return health_status


# def main():
#     """Main MVP function"""
#     logger.info("Starting RAG System...")
#     logger.info(f"Protocol: {'gRPC' if USE_GRPC else 'HTTP'}")
#
#     # Initialize system
#     vector_db = VectorDB(USE_GRPC)
#
#     # Sample data - your actual metadata
#     sample_datasets = [
#         DatasetMetadataWithContent(
#             id="4db3895e-92a9-4bb7-bb33-f792178d331f",
#             title="Landtagswahl 2024: Wahlbezirksergebnisse",
#             description="Der vorliegende Datensatz präsentiert die Ergebnisse der Wahl zum Sächsischen Landtag 2024 in den Leipziger Wahlbezirken.",
#             organization="Amt für Statistik und Wahlen",
#             metadata_created="2024-09-18T09:50:49.841530",
#             metadata_modified="2024-11-15T14:04:14.392074",
#             city="Leipzig",
#             state="Saxony",
#             country="Germany",
#         ),
#         DatasetMetadataWithContent(
#             id="bd1e8543af3e43eeac6b7423abcb424b",
#             title="Bodenrichtwerte_1998",
#             description="Bodenrichtwerte für Chemnitz aus dem Jahr 1998",
#             organization="Stadtverwaltung Chemnitz",
#             metadata_created="1998-01-01T00:00:00",
#             metadata_modified="2024-01-01T00:00:00",
#             city="Chemnitz",
#             state="Saxony",
#             country="Germany",
#         ),
#         DatasetMetadataWithContent(
#             id="d0d8ba5f-40b1-4c4e-84de-3e9e91035add",
#             title="Einwohnerinnen und Einwohner in den Ortsteilen Berlins am 31.12.2011",
#             description="Einwohnerinnen und Einwohner in den Ortsteilen Berlins am 31.12.2011",
#             organization="Amt für Statistik Berlin-Brandenburg",
#             metadata_created="2012-01-15T00:00:00",
#             metadata_modified="2024-01-01T00:00:00",
#             city="Berlin",
#             state="Berlin",
#             country="Germany",
#         ),
#         DatasetMetadataWithContent(
#             id="dataset_4",
#             title="Einwohner - Wanderungen - bezogen auf Dresdner Basiswohnung",
#             description="Wanderungsbewegungen und Migrationsdaten für Dresden",
#             city="Dresden",
#             organization="Landeshauptstadt Dresden",
#             metadata_created="2018-01-01T00:00:00",
#             metadata_modified="2024-01-01T00:00:00",
#             state="Saxony",
#             country="Germany",
#         ),
#         DatasetMetadataWithContent(
#             id="dataset_5",
#             title="Kommunalwahl 2024: Ergebnisse Leipzig",
#             description="Detaillierte Ergebnisse der Kommunalwahl 2024 in Leipzig nach Wahlbezirken",
#             organization="Amt für Statistik und Wahlen",
#             metadata_created="2024-06-01T00:00:00",
#             metadata_modified="2024-06-15T00:00:00",
#             city="Leipzig",
#             state="Saxony",
#             country="Germany",
#         ),
#     ]
#
#     # Index the data
#     vector_db.index_datasets(sample_datasets)
#
#     logger.info("\n" + "=" * 50)
#     logger.info("DATA INDEXED SUCCESSFULLY!")
#     logger.info("=" * 50)
#
#     # Show collection stats
#     vector_db.get_stats()
#
#     # Example queries
#     queries = [
#         # Request in different language without filter
#         ("Какая явка была на выборах в Лейпциге?", None),
#         # Perfectly precise request
#         ("Kommunalwahl 2024: Ergebnisse Leipzig", "Leipzig"),
#         # ("Bevölkerung Berlin", "Berlin"),
#         # ("Wahlergebnisse 2024", None),
#         # ("migration data Dresden", "Dresden"),
#     ]
#
#     # Run searches
#     for query, city_filter in queries:
#         logger.info("\n" + "-" * 50)
#         results = vector_db.search(query, city_filter)
#         logger.info(f"\nFound {len(results)} results:")
#         for i, result in enumerate(results, 1):
#             logger.info(f"\n{i}. Score: {result.score:.4f}")
#             logger.info(f"   Title: {result.payload['title']}")
#             logger.info(f"   City: {result.payload['city']}")
#             logger.info(f"   Organization: {result.payload['organization']}")
#             if result.payload.get("description"):
#                 desc = (
#                     result.payload["description"][:200] + "..."
#                     if len(result.payload["description"]) > 200
#                     else result.payload["description"]
#                 )
#                 logger.info(f"   Description: {desc}")
#
#         time.sleep(0.5)  # Small delay between queries
#
#     # Demonstrate batch search (only if using gRPC)
#     if vector_db.use_grpc:
#         logger.info("\n" + "=" * 50)
#         logger.info("BATCH SEARCH EXAMPLE (gRPC only)")
#         logger.info("=" * 50)
#
#         batch_queries = ["election", "Bodenrichtwerte", "population"]
#         vector_db.batch_search(batch_queries, limit=3)
#
#     logger.info("\n" + "=" * 50)
#     logger.info("MVP DEMO COMPLETE!")
#     logger.info("=" * 50)
#
#
def run_dev():
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


def run_start():
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)


#
#
# if __name__ == "__main__":
#     main()
# run_dev()
# logger.info("http://localhost:8000/docs")

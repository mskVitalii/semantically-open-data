import time

from config.config import USE_GRPC
from datasets.datasets_metadata import DatasetMetadata
from infrastructure.logger import get_logger
from vector_search.vector_db import VectorDB

logger = get_logger(__name__)


def benchmark_protocols():
    """Quick benchmark between gRPC and HTTP"""
    logger.info("\n" + "=" * 50)
    logger.info("BENCHMARKING gRPC vs HTTP")
    logger.info("=" * 50)

    # Test data
    test_query = "election results Leipzig"

    # Test HTTP
    start = time.time()
    rag_http = VectorDB(use_grpc=False)
    rag_http.search(test_query, limit=5)
    http_time = time.time() - start

    # Test gRPC
    start = time.time()
    rag_grpc = VectorDB(use_grpc=True)
    rag_grpc.search(test_query, limit=5)
    grpc_time = time.time() - start

    logger.info("\nBenchmark Results:")
    logger.info(f"  HTTP search time: {http_time:.3f}s")
    logger.info(f"  gRPC search time: {grpc_time:.3f}s")
    logger.info(f"  gRPC is {http_time / grpc_time:.2f}x faster")


def main():
    """Main MVP function"""
    logger.info("Starting RAG System...")
    logger.info(f"Protocol: {'gRPC' if USE_GRPC else 'HTTP'}")

    # Initialize system
    rag = VectorDB(USE_GRPC)

    # Sample data - your actual metadata
    sample_datasets = [
        DatasetMetadata(
            id="4db3895e-92a9-4bb7-bb33-f792178d331f",
            title="Landtagswahl 2024: Wahlbezirksergebnisse",
            description="Der vorliegende Datensatz präsentiert die Ergebnisse der Wahl zum Sächsischen Landtag 2024 in den Leipziger Wahlbezirken.",
            city="Leipzig",
            organization="Amt für Statistik und Wahlen",
            metadata_created="2024-09-18T09:50:49.841530",
            metadata_modified="2024-11-15T14:04:14.392074",
        ),
        DatasetMetadata(
            id="bd1e8543af3e43eeac6b7423abcb424b",
            title="Bodenrichtwerte_1998",
            description="Bodenrichtwerte für Chemnitz aus dem Jahr 1998",
            city="Chemnitz",
            organization="Stadtverwaltung Chemnitz",
            metadata_created="1998-01-01T00:00:00",
            metadata_modified="2024-01-01T00:00:00",
        ),
        DatasetMetadata(
            id="d0d8ba5f-40b1-4c4e-84de-3e9e91035add",
            title="Einwohnerinnen und Einwohner in den Ortsteilen Berlins am 31.12.2011",
            description="Einwohnerinnen und Einwohner in den Ortsteilen Berlins am 31.12.2011",
            city="Berlin",
            organization="Amt für Statistik Berlin-Brandenburg",
            metadata_created="2012-01-15T00:00:00",
            metadata_modified="2024-01-01T00:00:00",
        ),
        DatasetMetadata(
            id="dataset_4",
            title="Einwohner - Wanderungen - bezogen auf Dresdner Basiswohnung",
            description="Wanderungsbewegungen und Migrationsdaten für Dresden",
            city="Dresden",
            organization="Landeshauptstadt Dresden",
            metadata_created="2018-01-01T00:00:00",
            metadata_modified="2024-01-01T00:00:00",
        ),
        DatasetMetadata(
            id="dataset_5",
            title="Kommunalwahl 2024: Ergebnisse Leipzig",
            description="Detaillierte Ergebnisse der Kommunalwahl 2024 in Leipzig nach Wahlbezirken",
            city="Leipzig",
            organization="Amt für Statistik und Wahlen",
            metadata_created="2024-06-01T00:00:00",
            metadata_modified="2024-06-15T00:00:00",
        ),
    ]

    # Index the data
    rag.index_datasets(sample_datasets)

    logger.info("\n" + "=" * 50)
    logger.info("DATA INDEXED SUCCESSFULLY!")
    logger.info("=" * 50)

    # Show collection stats
    rag.get_stats()

    # Example queries
    queries = [
        ("Какая явка была на выборах в Лейпциге?", "Leipzig"),
        ("election results", None),
        ("Bevölkerung Berlin", "Berlin"),
        ("Wahlergebnisse 2024", None),
        ("migration data Dresden", "Dresden"),
    ]

    # Run searches
    for query, city_filter in queries:
        logger.info("\n" + "-" * 50)
        rag.search(query, city_filter)
        time.sleep(0.5)  # Small delay between queries

    # Demonstrate batch search (only if using gRPC)
    if rag.use_grpc:
        logger.info("\n" + "=" * 50)
        logger.info("BATCH SEARCH EXAMPLE (gRPC only)")
        logger.info("=" * 50)

        batch_queries = ["election", "Bodenrichtwerte", "population"]
        rag.batch_search(batch_queries, limit=3)

    logger.info("\n" + "=" * 50)
    logger.info("MVP DEMO COMPLETE!")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Uncomment to run benchmark
    benchmark_protocols()

    # Run main demo
    # main()

from pathlib import Path
from typing import Dict, Any, Tuple
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time

from infrastructure.logger import get_logger
from utils import extract_comprehensive_text

logger = get_logger(__name__)


def process_single_dataset(dataset_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single dataset folder.

    Args:
        dataset_path: Path to the dataset folder

    Returns:
        Tuple of (key, result_dict)
    """
    city_name = dataset_path.parent.name
    dataset_name = dataset_path.name
    key = f"{city_name}/{dataset_name}"

    try:
        # Extract comprehensive text using the utility function
        comprehensive_text = extract_comprehensive_text(dataset_path)

        result = {
            "path": str(dataset_path),
            "city": city_name,
            "dataset_name": dataset_name,
            "comprehensive_text": comprehensive_text,
            "status": "success",
        }

    except Exception as e:
        result = {
            "path": str(dataset_path),
            "city": city_name,
            "dataset_name": dataset_name,
            "error": str(e),
            "status": "failed",
        }

    return key, result


def process_city_datasets(city_path: Path, max_workers: int = None) -> Dict[str, Any]:
    """
    Process all datasets in a city directory using parallel processing.

    Args:
        city_path: Path to the city directory
        max_workers: Maximum number of worker processes

    Returns:
        Dictionary containing results for all datasets in the city
    """
    city_name = city_path.name
    dataset_folders = [d for d in city_path.iterdir() if d.is_dir()]

    logger.info(
        f"Processing {len(dataset_folders)} datasets in {city_name} with parallel workers"
    )

    results = {}
    successful = 0
    failed = 0

    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(process_single_dataset, dataset_path): dataset_path
            for dataset_path in dataset_folders
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_dataset):
            dataset_path = future_to_dataset[future]
            try:
                key, result = future.result()
                results[key] = result

                if result["status"] == "success":
                    successful += 1
                    logger.debug(f"✓ {key}")
                else:
                    failed += 1
                    logger.debug(f"✗ {key} - Error: {result['error']}")

            except Exception as e:
                logger.error(f"Unexpected error processing {dataset_path}: {str(e)}")
                failed += 1

    logger.info(f"Completed {city_name}: {successful} successful, {failed} failed")
    return results


def process_all_cities_parallel(
    datasets_path: str = "../datasets",
    city_workers: int = None,
    dataset_workers: int = None,
) -> Dict[str, Any]:
    """
    Process all cities in parallel, with each city processing its datasets in parallel.

    Args:
        datasets_path: Path to the datasets directory
        city_workers: Number of parallel city processors (default: number of cities)
        dataset_workers: Number of parallel dataset processors per city (default: CPU count)

    Returns:
        Dictionary containing all extraction results
    """
    base_path = Path(datasets_path)

    if not base_path.exists():
        logger.error(f"Datasets directory not found: {base_path}")
        return {}

    # Get all city directories
    city_dirs = [
        d
        for d in base_path.iterdir()
        if d.is_dir() and not d.name.startswith("__") and not d.name.endswith(".py")
    ]

    logger.info(
        f"Found {len(city_dirs)} cities to process: {[c.name for c in city_dirs]}"
    )

    # Default workers
    if dataset_workers is None:
        dataset_workers = multiprocessing.cpu_count()
    if city_workers is None:
        city_workers = min(len(city_dirs), 4)

    logger.info(
        f"Using {city_workers} city workers and {dataset_workers} dataset workers per city"
    )

    all_results = {}
    start_time = time.time()

    # Process cities in parallel
    with ThreadPoolExecutor(max_workers=city_workers) as executor:
        # Submit city processing tasks
        future_to_city = {
            executor.submit(
                process_city_datasets, city_path, dataset_workers
            ): city_path
            for city_path in city_dirs
        }

        # Collect results as cities complete
        for future in as_completed(future_to_city):
            city_path = future_to_city[future]
            try:
                city_results = future.result()
                all_results.update(city_results)
                logger.info(f"Completed processing {city_path.name}")
            except Exception as e:
                logger.error(f"Failed to process city {city_path.name}: {str(e)}")

    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

    return all_results


def save_results(results: Dict[str, Any], output_path: str = "extracted_texts.json"):
    """
    Save the extraction results to a JSON file.

    Args:
        results: Dictionary containing extraction results
        output_path: Path to save the results file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")


def generate_summary_report(results: Dict[str, Any]) -> str:
    """
    Generate a summary report of the extraction results.

    Args:
        results: Dictionary containing extraction results

    Returns:
        Formatted summary report string
    """
    report = ["=" * 60, "COMPREHENSIVE TEXT EXTRACTION SUMMARY", "=" * 60]

    # Group by city
    cities = {}
    for key, data in results.items():
        city = data["city"]
        if city not in cities:
            cities[city] = []
        cities[city].append(data)

    # Overall statistics
    total_datasets = len(results)
    total_successful = sum(1 for r in results.values() if r["status"] == "success")
    total_failed = total_datasets - total_successful

    report.append("\nOVERALL STATISTICS:")
    report.append(f"  Total datasets: {total_datasets}")
    report.append(f"  Successful: {total_successful}")
    report.append(f"  Failed: {total_failed}")
    report.append(f"  Success rate: {(total_successful / total_datasets * 100):.1f}%")

    # Report by city
    for city, datasets in sorted(cities.items()):
        report.append(f"\n{city.upper()}:")
        report.append("-" * 40)

        successful = [d for d in datasets if d["status"] == "success"]
        failed = [d for d in datasets if d["status"] == "failed"]

        report.append(f"  Total datasets: {len(datasets)}")
        report.append(f"  Successful: {len(successful)}")
        report.append(f"  Failed: {len(failed)}")

        # Show failed datasets (if any)
        if failed:
            report.append(f"\n  Failed datasets ({len(failed)}):")
            for dataset in failed[:10]:  # Show max 10 failures
                error_msg = (
                    dataset["error"][:50] + "..."
                    if len(dataset["error"]) > 50
                    else dataset["error"]
                )
                report.append(f"    ✗ {dataset['dataset_name']} - {error_msg}")
            if len(failed) > 10:
                report.append(f"    ... and {len(failed) - 10} more")

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def main():
    """
    Main function to run parallel comprehensive text extraction for all datasets.
    """
    logger.info("Starting parallel comprehensive text extraction for all datasets...")

    # You can adjust these parameters based on your system
    # city_workers: How many cities to process in parallel
    # dataset_workers: How many datasets to process in parallel within each city
    results = process_all_cities_parallel(
        datasets_path="../datasets",
        city_workers=4,  # Process 4 cities in parallel
        dataset_workers=8,  # Process 8 datasets in parallel per city
    )

    if results:
        # Save results to JSON
        save_results(results)

        # Generate and print summary report
        report = generate_summary_report(results)
        logger.info(report)

        # Quick stats
        successful = sum(1 for r in results.values() if r["status"] == "success")
        failed = len(results) - successful
        logger.info(
            f"\nFinal results: {successful} successful, {failed} failed out of {len(results)} total datasets"
        )
    else:
        logger.error("No results.")


if __name__ == "__main__":
    main()

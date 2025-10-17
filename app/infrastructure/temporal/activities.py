import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from temporalio import activity


@dataclass
class QueryLibraryParams:
    """Parameters for querying a library"""

    library_id: str
    query_embedding: List[float]
    k: int = 10
    metadata_filters: Optional[List[Dict[str, Any]]] = None


@dataclass
class QueryLibraryResult:
    """Result from querying a library"""

    results: List[Dict[str, Any]]
    query_time_ms: float
    total_results: int


@dataclass
class IndexLibraryParams:
    """Parameters for indexing a library"""

    library_id: str
    index_type: str


@activity.defn
async def query_library_activity(params: QueryLibraryParams) -> QueryLibraryResult:
    """
    Activity: Query a library for similar vectors

    This activity can be retried automatically by Temporal if it fails.
    It's idempotent - running it multiple times with the same input
    produces the same result.
    """
    # Use HTTP to call the API (activities run in separate process from API)
    import requests

    activity.logger.info(
        f"Querying library {params.library_id} with k={params.k}"
    )

    try:
        # Call the API via HTTP
        api_url = "http://localhost:8000"
        query_data = {
            "embedding": params.query_embedding,
            "k": params.k,
            "metadata_filters": params.metadata_filters,
        }

        response = requests.post(
            f"{api_url}/libraries/{params.library_id}/query",
            json=query_data,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"API returned {response.status_code}: {response.text}")

        result = response.json()

        activity.logger.info(
            f"Query completed: {result['total_results']} results in {result['query_time_ms']:.2f}ms"
        )

        return QueryLibraryResult(
            results=result['results'],
            query_time_ms=result['query_time_ms'],
            total_results=result['total_results']
        )

    except Exception as e:
        activity.logger.error(f"Query failed: {str(e)}")
        raise


@activity.defn
async def index_library_activity(params: IndexLibraryParams) -> Dict[str, Any]:
    """
    Activity: Index a library

    This is a long-running operation that can be retried if it fails.
    """
    # Use HTTP to call the API (activities run in separate process from API)
    import requests

    activity.logger.info(
        f"Indexing library {params.library_id} with type {params.index_type}"
    )

    try:
        start_time = time.time()

        # Call the API via HTTP
        api_url = "http://localhost:8000"
        response = requests.post(
            f"{api_url}/libraries/{params.library_id}/index",
            json={"index_type": params.index_type},
            timeout=600  # 10 minutes for indexing
        )

        if response.status_code != 200:
            raise Exception(f"API returned {response.status_code}: {response.text}")

        library = response.json()
        elapsed_time = (time.time() - start_time) * 1000

        activity.logger.info(f"Indexing completed in {elapsed_time:.2f}ms")

        return {
            "library_id": library["id"],
            "index_type": library["index_type"],
            "is_indexed": library["is_indexed"],
            "elapsed_time_ms": elapsed_time,
        }

    except Exception as e:
        activity.logger.error(f"Indexing failed: {str(e)}")
        raise


@activity.defn
async def batch_query_activity(
    params: List[QueryLibraryParams],
) -> List[QueryLibraryResult]:
    """
    Activity: Batch query multiple libraries

    Demonstrates how Temporal can handle batch operations.
    """
    activity.logger.info(f"Batch querying {len(params)} libraries")

    results = []
    for param in params:
        result = await query_library_activity(param)
        results.append(result)

    activity.logger.info(f"Batch query completed: {len(results)} queries executed")

    return results

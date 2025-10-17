"""
Temporal Activities

Activities are the building blocks of Temporal workflows.
Each activity performs a specific task and can be retried on failure.
"""

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
    # Import inside activity to avoid loading in workflow sandbox
    from app.api.dependencies import get_chunk_service, get_library_service
    from app.api.dto import MetadataFilter

    activity.logger.info(
        f"Querying library {params.library_id} with k={params.k}, "
        f"index_type={params.index_type if hasattr(params, 'index_type') else 'default'}"
    )

    try:
        library_service = get_library_service()

        # Convert filters
        filters = None
        if params.metadata_filters:
            filters = [
                MetadataFilter(
                    field=f["field"], operator=f["operator"], value=f["value"]
                )
                for f in params.metadata_filters
            ]

        # Query the library
        results, query_time = library_service.query_library(
            library_id=UUID(params.library_id),
            query_embedding=params.query_embedding,
            k=params.k,
            metadata_filters=filters,
        )

        # Convert results to dicts
        result_dicts = []
        chunk_service = get_chunk_service()

        for chunk, score in results:
            result_dicts.append(
                {
                    "chunk": {
                        "id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "text": chunk.text,
                        "embedding": chunk.embedding,
                        "metadata": {
                            "source": chunk.metadata.source,
                            "page_number": chunk.metadata.page_number,
                            "author": chunk.metadata.author,
                            "tags": chunk.metadata.tags,
                            "created_at": chunk.metadata.created_at.isoformat(),
                        },
                        "position": chunk.position,
                    },
                    "score": score,
                }
            )

        activity.logger.info(
            f"Query completed: {len(results)} results in {query_time:.2f}ms"
        )

        return QueryLibraryResult(
            results=result_dicts, query_time_ms=query_time, total_results=len(results)
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
    # Import inside activity to avoid loading in workflow sandbox
    from app.api.dependencies import get_library_service

    activity.logger.info(
        f"Indexing library {params.library_id} with type {params.index_type}"
    )

    try:
        library_service = get_library_service()

        start_time = time.time()

        # Index the library
        library = library_service.index_library(
            library_id=UUID(params.library_id), index_type=params.index_type
        )

        elapsed_time = (time.time() - start_time) * 1000

        activity.logger.info(f"Indexing completed in {elapsed_time:.2f}ms")

        return {
            "library_id": str(library.id),
            "index_type": library.index_type,
            "is_indexed": library.is_indexed,
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

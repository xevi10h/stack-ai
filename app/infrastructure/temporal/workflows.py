"""
Temporal Workflows

Workflows orchestrate activities and define the business logic.
They are durable and can handle failures, retries, and long-running operations.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.infrastructure.temporal.activities import (
        IndexLibraryParams,
        QueryLibraryParams,
        QueryLibraryResult,
        batch_query_activity,
        index_library_activity,
        query_library_activity,
    )


@dataclass
class QueryWorkflowParams:
    """Parameters for the query workflow"""

    library_id: str
    query_embedding: List[float]
    k: int = 10
    metadata_filters: Optional[List[Dict[str, Any]]] = None
    auto_index: bool = False
    index_type: str = "hnsw"


@workflow.defn
class QueryWorkflow:
    """
    Durable Query Workflow

    This workflow orchestrates querying a vector library with automatic
    indexing, retries, and signals for query updates.

    Features:
    - Automatic indexing if library is not indexed
    - Retry logic for failed queries
    - Signals to update query parameters
    - Queries to check workflow status
    """

    def __init__(self):
        self._status = "initializing"
        self._result: Optional[QueryLibraryResult] = None
        self._error: Optional[str] = None
        self._update_query_params: Optional[QueryWorkflowParams] = None

    @workflow.run
    async def run(self, params: QueryWorkflowParams) -> QueryLibraryResult:
        """
        Main workflow execution

        Args:
            params: Query parameters including library_id, query_embedding, k, etc.

        Returns:
            QueryLibraryResult with search results and metadata
        """
        workflow.logger.info(f"Starting query workflow for library {params.library_id}")
        self._status = "running"

        try:
            # Auto-index if requested
            if params.auto_index:
                self._status = "indexing"
                workflow.logger.info("Auto-indexing enabled, indexing library...")

                index_params = IndexLibraryParams(
                    library_id=params.library_id, index_type=params.index_type
                )

                await workflow.execute_activity(
                    index_library_activity,
                    index_params,
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=workflow.RetryPolicy(
                        maximum_attempts=3,
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=10),
                    ),
                )

                workflow.logger.info("Indexing completed")

            # Execute query activity
            self._status = "querying"
            query_params = QueryLibraryParams(
                library_id=params.library_id,
                query_embedding=params.query_embedding,
                k=params.k,
                metadata_filters=params.metadata_filters,
            )

            result = await workflow.execute_activity(
                query_library_activity,
                query_params,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=workflow.RetryPolicy(
                    maximum_attempts=5,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=10),
                    backoff_coefficient=2.0,
                ),
            )

            self._result = result
            self._status = "completed"

            workflow.logger.info(
                f"Query workflow completed: {result.total_results} results in {result.query_time_ms:.2f}ms"
            )

            # Wait for potential re-query signal
            await workflow.wait_condition(
                lambda: self._update_query_params is not None,
                timeout=timedelta(hours=1),
            )

            # If we got an update signal, re-run with new params
            if self._update_query_params:
                workflow.logger.info("Re-running query with updated parameters")
                return await self.run(self._update_query_params)

            return result

        except Exception as e:
            self._status = "failed"
            self._error = str(e)
            workflow.logger.error(f"Query workflow failed: {e}")
            raise

    @workflow.signal
    def update_query(self, new_params: QueryWorkflowParams):
        """
        Signal: Update query parameters

        This allows updating the query while the workflow is running.
        """
        workflow.logger.info("Received update_query signal")
        self._update_query_params = new_params

    @workflow.query
    def get_status(self) -> str:
        """
        Query: Get current workflow status

        Returns one of: initializing, indexing, querying, completed, failed
        """
        return self._status

    @workflow.query
    def get_result(self) -> Optional[Dict[str, Any]]:
        """
        Query: Get current result

        Returns the query result if available, None otherwise.
        """
        if self._result:
            return {
                "results": self._result.results,
                "query_time_ms": self._result.query_time_ms,
                "total_results": self._result.total_results,
            }
        return None

    @workflow.query
    def get_error(self) -> Optional[str]:
        """
        Query: Get error message if workflow failed
        """
        return self._error


@workflow.defn
class BatchQueryWorkflow:
    """
    Batch Query Workflow

    Executes multiple queries in parallel with proper orchestration.
    """

    def __init__(self):
        self._status = "initializing"
        self._completed_queries = 0
        self._total_queries = 0

    @workflow.run
    async def run(self, queries: List[QueryWorkflowParams]) -> List[QueryLibraryResult]:
        """
        Execute multiple queries in parallel

        Args:
            queries: List of query parameters

        Returns:
            List of query results
        """
        self._total_queries = len(queries)
        self._status = "running"

        workflow.logger.info(
            f"Starting batch query workflow with {len(queries)} queries"
        )

        # Convert to activity params
        query_params = [
            QueryLibraryParams(
                library_id=q.library_id,
                query_embedding=q.query_embedding,
                k=q.k,
                metadata_filters=q.metadata_filters,
            )
            for q in queries
        ]

        # Execute batch query activity
        results = await workflow.execute_activity(
            batch_query_activity,
            query_params,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=workflow.RetryPolicy(maximum_attempts=3),
        )

        self._completed_queries = len(results)
        self._status = "completed"

        workflow.logger.info(f"Batch query completed: {len(results)} queries executed")

        return results

    @workflow.query
    def get_progress(self) -> Dict[str, Any]:
        """
        Query: Get batch query progress
        """
        return {
            "status": self._status,
            "completed": self._completed_queries,
            "total": self._total_queries,
            "progress_pct": (
                (self._completed_queries / self._total_queries * 100)
                if self._total_queries > 0
                else 0
            ),
        }

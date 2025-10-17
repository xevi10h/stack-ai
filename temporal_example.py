"""
Example: Using Temporal Workflows for Durable Query Execution

This example demonstrates how to use Temporal workflows to execute
vector database queries with automatic retries, signals, and queries.
"""

import asyncio
import random
from uuid import uuid4

from temporalio.client import Client

from app.infrastructure.temporal.workflows import QueryWorkflow, QueryWorkflowParams


def generate_random_embedding(dimension: int = 1024) -> list[float]:
    """Generate a random embedding vector"""
    return [random.random() for _ in range(dimension)]


async def main():
    """
    Demonstrates Temporal workflow features:
    1. Starting a durable query workflow
    2. Checking workflow status via queries
    3. Updating query parameters via signals
    """

    # Connect to Temporal server
    print("Connecting to Temporal server...")
    client = await Client.connect("localhost:7233")
    print("✓ Connected to Temporal")
    print()

    # Create workflow parameters
    workflow_id = f"query-workflow-{uuid4()}"
    params = QueryWorkflowParams(
        library_id="8ea05217-d498-42c0-b575-ac2dbfde8f39",  # Replace with actual library ID
        query_embedding=generate_random_embedding(),  # Uses default 1024 dimensions
        k=10,
        auto_index=False,  # Set to True to auto-index before querying
        index_type="hnsw",
    )

    print(f"Starting workflow: {workflow_id}")
    print(f"Library ID: {params.library_id}")
    print(f"K: {params.k}")
    print()

    # Start the workflow
    handle = await client.start_workflow(
        QueryWorkflow.run,
        params,
        id=workflow_id,
        task_queue="vector-db-queue",
    )

    print("✓ Workflow started")
    print()

    # Check workflow status
    print("Checking workflow status...")
    status = await handle.query(QueryWorkflow.get_status)
    print(f"Status: {status}")
    print()

    # Wait a bit for the workflow to process
    await asyncio.sleep(2)

    # Check status again
    status = await handle.query(QueryWorkflow.get_status)
    print(f"Status after processing: {status}")
    print()

    # Wait for workflow to complete
    print("Waiting for workflow to complete...")
    try:
        final_result = await handle.result()
        print("✓ Workflow completed successfully!")
        print(f"  Total results: {final_result.total_results}")
        print(f"  Query time: {final_result.query_time_ms:.2f}ms")
        print()

        # Show first few results
        print("Top results:")
        for i, res in enumerate(final_result.results[:3], 1):
            print(f"  {i}. Score: {res['score']:.4f}")
            print(f"     Text: {res['text'][:80]}...")
            print()
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        error = await handle.query(QueryWorkflow.get_error)
        if error:
            print(f"  Error details: {error}")

    print()
    print("Example completed!")


async def batch_query_example():
    """
    Example of batch querying multiple libraries
    """
    print("\n=== Batch Query Example ===\n")

    client = await Client.connect("localhost:7233")

    # Create multiple queries
    queries = [
        QueryWorkflowParams(
            library_id="library-1",
            query_embedding=generate_random_embedding(),
            k=10,
        ),
        QueryWorkflowParams(
            library_id="library-2",
            query_embedding=generate_random_embedding(),
            k=5,
        ),
        QueryWorkflowParams(
            library_id="library-3",
            query_embedding=generate_random_embedding(),
            k=15,
        ),
    ]

    from app.infrastructure.temporal.workflows import BatchQueryWorkflow

    handle = await client.start_workflow(
        BatchQueryWorkflow.run,
        queries,
        id=f"batch-query-{uuid4()}",
        task_queue="vector-db-queue",
    )

    print(f"Started batch query workflow with {len(queries)} queries")

    # Check progress
    while True:
        progress = await handle.query(BatchQueryWorkflow.get_progress)
        print(
            f"Progress: {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)"
        )

        if progress["status"] == "completed":
            break

        await asyncio.sleep(1)

    results = await handle.result()
    print(f"\n✓ Batch query completed: {len(results)} queries executed")


if __name__ == "__main__":
    print("=== Temporal Workflow Example ===\n")
    print("Make sure you have:")
    print("1. Temporal server running (docker-compose up temporal)")
    print("2. Worker running (python -m app.infrastructure.temporal.worker)")
    print("3. API server running (uvicorn app.main:app)")
    print("4. A library created with some documents and chunks")
    print()
    input("Press Enter to continue...")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise

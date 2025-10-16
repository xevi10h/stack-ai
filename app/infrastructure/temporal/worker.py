"""
Temporal Worker

The worker runs workflows and activities. It connects to the Temporal server
and executes tasks from the task queue.
"""

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from app.infrastructure.temporal.activities import (
    batch_query_activity,
    index_library_activity,
    query_library_activity,
)
from app.infrastructure.temporal.workflows import BatchQueryWorkflow, QueryWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_worker(
    temporal_host: str = "localhost:7233", task_queue: str = "vector-db-queue"
):
    """
    Start the Temporal worker

    Args:
        temporal_host: Temporal server address
        task_queue: Name of the task queue to poll
    """
    logger.info(f"Connecting to Temporal server at {temporal_host}")

    # Connect to Temporal server
    client = await Client.connect(temporal_host)

    logger.info(f"Starting worker on task queue: {task_queue}")

    # Create worker with workflows and activities
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[QueryWorkflow, BatchQueryWorkflow],
        activities=[
            query_library_activity,
            index_library_activity,
            batch_query_activity,
        ],
    )

    # Run the worker
    logger.info("Worker started. Waiting for tasks...")
    await worker.run()


def main():
    """Main entry point for the worker"""
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        raise


if __name__ == "__main__":
    main()

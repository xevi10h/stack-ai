"""
Vector Database Python SDK

A Python client library for interacting with the Vector Database API.
"""

from vector_db_client.client import VectorDBClient
from vector_db_client.exceptions import (
    VectorDBAPIError,
    VectorDBConnectionError,
    VectorDBNotFoundError,
    VectorDBValidationError,
)
from vector_db_client.models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    Library,
    LibraryMetadata,
    MetadataFilter,
    QueryResult,
)

__version__ = "1.0.0"

__all__ = [
    "VectorDBClient",
    "VectorDBAPIError",
    "VectorDBConnectionError",
    "VectorDBNotFoundError",
    "VectorDBValidationError",
    "Library",
    "LibraryMetadata",
    "Document",
    "DocumentMetadata",
    "Chunk",
    "ChunkMetadata",
    "MetadataFilter",
    "QueryResult",
]

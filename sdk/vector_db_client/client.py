"""
Vector Database Client

Main client class for interacting with the Vector Database API.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import requests
from vector_db_client.exceptions import (
    VectorDBAPIError,
    VectorDBConnectionError,
    VectorDBNotFoundError,
    VectorDBRateLimitError,
    VectorDBUnauthorizedError,
    VectorDBValidationError,
)
from vector_db_client.models import (
    Chunk,
    Document,
    Library,
    MetadataFilter,
    QueryResult,
)


class VectorDBClient:
    """
    Python client for Vector Database API

    Example usage:
        client = VectorDBClient(base_url="http://localhost:8000")

        # Create a library
        library = client.create_library(
            name="My Library",
            description="A collection of documents",
            tags=["ml", "ai"],
            embedding_dimension=384
        )

        # Add documents and chunks
        document = client.create_document(
            library_id=library.id,
            title="Document Title",
            source="document.pdf"
        )

        chunk = client.create_chunk(
            document_id=document.id,
            text="Sample text",
            embedding=[0.1, 0.2, ...],
            source="document.pdf",
            position=0
        )

        # Index the library
        client.index_library(library.id, index_type="hnsw")

        # Query
        results = client.query_library(
            library_id=library.id,
            query_embedding=[0.1, 0.2, ...],
            k=10
        )
    """

    def __init__(
        self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None
    ):
        """
        Initialize the Vector Database client

        Args:
            base_url: The base URL of the Vector Database API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method, url=url, json=json, params=params, timeout=30
            )

            if response.status_code == 401:
                raise VectorDBUnauthorizedError()
            elif response.status_code == 404:
                raise VectorDBNotFoundError("Resource", endpoint)
            elif response.status_code == 422:
                raise VectorDBValidationError(
                    response.json().get("detail", "Validation error"),
                    status_code=422,
                )
            elif response.status_code == 429:
                raise VectorDBRateLimitError()
            elif response.status_code >= 400:
                error_msg = response.json().get("detail", "Unknown error")
                raise VectorDBAPIError(error_msg, status_code=response.status_code)

            response.raise_for_status()

            if response.status_code == 204:
                return None

            return response.json()

        except requests.exceptions.ConnectionError as e:
            raise VectorDBConnectionError(f"Failed to connect to {url}: {e}")
        except requests.exceptions.Timeout as e:
            raise VectorDBConnectionError(f"Request to {url} timed out: {e}")
        except requests.exceptions.RequestException as e:
            raise VectorDBAPIError(f"Request failed: {e}")

    def create_library(
        self,
        name: str,
        embedding_dimension: int,
        description: Optional[str] = None,
        tags: List[str] = None,
    ) -> Library:
        """Create a new library"""
        data = {
            "name": name,
            "embedding_dimension": embedding_dimension,
            "description": description,
            "tags": tags or [],
        }
        response = self._request("POST", "/libraries", json=data)
        return Library.from_dict(response)

    def get_library(self, library_id: UUID) -> Library:
        """Get a library by ID"""
        response = self._request("GET", f"/libraries/{library_id}")
        return Library.from_dict(response)

    def list_libraries(self) -> List[Library]:
        """List all libraries"""
        response = self._request("GET", "/libraries")
        return [Library.from_dict(lib) for lib in response]

    def update_library(
        self,
        library_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Library:
        """Update a library"""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if tags is not None:
            data["tags"] = tags

        response = self._request("PUT", f"/libraries/{library_id}", json=data)
        return Library.from_dict(response)

    def delete_library(self, library_id: UUID) -> None:
        """Delete a library"""
        self._request("DELETE", f"/libraries/{library_id}")

    def index_library(self, library_id: UUID, index_type: str = "flat") -> Library:
        """
        Index a library with the specified algorithm

        Args:
            library_id: The library to index
            index_type: One of 'flat', 'hnsw', or 'ivf'
        """
        data = {"index_type": index_type}
        response = self._request("POST", f"/libraries/{library_id}/index", json=data)
        return Library.from_dict(response)

    def query_library(
        self,
        library_id: UUID,
        query_embedding: List[float],
        k: int = 10,
        metadata_filters: Optional[List[MetadataFilter]] = None,
    ) -> tuple[List[QueryResult], float]:
        """
        Query a library for similar vectors

        Args:
            library_id: The library to query
            query_embedding: The query vector
            k: Number of results to return
            metadata_filters: Optional metadata filters

        Returns:
            Tuple of (results, query_time_ms)
        """
        data = {"embedding": query_embedding, "k": k}

        if metadata_filters:
            data["metadata_filters"] = [f.to_dict() for f in metadata_filters]

        response = self._request("POST", f"/libraries/{library_id}/query", json=data)

        results = [QueryResult.from_dict(r) for r in response["results"]]
        query_time = response["query_time_ms"]

        return results, query_time

    def create_document(
        self,
        library_id: UUID,
        title: str,
        source: str,
        author: Optional[str] = None,
        tags: List[str] = None,
        language: Optional[str] = None,
    ) -> Document:
        """Create a new document in a library"""
        data = {
            "title": title,
            "source": source,
            "author": author,
            "tags": tags or [],
            "language": language,
        }
        response = self._request(
            "POST", f"/libraries/{library_id}/documents", json=data
        )
        return Document.from_dict(response)

    def get_document(self, document_id: UUID) -> Document:
        """Get a document by ID"""
        response = self._request("GET", f"/documents/{document_id}")
        return Document.from_dict(response)

    def list_documents(self, library_id: UUID) -> List[Document]:
        """List all documents in a library"""
        response = self._request("GET", f"/libraries/{library_id}/documents")
        return [Document.from_dict(doc) for doc in response]

    def update_document(
        self,
        document_id: UUID,
        title: Optional[str] = None,
        source: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> Document:
        """Update a document"""
        data = {}
        if title is not None:
            data["title"] = title
        if source is not None:
            data["source"] = source
        if author is not None:
            data["author"] = author
        if tags is not None:
            data["tags"] = tags
        if language is not None:
            data["language"] = language

        response = self._request("PUT", f"/documents/{document_id}", json=data)
        return Document.from_dict(response)

    def delete_document(self, document_id: UUID) -> None:
        """Delete a document"""
        self._request("DELETE", f"/documents/{document_id}")

    def create_chunk(
        self,
        document_id: UUID,
        text: str,
        embedding: List[float],
        source: str,
        position: int,
        page_number: Optional[int] = None,
        author: Optional[str] = None,
        tags: List[str] = None,
    ) -> Chunk:
        """Create a new chunk in a document"""
        data = {
            "text": text,
            "embedding": embedding,
            "source": source,
            "position": position,
            "page_number": page_number,
            "author": author,
            "tags": tags or [],
        }
        response = self._request("POST", f"/documents/{document_id}/chunks", json=data)
        return Chunk.from_dict(response)

    def get_chunk(self, chunk_id: UUID) -> Chunk:
        """Get a chunk by ID"""
        response = self._request("GET", f"/chunks/{chunk_id}")
        return Chunk.from_dict(response)

    def list_chunks(self, document_id: UUID) -> List[Chunk]:
        """List all chunks in a document"""
        response = self._request("GET", f"/documents/{document_id}/chunks")
        return [Chunk.from_dict(chunk) for chunk in response]

    def update_chunk(
        self,
        chunk_id: UUID,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        source: Optional[str] = None,
        position: Optional[int] = None,
        page_number: Optional[int] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Chunk:
        """Update a chunk"""
        data = {}
        if text is not None:
            data["text"] = text
        if embedding is not None:
            data["embedding"] = embedding
        if source is not None:
            data["source"] = source
        if position is not None:
            data["position"] = position
        if page_number is not None:
            data["page_number"] = page_number
        if author is not None:
            data["author"] = author
        if tags is not None:
            data["tags"] = tags

        response = self._request("PUT", f"/chunks/{chunk_id}", json=data)
        return Chunk.from_dict(response)

    def delete_chunk(self, chunk_id: UUID) -> None:
        """Delete a chunk"""
        self._request("DELETE", f"/chunks/{chunk_id}")

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._request("GET", "/health")

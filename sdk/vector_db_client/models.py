"""
SDK Data Models

These mirror the API response models for type safety and autocomplete.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


class LibraryMetadata:
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        tags: List[str] = None,
        embedding_dimension: int = 0,
        total_documents: int = 0,
        total_chunks: int = 0,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.embedding_dimension = embedding_dimension
        self.total_documents = total_documents
        self.total_chunks = total_chunks
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LibraryMetadata":
        return cls(
            name=data["name"],
            description=data.get("description"),
            tags=data.get("tags", []),
            embedding_dimension=data.get("embedding_dimension", 0),
            total_documents=data.get("total_documents", 0),
            total_chunks=data.get("total_chunks", 0),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else None
            ),
        )


class Library:
    def __init__(
        self,
        id: UUID,
        metadata: LibraryMetadata,
        is_indexed: bool = False,
        index_type: Optional[str] = None,
        document_ids: List[UUID] = None,
    ):
        self.id = id
        self.metadata = metadata
        self.is_indexed = is_indexed
        self.index_type = index_type
        self.document_ids = document_ids or []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Library":
        # API returns flat structure, not nested metadata
        metadata = LibraryMetadata(
            name=data.get("name", ""),
            description=data.get("description"),
            tags=data.get("tags", []),
            embedding_dimension=data.get("embedding_dimension", 0),
            total_documents=data.get("total_documents", 0),
            total_chunks=data.get("total_chunks", 0),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else None
            ),
        )

        return cls(
            id=UUID(data["id"]),
            metadata=metadata,
            is_indexed=data.get("is_indexed", False),
            index_type=data.get("index_type"),
            document_ids=[UUID(doc_id) for doc_id in data.get("document_ids", [])],
        )


class DocumentMetadata:
    def __init__(
        self,
        title: str,
        source: str,
        author: Optional[str] = None,
        tags: List[str] = None,
        language: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.title = title
        self.source = source
        self.author = author
        self.tags = tags or []
        self.language = language
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        return cls(
            title=data["title"],
            source=data["source"],
            author=data.get("author"),
            tags=data.get("tags", []),
            language=data.get("language"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else None
            ),
        )


class Document:
    def __init__(
        self,
        id: UUID,
        library_id: UUID,
        metadata: DocumentMetadata,
        chunk_ids: List[UUID] = None,
    ):
        self.id = id
        self.library_id = library_id
        self.metadata = metadata
        self.chunk_ids = chunk_ids or []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        # API returns flat structure, not nested metadata
        metadata = DocumentMetadata(
            title=data.get("title", ""),
            source=data.get("source", ""),
            author=data.get("author"),
            tags=data.get("tags", []),
            language=data.get("language"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else None
            ),
        )

        return cls(
            id=UUID(data["id"]),
            library_id=UUID(data["library_id"]),
            metadata=metadata,
            chunk_ids=[UUID(chunk_id) for chunk_id in data.get("chunk_ids", [])],
        )


class ChunkMetadata:
    def __init__(
        self,
        source: str,
        page_number: Optional[int] = None,
        author: Optional[str] = None,
        tags: List[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.source = source
        self.page_number = page_number
        self.author = author
        self.tags = tags or []
        self.created_at = created_at

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        return cls(
            source=data["source"],
            page_number=data.get("page_number"),
            author=data.get("author"),
            tags=data.get("tags", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else None
            ),
        )


class Chunk:
    def __init__(
        self,
        id: UUID,
        document_id: UUID,
        text: str,
        embedding: List[float],
        metadata: ChunkMetadata,
        position: int,
    ):
        self.id = id
        self.document_id = document_id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata
        self.position = position

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        # API returns flat structure, not nested metadata
        metadata = ChunkMetadata(
            source=data.get("source", ""),
            page_number=data.get("page_number"),
            author=data.get("author"),
            tags=data.get("tags", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else None
            ),
        )

        return cls(
            id=UUID(data["id"]),
            document_id=UUID(data["document_id"]),
            text=data["text"],
            embedding=data["embedding"],
            metadata=metadata,
            position=data["position"],
        )


class MetadataFilter:
    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "operator": self.operator, "value": self.value}


class QueryResult:
    def __init__(
        self,
        chunk_id: UUID,
        document_id: UUID,
        text: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.score = score
        self.metadata = metadata

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResult":
        return cls(
            chunk_id=UUID(data["chunk_id"]),
            document_id=UUID(data["document_id"]),
            text=data["text"],
            score=data["score"],
            metadata=data.get("metadata", {}),
        )

from datetime import UTC, datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChunkMetadata(BaseModel):
    source: str
    page_number: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str = Field(min_length=1)
    embedding: List[float] = Field(min_length=1)
    metadata: ChunkMetadata
    position: int = Field(ge=0)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numeric values")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "This is a sample chunk of text",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": "document.pdf",
                    "page_number": 1,
                    "author": "John Doe",
                    "tags": ["sample", "test"],
                },
                "position": 0,
            }
        }
    )


class DocumentMetadata(BaseModel):
    title: str
    source: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    language: Optional[str] = None


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    library_id: UUID
    metadata: DocumentMetadata
    chunk_ids: List[UUID] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "library_id": "123e4567-e89b-12d3-a456-426614174000",
                "metadata": {
                    "title": "Sample Document",
                    "source": "document.pdf",
                    "author": "John Doe",
                    "tags": ["sample", "test"],
                    "language": "en",
                },
            }
        }
    )


class LibraryMetadata(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: List[str] = Field(default_factory=list)
    embedding_dimension: int = Field(gt=0)
    total_documents: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)


class Library(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    metadata: LibraryMetadata
    document_ids: List[UUID] = Field(default_factory=list)
    is_indexed: bool = Field(default=False)
    index_type: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metadata": {
                    "name": "My Library",
                    "description": "A collection of documents",
                    "tags": ["research", "papers"],
                    "embedding_dimension": 384,
                }
            }
        }
    )

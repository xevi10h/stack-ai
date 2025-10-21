from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CreateLibraryRequest(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class UpdateLibraryRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class LibraryResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    tags: List[str]
    embedding_dimension: int
    total_documents: int
    total_chunks: int
    created_at: datetime
    updated_at: datetime
    is_indexed: bool
    index_type: Optional[str]


class CreateDocumentRequest(BaseModel):
    title: str
    source: str
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    language: Optional[str] = None


class UpdateDocumentRequest(BaseModel):
    title: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    language: Optional[str] = None


class DocumentResponse(BaseModel):
    id: UUID
    library_id: UUID
    title: str
    source: str
    author: Optional[str]
    tags: List[str]
    language: Optional[str]
    created_at: datetime
    updated_at: datetime
    chunk_count: int


class CreateChunkRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str
    page_number: Optional[int] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    position: int = Field(ge=0)


class UpdateChunkRequest(BaseModel):
    text: Optional[str] = Field(None, min_length=1)
    source: Optional[str] = None
    page_number: Optional[int] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    position: Optional[int] = Field(None, ge=0)


class ChunkResponse(BaseModel):
    id: UUID
    document_id: UUID
    text: str
    embedding: List[float]
    source: str
    page_number: Optional[int]
    author: Optional[str]
    tags: List[str]
    created_at: datetime
    position: int


class IndexLibraryRequest(BaseModel):
    index_type: str = Field(pattern="^(flat|hnsw|ivf)$")


class MetadataFilter(BaseModel):
    field: str
    operator: str = Field(pattern="^(eq|ne|gt|gte|lt|lte|in|contains)$")
    value: Any


class QueryRequest(BaseModel):
    query_text: str = Field(min_length=1)
    k: int = Field(gt=0, le=100, default=10)
    metadata_filters: Optional[List[MetadataFilter]] = None


class QueryResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    text: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    results: List[QueryResult]
    total_results: int
    query_time_ms: float

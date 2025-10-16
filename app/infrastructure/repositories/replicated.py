"""
Replicated Repository Wrappers

These wrappers add replication capabilities to existing repositories.
They delegate to the underlying repository for local storage and use
ReplicationManager for async replication to follower nodes.
"""

import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.models import Chunk, Document, Library
from app.infrastructure.replication.node import ReplicationManager
from app.infrastructure.repositories.base import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)


class ReplicatedLibraryRepository(LibraryRepository):
    """
    Wraps a LibraryRepository with replication capabilities.

    All write operations (create, update, delete) are:
    1. Applied locally first
    2. Replicated asynchronously to followers
    """

    def __init__(
        self, local_repo: LibraryRepository, replication_manager: ReplicationManager
    ):
        self._local_repo = local_repo
        self._replication_manager = replication_manager

    def create(self, library: Library) -> Library:
        result = self._local_repo.create(library)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="create",
                entity_type="library",
                entity_id=result.id,
                data=self._library_to_dict(result),
            )
        )

        return result

    def get(self, library_id: UUID) -> Optional[Library]:
        return self._local_repo.get(library_id)

    def list_all(self) -> List[Library]:
        return self._local_repo.list_all()

    def update(self, library: Library) -> Library:
        result = self._local_repo.update(library)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="update",
                entity_type="library",
                entity_id=result.id,
                data=self._library_to_dict(result),
            )
        )

        return result

    def delete(self, library_id: UUID) -> None:
        self._local_repo.delete(library_id)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="delete",
                entity_type="library",
                entity_id=library_id,
                data={},
            )
        )

    def _library_to_dict(self, library: Library) -> Dict[str, Any]:
        return {
            "id": str(library.id),
            "metadata": {
                "name": library.metadata.name,
                "description": library.metadata.description,
                "tags": library.metadata.tags,
                "embedding_dimension": library.metadata.embedding_dimension,
                "total_documents": library.metadata.total_documents,
                "total_chunks": library.metadata.total_chunks,
                "created_at": library.metadata.created_at.isoformat(),
                "updated_at": library.metadata.updated_at.isoformat(),
            },
            "is_indexed": library.is_indexed,
            "index_type": library.index_type,
            "document_ids": [str(doc_id) for doc_id in library.document_ids],
        }


class ReplicatedDocumentRepository(DocumentRepository):
    """
    Wraps a DocumentRepository with replication capabilities.
    """

    def __init__(
        self, local_repo: DocumentRepository, replication_manager: ReplicationManager
    ):
        self._local_repo = local_repo
        self._replication_manager = replication_manager

    def create(self, document: Document) -> Document:
        result = self._local_repo.create(document)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="create",
                entity_type="document",
                entity_id=result.id,
                data=self._document_to_dict(result),
            )
        )

        return result

    def get(self, document_id: UUID) -> Optional[Document]:
        return self._local_repo.get(document_id)

    def list_by_library(self, library_id: UUID) -> List[Document]:
        return self._local_repo.list_by_library(library_id)

    def update(self, document: Document) -> Document:
        result = self._local_repo.update(document)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="update",
                entity_type="document",
                entity_id=result.id,
                data=self._document_to_dict(result),
            )
        )

        return result

    def delete(self, document_id: UUID) -> None:
        self._local_repo.delete(document_id)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="delete",
                entity_type="document",
                entity_id=document_id,
                data={},
            )
        )

    def _document_to_dict(self, document: Document) -> Dict[str, Any]:
        return {
            "id": str(document.id),
            "library_id": str(document.library_id),
            "metadata": {
                "title": document.metadata.title,
                "source": document.metadata.source,
                "author": document.metadata.author,
                "tags": document.metadata.tags,
                "language": document.metadata.language,
                "created_at": document.metadata.created_at.isoformat(),
                "updated_at": document.metadata.updated_at.isoformat(),
            },
            "chunk_ids": [str(chunk_id) for chunk_id in document.chunk_ids],
        }


class ReplicatedChunkRepository(ChunkRepository):
    """
    Wraps a ChunkRepository with replication capabilities.
    """

    def __init__(
        self, local_repo: ChunkRepository, replication_manager: ReplicationManager
    ):
        self._local_repo = local_repo
        self._replication_manager = replication_manager

    def create(self, chunk: Chunk) -> Chunk:
        result = self._local_repo.create(chunk)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="create",
                entity_type="chunk",
                entity_id=result.id,
                data=self._chunk_to_dict(result),
            )
        )

        return result

    def get(self, chunk_id: UUID) -> Optional[Chunk]:
        return self._local_repo.get(chunk_id)

    def list_by_document(self, document_id: UUID) -> List[Chunk]:
        return self._local_repo.list_by_document(document_id)

    def list_by_library(self, library_id: UUID) -> List[Chunk]:
        return self._local_repo.list_by_library(library_id)

    def update(self, chunk: Chunk) -> Chunk:
        result = self._local_repo.update(chunk)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="update",
                entity_type="chunk",
                entity_id=result.id,
                data=self._chunk_to_dict(result),
            )
        )

        return result

    def delete(self, chunk_id: UUID) -> None:
        self._local_repo.delete(chunk_id)

        asyncio.create_task(
            self._replication_manager.replicate(
                operation="delete",
                entity_type="chunk",
                entity_id=chunk_id,
                data={},
            )
        )

    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        return {
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
        }

import os

from app.application.services import ChunkService, DocumentService, LibraryService
from app.infrastructure.replication.node import HealthChecker, ReplicationManager
from app.infrastructure.repositories.memory import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
)
from app.infrastructure.repositories.replicated import (
    ReplicatedChunkRepository,
    ReplicatedDocumentRepository,
    ReplicatedLibraryRepository,
)

REPLICATION_ENABLED = os.getenv("REPLICATION_ENABLED", "false").lower() == "true"

_replication_manager = ReplicationManager()
_health_checker = HealthChecker(_replication_manager)

_local_library_repo = InMemoryLibraryRepository()
_local_document_repo = InMemoryDocumentRepository()
_local_chunk_repo = InMemoryChunkRepository()

if REPLICATION_ENABLED:
    _library_repo = ReplicatedLibraryRepository(
        _local_library_repo, _replication_manager
    )
    _document_repo = ReplicatedDocumentRepository(
        _local_document_repo, _replication_manager
    )
    _chunk_repo = ReplicatedChunkRepository(_local_chunk_repo, _replication_manager)
else:
    _library_repo = _local_library_repo
    _document_repo = _local_document_repo
    _chunk_repo = _local_chunk_repo

_library_service = LibraryService(_library_repo, _document_repo, _chunk_repo)
_document_service = DocumentService(_library_repo, _document_repo, _chunk_repo)
_chunk_service = ChunkService(_library_repo, _document_repo, _chunk_repo)


def get_library_service() -> LibraryService:
    return _library_service


def get_document_service() -> DocumentService:
    return _document_service


def get_chunk_service() -> ChunkService:
    return _chunk_service


def get_replication_manager() -> ReplicationManager:
    return _replication_manager


def get_health_checker() -> HealthChecker:
    return _health_checker

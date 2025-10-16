from app.application.services import ChunkService, DocumentService, LibraryService
from app.infrastructure.repositories.memory import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
)

_library_repo = InMemoryLibraryRepository()
_document_repo = InMemoryDocumentRepository()
_chunk_repo = InMemoryChunkRepository()

_library_service = LibraryService(_library_repo, _document_repo, _chunk_repo)
_document_service = DocumentService(_library_repo, _document_repo, _chunk_repo)
_chunk_service = ChunkService(_library_repo, _document_repo, _chunk_repo)


def get_library_service() -> LibraryService:
    return _library_service


def get_document_service() -> DocumentService:
    return _document_service


def get_chunk_service() -> ChunkService:
    return _chunk_service

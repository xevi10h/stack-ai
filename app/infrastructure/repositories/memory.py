from contextlib import contextmanager
from threading import RLock
from typing import Dict, List, Optional
from uuid import UUID

from app.domain.exceptions import EntityAlreadyExistsError, EntityNotFoundError
from app.domain.models import Chunk, Document, Library
from app.infrastructure.repositories.base import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)


class ReadWriteLock:
    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._lock = RLock()

    @contextmanager
    def read_lock(self):
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write_lock(self):
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()

    def _acquire_read(self):
        with self._lock:
            while self._writers > 0:
                pass
            self._readers += 1

    def _release_read(self):
        with self._lock:
            self._readers -= 1

    def _acquire_write(self):
        with self._lock:
            while self._readers > 0 or self._writers > 0:
                pass
            self._writers += 1

    def _release_write(self):
        with self._lock:
            self._writers -= 1


class InMemoryLibraryRepository(LibraryRepository):
    def __init__(self):
        self._storage: Dict[UUID, Library] = {}
        self._lock = ReadWriteLock()

    def create(self, library: Library) -> Library:
        with self._lock.write_lock():
            if library.id in self._storage:
                raise EntityAlreadyExistsError("Library", library.id)
            self._storage[library.id] = library
            return library

    def get(self, library_id: UUID) -> Optional[Library]:
        with self._lock.read_lock():
            return self._storage.get(library_id)

    def list_all(self) -> List[Library]:
        with self._lock.read_lock():
            return list(self._storage.values())

    def update(self, library: Library) -> Library:
        with self._lock.write_lock():
            if library.id not in self._storage:
                raise EntityNotFoundError("Library", library.id)
            self._storage[library.id] = library
            return library

    def delete(self, library_id: UUID) -> None:
        with self._lock.write_lock():
            if library_id not in self._storage:
                raise EntityNotFoundError("Library", library_id)
            del self._storage[library_id]


class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self):
        self._storage: Dict[UUID, Document] = {}
        self._library_index: Dict[UUID, List[UUID]] = {}
        self._lock = ReadWriteLock()

    def create(self, document: Document) -> Document:
        with self._lock.write_lock():
            if document.id in self._storage:
                raise EntityAlreadyExistsError("Document", document.id)

            self._storage[document.id] = document

            if document.library_id not in self._library_index:
                self._library_index[document.library_id] = []
            self._library_index[document.library_id].append(document.id)

            return document

    def get(self, document_id: UUID) -> Optional[Document]:
        with self._lock.read_lock():
            return self._storage.get(document_id)

    def list_by_library(self, library_id: UUID) -> List[Document]:
        with self._lock.read_lock():
            document_ids = self._library_index.get(library_id, [])
            return [
                self._storage[doc_id]
                for doc_id in document_ids
                if doc_id in self._storage
            ]

    def update(self, document: Document) -> Document:
        with self._lock.write_lock():
            if document.id not in self._storage:
                raise EntityNotFoundError("Document", document.id)

            old_document = self._storage[document.id]
            if old_document.library_id != document.library_id:
                if old_document.library_id in self._library_index:
                    self._library_index[old_document.library_id].remove(document.id)

                if document.library_id not in self._library_index:
                    self._library_index[document.library_id] = []
                self._library_index[document.library_id].append(document.id)

            self._storage[document.id] = document
            return document

    def delete(self, document_id: UUID) -> None:
        with self._lock.write_lock():
            if document_id not in self._storage:
                raise EntityNotFoundError("Document", document_id)

            document = self._storage[document_id]
            if document.library_id in self._library_index:
                self._library_index[document.library_id].remove(document_id)

            del self._storage[document_id]


class InMemoryChunkRepository(ChunkRepository):
    def __init__(self):
        self._storage: Dict[UUID, Chunk] = {}
        self._document_index: Dict[UUID, List[UUID]] = {}
        self._lock = ReadWriteLock()

    def create(self, chunk: Chunk) -> Chunk:
        with self._lock.write_lock():
            if chunk.id in self._storage:
                raise EntityAlreadyExistsError("Chunk", chunk.id)

            self._storage[chunk.id] = chunk

            if chunk.document_id not in self._document_index:
                self._document_index[chunk.document_id] = []
            self._document_index[chunk.document_id].append(chunk.id)

            return chunk

    def get(self, chunk_id: UUID) -> Optional[Chunk]:
        with self._lock.read_lock():
            return self._storage.get(chunk_id)

    def list_by_document(self, document_id: UUID) -> List[Chunk]:
        with self._lock.read_lock():
            chunk_ids = self._document_index.get(document_id, [])
            return [
                self._storage[chunk_id]
                for chunk_id in chunk_ids
                if chunk_id in self._storage
            ]

    def list_by_library(self, library_id: UUID) -> List[Chunk]:
        with self._lock.read_lock():
            return list(self._storage.values())

    def update(self, chunk: Chunk) -> Chunk:
        with self._lock.write_lock():
            if chunk.id not in self._storage:
                raise EntityNotFoundError("Chunk", chunk.id)

            old_chunk = self._storage[chunk.id]
            if old_chunk.document_id != chunk.document_id:
                if old_chunk.document_id in self._document_index:
                    self._document_index[old_chunk.document_id].remove(chunk.id)

                if chunk.document_id not in self._document_index:
                    self._document_index[chunk.document_id] = []
                self._document_index[chunk.document_id].append(chunk.id)

            self._storage[chunk.id] = chunk
            return chunk

    def delete(self, chunk_id: UUID) -> None:
        with self._lock.write_lock():
            if chunk_id not in self._storage:
                raise EntityNotFoundError("Chunk", chunk_id)

            chunk = self._storage[chunk_id]
            if chunk.document_id in self._document_index:
                self._document_index[chunk.document_id].remove(chunk_id)

            del self._storage[chunk_id]

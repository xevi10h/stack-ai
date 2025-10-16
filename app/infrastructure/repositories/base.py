from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from app.domain.models import Chunk, Document, Library


class LibraryRepository(ABC):
    @abstractmethod
    def create(self, library: Library) -> Library:
        pass

    @abstractmethod
    def get(self, library_id: UUID) -> Optional[Library]:
        pass

    @abstractmethod
    def list_all(self) -> List[Library]:
        pass

    @abstractmethod
    def update(self, library: Library) -> Library:
        pass

    @abstractmethod
    def delete(self, library_id: UUID) -> None:
        pass


class DocumentRepository(ABC):
    @abstractmethod
    def create(self, document: Document) -> Document:
        pass

    @abstractmethod
    def get(self, document_id: UUID) -> Optional[Document]:
        pass

    @abstractmethod
    def list_by_library(self, library_id: UUID) -> List[Document]:
        pass

    @abstractmethod
    def update(self, document: Document) -> Document:
        pass

    @abstractmethod
    def delete(self, document_id: UUID) -> None:
        pass


class ChunkRepository(ABC):
    @abstractmethod
    def create(self, chunk: Chunk) -> Chunk:
        pass

    @abstractmethod
    def get(self, chunk_id: UUID) -> Optional[Chunk]:
        pass

    @abstractmethod
    def list_by_document(self, document_id: UUID) -> List[Chunk]:
        pass

    @abstractmethod
    def list_by_library(self, library_id: UUID) -> List[Chunk]:
        pass

    @abstractmethod
    def update(self, chunk: Chunk) -> Chunk:
        pass

    @abstractmethod
    def delete(self, chunk_id: UUID) -> None:
        pass

import time
from datetime import UTC, datetime
from typing import List, Optional, Tuple
from uuid import UUID

from app.api.dto import MetadataFilter
from app.domain.exceptions import (
    EmptyLibraryError,
    EntityNotFoundError,
    InvalidEmbeddingDimensionError,
    LibraryNotIndexedError,
)
from app.domain.models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    Library,
    LibraryMetadata,
)
from app.infrastructure.indexing.base import VectorIndex
from app.infrastructure.indexing.factory import IndexFactory
from app.infrastructure.repositories.base import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)


class LibraryService:
    def __init__(
        self,
        library_repo: LibraryRepository,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
    ):
        self._library_repo = library_repo
        self._document_repo = document_repo
        self._chunk_repo = chunk_repo
        self._indexes: dict[UUID, VectorIndex] = {}

    def create_library(
        self,
        name: str,
        description: Optional[str],
        tags: List[str],
        embedding_dimension: int,
    ) -> Library:
        metadata = LibraryMetadata(
            name=name,
            description=description,
            tags=tags,
            embedding_dimension=embedding_dimension,
        )
        library = Library(metadata=metadata)
        return self._library_repo.create(library)

    def get_library(self, library_id: UUID) -> Library:
        library = self._library_repo.get(library_id)
        if not library:
            raise EntityNotFoundError("Library", library_id)
        return library

    def list_libraries(self) -> List[Library]:
        return self._library_repo.list_all()

    def update_library(
        self,
        library_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Library:
        library = self.get_library(library_id)

        if name is not None:
            library.metadata.name = name
        if description is not None:
            library.metadata.description = description
        if tags is not None:
            library.metadata.tags = tags

        library.metadata.updated_at = datetime.now(UTC)

        return self._library_repo.update(library)

    def delete_library(self, library_id: UUID) -> None:
        library = self.get_library(library_id)

        documents = self._document_repo.list_by_library(library_id)
        for document in documents:
            chunks = self._chunk_repo.list_by_document(document.id)
            for chunk in chunks:
                self._chunk_repo.delete(chunk.id)
            self._document_repo.delete(document.id)

        if library_id in self._indexes:
            del self._indexes[library_id]

        self._library_repo.delete(library_id)

    def index_library(self, library_id: UUID, index_type: str) -> Library:
        library = self.get_library(library_id)

        documents = self._document_repo.list_by_library(library_id)

        all_chunks = []
        for document in documents:
            chunks = self._chunk_repo.list_by_document(document.id)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise EmptyLibraryError(library_id)

        index = IndexFactory.create_index(index_type)

        vectors = [(chunk.id, chunk.embedding) for chunk in all_chunks]
        index.build(vectors)

        self._indexes[library_id] = index

        library.is_indexed = True
        library.index_type = index_type
        library.metadata.updated_at = datetime.now(UTC)

        return self._library_repo.update(library)

    def query_library(
        self,
        library_id: UUID,
        query_embedding: List[float],
        k: int,
        metadata_filters: Optional[List[MetadataFilter]] = None,
    ) -> Tuple[List[Tuple[Chunk, float]], float]:
        library = self.get_library(library_id)

        if not library.is_indexed:
            raise LibraryNotIndexedError(library_id)

        if len(query_embedding) != library.metadata.embedding_dimension:
            raise InvalidEmbeddingDimensionError(
                library.metadata.embedding_dimension,
                len(query_embedding),
            )

        if library_id not in self._indexes:
            raise LibraryNotIndexedError(library_id)

        start_time = time.time()

        index = self._indexes[library_id]
        results = index.search(query_embedding, k)

        filtered_results = []
        for chunk_id, score in results:
            chunk = self._chunk_repo.get(chunk_id)
            if chunk and self._apply_metadata_filters(chunk, metadata_filters):
                filtered_results.append((chunk, score))

        query_time = (time.time() - start_time) * 1000

        return filtered_results[:k], query_time

    def _apply_metadata_filters(
        self, chunk: Chunk, filters: Optional[List[MetadataFilter]]
    ) -> bool:
        if not filters:
            return True

        for filter_item in filters:
            field = filter_item.field
            operator = filter_item.operator
            value = filter_item.value

            chunk_value = getattr(chunk.metadata, field, None)

            if chunk_value is None:
                return False

            if operator == "eq" and chunk_value != value:
                return False
            elif operator == "ne" and chunk_value == value:
                return False
            elif operator == "gt" and not (chunk_value > value):
                return False
            elif operator == "gte" and not (chunk_value >= value):
                return False
            elif operator == "lt" and not (chunk_value < value):
                return False
            elif operator == "lte" and not (chunk_value <= value):
                return False
            elif operator == "in" and chunk_value not in value:
                return False
            elif operator == "contains":
                if isinstance(chunk_value, str) and value not in chunk_value:
                    return False
                elif isinstance(chunk_value, list) and value not in chunk_value:
                    return False

        return True


class DocumentService:
    def __init__(
        self,
        library_repo: LibraryRepository,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
    ):
        self._library_repo = library_repo
        self._document_repo = document_repo
        self._chunk_repo = chunk_repo

    def create_document(
        self,
        library_id: UUID,
        title: str,
        source: str,
        author: Optional[str],
        tags: List[str],
        language: Optional[str],
    ) -> Document:
        library = self._library_repo.get(library_id)
        if not library:
            raise EntityNotFoundError("Library", library_id)

        metadata = DocumentMetadata(
            title=title,
            source=source,
            author=author,
            tags=tags,
            language=language,
        )
        document = Document(library_id=library_id, metadata=metadata)
        created_document = self._document_repo.create(document)

        library.document_ids.append(created_document.id)
        library.metadata.total_documents += 1
        library.metadata.updated_at = datetime.now(UTC)
        self._library_repo.update(library)

        return created_document

    def get_document(self, document_id: UUID) -> Document:
        document = self._document_repo.get(document_id)
        if not document:
            raise EntityNotFoundError("Document", document_id)
        return document

    def list_documents_by_library(self, library_id: UUID) -> List[Document]:
        library = self._library_repo.get(library_id)
        if not library:
            raise EntityNotFoundError("Library", library_id)

        return self._document_repo.list_by_library(library_id)

    def update_document(
        self,
        document_id: UUID,
        title: Optional[str] = None,
        source: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> Document:
        document = self.get_document(document_id)

        if title is not None:
            document.metadata.title = title
        if source is not None:
            document.metadata.source = source
        if author is not None:
            document.metadata.author = author
        if tags is not None:
            document.metadata.tags = tags
        if language is not None:
            document.metadata.language = language

        document.metadata.updated_at = datetime.now(UTC)

        return self._document_repo.update(document)

    def delete_document(self, document_id: UUID) -> None:
        document = self.get_document(document_id)

        chunks = self._chunk_repo.list_by_document(document_id)
        for chunk in chunks:
            self._chunk_repo.delete(chunk.id)

        library = self._library_repo.get(document.library_id)
        if library:
            if document_id in library.document_ids:
                library.document_ids.remove(document_id)
            library.metadata.total_documents = max(
                0, library.metadata.total_documents - 1
            )
            library.metadata.total_chunks = max(
                0, library.metadata.total_chunks - len(chunks)
            )
            library.metadata.updated_at = datetime.now(UTC)
            self._library_repo.update(library)

        self._document_repo.delete(document_id)


class ChunkService:
    def __init__(
        self,
        library_repo: LibraryRepository,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
    ):
        self._library_repo = library_repo
        self._document_repo = document_repo
        self._chunk_repo = chunk_repo

    def create_chunk(
        self,
        document_id: UUID,
        text: str,
        embedding: List[float],
        source: str,
        page_number: Optional[int],
        author: Optional[str],
        tags: List[str],
        position: int,
    ) -> Chunk:
        document = self._document_repo.get(document_id)
        if not document:
            raise EntityNotFoundError("Document", document_id)

        library = self._library_repo.get(document.library_id)
        if not library:
            raise EntityNotFoundError("Library", document.library_id)

        if len(embedding) != library.metadata.embedding_dimension:
            raise InvalidEmbeddingDimensionError(
                library.metadata.embedding_dimension,
                len(embedding),
            )

        metadata = ChunkMetadata(
            source=source,
            page_number=page_number,
            author=author,
            tags=tags,
        )
        chunk = Chunk(
            document_id=document_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
            position=position,
        )
        created_chunk = self._chunk_repo.create(chunk)

        document.chunk_ids.append(created_chunk.id)
        document.metadata.updated_at = datetime.now(UTC)
        self._document_repo.update(document)

        library.metadata.total_chunks += 1
        library.metadata.updated_at = datetime.now(UTC)
        self._library_repo.update(library)

        return created_chunk

    def get_chunk(self, chunk_id: UUID) -> Chunk:
        chunk = self._chunk_repo.get(chunk_id)
        if not chunk:
            raise EntityNotFoundError("Chunk", chunk_id)
        return chunk

    def list_chunks_by_document(self, document_id: UUID) -> List[Chunk]:
        document = self._document_repo.get(document_id)
        if not document:
            raise EntityNotFoundError("Document", document_id)

        return self._chunk_repo.list_by_document(document_id)

    def update_chunk(
        self,
        chunk_id: UUID,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        source: Optional[str] = None,
        page_number: Optional[int] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        position: Optional[int] = None,
    ) -> Chunk:
        chunk = self.get_chunk(chunk_id)

        document = self._document_repo.get(chunk.document_id)
        if not document:
            raise EntityNotFoundError("Document", chunk.document_id)

        library = self._library_repo.get(document.library_id)
        if not library:
            raise EntityNotFoundError("Library", document.library_id)

        if embedding is not None:
            if len(embedding) != library.metadata.embedding_dimension:
                raise InvalidEmbeddingDimensionError(
                    library.metadata.embedding_dimension,
                    len(embedding),
                )
            chunk.embedding = embedding

        if text is not None:
            chunk.text = text
        if source is not None:
            chunk.metadata.source = source
        if page_number is not None:
            chunk.metadata.page_number = page_number
        if author is not None:
            chunk.metadata.author = author
        if tags is not None:
            chunk.metadata.tags = tags
        if position is not None:
            chunk.position = position

        chunk.metadata.created_at = datetime.now(UTC)

        return self._chunk_repo.update(chunk)

    def delete_chunk(self, chunk_id: UUID) -> None:
        chunk = self.get_chunk(chunk_id)

        document = self._document_repo.get(chunk.document_id)
        if document:
            if chunk_id in document.chunk_ids:
                document.chunk_ids.remove(chunk_id)
            document.metadata.updated_at = datetime.now(UTC)
            self._document_repo.update(document)

            library = self._library_repo.get(document.library_id)
            if library:
                library.metadata.total_chunks = max(
                    0, library.metadata.total_chunks - 1
                )
                library.metadata.updated_at = datetime.now(UTC)
                self._library_repo.update(library)

        self._chunk_repo.delete(chunk_id)

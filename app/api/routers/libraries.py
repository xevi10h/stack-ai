from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_library_service
from app.api.dto import (
    CreateLibraryRequest,
    IndexLibraryRequest,
    LibraryResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    UpdateLibraryRequest,
)
from app.application.services import LibraryService

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post(
    "",
    response_model=LibraryResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_library(
    request: CreateLibraryRequest,
    service: LibraryService = Depends(get_library_service),
):
    library = service.create_library(
        name=request.name,
        description=request.description,
        tags=request.tags,
    )

    return LibraryResponse(
        id=library.id,
        name=library.metadata.name,
        description=library.metadata.description,
        tags=library.metadata.tags,
        embedding_dimension=library.metadata.embedding_dimension,
        total_documents=library.metadata.total_documents,
        total_chunks=library.metadata.total_chunks,
        created_at=library.metadata.created_at,
        updated_at=library.metadata.updated_at,
        is_indexed=library.is_indexed,
        index_type=library.index_type,
    )


@router.get("", response_model=List[LibraryResponse])
def list_libraries(service: LibraryService = Depends(get_library_service)):
    libraries = service.list_libraries()

    return [
        LibraryResponse(
            id=library.id,
            name=library.metadata.name,
            description=library.metadata.description,
            tags=library.metadata.tags,
            embedding_dimension=library.metadata.embedding_dimension,
            total_documents=library.metadata.total_documents,
            total_chunks=library.metadata.total_chunks,
            created_at=library.metadata.created_at,
            updated_at=library.metadata.updated_at,
            is_indexed=library.is_indexed,
            index_type=library.index_type,
        )
        for library in libraries
    ]


@router.get("/{library_id}", response_model=LibraryResponse)
def get_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    library = service.get_library(library_id)

    return LibraryResponse(
        id=library.id,
        name=library.metadata.name,
        description=library.metadata.description,
        tags=library.metadata.tags,
        embedding_dimension=library.metadata.embedding_dimension,
        total_documents=library.metadata.total_documents,
        total_chunks=library.metadata.total_chunks,
        created_at=library.metadata.created_at,
        updated_at=library.metadata.updated_at,
        is_indexed=library.is_indexed,
        index_type=library.index_type,
    )


@router.patch("/{library_id}", response_model=LibraryResponse)
def update_library(
    library_id: UUID,
    request: UpdateLibraryRequest,
    service: LibraryService = Depends(get_library_service),
):
    library = service.update_library(
        library_id=library_id,
        name=request.name,
        description=request.description,
        tags=request.tags,
    )

    return LibraryResponse(
        id=library.id,
        name=library.metadata.name,
        description=library.metadata.description,
        tags=library.metadata.tags,
        embedding_dimension=library.metadata.embedding_dimension,
        total_documents=library.metadata.total_documents,
        total_chunks=library.metadata.total_chunks,
        created_at=library.metadata.created_at,
        updated_at=library.metadata.updated_at,
        is_indexed=library.is_indexed,
        index_type=library.index_type,
    )


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    service.delete_library(library_id)


@router.post("/{library_id}/index", response_model=LibraryResponse)
def index_library(
    library_id: UUID,
    request: IndexLibraryRequest,
    service: LibraryService = Depends(get_library_service),
):
    library = service.index_library(library_id, request.index_type)

    return LibraryResponse(
        id=library.id,
        name=library.metadata.name,
        description=library.metadata.description,
        tags=library.metadata.tags,
        embedding_dimension=library.metadata.embedding_dimension,
        total_documents=library.metadata.total_documents,
        total_chunks=library.metadata.total_chunks,
        created_at=library.metadata.created_at,
        updated_at=library.metadata.updated_at,
        is_indexed=library.is_indexed,
        index_type=library.index_type,
    )


@router.post("/{library_id}/query", response_model=QueryResponse)
def query_library(
    library_id: UUID,
    request: QueryRequest,
    service: LibraryService = Depends(get_library_service),
):
    results, query_time = service.query_library(
        library_id=library_id,
        query_text=request.query_text,
        k=request.k,
        metadata_filters=request.metadata_filters,
    )

    query_results = [
        QueryResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=score,
            metadata={
                "source": chunk.metadata.source,
                "page_number": chunk.metadata.page_number,
                "author": chunk.metadata.author,
                "tags": chunk.metadata.tags,
                "created_at": chunk.metadata.created_at.isoformat(),
                "position": chunk.position,
            },
        )
        for chunk, score in results
    ]

    return QueryResponse(
        results=query_results,
        total_results=len(query_results),
        query_time_ms=query_time,
    )

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_chunk_service
from app.api.dto import (
    ChunkResponse,
    CreateChunkRequest,
    UpdateChunkRequest,
)
from app.application.services import ChunkService

router = APIRouter(
    prefix="/libraries/{library_id}/documents/{document_id}/chunks",
    tags=["chunks"],
)


@router.post(
    "",
    response_model=ChunkResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_chunk(
    library_id: UUID,
    document_id: UUID,
    request: CreateChunkRequest,
    service: ChunkService = Depends(get_chunk_service),
):
    chunk = service.create_chunk(
        document_id=document_id,
        text=request.text,
        source=request.source,
        page_number=request.page_number,
        author=request.author,
        tags=request.tags,
        position=request.position,
    )

    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        source=chunk.metadata.source,
        page_number=chunk.metadata.page_number,
        author=chunk.metadata.author,
        tags=chunk.metadata.tags,
        created_at=chunk.metadata.created_at,
        position=chunk.position,
    )


@router.get("", response_model=List[ChunkResponse])
def list_chunks(
    library_id: UUID,
    document_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
):
    chunks = service.list_chunks_by_document(document_id)

    return [
        ChunkResponse(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            embedding=chunk.embedding,
            source=chunk.metadata.source,
            page_number=chunk.metadata.page_number,
            author=chunk.metadata.author,
            tags=chunk.metadata.tags,
            created_at=chunk.metadata.created_at,
            position=chunk.position,
        )
        for chunk in chunks
    ]


@router.get("/{chunk_id}", response_model=ChunkResponse)
def get_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
):
    chunk = service.get_chunk(chunk_id)

    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        source=chunk.metadata.source,
        page_number=chunk.metadata.page_number,
        author=chunk.metadata.author,
        tags=chunk.metadata.tags,
        created_at=chunk.metadata.created_at,
        position=chunk.position,
    )


@router.patch("/{chunk_id}", response_model=ChunkResponse)
def update_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    request: UpdateChunkRequest,
    service: ChunkService = Depends(get_chunk_service),
):
    chunk = service.update_chunk(
        chunk_id=chunk_id,
        text=request.text,
        source=request.source,
        page_number=request.page_number,
        author=request.author,
        tags=request.tags,
        position=request.position,
    )

    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        source=chunk.metadata.source,
        page_number=chunk.metadata.page_number,
        author=chunk.metadata.author,
        tags=chunk.metadata.tags,
        created_at=chunk.metadata.created_at,
        position=chunk.position,
    )


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
):
    service.delete_chunk(chunk_id)

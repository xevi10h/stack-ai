from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_document_service
from app.api.dto import (
    CreateDocumentRequest,
    DocumentResponse,
    UpdateDocumentRequest,
)
from app.application.services import DocumentService

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    library_id: UUID,
    request: CreateDocumentRequest,
    service: DocumentService = Depends(get_document_service),
):
    document = service.create_document(
        library_id=library_id,
        title=request.title,
        source=request.source,
        author=request.author,
        tags=request.tags,
        language=request.language,
    )

    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        title=document.metadata.title,
        source=document.metadata.source,
        author=document.metadata.author,
        tags=document.metadata.tags,
        language=document.metadata.language,
        created_at=document.metadata.created_at,
        updated_at=document.metadata.updated_at,
        chunk_count=len(document.chunk_ids),
    )


@router.get("", response_model=List[DocumentResponse])
def list_documents(
    library_id: UUID,
    service: DocumentService = Depends(get_document_service),
):
    documents = service.list_documents_by_library(library_id)

    return [
        DocumentResponse(
            id=document.id,
            library_id=document.library_id,
            title=document.metadata.title,
            source=document.metadata.source,
            author=document.metadata.author,
            tags=document.metadata.tags,
            language=document.metadata.language,
            created_at=document.metadata.created_at,
            updated_at=document.metadata.updated_at,
            chunk_count=len(document.chunk_ids),
        )
        for document in documents
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    library_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
):
    document = service.get_document(document_id)

    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        title=document.metadata.title,
        source=document.metadata.source,
        author=document.metadata.author,
        tags=document.metadata.tags,
        language=document.metadata.language,
        created_at=document.metadata.created_at,
        updated_at=document.metadata.updated_at,
        chunk_count=len(document.chunk_ids),
    )


@router.patch("/{document_id}", response_model=DocumentResponse)
def update_document(
    library_id: UUID,
    document_id: UUID,
    request: UpdateDocumentRequest,
    service: DocumentService = Depends(get_document_service),
):
    document = service.update_document(
        document_id=document_id,
        title=request.title,
        source=request.source,
        author=request.author,
        tags=request.tags,
        language=request.language,
    )

    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        title=document.metadata.title,
        source=document.metadata.source,
        author=document.metadata.author,
        tags=document.metadata.tags,
        language=document.metadata.language,
        created_at=document.metadata.created_at,
        updated_at=document.metadata.updated_at,
        chunk_count=len(document.chunk_ids),
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    library_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
):
    service.delete_document(document_id)

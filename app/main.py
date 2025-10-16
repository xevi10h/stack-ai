from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.api.error_handlers import (
    empty_library_handler,
    entity_already_exists_handler,
    entity_not_found_handler,
    generic_error_handler,
    invalid_embedding_dimension_handler,
    invalid_metadata_filter_handler,
    library_not_indexed_handler,
    validation_error_handler,
    value_error_handler,
)
from app.api.routers import chunks, documents, libraries, nodes
from app.domain.exceptions import (
    EmptyLibraryError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidEmbeddingDimensionError,
    InvalidMetadataFilterError,
    LibraryNotIndexedError,
)

app = FastAPI(
    title="Vector Database API",
    description="A REST API for indexing and querying documents in a Vector Database",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(libraries.router)
app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(nodes.router)

app.add_exception_handler(EntityNotFoundError, entity_not_found_handler)
app.add_exception_handler(EntityAlreadyExistsError, entity_already_exists_handler)
app.add_exception_handler(LibraryNotIndexedError, library_not_indexed_handler)
app.add_exception_handler(
    InvalidEmbeddingDimensionError, invalid_embedding_dimension_handler
)
app.add_exception_handler(EmptyLibraryError, empty_library_handler)
app.add_exception_handler(InvalidMetadataFilterError, invalid_metadata_filter_handler)
app.add_exception_handler(ValidationError, validation_error_handler)
app.add_exception_handler(ValueError, value_error_handler)
app.add_exception_handler(Exception, generic_error_handler)


@app.get("/", tags=["health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Vector Database API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["health"])
def detailed_health_check():
    return {
        "status": "healthy",
        "service": "Vector Database API",
        "version": "1.0.0",
        "available_indices": ["flat", "hnsw", "ivf"],
    }

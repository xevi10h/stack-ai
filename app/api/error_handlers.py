from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.domain.exceptions import (
    EmptyLibraryError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidEmbeddingDimensionError,
    InvalidMetadataFilterError,
    LibraryNotIndexedError,
)


def create_error_response(status_code: int, message: str, details: str = None):
    content = {"error": message}
    if details:
        content["details"] = details
    return JSONResponse(status_code=status_code, content=content)


async def entity_not_found_handler(request: Request, exc: EntityNotFoundError):
    return create_error_response(
        status.HTTP_404_NOT_FOUND,
        f"{exc.entity_type} not found",
        str(exc),
    )


async def entity_already_exists_handler(
    request: Request, exc: EntityAlreadyExistsError
):
    return create_error_response(
        status.HTTP_409_CONFLICT,
        f"{exc.entity_type} already exists",
        str(exc),
    )


async def library_not_indexed_handler(request: Request, exc: LibraryNotIndexedError):
    return create_error_response(
        status.HTTP_400_BAD_REQUEST,
        "Library not indexed",
        str(exc),
    )


async def invalid_embedding_dimension_handler(
    request: Request, exc: InvalidEmbeddingDimensionError
):
    return create_error_response(
        status.HTTP_400_BAD_REQUEST,
        "Invalid embedding dimension",
        str(exc),
    )


async def empty_library_handler(request: Request, exc: EmptyLibraryError):
    return create_error_response(
        status.HTTP_400_BAD_REQUEST,
        "Library is empty",
        str(exc),
    )


async def invalid_metadata_filter_handler(
    request: Request, exc: InvalidMetadataFilterError
):
    return create_error_response(
        status.HTTP_400_BAD_REQUEST,
        "Invalid metadata filter",
        str(exc),
    )


async def validation_error_handler(request: Request, exc: ValidationError):
    return create_error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "Validation error",
        str(exc),
    )


async def value_error_handler(request: Request, exc: ValueError):
    return create_error_response(
        status.HTTP_400_BAD_REQUEST,
        "Invalid value",
        str(exc),
    )


async def generic_error_handler(request: Request, exc: Exception):
    return create_error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal server error",
        str(exc),
    )

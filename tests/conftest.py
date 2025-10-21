import os
import pytest
from unittest.mock import Mock, patch

# Set mock API key before importing dependencies
os.environ["COHERE_API_KEY"] = "mock-api-key-for-testing"

# Mock the Cohere client to avoid API calls during import
with patch('cohere.Client') as mock_cohere:
    mock_instance = Mock()
    mock_instance.embed.return_value = Mock(embeddings=[[0.0] * 384])
    mock_cohere.return_value = mock_instance

    from app.api.dependencies import (
        _library_repo,
        _document_repo,
        _chunk_repo,
    )
    from app.application.services import ChunkService, DocumentService, LibraryService
    from app.infrastructure.embeddings.mock_embedding import MockEmbeddingService
    from app.main import app
    from app.api import dependencies


@pytest.fixture(autouse=True)
def reset_repositories():
    """Reset repositories before each test"""
    # Clear all data from repositories
    _library_repo._storage.clear()
    _document_repo._storage.clear()
    _document_repo._library_index.clear()
    _chunk_repo._storage.clear()
    _chunk_repo._document_index.clear()
    yield


@pytest.fixture(autouse=True)
def use_mock_embedding_service():
    """Use mock embedding service for all tests"""
    # Create mock embedding service
    mock_embedding_service = MockEmbeddingService(dimension=384)

    # Override the services with mock embedding service
    original_library_service = dependencies._library_service
    original_chunk_service = dependencies._chunk_service
    original_embedding_service = dependencies._embedding_service

    dependencies._embedding_service = mock_embedding_service
    dependencies._library_service = LibraryService(
        _library_repo, _document_repo, _chunk_repo, mock_embedding_service
    )
    dependencies._chunk_service = ChunkService(
        _library_repo, _document_repo, _chunk_repo, mock_embedding_service
    )

    yield

    # Restore original services
    dependencies._embedding_service = original_embedding_service
    dependencies._library_service = original_library_service
    dependencies._chunk_service = original_chunk_service

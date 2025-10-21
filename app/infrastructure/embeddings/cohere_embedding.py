import os
from typing import List

import cohere

from app.infrastructure.embeddings.base import EmbeddingService


class CohereEmbeddingService(EmbeddingService):
    """
    Cohere embedding service implementation

    Uses Cohere's embed API to generate embeddings for text
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-english-light-v3.0",
        input_type: str = "search_document",
    ):
        """
        Initialize Cohere embedding service

        Args:
            api_key: Cohere API key (if None, reads from COHERE_API_KEY env var)
            model: Model to use for embeddings
            input_type: Type of input ("search_document" for indexing, "search_query" for querying)
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key must be provided or set in COHERE_API_KEY environment variable"
            )

        self.model = model
        self.input_type = input_type
        self.client = cohere.Client(self.api_key)

        # Dimension for embed-english-light-v3.0 is 384
        self._dimension = 384

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        response = self.client.embed(
            texts=[text], model=self.model, input_type=self.input_type
        )
        return response.embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        response = self.client.embed(
            texts=texts, model=self.model, input_type=self.input_type
        )
        return response.embeddings

    def get_dimension(self) -> int:
        """
        Get the dimensionality of embeddings

        Returns:
            384 for embed-english-light-v3.0
        """
        return self._dimension


def create_query_embedding_service(api_key: str | None = None) -> EmbeddingService:
    """
    Factory function to create embedding service for queries

    Args:
        api_key: Cohere API key

    Returns:
        EmbeddingService configured for search queries
    """
    return CohereEmbeddingService(
        api_key=api_key, model="embed-english-light-v3.0", input_type="search_query"
    )


def create_document_embedding_service(api_key: str | None = None) -> EmbeddingService:
    """
    Factory function to create embedding service for documents

    Args:
        api_key: Cohere API key

    Returns:
        EmbeddingService configured for document indexing
    """
    return CohereEmbeddingService(
        api_key=api_key,
        model="embed-english-light-v3.0",
        input_type="search_document",
    )

from typing import List

from app.infrastructure.embeddings.base import EmbeddingService


class MockEmbeddingService(EmbeddingService):
    """
    Mock embedding service for testing

    Returns deterministic embeddings based on text length and content
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize mock embedding service

        Args:
            dimension: Dimensionality of embeddings (default: 384)
        """
        self._dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a deterministic mock embedding for a single text

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Create deterministic embedding based on text
        # Use text length and hash to generate unique but consistent vectors
        embedding = [0.0] * self._dimension

        # Simple deterministic algorithm:
        # Use text hash to fill first few positions
        text_hash = hash(text)
        for i in range(min(10, self._dimension)):
            # Generate values between -1 and 1
            embedding[i] = ((text_hash + i * 31) % 2000 - 1000) / 1000.0

        # Use text length to influence some positions
        text_len = len(text)
        for i in range(10, min(20, self._dimension)):
            embedding[i] = ((text_len * (i + 1)) % 2000 - 1000) / 1000.0

        # Normalize the vector (make it unit length)
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        """
        Get the dimensionality of embeddings

        Returns:
            The configured dimension
        """
        return self._dimension

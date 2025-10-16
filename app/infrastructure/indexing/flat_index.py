import heapq
from typing import Dict, List, Tuple
from uuid import UUID

from app.infrastructure.indexing.base import VectorIndex
from app.infrastructure.indexing.utils import cosine_similarity


class FlatIndex(VectorIndex):
    def __init__(self):
        self._vectors: Dict[UUID, List[float]] = {}

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        self._vectors = {chunk_id: vector for chunk_id, vector in vectors}

    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        if not self._vectors:
            return []

        k = min(k, len(self._vectors))

        similarities = []
        for chunk_id, vector in self._vectors.items():
            similarity = cosine_similarity(query_vector, vector)
            similarities.append((-similarity, chunk_id, similarity))

        heapq.heapify(similarities)
        top_k = heapq.nsmallest(k, similarities)

        return [(chunk_id, similarity) for _, chunk_id, similarity in top_k]

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        self._vectors[chunk_id] = vector

    def remove(self, chunk_id: UUID) -> None:
        if chunk_id in self._vectors:
            del self._vectors[chunk_id]

    def clear(self) -> None:
        self._vectors.clear()

    def get_name(self) -> str:
        return "flat"

    def get_complexity(self) -> Tuple[str, str]:
        return ("O(n * d)", "O(n * d)")

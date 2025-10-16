from abc import ABC, abstractmethod
from typing import List, Tuple
from uuid import UUID


class VectorIndex(ABC):
    @abstractmethod
    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        pass

    @abstractmethod
    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        pass

    @abstractmethod
    def remove(self, chunk_id: UUID) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_complexity(self) -> Tuple[str, str]:
        pass

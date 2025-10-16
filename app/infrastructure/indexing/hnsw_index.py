import heapq
import math
import random
from typing import Dict, List, Set, Tuple
from uuid import UUID

from app.infrastructure.indexing.base import VectorIndex
from app.infrastructure.indexing.utils import cosine_similarity


class HNSWIndex(VectorIndex):
    def __init__(self, m: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.m = m
        self.m_max = m
        self.m_max0 = m * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1 / math.log(2.0)

        self._vectors: Dict[UUID, List[float]] = {}
        self._graph: Dict[int, Dict[UUID, Set[UUID]]] = {}
        self._entry_point: UUID | None = None
        self._node_levels: Dict[UUID, int] = {}

    def _get_random_level(self) -> int:
        return int(-math.log(random.uniform(0, 1)) * self.ml)

    def _distance(self, vec1: List[float], vec2: List[float]) -> float:
        return 1.0 - cosine_similarity(vec1, vec2)

    def _search_layer(
        self, query: List[float], entry_points: Set[UUID], ef: int, layer: int
    ) -> List[Tuple[float, UUID]]:
        visited = set(entry_points)
        candidates = [
            (-self._distance(query, self._vectors[ep]), ep) for ep in entry_points
        ]
        heapq.heapify(candidates)

        w = [(-dist, node) for dist, node in candidates]
        heapq.heapify(w)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > -w[0][0]:
                break

            if layer not in self._graph or current not in self._graph[layer]:
                continue

            for neighbor in self._graph[layer][current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query, self._vectors[neighbor])

                    if dist < -w[0][0] or len(w) < ef:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (-dist, neighbor))

                        if len(w) > ef:
                            heapq.heappop(w)

        return [(-dist, node) for dist, node in w]

    def _get_neighbors(
        self, query: List[float], candidates: List[Tuple[float, UUID]], m: int
    ) -> List[UUID]:
        candidates = sorted(candidates, key=lambda x: x[0])
        return [node for _, node in candidates[:m]]

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        self.clear()
        for chunk_id, vector in vectors:
            self.add(chunk_id, vector)

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        if chunk_id in self._vectors:
            return

        self._vectors[chunk_id] = vector
        level = self._get_random_level()
        self._node_levels[chunk_id] = level

        if self._entry_point is None:
            self._entry_point = chunk_id
            for lc in range(level + 1):
                if lc not in self._graph:
                    self._graph[lc] = {}
                self._graph[lc][chunk_id] = set()
            return

        nearest = self._entry_point
        entry_point_level = self._node_levels[self._entry_point]

        for lc in range(entry_point_level, level, -1):
            if lc in self._graph and nearest in self._graph[lc]:
                neighbors_at_layer = self._search_layer(vector, {nearest}, 1, lc)
                if neighbors_at_layer:
                    nearest = neighbors_at_layer[0][1]

        for lc in range(min(level, entry_point_level), -1, -1):
            if lc not in self._graph:
                self._graph[lc] = {}

            candidates = self._search_layer(vector, {nearest}, self.ef_construction, lc)

            m = self.m_max if lc > 0 else self.m_max0
            neighbors = self._get_neighbors(vector, candidates, m)

            if chunk_id not in self._graph[lc]:
                self._graph[lc][chunk_id] = set()

            for neighbor in neighbors:
                self._graph[lc][chunk_id].add(neighbor)
                if neighbor in self._graph[lc]:
                    self._graph[lc][neighbor].add(chunk_id)

                    if len(self._graph[lc][neighbor]) > m:
                        neighbor_candidates = [
                            (
                                self._distance(
                                    self._vectors[neighbor], self._vectors[n]
                                ),
                                n,
                            )
                            for n in self._graph[lc][neighbor]
                        ]
                        new_neighbors = self._get_neighbors(
                            self._vectors[neighbor], neighbor_candidates, m
                        )
                        self._graph[lc][neighbor] = set(new_neighbors)

            if candidates:
                nearest = candidates[0][1]

        if level > entry_point_level:
            self._entry_point = chunk_id

    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        if self._entry_point is None or not self._vectors:
            return []

        k = min(k, len(self._vectors))
        nearest = self._entry_point
        entry_level = self._node_levels[self._entry_point]

        for lc in range(entry_level, 0, -1):
            candidates = self._search_layer(query_vector, {nearest}, 1, lc)
            if candidates:
                nearest = candidates[0][1]

        candidates = self._search_layer(
            query_vector, {nearest}, max(self.ef_search, k), 0
        )

        results = sorted(candidates, key=lambda x: x[0])[:k]
        return [(node, 1.0 - dist) for dist, node in results]

    def remove(self, chunk_id: UUID) -> None:
        if chunk_id not in self._vectors:
            return

        level = self._node_levels[chunk_id]

        for lc in range(level + 1):
            if lc in self._graph and chunk_id in self._graph[lc]:
                for neighbor in self._graph[lc][chunk_id]:
                    if neighbor in self._graph[lc]:
                        self._graph[lc][neighbor].discard(chunk_id)

                del self._graph[lc][chunk_id]

        del self._vectors[chunk_id]
        del self._node_levels[chunk_id]

        if self._entry_point == chunk_id:
            if self._vectors:
                self._entry_point = next(iter(self._vectors.keys()))
            else:
                self._entry_point = None

    def clear(self) -> None:
        self._vectors.clear()
        self._graph.clear()
        self._entry_point = None
        self._node_levels.clear()

    def get_name(self) -> str:
        return "hnsw"

    def get_complexity(self) -> Tuple[str, str]:
        return ("O(n * log(n) * d)", "O(log(n) * d)")

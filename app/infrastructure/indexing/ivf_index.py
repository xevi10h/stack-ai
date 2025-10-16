import heapq
import random
from typing import Dict, List, Set, Tuple
from uuid import UUID

from app.infrastructure.indexing.base import VectorIndex
from app.infrastructure.indexing.utils import cosine_similarity, euclidean_distance


class IVFIndex(VectorIndex):
    def __init__(self, n_clusters: int = 10, n_probe: int = 3):
        self.n_clusters = n_clusters
        self.n_probe = n_probe

        self._vectors: Dict[UUID, List[float]] = {}
        self._centroids: List[List[float]] = []
        self._cluster_assignments: Dict[UUID, int] = {}
        self._inverted_lists: Dict[int, Set[UUID]] = {}
        self._is_trained = False

    def _kmeans_init(self, vectors: List[List[float]]) -> List[List[float]]:
        if len(vectors) <= self.n_clusters:
            return vectors.copy()

        centroids = [vectors[0]]

        for _ in range(1, self.n_clusters):
            distances = []
            for vec in vectors:
                min_dist = min(
                    euclidean_distance(vec, centroid) ** 2 for centroid in centroids
                )
                distances.append(min_dist)

            total_dist = sum(distances)
            if total_dist == 0:
                centroids.append(random.choice(vectors))
            else:
                probabilities = [d / total_dist for d in distances]
                cumulative = []
                total = 0
                for p in probabilities:
                    total += p
                    cumulative.append(total)

                r = random.random()
                for i, cum_prob in enumerate(cumulative):
                    if r <= cum_prob:
                        centroids.append(vectors[i])
                        break

        return centroids

    def _assign_to_clusters(self, vectors: List[List[float]]) -> List[int]:
        assignments = []
        for vec in vectors:
            min_dist = float("inf")
            closest_cluster = 0

            for cluster_idx, centroid in enumerate(self._centroids):
                dist = euclidean_distance(vec, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster_idx

            assignments.append(closest_cluster)

        return assignments

    def _update_centroids(
        self, vectors: List[List[float]], assignments: List[int]
    ) -> None:
        dimension = len(vectors[0])

        cluster_sums = [[0.0] * dimension for _ in range(self.n_clusters)]
        cluster_counts = [0] * self.n_clusters

        for vec, cluster_idx in zip(vectors, assignments):
            for dim in range(dimension):
                cluster_sums[cluster_idx][dim] += vec[dim]
            cluster_counts[cluster_idx] += 1

        for cluster_idx in range(self.n_clusters):
            if cluster_counts[cluster_idx] > 0:
                self._centroids[cluster_idx] = [
                    cluster_sums[cluster_idx][dim] / cluster_counts[cluster_idx]
                    for dim in range(dimension)
                ]

    def _train_kmeans(
        self, vectors: List[List[float]], max_iterations: int = 100
    ) -> None:
        self._centroids = self._kmeans_init(vectors)

        for iteration in range(max_iterations):
            assignments = self._assign_to_clusters(vectors)
            old_centroids = [centroid.copy() for centroid in self._centroids]
            self._update_centroids(vectors, assignments)

            converged = all(
                euclidean_distance(old, new) < 1e-6
                for old, new in zip(old_centroids, self._centroids)
            )

            if converged:
                break

        self._is_trained = True

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        self.clear()

        if not vectors:
            return

        self._vectors = {chunk_id: vector for chunk_id, vector in vectors}

        actual_clusters = min(self.n_clusters, len(vectors))
        self.n_clusters = actual_clusters

        vector_list = [vector for _, vector in vectors]
        self._train_kmeans(vector_list)

        self._inverted_lists = {i: set() for i in range(self.n_clusters)}

        for chunk_id, vector in vectors:
            cluster_idx = self._find_nearest_cluster(vector)
            self._cluster_assignments[chunk_id] = cluster_idx
            self._inverted_lists[cluster_idx].add(chunk_id)

    def _find_nearest_cluster(self, vector: List[float]) -> int:
        if not self._centroids:
            return 0

        min_dist = float("inf")
        nearest_cluster = 0

        for cluster_idx, centroid in enumerate(self._centroids):
            dist = euclidean_distance(vector, centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = cluster_idx

        return nearest_cluster

    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        if not self._is_trained or not self._vectors:
            return []

        k = min(k, len(self._vectors))

        cluster_distances = [
            (euclidean_distance(query_vector, centroid), cluster_idx)
            for cluster_idx, centroid in enumerate(self._centroids)
        ]
        cluster_distances.sort()

        n_probe = min(self.n_probe, len(self._centroids))
        clusters_to_search = [
            cluster_idx for _, cluster_idx in cluster_distances[:n_probe]
        ]

        candidates = []
        for cluster_idx in clusters_to_search:
            for chunk_id in self._inverted_lists[cluster_idx]:
                similarity = cosine_similarity(query_vector, self._vectors[chunk_id])
                candidates.append((-similarity, chunk_id, similarity))

        if not candidates:
            return []

        heapq.heapify(candidates)
        top_k = heapq.nsmallest(k, candidates)

        return [(chunk_id, similarity) for _, chunk_id, similarity in top_k]

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        self._vectors[chunk_id] = vector

        if not self._is_trained:
            return

        cluster_idx = self._find_nearest_cluster(vector)
        self._cluster_assignments[chunk_id] = cluster_idx
        self._inverted_lists[cluster_idx].add(chunk_id)

    def remove(self, chunk_id: UUID) -> None:
        if chunk_id not in self._vectors:
            return

        if chunk_id in self._cluster_assignments:
            cluster_idx = self._cluster_assignments[chunk_id]
            self._inverted_lists[cluster_idx].discard(chunk_id)
            del self._cluster_assignments[chunk_id]

        del self._vectors[chunk_id]

    def clear(self) -> None:
        self._vectors.clear()
        self._centroids.clear()
        self._cluster_assignments.clear()
        self._inverted_lists.clear()
        self._is_trained = False

    def get_name(self) -> str:
        return "ivf"

    def get_complexity(self) -> Tuple[str, str]:
        return ("O(n * k * d)", "O((n/k) * d)")

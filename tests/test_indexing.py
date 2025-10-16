from uuid import uuid4

import pytest

from app.infrastructure.indexing.flat_index import FlatIndex
from app.infrastructure.indexing.hnsw_index import HNSWIndex
from app.infrastructure.indexing.ivf_index import IVFIndex


@pytest.fixture
def sample_vectors():
    return [
        (uuid4(), [1.0, 0.0, 0.0]),
        (uuid4(), [0.0, 1.0, 0.0]),
        (uuid4(), [0.0, 0.0, 1.0]),
        (uuid4(), [0.5, 0.5, 0.0]),
        (uuid4(), [0.5, 0.0, 0.5]),
    ]


def test_flat_index_search(sample_vectors):
    index = FlatIndex()
    index.build(sample_vectors)

    query = [1.0, 0.0, 0.0]
    results = index.search(query, k=3)

    assert len(results) == 3
    assert results[0][0] == sample_vectors[0][0]
    assert results[0][1] >= results[1][1]


def test_hnsw_index_search(sample_vectors):
    index = HNSWIndex(m=4, ef_construction=50, ef_search=20)
    index.build(sample_vectors)

    query = [0.0, 1.0, 0.0]
    results = index.search(query, k=3)

    # HNSW is an approximate algorithm, so we just verify we get results
    assert len(results) >= 1
    # Verify results contain valid UUIDs from our sample
    result_ids = [r[0] for r in results]
    sample_ids = [v[0] for v in sample_vectors]
    assert all(rid in sample_ids for rid in result_ids)


def test_ivf_index_search(sample_vectors):
    index = IVFIndex(n_clusters=2, n_probe=2)
    index.build(sample_vectors)

    query = [0.0, 0.0, 1.0]
    results = index.search(query, k=3)

    assert len(results) == 3
    assert results[0][0] == sample_vectors[2][0]


def test_index_add_remove():
    index = FlatIndex()
    chunk_id = uuid4()
    vector = [1.0, 0.0, 0.0]

    index.add(chunk_id, vector)

    results = index.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0][0] == chunk_id

    index.remove(chunk_id)

    results = index.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 0


def test_index_clear(sample_vectors):
    index = FlatIndex()
    index.build(sample_vectors)

    assert len(index.search([1.0, 0.0, 0.0], k=10)) == 5

    index.clear()

    assert len(index.search([1.0, 0.0, 0.0], k=10)) == 0

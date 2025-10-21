import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_library():
    response = client.post(
        "/libraries",
        json={
            "name": "Test Library",
            "description": "A test library",
            "tags": ["test"],
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Library"
    assert data["embedding_dimension"] == 384  # Auto-determined from embedding service
    assert data["is_indexed"] is False


def test_create_and_query_library():
    library_response = client.post(
        "/libraries",
        json={
            "name": "Query Test Library",
            "description": "Testing queries",
            "tags": ["test"],
        },
    )
    assert library_response.status_code == 201
    library_id = library_response.json()["id"]

    document_response = client.post(
        f"/libraries/{library_id}/documents",
        json={
            "title": "Test Document",
            "source": "test.pdf",
            "tags": ["test"],
        },
    )
    assert document_response.status_code == 201
    document_id = document_response.json()["id"]

    chunk_response = client.post(
        f"/libraries/{library_id}/documents/{document_id}/chunks",
        json={
            "text": "This is a test chunk about machine learning",
            "source": "test.pdf",
            "tags": ["test"],
            "position": 0,
        },
    )
    assert chunk_response.status_code == 201

    index_response = client.post(
        f"/libraries/{library_id}/index",
        json={"index_type": "flat"},
    )
    assert index_response.status_code == 200
    assert index_response.json()["is_indexed"] is True

    query_response = client.post(
        f"/libraries/{library_id}/query",
        json={
            "query_text": "machine learning test",
            "k": 10,
        },
    )
    assert query_response.status_code == 200
    query_data = query_response.json()
    assert query_data["total_results"] == 1
    assert len(query_data["results"]) == 1


def test_list_libraries():
    response = client.get("/libraries")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_update_library():
    create_response = client.post(
        "/libraries",
        json={
            "name": "Original Name",
        },
    )
    library_id = create_response.json()["id"]

    update_response = client.patch(
        f"/libraries/{library_id}",
        json={"name": "Updated Name"},
    )

    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Updated Name"


def test_delete_library():
    create_response = client.post(
        "/libraries",
        json={
            "name": "To Delete",
        },
    )
    library_id = create_response.json()["id"]

    delete_response = client.delete(f"/libraries/{library_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/libraries/{library_id}")
    assert get_response.status_code == 404


def test_library_not_found():
    response = client.get("/libraries/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_chunk_creation():
    """Test that chunks are created successfully with auto-generated embeddings"""
    library_response = client.post(
        "/libraries",
        json={
            "name": "Test Library",
        },
    )
    library_id = library_response.json()["id"]

    document_response = client.post(
        f"/libraries/{library_id}/documents",
        json={"title": "Test Document", "source": "test.pdf"},
    )
    document_id = document_response.json()["id"]

    chunk_response = client.post(
        f"/libraries/{library_id}/documents/{document_id}/chunks",
        json={
            "text": "This is a test chunk with some content",
            "source": "test.pdf",
            "position": 0,
        },
    )

    assert chunk_response.status_code == 201
    chunk_data = chunk_response.json()
    assert chunk_data["text"] == "This is a test chunk with some content"
    assert len(chunk_data["embedding"]) == 384  # Auto-generated with correct dimension


def test_query_unindexed_library():
    library_response = client.post(
        "/libraries",
        json={
            "name": "Unindexed Library",
        },
    )
    library_id = library_response.json()["id"]

    query_response = client.post(
        f"/libraries/{library_id}/query",
        json={
            "query_text": "test query",
            "k": 10,
        },
    )

    assert query_response.status_code == 400

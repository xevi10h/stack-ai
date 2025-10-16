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
            "embedding_dimension": 384,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Library"
    assert data["embedding_dimension"] == 384
    assert data["is_indexed"] is False


def test_create_and_query_library():
    library_response = client.post(
        "/libraries",
        json={
            "name": "Query Test Library",
            "description": "Testing queries",
            "tags": ["test"],
            "embedding_dimension": 3,
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
            "text": "This is a test chunk",
            "embedding": [1.0, 0.0, 0.0],
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
            "embedding": [1.0, 0.0, 0.0],
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
            "embedding_dimension": 384,
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
            "embedding_dimension": 384,
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


def test_invalid_embedding_dimension():
    library_response = client.post(
        "/libraries",
        json={
            "name": "Test Library",
            "embedding_dimension": 3,
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
            "text": "Test",
            "embedding": [1.0, 0.0],
            "source": "test.pdf",
            "position": 0,
        },
    )

    assert chunk_response.status_code == 400


def test_query_unindexed_library():
    library_response = client.post(
        "/libraries",
        json={
            "name": "Unindexed Library",
            "embedding_dimension": 3,
        },
    )
    library_id = library_response.json()["id"]

    query_response = client.post(
        f"/libraries/{library_id}/query",
        json={
            "embedding": [1.0, 0.0, 0.0],
            "k": 10,
        },
    )

    assert query_response.status_code == 400

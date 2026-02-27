import requests
import numpy as np

from conftest import BASE_URL, create_index


def test_add_documents_no_metadata():
    vectors = [np.random.rand(4).tolist() for _ in range(5)]
    ids = list(range(5))

    response = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": "add_docs",
            "ids": ids,
            "vectors": vectors,
            "metadatas": None,
        },
    )
    assert response.status_code == 201, f"Failed to add documents: {response.text}"


def test_add_documents_with_metadata():
    vectors = [np.random.rand(4).tolist() for _ in range(5)]
    ids = list(range(5))
    metadatas = [
        {"name": f"doc_{i}", "integer": i, "float": i * 100 / 3.234} for i in range(5)
    ]

    response = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": "add_docs_metadata",
            "ids": ids,
            "vectors": vectors,
            "metadatas": metadatas,
        },
    )
    assert (
        response.status_code == 201
    ), f"Failed to add documents with metadata: {response.text}"


def test_delete_documents():
    name = "delete_docs_test"
    create_index(name)

    vectors = [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
    add_res = requests.post(
        f"{BASE_URL}/add_documents",
        json={"indexName": name, "ids": [0, 1, 2, 3], "vectors": vectors},
    )
    assert add_res.status_code == 201

    del_res = requests.delete(
        f"{BASE_URL}/delete_documents",
        json={"indexName": name, "ids": [1, 2]},
    )
    assert del_res.status_code == 200

    search_res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": name,
            "queryVector": [1, 0, 0, 0],
            "k": 4,
            "efSearch": 200,
        },
    )
    assert search_res.status_code == 200
    hits = search_res.json()["hits"]
    assert 1 not in hits, f"Deleted id 1 still in hits: {hits}"
    assert 2 not in hits, f"Deleted id 2 still in hits: {hits}"

    assert requests.get(f"{BASE_URL}/get_document/{name}/1").status_code == 404
    assert requests.get(f"{BASE_URL}/get_document/{name}/2").status_code == 404


def test_get_document():
    name = "get_doc_test"
    create_index(name)

    vector = [0.1, 0.2, 0.3, 0.4]
    metadata = {"name": "test_doc", "count": 42, "score": 3.14}
    add_res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": [7],
            "vectors": [vector],
            "metadatas": [metadata],
        },
    )
    assert add_res.status_code == 201

    doc_res = requests.get(f"{BASE_URL}/get_document/{name}/7")
    assert doc_res.status_code == 200
    doc = doc_res.json()
    assert doc["id"] == 7
    assert np.allclose(doc["vector"], vector, atol=1e-5)
    assert doc["metadata"]["name"] == "test_doc"
    assert doc["metadata"]["count"] == 42
    assert abs(doc["metadata"]["score"] - 3.14) < 1e-5

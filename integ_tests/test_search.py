import requests

from conftest import BASE_URL, create_index


def test_search_index_no_filter():
    document_vectors = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
    ]
    ids = list(range(len(document_vectors)))

    add_res = requests.post(
        f"{BASE_URL}/add_documents",
        json={"indexName": "search", "ids": ids, "vectors": document_vectors},
    )
    assert add_res.status_code == 201, f"Failed to add documents: {add_res.text}"

    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": "search",
            "queryVector": [1, 1, 1, 1],
            "k": 4,
            "efSearch": 200,
            "returnMetadata": True,
        },
    )
    assert response.status_code == 200, f"Search failed: {response.text}"

    results = response.json()
    assert len(results["hits"]) == 4, f"Expected 4 results, got {len(results['hits'])}"

    expected_order = [3, 2, 1, 0]
    assert (
        results["hits"] == expected_order
    ), f"Expected {expected_order}, got {results['hits']}"

    distances = results["distances"]
    assert all(
        earlier < later for earlier, later in zip(distances, distances[1:])
    ), f"Distances not in increasing order: {distances}"


def test_search_index_with_filter():
    document_vectors = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
    ]
    ids = list(range(len(document_vectors)))
    metadatas = [
        {"name": f"doc_{i}", "integer": i, "float": i * 100 / 3.234}
        for i in range(len(document_vectors))
    ]

    add_res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": "search_filters",
            "ids": ids,
            "vectors": document_vectors,
            "metadatas": metadatas,
        },
    )
    assert add_res.status_code == 201, f"Failed to add documents: {add_res.text}"

    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": "search_filters",
            "queryVector": [1, 1, 1, 1],
            "k": 4,
            "efSearch": 200,
            "filter": 'name = "doc_1" OR name = "doc_2"',
            "returnMetadata": True,
        },
    )
    assert response.status_code == 200, f"Search failed: {response.text}"

    results = response.json()
    assert len(results["hits"]) == 2, f"Expected 2 results, got {len(results['hits'])}"
    assert set(results["hits"]).issubset({1, 2}), f"Unexpected ids: {results['hits']}"


def test_search_without_return_metadata():
    name = "no_meta_search"
    create_index(name)
    requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": [0],
            "vectors": [[1, 1, 1, 1]],
            "metadatas": [{"tag": "hello"}],
        },
    )

    search_res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": name,
            "queryVector": [1, 1, 1, 1],
            "k": 1,
            "efSearch": 200,
        },
    )
    assert search_res.status_code == 200
    body = search_res.json()
    assert "metadatas" not in body, f"metadatas should be absent: {body}"

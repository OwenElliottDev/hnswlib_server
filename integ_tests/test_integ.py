import pytest
import requests
import numpy as np
import os

BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8685")

TEST_INDEX_NAMES = ["add_docs", "add_docs_metadata", "search", "search_filters"]


@pytest.fixture(scope="session", autouse=True)
def index_setup():
    for index_name in TEST_INDEX_NAMES:
        delete_data = {
            "indexName": index_name,
        }
        index_data = {
            "indexName": index_name,
            "dimension": 4,
            "indexType": "Approximate",
            "spaceType": "IP",
            "efConstruction": 200,
            "M": 16,
        }

        # Delete if exists
        requests.post(f"{BASE_URL}/delete_index", json=delete_data)

        # Create index for testing
        res = requests.post(f"{BASE_URL}/create_index", json=index_data)
        if res.status_code != 200:
            raise Exception(f"Failed to create index: {res.text}")
        else:
            print(f"Created index {index_name}")

    yield

    for index_name in TEST_INDEX_NAMES:
        delete_data = {"indexName": index_name}
        res = requests.post(f"{BASE_URL}/delete_index", json=delete_data)
        if res.status_code != 200:
            print(f"Failed to delete index {index_name} during cleanup: {res.text}")


def test_add_documents_no_metadata():
    vectors = [np.random.rand(4).tolist() for _ in range(5)]
    ids = list(range(5))

    add_docs_data = {
        "indexName": "add_docs",
        "ids": ids,
        "vectors": vectors,
        "metadatas": None,
    }

    response = requests.post(f"{BASE_URL}/add_documents", json=add_docs_data)
    assert response.status_code == 200, f"Failed to add documents: {response.text}"


def test_add_documents_with_metadata():
    vectors = [np.random.rand(4).tolist() for _ in range(5)]
    ids = list(range(5))

    metadatas = [
        {"name": f"doc_{i}", "integer": i, "float": i * 100 / 3.234} for i in range(5)
    ]

    add_docs_data = {
        "indexName": "add_docs_metadata",
        "ids": ids,
        "vectors": vectors,
        "metadatas": metadatas,
    }

    response = requests.post(f"{BASE_URL}/add_documents", json=add_docs_data)
    assert (
        response.status_code == 200
    ), f"Failed to add documents with metadata: {response.text}"


def test_search_index_no_filter():
    document_vectors = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
    ]
    ids = list(range(len(document_vectors)))

    add_documents_data = {
        "indexName": "search",
        "ids": ids,
        "vectors": document_vectors,
    }

    # Add documents first
    add_res = requests.post(f"{BASE_URL}/add_documents", json=add_documents_data)
    assert add_res.status_code == 200, f"Failed to add documents: {add_res.text}"

    query_vector = [1, 1, 1, 1]

    search_data = {
        "indexName": "search",
        "queryVector": query_vector,
        "k": 4,
        "efSearch": 200,
        "returnMetadata": True,
    }

    response = requests.post(f"{BASE_URL}/search", json=search_data)
    assert response.status_code == 200, f"Search failed: {response.text}"

    results = response.json()
    assert (
        len(results["hits"]) == 4
    ), f"Expected 4 results from search, got {len(results)}"

    expected_order = [3, 2, 1, 0]
    actual_order = results["hits"]
    assert (
        actual_order == expected_order
    ), f"Expected ranking {expected_order}, but got {actual_order}"
    distances = results["distances"]
    assert all(
        earlier < later for earlier, later in zip(distances, distances[1:])
    ), f"Distances are not in increasing order (i.e. dot products decreasing): {distances}"


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

    add_documents_data = {
        "indexName": "search_filters",
        "ids": ids,
        "vectors": document_vectors,
        "metadatas": metadatas,
    }

    add_res = requests.post(f"{BASE_URL}/add_documents", json=add_documents_data)
    assert add_res.status_code == 200, f"Failed to add documents: {add_res.text}"

    query_vector = [1, 1, 1, 1]

    search_data = {
        "indexName": "search_filters",
        "queryVector": query_vector,
        "k": 4,
        "efSearch": 200,
        "filter": 'name = "doc_1" OR name = "doc_2"',
        "returnMetadata": True,
    }

    response = requests.post(f"{BASE_URL}/search", json=search_data)
    assert response.status_code == 200, f"Search failed: {response.text}"

    results = response.json()
    assert (
        len(results["hits"]) == 2
    ), f"Expected 2 results from search with filter, got {len(results)}"

    # Ensure only doc_1 and doc_2 are returned
    returned_ids = results["hits"]
    assert set(returned_ids).issubset({1, 2}), f"Unexpected result ids: {returned_ids}"

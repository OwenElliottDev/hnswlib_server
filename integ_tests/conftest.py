import pytest
import requests
import os

BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8685")

TEST_INDEX_NAMES = ["add_docs", "add_docs_metadata", "search", "search_filters"]


def force_remove_index(name):
    requests.delete(f"{BASE_URL}/delete_index", json={"indexName": name})
    requests.delete(f"{BASE_URL}/delete_index_from_disk", json={"indexName": name})


def create_index(name, dimension=4, space_type="IP", vector_type="FLOAT32"):
    force_remove_index(name)
    res = requests.post(
        f"{BASE_URL}/create_index",
        json={
            "indexName": name,
            "dimension": dimension,
            "indexType": "Approximate",
            "spaceType": space_type,
            "vectorType": vector_type,
            "efConstruction": 200,
            "M": 16,
        },
    )
    assert res.status_code == 201, f"Failed to create index {name}: {res.text}"


def delete_index(name):
    return requests.delete(f"{BASE_URL}/delete_index", json={"indexName": name})


def delete_index_from_disk(name):
    return requests.delete(
        f"{BASE_URL}/delete_index_from_disk", json={"indexName": name}
    )


@pytest.fixture(scope="session", autouse=True)
def index_setup():
    for index_name in TEST_INDEX_NAMES:
        create_index(index_name)

    yield

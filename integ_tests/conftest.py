import time

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


def wait_for_replay_complete(name, timeout=30):
    """Poll index_status until WAL replay is done or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(f"{BASE_URL}/index_status/{name}")
        assert res.status_code == 200, f"index_status failed: {res.text}"
        status = res.json()
        if not status.get("replayingWal", False):
            assert (
                "walReplayError" not in status
            ), f"WAL replay error: {status['walReplayError']}"
            return status
        time.sleep(0.1)
    raise TimeoutError(f"WAL replay for '{name}' did not complete within {timeout}s")


@pytest.fixture(scope="session", autouse=True)
def index_setup():
    for index_name in TEST_INDEX_NAMES:
        create_index(index_name)

    yield

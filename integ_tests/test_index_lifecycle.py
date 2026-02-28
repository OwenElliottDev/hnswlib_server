import requests

from conftest import (
    BASE_URL,
    create_index,
    delete_index_from_disk,
    force_remove_index,
    wait_for_replay_complete,
)


def test_save_and_load_index():
    name = "save_load_test"
    create_index(name)

    vectors = [[1, 2, 3, 4], [5, 6, 7, 8]]
    metadatas = [{"tag": "first", "val": 10}, {"tag": "second", "val": 20}]
    add_res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": [0, 1],
            "vectors": vectors,
            "metadatas": metadatas,
        },
    )
    assert add_res.status_code == 201

    save_res = requests.post(f"{BASE_URL}/save_index", json={"indexName": name})
    assert save_res.status_code == 200

    requests.delete(f"{BASE_URL}/delete_index", json={"indexName": name})

    load_res = requests.post(f"{BASE_URL}/load_index", json={"indexName": name})
    assert load_res.status_code == 200, f"Load failed: {load_res.text}"

    wait_for_replay_complete(name)

    search_res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": name,
            "queryVector": [5, 6, 7, 8],
            "k": 2,
            "efSearch": 200,
            "returnMetadata": True,
        },
    )
    assert search_res.status_code == 200
    assert 1 in search_res.json()["hits"]

    doc_res = requests.get(f"{BASE_URL}/get_document/{name}/0")
    assert doc_res.status_code == 200
    doc = doc_res.json()
    assert doc["metadata"]["tag"] == "first"
    assert doc["metadata"]["val"] == 10


def test_delete_index_from_disk():
    name = "disk_delete_test"
    create_index(name)
    requests.post(
        f"{BASE_URL}/add_documents",
        json={"indexName": name, "ids": [0], "vectors": [[1, 2, 3, 4]]},
    )
    requests.post(f"{BASE_URL}/save_index", json={"indexName": name})
    requests.delete(f"{BASE_URL}/delete_index", json={"indexName": name})

    disk_del = delete_index_from_disk(name)
    assert disk_del.status_code == 200

    load_res = requests.post(f"{BASE_URL}/load_index", json={"indexName": name})
    assert (
        load_res.status_code == 404
    ), f"Expected 404 after disk delete, got {load_res.status_code}"


def test_async_wal_replay():
    """Verify that load_index returns immediately and status reports replay progress."""
    name = "async_replay_test"
    create_index(name)

    vectors = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    metadatas = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]
    add_res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": [0, 1, 2],
            "vectors": vectors,
            "metadatas": metadatas,
        },
    )
    assert add_res.status_code == 201

    del_res = requests.delete(
        f"{BASE_URL}/delete_documents",
        json={"indexName": name, "ids": [2]},
    )
    assert del_res.status_code == 200

    requests.delete(f"{BASE_URL}/delete_index", json={"indexName": name})

    load_res = requests.post(f"{BASE_URL}/load_index", json={"indexName": name})
    assert load_res.status_code == 200

    status_res = requests.get(f"{BASE_URL}/index_status/{name}")
    assert status_res.status_code == 200
    status = status_res.json()
    assert "replayingWal" in status

    if status["replayingWal"]:
        assert "walReplayProgress" in status
        progress = status["walReplayProgress"]
        assert "replayedAdds" in progress
        assert "totalAdds" in progress
        assert "percentComplete" in progress

    wait_for_replay_complete(name)

    search_res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": name,
            "queryVector": [5, 6, 7, 8],
            "k": 3,
            "efSearch": 200,
            "returnMetadata": True,
        },
    )
    assert search_res.status_code == 200
    hits = search_res.json()["hits"]
    assert 1 in hits
    assert 0 in hits
    assert 2 not in hits

    add_res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": [10],
            "vectors": [[1, 1, 1, 1]],
            "metadatas": [{"tag": "new"}],
        },
    )
    assert add_res.status_code == 201

    final_status = requests.get(f"{BASE_URL}/index_status/{name}").json()
    assert final_status["replayingWal"] is False
    assert "walReplayError" not in final_status

    force_remove_index(name)


def test_delete_index_from_disk_while_loaded():
    name = "disk_delete_loaded_test"
    create_index(name)
    res = delete_index_from_disk(name)
    assert res.status_code == 400

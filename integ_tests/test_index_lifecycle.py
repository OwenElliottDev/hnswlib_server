import requests

from conftest import BASE_URL, create_index, delete_index_from_disk, force_remove_index


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


def test_delete_index_from_disk_while_loaded():
    name = "disk_delete_loaded_test"
    create_index(name)
    res = delete_index_from_disk(name)
    assert res.status_code == 400

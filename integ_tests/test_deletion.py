import requests

from conftest import BASE_URL, create_index


def _status(name):
    res = requests.get(f"{BASE_URL}/index_status/{name}")
    assert res.status_code == 200, res.text
    return res.json()


def test_true_deletion_removes_and_reuses_slots():
    name = "deletion_test"
    create_index(name, dimension=4, space_type="L2")

    docs = {i: [float(i), 0.0, 0.0, 0.0] for i in range(10)}
    requests.post(
        f"{BASE_URL}/add_documents",
        json={"indexName": name, "ids": list(docs), "vectors": list(docs.values())},
    )
    assert _status(name)["currentElements"] == 10

    # delete three documents
    res = requests.delete(
        f"{BASE_URL}/delete_documents", json={"indexName": name, "ids": [3, 4, 5]}
    )
    assert res.status_code == 200

    # deleted ids must not appear in search results
    hits = requests.post(
        f"{BASE_URL}/search",
        json={"indexName": name, "queryVector": [4.0, 0, 0, 0], "k": 10},
    ).json()["hits"]
    assert all(h not in (3, 4, 5) for h in hits)

    # re-adding reuses the freed slots: currentElements stays at 10 (not 13)
    requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": [100, 101, 102],
            "vectors": [[3.0, 0, 0, 0], [4.0, 0, 0, 0], [5.0, 0, 0, 0]],
        },
    )
    status = _status(name)
    assert status["currentElements"] == 10
    assert status["deletedElements"] == 0

    hits = requests.post(
        f"{BASE_URL}/search",
        json={"indexName": name, "queryVector": [4.0, 0, 0, 0], "k": 3},
    ).json()["hits"]
    assert 101 in hits


def test_delete_missing_id_is_noop():
    name = "deletion_missing_test"
    create_index(name, dimension=4, space_type="L2")
    requests.post(
        f"{BASE_URL}/add_documents",
        json={"indexName": name, "ids": [0], "vectors": [[1.0, 1.0, 1.0, 1.0]]},
    )
    res = requests.delete(
        f"{BASE_URL}/delete_documents", json={"indexName": name, "ids": [999]}
    )
    assert res.status_code == 200
    assert _status(name)["currentElements"] == 1

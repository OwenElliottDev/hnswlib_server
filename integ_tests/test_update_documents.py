import requests
from conftest import BASE_URL, create_index, wait_for_replay_complete

TIMEOUT = 10


def add(name, ids, metadatas):
    res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": ids,
            "vectors": [[float(i), 1.0, 1.0, 1.0] for i in ids],
            "metadatas": metadatas,
        },
        timeout=TIMEOUT,
    )
    assert res.status_code == 201, res.text


def update(name, ids, metadatas):
    return requests.patch(
        f"{BASE_URL}/update_documents",
        json={"indexName": name, "ids": ids, "metadatas": metadatas},
        timeout=TIMEOUT,
    )


def metadata(name, doc_id):
    res = requests.get(f"{BASE_URL}/get_document/{name}/{doc_id}", timeout=TIMEOUT)
    assert res.status_code == 200, res.text
    return res.json()["metadata"]


def filter_hits(name, filter_str):
    res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": name,
            "queryVector": [1.0, 1.0, 1.0, 1.0],
            "k": 10,
            "filter": filter_str,
        },
        timeout=TIMEOUT,
    )
    assert res.status_code == 200, res.text
    return sorted(res.json()["hits"])


def reload_from_wal(name):
    requests.delete(
        f"{BASE_URL}/delete_index", json={"indexName": name}, timeout=TIMEOUT
    )
    res = requests.post(
        f"{BASE_URL}/load_index", json={"indexName": name}, timeout=TIMEOUT
    )
    assert res.status_code == 200, res.text
    wait_for_replay_complete(name)


def test_update_merges_into_existing_metadata():
    name = "update_basic"
    create_index(name)
    add(name, [1, 2], [{"color": "red", "size": 1}, {"color": "red", "size": 2}])

    res = update(name, [1], [{"color": "blue", "shape": "round"}])
    assert res.status_code == 200, res.text
    assert res.json() == {"errors": False, "results": [{"id": 1, "status": 200}]}

    assert metadata(name, 1) == {"color": "blue", "size": 1, "shape": "round"}
    assert filter_hits(name, 'color = "blue"') == [1]
    assert filter_hits(name, 'color = "red"') == [2]
    assert filter_hits(name, "size >= 1") == [1, 2]


def test_update_null_removes_field():
    name = "update_null"
    create_index(name)
    add(name, [1], [{"name": "shoe", "price": 10.0, "tags": ["a"]}])

    res = update(name, [1], [{"price": None, "name": "boot", "missing": None}])
    assert res.status_code == 200, res.text
    assert res.json()["errors"] is False

    assert metadata(name, 1) == {"name": "boot", "tags": ["a"]}
    assert filter_hits(name, "price > 0.0") == []
    assert filter_hits(name, 'name = "boot"') == [1]


def test_empty_update_keeps_metadata():
    name = "update_empty"
    create_index(name)
    add(name, [1], [{"v": 1}])

    assert update(name, [1], [{}]).json()["errors"] is False
    assert metadata(name, 1) == {"v": 1}


def test_update_reports_per_document_status():
    name = "update_partial"
    create_index(name)
    add(name, [1, 3], [{"v": 0}, {"v": 0}])

    res = update(name, [1, 7, 3], [{"v": 1}, {"v": 7}, {"v": 3}])
    assert res.status_code == 200, res.text
    assert res.json() == {
        "errors": True,
        "results": [
            {"id": 1, "status": 200},
            {"id": 7, "status": 404, "error": "Document not found"},
            {"id": 3, "status": 200},
        ],
    }
    assert metadata(name, 1) == {"v": 1}
    assert metadata(name, 3) == {"v": 3}
    res = requests.get(f"{BASE_URL}/get_document/{name}/7", timeout=TIMEOUT)
    assert res.status_code == 404


def test_missing_document_does_not_lock_its_id():
    name = "update_missing_unlocks"
    create_index(name)

    assert update(name, [5], [{"v": 1}]).json()["errors"] is True
    assert update(name, [5], [{"v": 1}]).json()["errors"] is True

    add(name, [5], [{"v": 0}])
    assert update(name, [5], [{"v": 2}]).json()["errors"] is False
    assert metadata(name, 5) == {"v": 2}


def test_update_request_errors():
    name = "update_errors"
    create_index(name)
    add(name, [1], [{"v": 0}])

    assert update(name, [1], []).status_code == 400
    assert update(name, [1, 2], [{"v": 1}]).status_code == 400
    assert update(name, [1], [{"v": {"nested": 1}}]).status_code == 400
    assert update(name, [1], [{"v": [None]}]).status_code == 400
    assert update(name, [1], ["not an object"]).status_code == 400
    assert update("no_such_index", [1], [{"v": 1}]).status_code == 404
    assert metadata(name, 1) == {"v": 0}


def test_update_of_wal_added_document_survives_replay():
    name = "update_replay_wal_add"
    create_index(name)
    add(name, [1], [{"v": 0, "keep": "yes", "drop": 1}])
    update(name, [1], [{"v": 1, "drop": None}])

    reload_from_wal(name)

    assert metadata(name, 1) == {"v": 1, "keep": "yes"}
    assert filter_hits(name, "v = 1") == [1]


def test_update_of_saved_document_survives_replay():
    name = "update_replay_snapshot"
    create_index(name)
    add(name, [1, 2], [{"v": 0, "keep": "yes", "drop": 1}, {"v": 0}])
    res = requests.post(
        f"{BASE_URL}/save_index", json={"indexName": name}, timeout=TIMEOUT
    )
    assert res.status_code == 200

    update(name, [1], [{"v": 1, "drop": None}])

    reload_from_wal(name)

    assert metadata(name, 1) == {"v": 1, "keep": "yes"}
    assert metadata(name, 2) == {"v": 0}
    assert filter_hits(name, "v = 1") == [1]


def test_update_then_delete_is_not_resurrected_by_replay():
    name = "update_replay_delete"
    create_index(name)
    add(name, [1, 2], [{"v": 0}, {"v": 0}])
    requests.post(f"{BASE_URL}/save_index", json={"indexName": name}, timeout=TIMEOUT)

    update(name, [1], [{"v": 1}])
    res = requests.delete(
        f"{BASE_URL}/delete_documents",
        json={"indexName": name, "ids": [1]},
        timeout=TIMEOUT,
    )
    assert res.status_code == 200

    reload_from_wal(name)

    res = requests.get(f"{BASE_URL}/get_document/{name}/1", timeout=TIMEOUT)
    assert res.status_code == 404
    assert filter_hits(name, "v >= 0") == [2]

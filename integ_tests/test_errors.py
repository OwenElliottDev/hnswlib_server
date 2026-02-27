import requests

from conftest import BASE_URL, create_index, delete_index


def test_search_nonexistent_index():
    res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": "nonexistent_idx_search",
            "queryVector": [1, 1, 1, 1],
            "k": 1,
            "efSearch": 200,
        },
    )
    assert res.status_code == 404


def test_get_document_nonexistent_index():
    res = requests.get(f"{BASE_URL}/get_document/nonexistent_idx_getdoc/0")
    assert res.status_code == 404


def test_get_document_nonexistent_document():
    name = "getdoc_missing_doc"
    create_index(name)
    res = requests.get(f"{BASE_URL}/get_document/{name}/99999")
    assert res.status_code == 404


def test_create_duplicate_index():
    name = "dup_create_test"
    create_index(name)
    res = requests.post(
        f"{BASE_URL}/create_index",
        json={
            "indexName": name,
            "dimension": 4,
            "indexType": "Approximate",
            "spaceType": "IP",
            "efConstruction": 200,
            "M": 16,
        },
    )
    assert res.status_code == 400


def test_delete_nonexistent_index():
    res = delete_index("nonexistent_idx_del")
    assert res.status_code == 404


def test_add_documents_mismatched_ids_vectors():
    name = "mismatch_test"
    create_index(name)
    res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": [0, 1],
            "vectors": [[1, 2, 3, 4]],
        },
    )
    assert res.status_code == 400


def test_delete_documents_nonexistent_index():
    res = requests.delete(
        f"{BASE_URL}/delete_documents",
        json={"indexName": "nonexistent_idx_deldocs", "ids": [0]},
    )
    assert res.status_code == 404


def test_add_documents_nonexistent_index():
    res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": "nonexistent_idx_adddocs",
            "ids": [0],
            "vectors": [[1, 2, 3, 4]],
        },
    )
    assert res.status_code == 404

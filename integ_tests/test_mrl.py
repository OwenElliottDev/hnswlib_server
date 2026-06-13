import requests

from conftest import BASE_URL, force_remove_index

# doc 1 matches the query on all 8 dims; doc 2 matches only on the first 4
# (the MRL scan prefix) but diverges on the tail; doc 3 is far on every dim.
DOCS = {
    1: [1, 1, 1, 1, 1, 1, 1, 1],
    2: [1, 1, 1, 1, 9, 9, 9, 9],
    3: [5, 5, 5, 5, 5, 5, 5, 5],
}
QUERY = [1.0, 1, 1, 1, 1, 1, 1, 1]


def _create_mrl(name, dimension=8, mrl_scan_dim=4, space_type="L2"):
    force_remove_index(name)
    res = requests.post(
        f"{BASE_URL}/create_index",
        json={
            "indexName": name,
            "dimension": dimension,
            "spaceType": space_type,
            "mrlScanDim": mrl_scan_dim,
            "efConstruction": 200,
            "M": 16,
        },
    )
    assert res.status_code == 201, res.text
    return name


def _add_docs(name):
    requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": list(DOCS),
            "vectors": [list(map(float, v)) for v in DOCS.values()],
        },
    )


def test_mrl_rerank_uses_full_dimension():
    name = _create_mrl("mrl_test")
    _add_docs(name)

    # with reranking, the full-dimension distance puts the exact match (doc 1) first
    res = requests.post(
        f"{BASE_URL}/search",
        json={"indexName": name, "queryVector": QUERY, "k": 3, "rerankSize": 3},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["hits"][0] == 1
    assert abs(body["distances"][0]) < 1e-3  # full-dim L2 to the exact match is ~0


def test_mrl_without_rerank_returns_all():
    name = _create_mrl("mrl_test_norerank")
    _add_docs(name)

    res = requests.post(
        f"{BASE_URL}/search",
        json={"indexName": name, "queryVector": QUERY, "k": 3},
    )
    assert res.status_code == 200
    assert set(res.json()["hits"]) == {1, 2, 3}


def test_mrl_scan_dim_must_be_smaller_than_dimension():
    name = "mrl_bad_dim"
    force_remove_index(name)
    res = requests.post(
        f"{BASE_URL}/create_index",
        json={"indexName": name, "dimension": 8, "mrlScanDim": 8, "spaceType": "L2"},
    )
    assert res.status_code == 400


def test_mrl_survives_save_load():
    name = _create_mrl("mrl_persist")
    _add_docs(name)
    assert requests.post(f"{BASE_URL}/save_index", json={"indexName": name}).ok
    requests.delete(f"{BASE_URL}/delete_index", json={"indexName": name})

    assert requests.post(f"{BASE_URL}/load_index", json={"indexName": name}).ok
    res = requests.post(
        f"{BASE_URL}/search",
        json={"indexName": name, "queryVector": QUERY, "k": 3, "rerankSize": 3},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["hits"][0] == 1
    assert abs(body["distances"][0]) < 1e-3
    force_remove_index(name)

import requests
import numpy as np

from conftest import BASE_URL, create_index


def test_vector_type_float16():
    name = "float16_test"
    create_index(name, vector_type="FLOAT16")

    vectors = [[1, 2, 3, 4], [5, 6, 7, 8]]
    requests.post(
        f"{BASE_URL}/add_documents",
        json={"indexName": name, "ids": [0, 1], "vectors": vectors},
    )

    search_res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": name,
            "queryVector": [5, 6, 7, 8],
            "k": 2,
            "efSearch": 200,
        },
    )
    assert search_res.status_code == 200
    assert search_res.json()["hits"][0] == 1

    doc_res = requests.get(f"{BASE_URL}/get_document/{name}/0")
    assert doc_res.status_code == 200
    assert np.allclose(doc_res.json()["vector"], [1, 2, 3, 4], atol=0.01)


def test_vector_type_bfloat16():
    name = "bfloat16_test"
    create_index(name, vector_type="BFLOAT16")

    vectors = [[1, 2, 3, 4], [5, 6, 7, 8]]
    requests.post(
        f"{BASE_URL}/add_documents",
        json={"indexName": name, "ids": [0, 1], "vectors": vectors},
    )

    search_res = requests.post(
        f"{BASE_URL}/search",
        json={
            "indexName": name,
            "queryVector": [5, 6, 7, 8],
            "k": 2,
            "efSearch": 200,
        },
    )
    assert search_res.status_code == 200
    assert search_res.json()["hits"][0] == 1

    doc_res = requests.get(f"{BASE_URL}/get_document/{name}/0")
    assert doc_res.status_code == 200
    assert np.allclose(doc_res.json()["vector"], [1, 2, 3, 4], atol=0.01)

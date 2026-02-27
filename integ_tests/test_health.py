import requests

from conftest import BASE_URL, create_index, force_remove_index


def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.text == "OK"


def test_list_indices():
    idx1, idx2 = "list_test_a", "list_test_b"
    create_index(idx1)
    create_index(idx2)
    response = requests.get(f"{BASE_URL}/list_indices")
    assert response.status_code == 200
    names = response.json()
    assert idx1 in names, f"{idx1} not in {names}"
    assert idx2 in names, f"{idx2} not in {names}"

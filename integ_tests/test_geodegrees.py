import requests

from conftest import BASE_URL, create_index, force_remove_index

# (latitude, longitude) in degrees
CITIES = {
    1: [51.5074, -0.1278],  # London
    2: [48.8566, 2.3522],  # Paris
    3: [40.7128, -74.0060],  # New York
    4: [-33.8688, 151.2093],  # Sydney
}


def test_geodegrees_nearest_and_km_distance():
    name = "geo_test"
    create_index(name, dimension=2, space_type="GEODEGREES")

    requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": name,
            "ids": list(CITIES),
            "vectors": list(CITIES.values()),
        },
    )

    # query just north of London
    res = requests.post(
        f"{BASE_URL}/search",
        json={"indexName": name, "queryVector": [51.6, -0.1], "k": 2},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["hits"][0] == 1  # London
    assert body["hits"][1] == 2  # Paris
    # great-circle London -> Paris is ~344 km; distances are returned in km
    assert 300.0 < body["distances"][1] < 380.0


def test_geodegrees_requires_dim_2():
    name = "geo_bad_dim"
    force_remove_index(name)
    res = requests.post(
        f"{BASE_URL}/create_index",
        json={"indexName": name, "dimension": 3, "spaceType": "GEODEGREES"},
    )
    assert res.status_code == 400

import pytest
import requests
from conftest import BASE_URL, create_index, delete_index

INDEX_NAME = "pagination"
N_DOCS = 50

VECTORS = [[float(i), 0.0, 0.0, 0.0] for i in range(N_DOCS)]
IDS = list(range(N_DOCS))
METADATAS = [{"position": i, "parity": "even" if i % 2 == 0 else "odd"} for i in IDS]


@pytest.fixture(scope="module", autouse=True)
def pagination_index():
    create_index(INDEX_NAME, space_type="L2")
    res = requests.post(
        f"{BASE_URL}/add_documents",
        json={
            "indexName": INDEX_NAME,
            "ids": IDS,
            "vectors": VECTORS,
            "metadatas": METADATAS,
        },
    )
    assert res.status_code == 201, f"Failed to add documents: {res.text}"
    yield
    delete_index(INDEX_NAME)


def search(**kwargs):
    body = {
        "indexName": INDEX_NAME,
        "queryVector": [0.0, 0.0, 0.0, 0.0],
        "efSearch": 200,
        **kwargs,
    }
    return requests.post(f"{BASE_URL}/search", json=body)


def similar(**kwargs):
    body = {"indexName": INDEX_NAME, "efSearch": 200, **kwargs}
    return requests.post(f"{BASE_URL}/similar", json=body)


def ok_json(res):
    assert res.status_code == 200, f"Request failed: {res.status_code} {res.text}"
    return res.json()


def test_search_offset_zero_matches_no_offset():
    without = ok_json(search(k=10))
    with_zero = ok_json(search(k=10, offset=0))
    assert with_zero == without
    assert without["hits"] == list(range(10))


def test_search_pages_concatenate_to_full_result():
    full = ok_json(search(k=30))
    hits, distances = [], []
    for offset in range(0, 30, 10):
        page = ok_json(search(k=10, offset=offset))
        assert len(page["hits"]) == 10
        hits += page["hits"]
        distances += page["distances"]
    assert hits == full["hits"] == list(range(30))
    assert distances == pytest.approx(full["distances"])


def test_search_page_is_offset_slice():
    page = ok_json(search(k=5, offset=12))
    assert page["hits"] == list(range(12, 17))
    assert page["distances"] == pytest.approx([float(i * i) for i in range(12, 17)])


def test_search_partial_last_page():
    page = ok_json(search(k=10, offset=45))
    assert page["hits"] == list(range(45, 50))
    assert len(page["distances"]) == 5


def test_search_offset_past_end_returns_empty():
    page = ok_json(search(k=10, offset=N_DOCS))
    assert page["hits"] == []
    assert page["distances"] == []

    page = ok_json(search(k=10, offset=N_DOCS + 100))
    assert page["hits"] == []


def test_search_offset_with_filter():
    page = ok_json(search(k=5, offset=5, filter='parity = "odd"'))
    assert page["hits"] == [11, 13, 15, 17, 19]


def test_search_offset_past_filtered_results():
    page = ok_json(search(k=10, offset=20, filter='parity = "odd"'))
    assert page["hits"] == [41, 43, 45, 47, 49]


def test_search_offset_metadata_aligned_with_hits():
    page = ok_json(search(k=4, offset=8, returnMetadata=True))
    assert page["hits"] == [8, 9, 10, 11]
    assert [m["position"] for m in page["metadatas"]] == page["hits"]


def test_search_offset_larger_than_ef_search():
    page = ok_json(search(k=5, offset=40, efSearch=10))
    assert page["hits"] == list(range(40, 45))


def test_search_negative_offset_rejected():
    res = search(k=5, offset=-1)
    assert res.status_code == 400, res.text


def test_similar_excludes_input_document_by_default():
    page = ok_json(similar(docId=0, k=5))
    assert page["hits"] == [1, 2, 3, 4, 5]


def test_similar_can_include_input_document():
    page = ok_json(similar(docId=0, k=5, excludeInputDocument=False))
    assert page["hits"] == [0, 1, 2, 3, 4]
    assert page["distances"][0] == pytest.approx(0.0)


def test_similar_pages_concatenate_to_full_result():
    full = ok_json(similar(docId=0, k=30))
    hits = []
    for offset in range(0, 30, 10):
        page = ok_json(similar(docId=0, k=10, offset=offset))
        assert len(page["hits"]) == 10
        hits += page["hits"]
    assert hits == full["hits"] == list(range(1, 31))
    assert 0 not in hits


def test_similar_partial_last_page_excludes_input_document():
    page = ok_json(similar(docId=0, k=10, offset=45))
    assert page["hits"] == [46, 47, 48, 49]


def test_similar_offset_past_end_returns_empty():
    page = ok_json(similar(docId=0, k=10, offset=N_DOCS))
    assert page["hits"] == []


def test_similar_offset_with_filter_and_metadata():
    page = ok_json(
        similar(docId=20, k=4, offset=2, filter='parity = "even"', returnMetadata=True)
    )
    assert len(page["hits"]) == 4
    assert all(h % 2 == 0 and h != 20 for h in page["hits"])
    assert set(page["hits"]) == {16, 24, 14, 26}
    assert [m["position"] for m in page["metadatas"]] == page["hits"]


def test_similar_unknown_document_returns_404():
    res = similar(docId=N_DOCS + 1000, k=5)
    assert res.status_code == 404, res.text


def test_similar_negative_offset_rejected():
    res = similar(docId=0, k=5, offset=-3)
    assert res.status_code == 400, res.text

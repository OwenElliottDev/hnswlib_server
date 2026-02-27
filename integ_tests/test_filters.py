import requests

from conftest import BASE_URL, create_index

FILTER_INDEX = "filter_ops_test"
FILTER_DOCS = [
    {
        "id": 0,
        "vector": [6, 0, 0, 0],
        "meta": {"name": "alice", "age": 25, "score": 88.5, "city": "new_york"},
    },
    {
        "id": 1,
        "vector": [5, 0, 0, 0],
        "meta": {"name": "bob", "age": 30, "score": 92.0, "city": "boston"},
    },
    {
        "id": 2,
        "vector": [4, 0, 0, 0],
        "meta": {"name": "carol", "age": 25, "score": 75.5, "city": "boston"},
    },
    {
        "id": 3,
        "vector": [3, 0, 0, 0],
        "meta": {"name": "dave", "age": 35, "score": 88.5, "city": "chicago"},
    },
    {
        "id": 4,
        "vector": [2, 0, 0, 0],
        "meta": {"name": "eve", "age": 40, "score": 95.0, "city": "new_york"},
    },
    {
        "id": 5,
        "vector": [1, 0, 0, 0],
        "meta": {"name": "frank", "age": 22, "score": 60.0, "city": "chicago"},
    },
]


class TestFilterOperators:
    @classmethod
    def setup_class(cls):
        create_index(FILTER_INDEX)
        res = requests.post(
            f"{BASE_URL}/add_documents",
            json={
                "indexName": FILTER_INDEX,
                "ids": [d["id"] for d in FILTER_DOCS],
                "vectors": [d["vector"] for d in FILTER_DOCS],
                "metadatas": [d["meta"] for d in FILTER_DOCS],
            },
        )
        assert res.status_code == 201

    def _search(self, filter_str, k=6):
        res = requests.post(
            f"{BASE_URL}/search",
            json={
                "indexName": FILTER_INDEX,
                "queryVector": [1, 0, 0, 0],
                "k": k,
                "efSearch": 200,
                "filter": filter_str,
                "returnMetadata": True,
            },
        )
        assert res.status_code == 200, f"Search failed: {res.text}"
        return res.json()

    def test_filter_not_equal_string(self):
        results = self._search('name != "alice"')
        assert len(results["hits"]) == 5
        assert 0 not in results["hits"]

    def test_filter_greater_than_integer(self):
        results = self._search("age > 30")
        assert set(results["hits"]) == {3, 4}

    def test_filter_less_than_integer(self):
        results = self._search("age < 25")
        assert set(results["hits"]) == {5}

    def test_filter_greater_equal_float(self):
        results = self._search("score >= 88.5")
        assert set(results["hits"]) == {0, 1, 3, 4}

    def test_filter_less_equal_float(self):
        results = self._search("score <= 75.5")
        assert set(results["hits"]) == {2, 5}

    def test_filter_and_operator(self):
        results = self._search('age > 24 AND city = "boston"')
        assert set(results["hits"]) == {1, 2}

    def test_filter_not_operator(self):
        results = self._search('NOT city = "chicago"')
        assert set(results["hits"]) == {0, 1, 2, 4}

    def test_filter_parenthesized_grouping(self):
        results = self._search('(city = "new_york" OR city = "boston") AND age > 26')
        assert set(results["hits"]) == {1, 4}

    def test_filter_mixed_type_comparisons(self):
        results = self._search('age >= 30 AND name = "bob"')
        assert set(results["hits"]) == {1}

    def test_filter_equal_string(self):
        results = self._search('city = "boston"')
        assert set(results["hits"]) == {1, 2}

    def test_filter_equal_integer(self):
        results = self._search("age = 25")
        assert set(results["hits"]) == {0, 2}

    def test_filter_equal_float(self):
        results = self._search("score = 88.5")
        assert set(results["hits"]) == {0, 3}

    def test_filter_not_equal_integer(self):
        results = self._search("age != 25")
        assert set(results["hits"]) == {1, 3, 4, 5}

    def test_filter_not_equal_float(self):
        results = self._search("score != 88.5")
        assert set(results["hits"]) == {1, 2, 4, 5}

    def test_filter_greater_than_float(self):
        results = self._search("score > 88.5")
        assert set(results["hits"]) == {1, 4}

    def test_filter_less_than_float(self):
        results = self._search("score < 75.5")
        assert set(results["hits"]) == {5}

    def test_filter_greater_equal_integer(self):
        results = self._search("age >= 35")
        assert set(results["hits"]) == {3, 4}

    def test_filter_less_equal_integer(self):
        results = self._search("age <= 25")
        assert set(results["hits"]) == {0, 2, 5}

    def test_filter_or_operator(self):
        results = self._search('city = "boston" OR city = "chicago"')
        assert set(results["hits"]) == {1, 2, 3, 5}

    def test_filter_matches_all_docs(self):
        results = self._search("age >= 0")
        assert set(results["hits"]) == {0, 1, 2, 3, 4, 5}

    def test_filter_matches_no_docs(self):
        results = self._search("age > 100")
        assert len(results["hits"]) == 0

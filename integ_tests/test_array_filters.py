import requests

from conftest import BASE_URL, create_index

ARRAY_INDEX = "array_filter_test"
ARRAY_DOCS = [
    {
        "id": 0,
        "vector": [6, 0, 0, 0],
        "meta": {
            "name": "alice",
            "age": 25,
            "tags": ["python", "cpp", "rust"],
            "scores": [90, 85, 78],
        },
    },
    {
        "id": 1,
        "vector": [5, 0, 0, 0],
        "meta": {
            "name": "bob",
            "age": 30,
            "tags": ["java", "python"],
            "scores": [70, 95],
        },
    },
    {
        "id": 2,
        "vector": [4, 0, 0, 0],
        "meta": {
            "name": "carol",
            "age": 25,
            "tags": ["rust", "go"],
            "scores": [88, 92],
        },
    },
    {
        "id": 3,
        "vector": [3, 0, 0, 0],
        "meta": {
            "name": "dave",
            "age": 35,
            "tags": ["javascript", "typescript"],
            "scores": [60, 65],
        },
    },
    {
        "id": 4,
        "vector": [2, 0, 0, 0],
        "meta": {
            "name": "eve",
            "age": 40,
            "tags": ["python", "java", "go"],
            "scores": [99, 100, 97],
        },
    },
]


class TestArrayFilters:
    @classmethod
    def setup_class(cls):
        create_index(ARRAY_INDEX)
        res = requests.post(
            f"{BASE_URL}/add_documents",
            json={
                "indexName": ARRAY_INDEX,
                "ids": [d["id"] for d in ARRAY_DOCS],
                "vectors": [d["vector"] for d in ARRAY_DOCS],
                "metadatas": [d["meta"] for d in ARRAY_DOCS],
            },
        )
        assert res.status_code == 201, f"Failed to add docs: {res.text}"

    def _search(self, filter_str, k=5):
        res = requests.post(
            f"{BASE_URL}/search",
            json={
                "indexName": ARRAY_INDEX,
                "queryVector": [1, 0, 0, 0],
                "k": k,
                "efSearch": 200,
                "filter": filter_str,
                "returnMetadata": True,
            },
        )
        assert res.status_code == 200, f"Search failed: {res.text}"
        return res.json()

    def test_in_string(self):
        results = self._search('name IN ["alice", "bob"]')
        assert set(results["hits"]) == {0, 1}

    def test_in_string_single(self):
        results = self._search('name IN ["carol"]')
        assert set(results["hits"]) == {2}

    def test_in_integer(self):
        results = self._search("age IN [25, 35]")
        assert set(results["hits"]) == {0, 2, 3}

    def test_in_no_match(self):
        results = self._search('name IN ["zach", "yvonne"]')
        assert len(results["hits"]) == 0

    def test_in_with_and(self):
        results = self._search('name IN ["alice", "bob", "carol"] AND age > 25')
        assert set(results["hits"]) == {1}

    def test_contains_substring(self):
        results = self._search('name CONTAINS "li"')
        assert set(results["hits"]) == {0}

    def test_contains_substring_multiple_matches(self):
        """'a' appears in alice, carol, dave"""
        results = self._search('name CONTAINS "a"')
        assert set(results["hits"]) == {0, 2, 3}

    def test_contains_substring_no_match(self):
        results = self._search('name CONTAINS "zzz"')
        assert len(results["hits"]) == 0

    def test_contains_array_element(self):
        """alice(0), bob(1), eve(4) have 'python' in tags"""
        results = self._search('tags CONTAINS "python"')
        assert set(results["hits"]) == {0, 1, 4}

    def test_contains_array_element_single_match(self):
        results = self._search('tags CONTAINS "typescript"')
        assert set(results["hits"]) == {3}

    def test_contains_array_element_no_match(self):
        results = self._search('tags CONTAINS "haskell"')
        assert len(results["hits"]) == 0

    def test_contains_array_with_and(self):
        """python users older than 30: bob(30) and eve(40)"""
        results = self._search('tags CONTAINS "python" AND age >= 30')
        assert set(results["hits"]) == {1, 4}

    def test_contains_array_with_not(self):
        """everyone except python users"""
        results = self._search('NOT tags CONTAINS "python"')
        assert set(results["hits"]) == {2, 3}

    def test_in_and_contains_combined(self):
        """name in list AND has tag"""
        results = self._search(
            'name IN ["alice", "bob", "carol"] AND tags CONTAINS "rust"'
        )
        assert set(results["hits"]) == {0, 2}

    def test_contains_array_with_or(self):
        results = self._search('tags CONTAINS "go" OR tags CONTAINS "typescript"')
        assert set(results["hits"]) == {2, 3, 4}


class TestArrayDocumentStorage:
    """Test that array metadata is stored and retrievable via get_document."""

    @classmethod
    def setup_class(cls):
        cls.index_name = "array_storage_test"
        create_index(cls.index_name)
        res = requests.post(
            f"{BASE_URL}/add_documents",
            json={
                "indexName": cls.index_name,
                "ids": [0],
                "vectors": [[1, 0, 0, 0]],
                "metadatas": [
                    {
                        "name": "test",
                        "tags": ["a", "b", "c"],
                        "nums": [10, 20, 30],
                    }
                ],
            },
        )
        assert res.status_code == 201, f"Failed to add docs: {res.text}"

    def test_get_document_with_arrays(self):
        res = requests.get(f"{BASE_URL}/get_document/{self.index_name}/0")
        assert res.status_code == 200
        doc = res.json()
        assert doc["metadata"]["name"] == "test"
        assert doc["metadata"]["tags"] == ["a", "b", "c"]
        assert doc["metadata"]["nums"] == [10, 20, 30]

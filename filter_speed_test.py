import requests
import numpy as np
import json
import timeit

BASE_URL = "http://localhost:8685"
DIMENSION = 2 # low because we care about filters here rather than HNSW traversal
TOTAL_DOCS = 1_000_000


def vector_to_json(vec):
    return [float(v) for v in vec]


def create_index():
    index_data = {
        "indexName": "test_index",
        "dimension": DIMENSION,
        "indexType": "Approximate",
        "spaceType": "L2",
        "efConstruction": 200,
        "M": 16,
    }
    requests.post(f"{BASE_URL}/create_index", json=index_data)


def add_documents(num_docs=1_000_000):
    vectors = [np.random.rand(DIMENSION).tolist() for _ in range(num_docs)]
    ids = list(range(num_docs))
    metadatas = [
        {"name": f"doc_{i}", "integer": i, "float_data": i * 3.14}
        for i in range(num_docs)
    ]
    add_docs_data = {
        "indexName": "test_index",
        "ids": ids,
        "vectors": vectors,
        "metadatas": metadatas,
    }
    print(f"Adding {len(vectors)} documents to the index...")
    requests.post(f"{BASE_URL}/add_documents", json=add_docs_data)


def search_index_with_filter(filter_string):
    np.random.seed(42)
    query_vector = np.random.rand(DIMENSION).tolist()
    search_data = {
        "indexName": "test_index",
        "queryVector": vector_to_json(query_vector),
        "k": 3,
        "efSearch": 200,
        "filter": filter_string,
        "returnMetadata": True,
    }
    response = requests.post(f"{BASE_URL}/search", json=search_data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Search failed: {response.text}"


def run_speed_tests():
    filters = [
        "",
        'name = "doc_50000"',
        "integer > 50000",
        "float_data < 50000.32",
        "float_data < 1000.32",
    ]
    for f in filters:
        n_runs = 100
        time_taken = timeit.timeit(lambda: search_index_with_filter(f), number=n_runs)
        avg_ms = (time_taken / n_runs) * 1000
        print(
            f"Filter '{f}': {avg_ms:.2f}ms per search on average | QPS: {1 / (time_taken / n_runs):.2f}"
        )


def delete_index():
    requests.post(f"{BASE_URL}/delete_index", json={"indexName": "test_index"})


print("Single threaded filter performance test...")
create_index()
add_documents(num_docs=TOTAL_DOCS)
run_speed_tests()
delete_index()

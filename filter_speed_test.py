import requests
import numpy as np
import json
import timeit

# Define the base URL for the server
BASE_URL = "http://localhost:8685"
DIMENSION = 2


# Helper to format vectors in JSON
def vector_to_json(vec):
    return [float(v) for v in vec]


# 1. Create an Index
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


# 2. Add Documents
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


# 3. Search with Filters and Benchmark
def search_index_with_filter(filter_string):
    np.random.seed(42)
    query_vector = np.random.rand(DIMENSION).tolist()
    search_data = {
        "indexName": "test_index",
        "queryVector": vector_to_json(query_vector),
        "k": 3,
        "efSearch": 200,
        "filter": filter_string,
    }
    response = requests.post(f"{BASE_URL}/search", json=search_data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Search failed: {response.text}"


# 4. Timing the filtering process with different conditions
def run_speed_tests():
    filters = [
        "",  # No filter
        'name = "doc_50000"',  # Exact match filter
        "integer > 50000",  # Range filter
        "float_data < 50000.32",  # Less than filter
        "float_data < 1000.32",  # Less than filter with less
    ]
    for f in filters:
        n_runs = 100
        time_taken = timeit.timeit(lambda: search_index_with_filter(f), number=n_runs)
        print(
            f"Filter '{f}': {time_taken / n_runs:.4f} seconds per search on average | QPS: {1 / (time_taken / n_runs):.2f}"
        )


# Main Execution
create_index()
add_documents()  # Adding 100k documents to the index
run_speed_tests()  # Run and time searches with different filters

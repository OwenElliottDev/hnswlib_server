import requests
import numpy as np
import json
import timeit

# Define the base URL for the server
BASE_URL = "http://localhost:8685"

# Helper to format vectors in JSON
def vector_to_json(vec):
    return [float(v) for v in vec]

# 1. Create an Index
def create_index():
    index_data = {
        "index_name": "test_index",
        "dimension": 4,
        "index_type": "Approximate",
        "space_type": "L2",
        "ef_construction": 200,
        "M": 16,
    }
    requests.post(f"{BASE_URL}/create_index", json=index_data)

# 2. Add Documents
def add_documents(num_docs=10000):
    vectors = [np.random.rand(4).tolist() for _ in range(num_docs)]
    ids = list(range(num_docs))
    metadatas = [{"name": f"doc_{i}", "integer": i, "float": i * 100 / 3.234} for i in range(num_docs)]
    add_docs_data = {"index_name": "test_index", "ids": ids, "vectors": vectors, "metadatas": metadatas}
    requests.post(f"{BASE_URL}/add_documents", json=add_docs_data)

# 3. Search with Filters and Benchmark
def search_index_with_filter(filter_string):
    np.random.seed(42)
    query_vector = np.random.rand(4).tolist()
    search_data = {
        "index_name": "test_index",
        "query_vector": vector_to_json(query_vector),
        "k": 3,
        "ef_search": 200,
        "filter": filter_string
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
        "float < 15000",  # Less than filter
    ]
    for f in filters:
        time_taken = timeit.timeit(lambda: search_index_with_filter(f), number=100)
        print(f"Filter '{f}': {time_taken / 100:.4f} seconds per search on average | QPS: {1 / (time_taken / 100):.2f}")

# Main Execution
create_index()
add_documents()  # Adding 100k documents to the index
run_speed_tests()  # Run and time searches with different filters

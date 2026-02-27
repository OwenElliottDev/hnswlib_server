import requests
import numpy as np
import json

# Define the base URL for the server
BASE_URL = "http://localhost:8685"


# Helper to format vectors in JSON
def vector_to_json(vec):
    return [float(v) for v in vec]


# 1. Create an Index
def create_index():
    index_data = {
        "indexName": "test_index",
        "dimension": 4,
        "indexType": "Approximate",
        "spaceType": "IP",
        "efConstruction": 200,
        "M": 16,
    }

    response = requests.post(f"{BASE_URL}/create_index", json=index_data)

    if response.status_code == 200:
        print("Index created successfully.")
    else:
        print(f"Failed to create index: {response.text}")


# 2. Add Documents
def add_documents():
    # Generate some sample vectors
    vectors = [np.random.rand(4).tolist() for _ in range(5)]
    ids = list(range(5))

    metadatas = [
        {"name": f"doc_{i}", "integer": i, "float": i * 100 / 3.234} for i in range(5)
    ]

    add_docs_data = {
        "indexName": "test_index",
        "ids": ids,
        "vectors": vectors,
        "metadatas": metadatas,
    }

    response = requests.post(f"{BASE_URL}/add_documents", json=add_docs_data)

    if response.status_code == 200:
        print("Documents added successfully.")
    else:
        print(f"Failed to add documents: {response.text}")


# 3. Search in the Index
def search_index(filter_string=None):
    # Generate a random query vector
    np.random.seed(42)
    query_vector = np.random.rand(4).tolist()

    search_data = {
        "indexName": "test_index",
        "queryVector": vector_to_json(query_vector),
        "k": 3,  # Find the 3 nearest neighbors
        "efSearch": 200,
        "filter": "",
        "returnMetadata": True,
    }

    if filter_string:
        search_data["filter"] = filter_string
    print(search_data)
    response = requests.post(f"{BASE_URL}/search", json=search_data)

    if response.status_code == 200:
        results = response.json()
        print("Search results:", results)
    else:
        print(f"Search failed: {response.text}")


# 4. Save the Index
def save_index():
    save_data = {
        "indexName": "test_index",
    }

    response = requests.post(f"{BASE_URL}/save_index", json=save_data)

    if response.status_code == 200:
        print("Index saved successfully.")
    else:
        print(f"Failed to save index: {response.text}")


# 5. Delete the Index
def delete_index():
    delete_data = {
        "indexName": "test_index",
    }

    response = requests.delete(f"{BASE_URL}/delete_index", json=delete_data)

    if response.status_code == 200:
        print("Index deleted successfully.")
    else:
        print(f"Failed to delete index: {response.text}")


# 6. Load the Index
def load_index():
    load_data = {
        "indexName": "test_index",
    }

    response = requests.post(f"{BASE_URL}/load_index", json=load_data)

    if response.status_code == 200:
        print("Index loaded successfully.")
    else:
        print(f"Failed to load index: {response.text}")


# 7. Delete the index from disk
def delete_index_from_disk():
    delete_data = {
        "indexName": "test_index",
    }

    response = requests.delete(f"{BASE_URL}/delete_index_from_disk", json=delete_data)

    if response.status_code == 200:
        print("Index deleted from disk successfully.")
    else:
        print(f"Failed to delete index from disk: {response.text}")


# Test the Server

create_index()
add_documents()
search_index()
search_index('name = "doc_2"')
search_index("integer > 2")
save_index()
delete_index()
load_index()
search_index()
search_index('name = "doc_2"')
delete_index()
delete_index_from_disk()

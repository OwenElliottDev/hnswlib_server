import pycurl
import json
import time
import random
from tqdm import tqdm
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# Global parameters
INDEX_NAME = "benchmark"
DIMENSION = 512  # Dimension of the vectors
NUM_DOC_BATCHES = 100  # Number of document batches to add
DOC_BATCH_SIZE = 100  # Number of documents to add in each batch
NUM_QUERIES = 10000  # Number of queries to run
VECTOR_RANGE = (0.0, 1.0)  # Range for random vector values
K = 100  # Number of nearest neighbors to retrieve
EF_CONSTRUCTION = 512  # HNSW index construction parameter
EF_SEARCH = 512  # HNSW search parameter
ADD_DOCS_CLIENTS = 10  # Number of parallel clients for adding documents
SEARCH_CLIENTS = 20  # Number of parallel clients for querying

# Helper function to send POST requests
def send_post_request(url, data):
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.POST, 1)
    c.setopt(c.POSTFIELDS, json.dumps(data))
    c.setopt(c.HTTPHEADER, ["Content-Type: application/json"])
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()

    body = buffer.getvalue().decode("utf-8")
    return body

# URL setup
server_url = "http://localhost:8685"
create_index_url = f"{server_url}/create_index"
add_documents_url = f"{server_url}/add_documents"
search_url = f"{server_url}/search"

# Function to generate random vectors
def generate_random_vector(dim, range_min, range_max):
    return [random.uniform(range_min, range_max) for _ in range(dim)]

# Pre-generate all documents
def generate_all_documents():
    all_docs = []
    for _ in range(NUM_DOC_BATCHES):
        batch_vectors = [
            generate_random_vector(DIMENSION, VECTOR_RANGE[0], VECTOR_RANGE[1])
            for _ in range(DOC_BATCH_SIZE)
        ]
        all_docs.append(batch_vectors)
    return all_docs

# Pre-generate all queries
def generate_all_queries():
    queries = [
        generate_random_vector(DIMENSION, VECTOR_RANGE[0], VECTOR_RANGE[1])
        for _ in range(NUM_QUERIES)
    ]
    return queries

# Function to create index
def create_index():
    create_index_data = {
        "index_name": INDEX_NAME,
        "dimension": DIMENSION,
        "index_type": "Approximate",
        "space_type": "IP",
        "M": 16,
        "ef_construction": EF_CONSTRUCTION,
    }
    send_post_request(create_index_url, create_index_data)

ADDED_DOCS = 0

# Function to add documents
def add_documents(batch_vectors):
    global ADDED_DOCS
    add_documents_data = {
        "index_name": INDEX_NAME,
        "ids": list(range(ADDED_DOCS, ADDED_DOCS + DOC_BATCH_SIZE)),
        "vectors": batch_vectors,
        "metadatas": [],
    }
    ADDED_DOCS += DOC_BATCH_SIZE
    send_post_request(add_documents_url, add_documents_data)

# Function to search the index
def search_index(query_vector):
    search_data = {
        "index_name": INDEX_NAME,
        "query_vector": query_vector,
        "k": K,
        "ef_search": EF_SEARCH,
        "filter": "",
    }
    send_post_request(search_url, search_data)

# Parallel function for adding documents
def parallel_add_documents(all_docs):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=ADD_DOCS_CLIENTS) as executor:
        list(tqdm(executor.map(add_documents, all_docs), total=len(all_docs)))

    total_time = time.time() - start_time
    avg_time_per_doc = total_time / (NUM_DOC_BATCHES * DOC_BATCH_SIZE)
    print(f"Average time per document: {avg_time_per_doc:.6f} seconds")
    docs_per_sec = (NUM_DOC_BATCHES * DOC_BATCH_SIZE) / total_time
    print(f"Documents per second: {docs_per_sec:.2f}")
    return avg_time_per_doc

# Parallel function for querying the index
def parallel_search_queries(queries):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=SEARCH_CLIENTS) as executor:
        list(tqdm(executor.map(search_index, queries), total=len(queries)))

    total_time = time.time() - start_time
    avg_time_per_query = total_time / NUM_QUERIES
    qps = NUM_QUERIES / total_time
    print(f"Average time per query: {avg_time_per_query:.6f} seconds")
    print(f"Queries per second (QPS): {qps:.2f}")
    return avg_time_per_query, qps

# Pre-generate all the data
print("Generating documents and queries...")
all_docs = generate_all_documents()
all_queries = generate_all_queries()

# Running the speed tests
print("Creating index...")
create_index()

print("\nAdding documents with parallel clients...")
add_time_per_doc = parallel_add_documents(all_docs)

print("\nRunning queries with parallel clients...")
avg_query_time, qps = parallel_search_queries(all_queries)

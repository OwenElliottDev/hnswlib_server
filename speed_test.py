import argparse
import pycurl
import json
import time
import numpy as np
from tqdm import tqdm
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

parser = argparse.ArgumentParser()
parser.add_argument("--vector-type", choices=["FLOAT32", "FLOAT16", "BFLOAT16"], default="FLOAT32")
args = parser.parse_args()
VECTOR_TYPE = args.vector_type

# Global parameters
INDEX_NAME = "benchmark"
DIMENSION = 512
NUM_DOC_BATCHES = 1000
DOC_BATCH_SIZE = 100
NUM_QUERIES = 10000
VECTOR_RANGE = (-1.0, 1.0)
K = 100
M = 16
EF_CONSTRUCTION = 512
EF_SEARCH = 512
ADD_DOCS_CLIENTS = 20
SEARCH_CLIENTS = 100

server_url = "http://localhost:8685"
create_index_url = f"{server_url}/create_index"
delete_index_url = f"{server_url}/delete_index"
add_documents_url = f"{server_url}/add_documents"
search_url = f"{server_url}/search"


class CurlPool:
    def __init__(self, max_size):
        self.pool = Queue(maxsize=max_size)
        for _ in range(max_size):
            curl = pycurl.Curl()
            curl.setopt(pycurl.CONNECTTIMEOUT, 5)
            curl.setopt(pycurl.TIMEOUT, 10)
            curl.setopt(
                pycurl.HTTPHEADER,
                ["Content-Type: application/json", "Connection: keep-alive"],
            )
            self.pool.put(curl)

    def acquire(self):
        return self.pool.get()

    def release(self, curl):
        self.pool.put(curl)

    def close_all(self):
        while not self.pool.empty():
            curl = self.pool.get()
            curl.close()


curl_pool = CurlPool(max_size=SEARCH_CLIENTS)


def send_post_request(url, data):
    buffer = BytesIO()
    curl = curl_pool.acquire()
    try:
        curl.setopt(curl.URL, url)
        curl.setopt(curl.POSTFIELDS, json.dumps(data))
        curl.setopt(curl.WRITEDATA, buffer)
        curl.perform()
        response = buffer.getvalue().decode("utf-8")
    finally:
        buffer.truncate(0)
        buffer.seek(0)
        curl_pool.release(curl)
    return response


def generate_random_vector(dim, range_min, range_max):
    vec = np.random.uniform(range_min, range_max, dim)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def generate_all_documents():
    all_docs = []
    for _ in range(NUM_DOC_BATCHES):
        batch_vectors = [
            generate_random_vector(DIMENSION, VECTOR_RANGE[0], VECTOR_RANGE[1])
            for _ in range(DOC_BATCH_SIZE)
        ]
        all_docs.append(batch_vectors)
    return all_docs


def generate_all_queries():
    return [
        generate_random_vector(DIMENSION, VECTOR_RANGE[0], VECTOR_RANGE[1])
        for _ in range(NUM_QUERIES)
    ]


def create_index():
    create_index_data = {
        "indexName": INDEX_NAME,
        "dimension": DIMENSION,
        "M": M,
        "efConstruction": EF_CONSTRUCTION,
        "vectorType": VECTOR_TYPE,
    }
    send_post_request(create_index_url, create_index_data)


def delete_index():
    delete_index_data = {"indexName": INDEX_NAME}
    send_post_request(delete_index_url, delete_index_data)


import threading
_doc_id_lock = threading.Lock()
ADDED_DOCS = 0


def add_documents(batch_vectors):
    global ADDED_DOCS
    with _doc_id_lock:
        start_id = ADDED_DOCS
        ADDED_DOCS += DOC_BATCH_SIZE
    add_documents_data = {
        "indexName": INDEX_NAME,
        "ids": list(range(start_id, start_id + DOC_BATCH_SIZE)),
        "vectors": batch_vectors,
    }
    start = time.perf_counter()
    send_post_request(add_documents_url, add_documents_data)
    return time.perf_counter() - start


def search_index(query_vector):
    search_data = {
        "indexName": INDEX_NAME,
        "queryVector": query_vector,
        "k": K,
        "efSearch": EF_SEARCH,
    }
    start = time.perf_counter()
    send_post_request(search_url, search_data)
    return time.perf_counter() - start


def parallel_add_documents(all_docs):
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=ADD_DOCS_CLIENTS) as executor:
        latencies = list(tqdm(executor.map(add_documents, all_docs), total=len(all_docs)))
    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 / DOC_BATCH_SIZE
    print(f"Average latency per document: {avg_latency_ms:.4f}ms")
    print(
        f"Average documents per second: {NUM_DOC_BATCHES * DOC_BATCH_SIZE / elapsed:.2f}"
    )


def parallel_search_queries(queries):
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=SEARCH_CLIENTS) as executor:
        latencies = list(tqdm(executor.map(search_index, queries), total=len(queries)))
    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    print(f"Average latency per query: {avg_latency_ms:.4f}ms")
    print(f"Average queries per second: {NUM_QUERIES / elapsed:.2f}")


print(f"Vector type: {VECTOR_TYPE}")
print("Generating documents and queries...")
all_docs = generate_all_documents()
all_queries = generate_all_queries()

print("Creating index...")
create_index()

print("\nAdding documents with parallel clients...")
parallel_add_documents(all_docs)

print("\nRunning queries with parallel clients...")
parallel_search_queries(all_queries)

delete_index()
curl_pool.close_all()

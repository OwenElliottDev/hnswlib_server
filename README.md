# HNSWLib Server

This is a lightweight HTTP server that wraps the HNSWLib Library. It allows you to build a HNSW index and query it using a simple REST API. The server is written in C++.

A bunch of optimisations are applied for compiling and static linking is also used to make it self-contained. The binary is about 2.6MB and we can package it into a Docker container with a busybox base image to make a portable version that is 7.3MB.

## Features

### Filtering

This server supports arbitrary metadata in documents on top of HNSWLib, these can be used for filtering as well. e.g.
    
```json
{
    "indexName": "test_index",
    "ids": [0, 1, 2, 3, 4],
    "vectors": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.2]],
    "metadatas": [{"name": "doc_0"}, {"some_number": 2}, {"some_float": 0.4234}, {"name": "doc_3", "category": "cool"}, {"name": "doc_4", "category": "cool"}]
}
```

You can then filter by metadata like so:

```json
{
    "indexName": "test_index",
    "query_vector": [0.1, 0.2, 0.3, 0.4],
    "k": 5,
    "ef_search": 200,
    "filter": "(category = \"cool\" AND name = \"doc_3\") OR some_number > 1"
}
```

Filtering supports grouping with parentheses, the following operators are supported:

Comparison operators: `=`, `!=`, `>`, `<`, `>=`, `<=`.

Logical operators: `AND`, `OR`, `NOT`.

### 

## Purpose

This is more of a personal project to get better as C++ and write some more complicated data structures. The goal was to hit a good balance of performance, feature completeness, and simplicity. With support for arbitrary metadata on documents and the ability to filter these datatypes, I think it's a good start. It's also pretty fast:

```python
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
SEARCH_CLIENTS = 20
```

For a total of 100k docs on an M1 Macbook Pro:
```
Average time per document: 0.000548 seconds
Average documents per second: 1825.57
Average time per query: 0.000865 seconds
Average queries per second: 1156.02
```

## Building

To build the server you need to have the submodules initialized. You can do this by running:

```bash
git submodule update --init --recursive
```

Then you can build the server by running:

```bash
mkdir build
cd build
cmake ..
make
```

## Running

Run the server by executing the binary from the `build` directory:

```bash
./bin/server
```

## Docker

### Building

```bash
docker build -t hnswlib_server .
```

### Running

```bash
docker run -p 8685:8685 hnswlib_server
```

# API Docs

## `POST /index`

Creates a new index with the given parameters. Valid `space_type` values are `L2`, and `IP`. If you want cosine similarity, use `IP` and unit normalize your vectors.

### Request

```json
{
    "indexName": "test_index",
    "dimension": 4,
    "indexType": "Approximate",
    "spaceType": "IP",
    "efConstruction": 200,
    "M": 16
}
```

### Response

- `200 OK`: Index created successfully.

## `POST /add_documents`

Adds documents to the index.

### Request

```json
{
    "indexName": "test_index",
    "ids": [0, 1, 2, 3, 4],
    "vectors": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.2]],
    "metadatas": [{"name": "doc_0"}, {"name": "doc_1"}, {"name": "doc_2"}, {"name": "doc_3"}, {"name": "doc_4"}]
}
```

### Response

- `200 OK`: Documents added successfully.

## `POST /search`

Searches for the nearest neighbors of a query vector in the index.

### Request

```json
{
    "indexName": "test_index",
    "queryVector": [0.1, 0.2, 0.3, 0.4],
    "k": 5,
    "efSearch": 200,
    "filter": ""
}
```

### Response

- `200 OK`: Returns a JSON array of the nearest neighbors.

## `POST /save_index`

Saves the index to disk.

### Request

```json
{
    "indexName": "test_index"
}
```

### Response

- `200 OK`: Index saved successfully.

## `POST /delete_index`

Deletes the index from memory.

### Request

```json
{
    "indexName": "test_index"
}
```

### Response

- `200 OK`: Index deleted successfully.

## `POST /load_index`

Loads the index from disk.

### Request

```json
{
    "indexName": "test_index"
}
```

### Response

- `200 OK`: Index loaded successfully.

## `POST /delete_index_from_disk`

Deletes the index from disk.

### Request

```json
{
    "indexName": "test_index"
}
```

### Response

- `200 OK`: Index deleted from disk successfully."[]


# Testing

You can run the tests by running:

```bash
rm -rf build && cmake -B build -S . && cmake --build build -j 8
./build/test_filters
./build/test_data_store
```

## Integration Tests

Integration tests are located in the `integ_tests` directory. You can run them using `pytest`. Dependencies are managed with `uv` for integration tests.

```bash
uv sync --dev
```

With HNSWLiv server running, you can execute the integration tests:

```bash
uv run pytest
```
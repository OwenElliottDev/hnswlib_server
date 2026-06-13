# HNSWLib Server

This is a lightweight HTTP server that wraps the HNSWLib Library. It allows you to build a HNSW index and query it using a simple REST API. The server is written in C++.

A bunch of optimisations are applied for compiling and static linking is also used to make it self-contained. The binary is about 3MB and we can package it into a Docker container with a scratch base image to make a portable version that is 4.88MB.

## Quick Start

```bash
docker run -p 8685:8685 -v ./indices:/indices owenelliottdev/hnswlib_server:latest
```

Or using [docker-compose.yml](docker-compose.yml):

```bash
docker compose up
```

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
    "queryVector": [0.1, 0.2, 0.3, 0.4],
    "k": 5,
    "efSearch": 200,
    "filter": "(category = \"cool\" AND name = \"doc_3\") OR some_number > 1"
}
```

Metadata values can also be arrays:

```json
{
    "metadatas": [{"name": "alice", "tags": ["python", "cpp"]}, {"name": "bob", "tags": ["java"]}]
}
```

Filtering supports grouping with parentheses, the following operators are supported:

Comparison operators: `=`, `!=`, `>`, `<`, `>=`, `<=`.

Logical operators: `AND`, `OR`, `NOT`.

`IN` - checks if a field value matches any value in an array: `name IN ["alice", "bob"]`.

`CONTAINS` - substring match on string fields: `name CONTAINS "lic"`. Element membership on array fields: `tags CONTAINS "python"`.

### Vector Types

The server supports multiple vector storage types to trade off precision for memory:

- `FLOAT32` (default) - Full precision 32-bit floats
- `FLOAT16` - Half precision 16-bit floats
- `BFLOAT16` - Brain float 16-bit

Set the `vectorType` field when creating an index. Vectors are always sent/received as FLOAT32 in the API and converted internally.

### Geographic Distance

The `GEODEGREES` space computes the great-circle (haversine) distance in **kilometers** between two `(latitude, longitude)` points given in degrees. It requires `dimension: 2` and `FLOAT32` vectors.

```json
{
    "indexName": "places",
    "dimension": 2,
    "spaceType": "GEODEGREES"
}
```

Vectors are `[latitude, longitude]` and search distances are returned in kilometers.

### Matryoshka (MRL) Embeddings

For [Matryoshka](https://arxiv.org/abs/2205.13147) embeddings the graph can be built and scanned using only the first `mrlScanDim` dimensions while full-dimensional vectors are still stored. This makes index construction and traversal cheaper, and queries can optionally rerank the best candidates at full dimensionality for accuracy.

Set `mrlScanDim` (smaller than `dimension`) when creating the index:

```json
{
    "indexName": "mrl_index",
    "dimension": 768,
    "spaceType": "IP",
    "mrlScanDim": 192
}
```

At query time, pass `rerankSize` to rerank the best `rerankSize` scan-dimension candidates using the full-dimension distance and return the top `k`:

```json
{
    "indexName": "mrl_index",
    "queryVector": [ ... 768 values ... ],
    "k": 10,
    "rerankSize": 100
}
```

If `rerankSize` is `0` (or omitted), results are ranked using only the `mrlScanDim` dimensions. MRL is supported for the `L2` and `IP` spaces (and their `FLOAT16`/`BFLOAT16` variants).

### Deletion

`delete_documents` removes elements from the graph and repairs the surrounding neighborhood (delete-and-reconnect), rather than only marking them as deleted. This keeps query throughput from degrading as deletions accumulate. Freed slots are reused by subsequent `add_documents` calls, so the index does not grow unbounded under delete/insert churn.

### Write-Ahead Logging (WAL)

All add and delete operations are logged to a write-ahead log for durability. When an index is loaded from disk, the WAL is replayed to recover any operations that occurred after the last save. The WAL automatically compacts when it exceeds 64MB.

The fsync interval can be configured via the `WAL_FSYNC_INTERVAL_MS` environment variable (default: 1000ms).

## Examples

Runnable end-to-end demos live in [`examples/`](examples/):

- [`word-search/`](examples/word-search/) — semantic word search over GloVe vectors stored as **bfloat16**, with a browser UI and vector arithmetic (`king - man + woman ≈ queen`).
- [`airports-geo/`](examples/airports-geo/) — nearest-airport search over ~80k airports using the **`geodegrees`** space, with metadata filtering and an interactive Leaflet map.
- [`wal-recovery/`](examples/wal-recovery/) — crash recovery that serves **live search traffic while the WAL replays** on a large index.

## Purpose

This is more of a personal project to get better as C++. The goal was to hit a good balance of performance, feature completeness, and simplicity. With support for arbitrary metadata on documents and the ability to filter these datatypes, I think it's a good start. It's also pretty fast:

```python
INDEX_NAME = "benchmark"
DIMENSION = 512
NUM_DOC_BATCHES = 10000
DOC_BATCH_SIZE = 100
NUM_QUERIES = 10000
VECTOR_RANGE = (-1.0, 1.0)
K = 100
M = 16
EF_CONSTRUCTION = 512
EF_SEARCH = 512
ADD_DOCS_CLIENTS = 20
SEARCH_CLIENTS = 100
```

For a total of 1 million float32 vectors on a 9950X:
```
Average latency per document: 12.7905ms
Documents per second: 1561.50
Average latency per query: 61.8965ms
Queries per second: 1466.23
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

## Formatting

`src/` is formatted with clang-format (`.clang-format`: LLVM, 140 cols); CI checks it. Globs are unquoted so the shell expands them.

Format in place:

```bash
uvx clang-format@20.1.7 -i src/*.cpp src/*.hpp
```

## Running

Run the server by executing the binary from the `build` directory:

```bash
./bin/server
```

## Docker Build

### Building

```bash
docker build -t hnswlib_server .
```

### Running

```bash
docker run -p 8685:8685 hnswlib_server
```

# API Docs

## `GET /health`

Health check endpoint.

### Response

- `200 OK`: Returns `"OK"`.

## `GET /list_indices`

Lists all currently loaded indices.

### Response

- `200 OK`: Returns a JSON array of index names.

## `POST /create_index`

Creates a new index with the given parameters. Valid `spaceType` values are `L2`, `IP`, and `GEODEGREES`. If you want cosine similarity, use `IP` and unit normalize your vectors. Valid `vectorType` values are `FLOAT32` (default), `FLOAT16`, and `BFLOAT16`.

`mrlScanDim` (optional, default `0`) enables [Matryoshka (MRL)](#matryoshka-mrl-embeddings) embeddings: the graph is built and scanned using only the first `mrlScanDim` dimensions while full vectors are stored. It must be smaller than `dimension`.

### Request

```json
{
    "indexName": "test_index",
    "dimension": 4,
    "indexType": "Approximate",
    "spaceType": "IP",
    "vectorType": "FLOAT32",
    "efConstruction": 200,
    "M": 16,
    "mrlScanDim": 0
}
```

### Response

- `201 Created`: Index created successfully.
- `400 Bad Request`: Invalid configuration (e.g. `GEODEGREES` with `dimension != 2`, or `mrlScanDim >= dimension`).

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

- `201 Created`: Documents added successfully.

## `POST /search`

Searches for the nearest neighbors of a query vector in the index. Set `returnMetadata` to `true` to include document metadata in the response. For [MRL](#matryoshka-mrl-embeddings) indexes, set `rerankSize` to rerank the best `rerankSize` scan-dimension candidates at full dimensionality (ignored for non-MRL indexes).

### Request

```json
{
    "indexName": "test_index",
    "queryVector": [0.1, 0.2, 0.3, 0.4],
    "k": 5,
    "efSearch": 200,
    "filter": "",
    "returnMetadata": false,
    "rerankSize": 0
}
```

### Response

- `200 OK`: Returns JSON with `hits` (array of IDs), `distances` (array of distances), and optionally `metadatas` (array of metadata objects if `returnMetadata` is `true`).

## `DELETE /delete_documents`

Deletes documents from an index by their IDs. Elements are removed from the graph and the surrounding neighborhood is repaired (delete-and-reconnect); freed slots are reused by later `add_documents` calls. Deleting an ID that is not present is a no-op.

### Request

```json
{
    "indexName": "test_index",
    "ids": [0, 1, 2]
}
```

### Response

- `200 OK`: Documents deleted successfully.

## `GET /get_document/<indexName>/<id>`

Retrieves a specific document by index name and ID.

### Response

- `200 OK`: Returns JSON with `id`, `vector`, and `metadata` fields.
- `404 Not Found`: Index or document not found.

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

## `DELETE /delete_index`

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

## `DELETE /delete_index_from_disk`

Deletes the index from disk.

### Request

```json
{
    "indexName": "test_index"
}
```

### Response

- `200 OK`: Index deleted from disk successfully.


# Testing

You can run the tests by running:

```bash
rm -rf build && cmake -B build -S . && cmake --build build -j 8
./build/test_filters
./build/test_data_store
./build/test_save_load
./build/test_wal
```

## Integration Tests

Integration tests are located in the `integ_tests` directory. You can run them using `pytest`. Dependencies are managed with `uv` for integration tests.

```bash
uv sync --dev
```

With HNSWLib server running, you can execute the integration tests:

```bash
uv run pytest
```
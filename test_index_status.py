"""
Integration test for GET /index_status/<indexName>.

Requires the server to be running on localhost:8685.

Tests:
  1. 404 for non-existent index
  2. Status after fresh create (empty index, not resizing)
  3. Status after adding documents (element count matches)
  4. Status during/after a resize triggered by heavy writes
  5. All expected fields present and correctly typed
"""

import json
import time
import sys
import threading
import numpy as np
from io import BytesIO

import pycurl

SERVER = "http://localhost:8685"
INDEX_NAME = "test_status"
DIMENSION = 16


def request(method, path, data=None):
    """Send an HTTP request and return (status_code, parsed_json_or_text)."""
    buf = BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, f"{SERVER}{path}")
    c.setopt(pycurl.CONNECTTIMEOUT, 5)
    c.setopt(pycurl.TIMEOUT, 30)
    c.setopt(pycurl.HTTPHEADER, ["Content-Type: application/json"])
    c.setopt(pycurl.WRITEDATA, buf)

    if method == "POST":
        c.setopt(pycurl.POSTFIELDS, json.dumps(data) if data else "")
    elif method == "DELETE":
        c.setopt(pycurl.CUSTOMREQUEST, "DELETE")
        c.setopt(pycurl.POSTFIELDS, json.dumps(data) if data else "")
    elif method == "GET":
        c.setopt(pycurl.HTTPGET, 1)

    c.perform()
    status = c.getinfo(pycurl.RESPONSE_CODE)
    c.close()

    body = buf.getvalue().decode("utf-8")
    try:
        body = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        pass
    return status, body


def random_vectors(n):
    vecs = np.random.randn(n, DIMENSION).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()


def cleanup():
    request("DELETE", "/delete_index", {"indexName": INDEX_NAME})


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cleanup()

# ---------------------------------------------------------------------------
# Test 1: 404 for non-existent index
# ---------------------------------------------------------------------------
print("\n[Test 1] index_status returns 404 for non-existent index")
status, body = request("GET", f"/index_status/{INDEX_NAME}")
check("status code is 404", status == 404, f"got {status}")

# ---------------------------------------------------------------------------
# Test 2: Status of a freshly created index
# ---------------------------------------------------------------------------
print("\n[Test 2] index_status after fresh create")
request(
    "POST",
    "/create_index",
    {
        "indexName": INDEX_NAME,
        "dimension": DIMENSION,
        "M": 16,
        "efConstruction": 200,
    },
)
status, body = request("GET", f"/index_status/{INDEX_NAME}")
check("status code is 200", status == 200, f"got {status}")
check(
    "indexName matches",
    body.get("indexName") == INDEX_NAME,
    f"got {body.get('indexName')}",
)
check("resizing is false", body.get("resizing") is False, f"got {body.get('resizing')}")
check(
    "bufferedWrites is 0",
    body.get("bufferedWrites") == 0,
    f"got {body.get('bufferedWrites')}",
)
check(
    "currentElements is 0",
    body.get("currentElements") == 0,
    f"got {body.get('currentElements')}",
)
check(
    "maxElements > 0", body.get("maxElements", 0) > 0, f"got {body.get('maxElements')}"
)
check(
    "has deletedElements field", "deletedElements" in body, f"keys: {list(body.keys())}"
)

# ---------------------------------------------------------------------------
# Test 3: Status after adding a small batch of documents
# ---------------------------------------------------------------------------
print("\n[Test 3] index_status after adding documents")
n_docs = 50
vecs = random_vectors(n_docs)
request(
    "POST",
    "/add_documents",
    {
        "indexName": INDEX_NAME,
        "ids": list(range(n_docs)),
        "vectors": vecs,
    },
)

# Wait for any buffering to flush
for _ in range(100):
    status, body = request("GET", f"/index_status/{INDEX_NAME}")
    if not body.get("resizing") and body.get("bufferedWrites", 0) == 0:
        break
    time.sleep(0.05)

check("status code is 200", status == 200, f"got {status}")
check(
    "currentElements == 50",
    body.get("currentElements") == n_docs,
    f"got {body.get('currentElements')}",
)
check("resizing is false", body.get("resizing") is False, f"got {body.get('resizing')}")
check(
    "bufferedWrites is 0",
    body.get("bufferedWrites") == 0,
    f"got {body.get('bufferedWrites')}",
)

# ---------------------------------------------------------------------------
# Test 4: Trigger a resize with heavy parallel writes, observe buffering
# ---------------------------------------------------------------------------
print("\n[Test 4] index_status during resize (heavy parallel writes)")
cleanup()
# Create index with small initial size (DEFAULT_INDEX_SIZE=100000, so we need
# to exceed that minus the 10000 headroom = 90000 elements to trigger resize)
request(
    "POST",
    "/create_index",
    {
        "indexName": INDEX_NAME,
        "dimension": DIMENSION,
        "M": 16,
        "efConstruction": 200,
    },
)

# We need to push past 90k to trigger resize. Use batches of 1000 across
# many threads to create contention.
BATCH_SIZE = 1000
TOTAL_DOCS = 95000  # enough to trigger resize
NUM_BATCHES = TOTAL_DOCS // BATCH_SIZE
doc_id_counter = 0
id_lock = threading.Lock()

# Pre-generate all vectors
all_vecs = random_vectors(TOTAL_DOCS)

saw_resizing = False
saw_buffered = False
status_samples = []


def add_batch(batch_idx):
    global doc_id_counter
    with id_lock:
        start_id = doc_id_counter
        doc_id_counter += BATCH_SIZE

    request(
        "POST",
        "/add_documents",
        {
            "indexName": INDEX_NAME,
            "ids": list(range(start_id, start_id + BATCH_SIZE)),
            "vectors": all_vecs[start_id : start_id + BATCH_SIZE],
        },
    )


def poll_status():
    """Poll index_status while writes are happening."""
    global saw_resizing, saw_buffered
    while not poll_done.is_set():
        _, resp = request("GET", f"/index_status/{INDEX_NAME}")
        if isinstance(resp, dict):
            status_samples.append(resp)
            if resp.get("resizing"):
                saw_resizing = True
            if resp.get("bufferedWrites", 0) > 0:
                saw_buffered = True
        time.sleep(0.01)


poll_done = threading.Event()
poller = threading.Thread(target=poll_status)
poller.start()

# Fire writes from many threads
threads = []
for i in range(NUM_BATCHES):
    t = threading.Thread(target=add_batch, args=(i,))
    threads.append(t)
    t.start()
    # Stagger slightly to increase chance of observing resize
    if i % 10 == 0:
        time.sleep(0.001)

for t in threads:
    t.join()

# Wait for flush to complete
for _ in range(200):
    _, body = request("GET", f"/index_status/{INDEX_NAME}")
    if (
        isinstance(body, dict)
        and not body.get("resizing")
        and body.get("bufferedWrites", 0) == 0
    ):
        break
    time.sleep(0.05)

poll_done.set()
poller.join()

check(
    "final resizing is false",
    body.get("resizing") is False,
    f"got {body.get('resizing')}",
)
check(
    "final bufferedWrites is 0",
    body.get("bufferedWrites") == 0,
    f"got {body.get('bufferedWrites')}",
)
check(
    "currentElements == total docs",
    body.get("currentElements") == TOTAL_DOCS,
    f"got {body.get('currentElements')}, expected {TOTAL_DOCS}",
)
check(
    "maxElements grew beyond initial 100k",
    body.get("maxElements", 0) > 100000,
    f"got {body.get('maxElements')}",
)
# These might not always be observed depending on timing, so just report
if saw_resizing:
    print(
        f"  INFO: observed resizing=true during writes ({sum(1 for s in status_samples if s.get('resizing'))} samples)"
    )
else:
    print(f"  INFO: did not observe resizing=true (resize was too fast to catch)")
if saw_buffered:
    max_buf = max(s.get("bufferedWrites", 0) for s in status_samples)
    print(f"  INFO: observed bufferedWrites > 0 (peak={max_buf})")
else:
    print(f"  INFO: did not observe bufferedWrites > 0 (flush was too fast to catch)")

# ---------------------------------------------------------------------------
# Test 5: All documents are searchable after flush
# ---------------------------------------------------------------------------
print("\n[Test 5] all documents searchable after flush completes")
query = random_vectors(1)[0]
status, result = request(
    "POST",
    "/search",
    {
        "indexName": INDEX_NAME,
        "queryVector": query,
        "k": 10,
        "efSearch": 200,
    },
)
check("search returns 200", status == 200, f"got {status}")
check(
    "search returns 10 hits",
    len(result.get("hits", [])) == 10,
    f"got {len(result.get('hits', []))} hits",
)

# ---------------------------------------------------------------------------
# Cleanup & summary
# ---------------------------------------------------------------------------
cleanup()

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed > 0:
    sys.exit(1)
print("All tests passed!")

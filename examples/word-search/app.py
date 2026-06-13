# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "flask",
#     "numpy",
#     "requests",
# ]
# ///
"""Web UI for semantic word search backed by the hnswlib server.

Type a word to find its nearest neighbours, or do vector arithmetic like
"king - man + woman" (a classic GloVe analogy that lands near "queen").

The browser talks only to this Flask app (same origin); Flask resolves the
query words to a vector and proxies the search to the hnswlib server.
"""

import argparse
import os
import re
import time

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory

from glove_common import INDEX_NAME, load_vectors

SERVER = os.getenv("HNSW_SERVER", "http://localhost:8685")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__)
WORDS = []
WORD_TO_VEC = {}
DIM = 0

# split an expression like "king - man + woman" into [("+","king"),("-","man"),...]
TERM_RE = re.compile(r"([+-]?)\s*([a-zA-Z][a-zA-Z'\-]*)")


def build_query_vector(expr):
    terms = TERM_RE.findall(expr.lower())
    if not terms:
        return None, [], []
    vec = None
    used, missing = [], []
    for sign, word in terms:
        if word not in WORD_TO_VEC:
            missing.append(word)
            continue
        used.append(word)
        v = WORD_TO_VEC[word] * (-1.0 if sign == "-" else 1.0)
        vec = v.copy() if vec is None else vec + v
    if vec is None:
        return None, used, missing
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec, used, missing


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify({"vocab": len(WORDS), "dim": DIM})


@app.route("/api/search", methods=["POST"])
def search():
    body = request.get_json(force=True)
    expr = (body.get("query") or "").strip()
    k = int(body.get("k", 15))
    vec, used, missing = build_query_vector(expr)
    if vec is None:
        msg = (
            f"none of those words are in the vocabulary: {missing}"
            if missing
            else "empty query"
        )
        return jsonify({"error": msg, "missing": missing}), 400

    # ask for extra results so we can drop the input words themselves
    t0 = time.perf_counter()
    res = requests.post(
        f"{SERVER}/search",
        json={
            "indexName": INDEX_NAME,
            "queryVector": vec.tolist(),
            "k": k + len(used),
            "efSearch": 256,
            "returnMetadata": True,
        },
    )
    server_ms = round((time.perf_counter() - t0) * 1000, 2)
    if not res.ok:
        return jsonify({"error": f"server: {res.status_code} {res.text}"}), 502
    data = res.json()
    used_set = set(used)
    hits = []
    for dist, m in zip(data["distances"], data.get("metadatas", [])):
        word = m.get("word")
        if word in used_set:
            continue
        # IP space stores distance = 1 - cosine for normalized vectors
        hits.append({"word": word, "similarity": round(1.0 - dist, 4)})
        if len(hits) >= k:
            break
    return jsonify(
        {
            "query": expr,
            "used": used,
            "missing": missing,
            "server_ms": server_ms,
            "results": hits,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=100, choices=[50, 100, 200, 300])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="must match the --limit used in load.py so the vocabulary lines up (default: all)",
    )
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    print(
        f"Loading vocabulary ({'all' if args.limit is None else 'top ' + str(args.limit)} GloVe {args.dim}d words) ..."
    )
    global WORDS, WORD_TO_VEC, DIM
    WORDS, vectors = load_vectors(args.dim, args.limit)
    WORD_TO_VEC = {w: vectors[i] for i, w in enumerate(WORDS)}
    DIM = int(vectors.shape[1])
    print(f"  {len(WORDS)} words ready")
    print(f"Open http://localhost:{args.port}  (hnswlib server: {SERVER})")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()

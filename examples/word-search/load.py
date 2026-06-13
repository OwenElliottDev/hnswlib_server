# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "requests",
# ]
# ///
"""Ingest GloVe word vectors into the hnswlib server as a BFLOAT16 index.

Vectors are L2-normalized and the index uses the IP (inner product) space, so
nearest neighbours by inner product are the highest-cosine-similarity words.
Storing them as BFLOAT16 halves the memory footprint; vectors are sent over the
API as FLOAT32 and converted server-side.
"""

import argparse
import os

import requests

from glove_common import INDEX_NAME, load_vectors

SERVER = os.getenv("HNSW_SERVER", "http://localhost:8685")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=100, choices=[50, 100, 200, 300])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="number of most-frequent words to load (default: all ~400k)",
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    print(
        f"Loading {'all' if args.limit is None else 'top ' + str(args.limit)} GloVe {args.dim}d vectors ..."
    )
    words, vectors = load_vectors(args.dim, args.limit)
    print(f"  {len(words)} words, dim={vectors.shape[1]}")

    # recreate the index fresh
    requests.delete(f"{SERVER}/delete_index", json={"indexName": INDEX_NAME})
    requests.delete(f"{SERVER}/delete_index_from_disk", json={"indexName": INDEX_NAME})
    res = requests.post(
        f"{SERVER}/create_index",
        json={
            "indexName": INDEX_NAME,
            "dimension": vectors.shape[1],
            "spaceType": "IP",
            "vectorType": "BFLOAT16",
            "efConstruction": 256,
            "M": 16,
        },
    )
    res.raise_for_status()
    print(f"Created index '{INDEX_NAME}' (IP space, BFLOAT16)")

    total = len(words)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_ids = list(range(start, end))
        batch_vecs = vectors[start:end].tolist()
        batch_meta = [{"word": words[i]} for i in batch_ids]
        res = requests.post(
            f"{SERVER}/add_documents",
            json={
                "indexName": INDEX_NAME,
                "ids": batch_ids,
                "vectors": batch_vecs,
                "metadatas": batch_meta,
            },
        )
        res.raise_for_status()
        print(f"  added {end}/{total}", end="\r")

    print(f"\nDone. Loaded {total} words into '{INDEX_NAME}'.")
    print("Start the UI with:  python app.py --dim", args.dim)


if __name__ == "__main__":
    main()

# Word search (GloVe + bfloat16 + UI)

Semantic word search over [GloVe](https://nlp.stanford.edu/projects/glove/) word
vectors. The embeddings are pre-trained and downloadable, so **no embedding model
is required** — every word already has a vector with real meaning. Vectors are
stored in the index as **bfloat16** (half the memory of float32), and search uses
the `IP` space over L2-normalized vectors, which is exactly cosine similarity.

The UI also supports **vector arithmetic** — the classic `king - man + woman ≈ queen`.

```
browser ──▶ Flask app (this dir) ──▶ hnswlib server (:8685)
            resolves words→vector       BFLOAT16 IP index
```

## Run it

1. Start the hnswlib server in Docker (build the image once from the repo root —
   see [`../README.md`](../README.md)):
   ```bash
   docker run --rm -p 8685:8685 -v "$PWD/indices:/indices" hnswlib_server:local
   ```

2. Download the vectors (one-time; the GloVe 6B archive is ~822 MB):
   ```bash
   cd examples/word-search
   uv run download_data.py --dim 100
   ```

3. Load the full ~400k-word vocabulary into the server as bfloat16:
   ```bash
   uv run load.py --dim 100
   ```

4. Start the search UI and open <http://localhost:5001>:
   ```bash
   uv run app.py --dim 100
   ```

   `load.py` and `app.py` both default to the full vocabulary; if you pass
   `--limit` to one, pass the same to the other so the words line up. (A smaller
   `--limit 100000` loads faster and keeps results to the most common words.)

## Things to try

- `coffee`, `quantum`, `barcelona` — nearest neighbours by meaning
- `king - man + woman` — gender analogy (lands near *queen*)
- `paris - france + japan` — capital analogy (lands near *tokyo*)
- `walking - walk + swim` — morphology analogy (lands near *swimming*)

## How it works

- `download_data.py` fetches GloVe 6B and extracts one dimensionality.
- `load.py` L2-normalizes the vectors and creates a `BFLOAT16` / `IP` index, then
  bulk-adds words with their text stored as metadata (`{"word": "..."}`).
- `app.py` keeps the vocabulary in memory to turn a query expression into a vector
  (summing `+`/`-` terms), then calls `/search` with `returnMetadata: true` and
  converts the IP distance back to a cosine similarity for display.

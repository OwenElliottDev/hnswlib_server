# Examples

Runnable demos of the hnswlib server. Each lives in its own folder with a README
and uses [`uv`](https://github.com/astral-sh/uv) to pull its dependencies on the
fly — no virtualenv setup required.

All examples talk to the server over HTTP. Run it in Docker — build the image
once from the repo root (the published image predates the 0.10.0 features these
examples use, e.g. `geodegrees`), then start it:

```bash
docker build -t hnswlib_server:local .
docker run --rm -p 8685:8685 -v "$PWD/indices:/indices" hnswlib_server:local
```

That listens on `:8685` and persists indices to `./indices`. The web UIs default
to `http://localhost:8685` (override with the `HNSW_SERVER` env var). The WAL
recovery demo starts and stops its own container (image via `HNSW_IMAGE`).

> Prefer a local build? Every example also works against `./build/bin/server`
> (`cmake -S . -B build && cmake --build build`); the WAL recovery demo takes a
> `--binary` flag for that.

| Example | What it shows |
| --- | --- |
| [`word-search/`](word-search/) | Semantic word search over GloVe vectors stored as **bfloat16**, with cosine similarity (IP space) and vector arithmetic (`king - man + woman ≈ queen`). Browser UI. |
| [`airports-geo/`](airports-geo/) | Nearest-airport search over ~80k airports using the **`geodegrees`** space (great-circle km), with metadata **filtering** by type/country/name/elevation. Interactive Leaflet map. |
| [`wal-recovery/`](wal-recovery/) | Crash recovery via the **write-ahead log**: kill a large index, then serve **live search traffic while the WAL replays** in the background. |

Each example downloads its own data into a local `data/` folder (gitignored) on
first run. See the per-example README for exact commands.

# Airports at the edge (in-browser search)

The [airports-geo](../airports-geo/) demo with no backend at query time. The browser
downloads the server's saved `airports` index once, loads it into the
[WebAssembly build](../../wasm/), and then runs every nearest-neighbour search and
filter locally in a few milliseconds.

```
hnswlib server ──save_index──▶ indices/airports.{bin,json,data} ──static files──▶ browser (wasm)
                                                                    map click → local search
```

Shift-click the map to add your own airfield. It's inserted into the in-browser index
and appears in results immediately.

## Run it

1. Build the WASM module (see [`wasm/README.md`](../../wasm/README.md) for installing Emscripten):
   ```bash
   ./wasm/build.sh
   ```

2. Build the airports index on the server, following steps 1–3 of
   [airports-geo](../airports-geo/README.md), then save it to disk:
   ```bash
   curl -X POST localhost:8685/save_index -d '{"indexName": "airports"}'
   ```
   This writes `indices/airports.bin`, `.json` and `.data` (about 38 MB uncompressed for
   85k airports). Once saved, the server is no longer needed.

3. Serve the page, the WASM build and the index files, then open <http://localhost:5003>:
   ```bash
   uv run examples/airports-edge/serve.py
   ```
   `serve.py` is a plain static file server. In production the same files can sit on any
   CDN or static host.

## How it works

- `VectorIndex.fromUrl('./indices', 'airports')` fetches the three files in parallel and
  loads them into wasm memory. That takes under a second for this dataset.
- The page builds the same filter expressions as `airports-geo/app.py`, for example
  `type IN ["large_airport"] AND iso_country = "JP"`, and calls `index.search(...)` directly.
- Results, distances (km from the `GEODEGREES` space) and metadata are identical to what
  the server returns for the same query.

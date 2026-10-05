# hnswlib edge (WebAssembly)

The server's index and metadata store, compiled to WebAssembly so vector search with
filtering runs directly in a browser, a Web Worker, Node, Deno or an edge runtime: no
HTTP server involved.

It reads and writes the server's own index files, so the typical flow is:

```
server: create_index → add_documents → save_index      indices/<name>.bin|.json|.data
                                                              │  host anywhere (CDN, S3, static site)
browser: VectorIndex.fromUrl(baseUrl, name) → search  ◀───────┘
```

Indexes built or modified in the browser can go the other way too: `save()` produces
the same three files, which the server loads with `/load_index`.

What's included: HNSW search over every space and vector type the server supports
(`IP`, `L2`, `GEODEGREES`; `FLOAT32`, `FLOAT16`, `BFLOAT16`; MRL with reranking), the
full filter DSL, document add/upsert/delete and get-by-id. Not included: the WAL and
background threads. Everything runs synchronously on the calling thread.

## Build

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html):

```bash
git clone https://github.com/emscripten-core/emsdk.git && cd emsdk
./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh
cd /path/to/hnswlib_server
./wasm/build.sh
```

This writes `wasm/dist/`:

| File | |
| --- | --- |
| `hnswlib-edge.mjs` | The API below: import this |
| `hnswlib-edge.d.ts` | TypeScript types |
| `hnswlib_edge.mjs`, `hnswlib_edge.wasm` | Emscripten module (~100 KB JS + ~450 KB wasm, before compression) |

Serve all four files from the same directory. `.wasm` should be served as
`application/wasm` and `.mjs` as `text/javascript`.

## Releases

Pushing a `v<version>` tag that matches the repo's `VERSION` file runs
`.github/workflows/publish-wasm.yml`. It builds and tests the module, commits the four
`dist/` files to the orphan `wasm-dist` branch, and tags that commit `wasm-v<version>`.
Use a pinned release straight from a CDN:

```js
import { VectorIndex, version } from 'https://cdn.jsdelivr.net/gh/OwenPendrighElliott/hnswlib_server@wasm-v0.3.0/hnswlib-edge.mjs';
```

`VERSION` is the single source for the release number. The server's `/version`, the
wasm module and the exported `version` constant are all built from it.

## Usage

```js
import { VectorIndex } from './hnswlib-edge.mjs';

// load an index saved by the server (fetches products.bin, products.json, products.data)
const index = await VectorIndex.fromUrl('https://cdn.example.com/indices', 'products');

const { hits, distances, metadatas } = index.search(queryVector, {
  k: 10,
  efSearch: 128,
  filter: 'category = "shoes" AND price < 100.0',
  returnMetadata: true,
});
```

Request and response shapes mirror the server's HTTP API:

```js
// create from scratch: same settings as /create_index
const index = await VectorIndex.create({ dimension: 384, spaceType: 'IP', vectorType: 'FLOAT16' });

// same shape as /add_documents; vectors may also be one flat Float32Array
index.addDocuments({
  ids: [1, 2],
  vectors: [vec1, vec2],
  metadatas: [{ category: 'shoes', price: 59.5 }, { category: 'hats', price: 20.0 }],
});
index.addDocuments({ ids: [1], vectors: [newVec1], metadatas: [{ category: 'boots' }] }); // upsert
index.deleteDocuments([2]);

index.getDocument(1);  // { id, vector: Float32Array, metadata } or null
index.status();        // { currentElements, maxElements, deletedElements }
index.settings;        // contents of <name>.json

// persist: write these as indices/<name>.bin/.json/.data for the server,
// or keep them in OPFS / IndexedDB and pass them back to VectorIndex.load()
const { bin, settings, data } = index.save();
const again = await VectorIndex.load({ bin, settings, data });

index.dispose(); // frees wasm memory; also works with `using index = ...`
```

In Node or Deno, read the files yourself:

```js
import { readFile } from 'node:fs/promises';
const [bin, settings, data] = await Promise.all(['bin', 'json', 'data'].map((e) => readFile(`indices/products.${e}`)));
const index = await VectorIndex.load({ bin, settings: settings.toString(), data });
```

If the `.wasm` file isn't next to the `.mjs`, call `init({ locateFile: (f) => '/assets/' + f })`
before creating an index.

### Numbers in metadata

The filter engine keeps integers and floats separate, so a field should hold one numeric
type. JavaScript can't tell `3` from `3.0`: `JSON.stringify(3.0)` is `"3"`. That means
a float field like `price` silently becomes an integer for whole values, and range
filters on it then return the wrong results. List such fields in the index settings and
they're always stored as floats:

```js
await VectorIndex.create({ dimension: 384, doubleFields: ['price', 'rating'] });
```

`doubleFields` is saved in `<name>.json`. For an index built by the server it isn't
needed for existing data, because Python and the server keep the types. Add it to the
settings if you'll add documents in the browser. The same rule applies to filter
literals: write `price < 100.0` for a float field.

Integers in metadata are 32-bit in WASM. Loading or adding a value outside ±2^31
throws, so store large values such as millisecond timestamps as floats or strings.

## Performance notes

- Memory is the main limit. wasm32 caps the heap at 4 GB, and browsers are happier well
  below that. The heap grows on demand but never shrinks. Budget roughly
  `N × (dimension × bytes-per-value + 2·M·4 + ~100)` bytes, plus metadata. `FLOAT16`,
  `BFLOAT16` and MRL indexes cut vector memory.
- The default `efSearch` is 512 for parity with the server. Lowering it (64–128) is
  usually much faster at a small recall cost.
- Loaded indexes are sized to their contents, so they don't carry the server's spare
  capacity, and they grow geometrically if you add documents.
- Search is synchronous. For large indexes, or to keep the UI responsive while loading,
  run the index inside a Web Worker.
- The `.bin` and `.data` files compress well. Serve them with gzip or brotli.
- Distance kernels are portable C++ auto-vectorised with WASM SIMD (`-msimd128`), not
  the hand-written SSE/AVX/NEON paths used natively. Results match the server (verified
  by `test/compat.mjs`), with differences in the last few bits of float distances.

## File compatibility

hnswlib writes `size_t` fields and labels straight from memory, so its files depend on
the platform's word size. Index files are always kept in the 64-bit layout the server
writes. `src/hnsw_format.hpp` translates to and from wasm32's in-memory layout on load
and save. The metadata `.data` format uses fixed-width integers and is identical on every
platform.

## Tests

```bash
node --test wasm/test/test.mjs          # unit tests against the wasm build

# server <-> wasm round trips (run the server from the repo root first)
./build/bin/server &
node wasm/test/compat.mjs
```

## Layout

| Path | |
| --- | --- |
| `src/edge_index.{hpp,cpp}` | One index: hnswlib graph + `DataStore`, mirroring the server's semantics |
| `src/hnsw_format.hpp` | 64-bit ↔ native translation of hnswlib index files |
| `src/bindings.cpp` | Embind surface |
| `js/hnswlib-edge.mjs` | Public JS API |
| `../src/index_utils.hpp` | Space construction, vector conversion and kNN dispatch shared with the server |

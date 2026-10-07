# HNSWLib Edge (WebAssembly)

This is the server's index and filtering engine compiled to WebAssembly. It lets you load an index and search it directly in a browser, Web Worker, Node, Deno or edge runtime without running the HTTP server.

It reads and writes the same index files as the server (`<name>.bin`, `<name>.json` and `<name>.data`). The usual workflow is to build an index on the server, save it, host the files somewhere static like a CDN or S3 bucket, and load them in the browser with `VectorIndex.fromUrl`. It works the other way too: an index built or modified in the browser can be saved and loaded into the server with `/load_index`.

It supports the same spaces (`IP`, `L2`, `GEODEGREES`), vector types (`FLOAT32`, `FLOAT16`, `BFLOAT16`), MRL reranking and filter syntax as the server, along with adding, deleting and fetching documents. There's no WAL and no background threads, everything runs synchronously on the calling thread.

## Quick Start

```js
import { VectorIndex } from './hnswlib-edge.mjs';

// fetches products.bin, products.json and products.data
const index = await VectorIndex.fromUrl('https://cdn.example.com/indices', 'products');

const { hits, distances, metadatas } = index.search(queryVector, {
  k: 10,
  efSearch: 128,
  filter: 'category = "shoes" AND price < 100.0',
  returnMetadata: true,
});
```

Pinned releases can be imported straight from a CDN:

```js
import { VectorIndex, version } from 'https://cdn.jsdelivr.net/gh/OwenPendrighElliott/hnswlib_server@wasm-v0.3.0/hnswlib-edge.mjs';
```

## Building

You need the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html) installed and activated:

```bash
git clone https://github.com/emscripten-core/emsdk.git && cd emsdk
./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh
```

Then from the root of this repo:

```bash
./wasm/build.sh
```

This writes four files to `wasm/dist/`. `hnswlib-edge.mjs` is the one you import and `hnswlib-edge.d.ts` has the TypeScript types. `hnswlib_edge.mjs` and `hnswlib_edge.wasm` are the Emscripten module (about 100KB of JS and 450KB of wasm before compression).

Serve all four from the same directory, with `.wasm` as `application/wasm` and `.mjs` as `text/javascript`. If the `.wasm` file lives somewhere else, call `init` before creating an index:

```js
import { init } from './hnswlib-edge.mjs';
await init({ locateFile: (f) => '/assets/' + f });
```

## Releases

Pushing a `v<version>` tag that matches the `VERSION` file runs `.github/workflows/publish-wasm.yml`. It builds and tests the module, commits the `dist/` files to the orphan `wasm-dist` branch and tags that commit `wasm-v<version>`.

`VERSION` is the only place the version number lives. The server's `/version`, the wasm module and the exported `version` constant all come from it.

## Usage

The methods take the same request shapes as the server's HTTP API.

### Creating an index

Settings are the same as `/create_index`:

```js
const index = await VectorIndex.create({ dimension: 384, spaceType: 'IP', vectorType: 'FLOAT16' });
```

### Adding and deleting documents

Same shape as `/add_documents`. Vectors can be an array of arrays or one flat `Float32Array`. Adding an ID that already exists replaces it.

```js
index.addDocuments({
  ids: [1, 2],
  vectors: [vec1, vec2],
  metadatas: [{ category: 'shoes', price: 59.5 }, { category: 'hats', price: 20.0 }],
});

index.deleteDocuments([2]);
```

### Searching

Same options as `/search`. Use `offset` to page through results:

```js
const page1 = index.search(queryVector, { k: 10 });
const page2 = index.search(queryVector, { k: 10, offset: 10 });
```

### Similar documents

`similar` searches using a stored document's vector as the query, the same as `/similar`. The document itself is left out of the results unless you set `excludeInputDocument: false`. It throws if the ID isn't in the index.

```js
const { hits } = index.similar(42, { k: 10, filter: 'category = "shoes"' });
```

### Fetching documents and status

```js
index.getDocument(1);  // { id, vector, metadata }, or null if the ID isn't in the index
index.contains(1);     // true
index.status();        // { currentElements, maxElements, deletedElements }
index.settings;        // contents of <name>.json
```

### Saving and loading

`save` returns the bytes of the three index files. Write them to `indices/<name>.bin`, `.json` and `.data` to load them into the server, or keep them in OPFS or IndexedDB and pass them back to `VectorIndex.load`:

```js
const { bin, settings, data } = index.save();
const again = await VectorIndex.load({ bin, settings, data });
```

In Node or Deno you can read the server's files yourself:

```js
import { readFile } from 'node:fs/promises';
const [bin, settings, data] = await Promise.all(['bin', 'json', 'data'].map((e) => readFile(`indices/products.${e}`)));
const index = await VectorIndex.load({ bin, settings: settings.toString(), data });
```

Call `dispose` when you're done with an index to free its wasm memory. It also works with `using index = ...`.

## Numbers in Metadata

The filter engine keeps integers and floats separate, so each numeric field should hold one type. JavaScript can't tell `3` from `3.0`, so a float field like `price` quietly turns into an integer for whole values, and range filters on it then give wrong results. To avoid this, list float fields in `doubleFields` when creating the index and they'll always be stored as floats:

```js
await VectorIndex.create({ dimension: 384, doubleFields: ['price', 'rating'] });
```

`doubleFields` is saved in `<name>.json`. Indexes built by the server don't need it for their existing data, because Python and the server keep the types, but add it to the settings if you'll be adding documents in the browser. The same applies to filters: write `price < 100.0` for a float field.

Integers are 32-bit in WASM. Adding or loading a value outside ±2^31 throws, so store things like millisecond timestamps as floats or strings.

## Performance

Memory is the main limit. wasm32 caps the heap at 4GB, and browsers are happier well below that. The heap grows as needed but never shrinks. As a rough guide an index takes `N × (dimension × bytes per value + 2 × M × 4 + 100)` bytes plus metadata. `FLOAT16`, `BFLOAT16` and MRL indexes all reduce vector memory.

The default `efSearch` is 512 to match the server. Dropping it to 64 to 128 is usually a lot faster for a small loss in recall.

Loaded indexes are sized to fit their contents, so they don't carry the server's spare capacity. They grow geometrically if you add documents.

Search is synchronous, so for large indexes, or to keep the UI responsive while loading, run the index in a Web Worker. The `.bin` and `.data` files compress well, so serve them with gzip or brotli.

The distance functions are plain C++ auto-vectorised with WASM SIMD (`-msimd128`) instead of the hand-written SSE, AVX and NEON versions used natively. Results match the server, but float distances can differ in the last few bits.

## File Compatibility

hnswlib writes `size_t` fields and labels straight from memory, so its files depend on the platform's word size. Index files are always kept in the 64-bit layout the server writes, and `src/hnsw_format.hpp` converts to and from wasm32's layout when loading and saving. The `.data` metadata format uses fixed-width integers, so it's the same everywhere.

## Code Layout

- `src/edge_index.hpp` and `src/edge_index.cpp`: a single index, the hnswlib graph plus a `DataStore`, with the same behaviour as the server.
- `src/hnsw_format.hpp`: converts hnswlib index files between the 64-bit and native layouts.
- `src/bindings.cpp`: the Embind bindings.
- `js/hnswlib-edge.mjs`: the public JS API.
- `../src/index_utils.hpp`: space construction, vector conversion, kNN search and pagination, shared with the server.

# Testing

The unit tests run against the wasm build:

```bash
node --test wasm/test/test.mjs
```

The compatibility tests check that the server and wasm build give the same results and can load each other's files. Start the server from the repo root first:

```bash
./build/bin/server &
node wasm/test/compat.mjs
```

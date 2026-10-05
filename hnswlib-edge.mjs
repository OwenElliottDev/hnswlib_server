// High-level API over the Embind module. Mirrors the HTTP server's request and
// response shapes, and reads/writes the server's index files
// (<name>.bin, <name>.json, <name>.data) so indexes move freely between the two.

import createHnswlibEdgeModule from './hnswlib_edge.mjs';

/** Release version, stamped from the repo's VERSION file by build.sh. */
export const version = '0.3.1';

let modulePromise = null;

/**
 * Initialises the WASM module. Optional: every factory calls it. Call it
 * yourself first to pass Emscripten options such as `locateFile` when the
 * .wasm file is not served next to this script.
 */
export function init(options = {}) {
  if (!modulePromise) {
    modulePromise = createHnswlibEdgeModule(options).catch((err) => {
      modulePromise = null;
      throw err;
    });
  }
  return modulePromise;
}

let tmpCounter = 0;
const tmpPath = (ext) => `/tmp/hnswlib-edge-${Date.now()}-${++tmpCounter}.${ext}`;

function unlinkQuietly(FS, path) {
  try {
    FS.unlink(path);
  } catch {
    // already gone
  }
}

function toBytes(src, what) {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  if (ArrayBuffer.isView(src)) return new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
  throw new TypeError(`${what} must be a Uint8Array, ArrayBuffer or typed array`);
}

function settingsToJson(settings) {
  if (typeof settings === 'string') return settings;
  if (settings && typeof settings === 'object') return JSON.stringify(settings);
  throw new TypeError('settings must be an object or JSON string');
}

function flattenVectors(vectors, dimension) {
  if (vectors instanceof Float32Array) return vectors;
  if (!Array.isArray(vectors)) throw new TypeError('vectors must be a Float32Array or an array of vectors');
  const flat = new Float32Array(vectors.length * dimension);
  vectors.forEach((v, i) => {
    if (v.length !== dimension) {
      throw new Error(`Vector ${i} has dimension ${v.length}, expected ${dimension}`);
    }
    flat.set(v, i * dimension);
  });
  return flat;
}

async function fetchBytes(fetchFn, url, requestInit, optional) {
  const res = await fetchFn(url, requestInit);
  if (!res.ok) {
    if (optional && res.status === 404) return null;
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

export class VectorIndex {
  #module;
  #handle;
  #settings;

  /** @private Use VectorIndex.create / load / fromUrl. */
  constructor(module, handle) {
    this.#module = module;
    this.#handle = handle;
    this.#settings = JSON.parse(handle.settings());
  }

  /**
   * Creates an empty index. Settings match the server's /create_index body:
   * { dimension, spaceType?: 'IP'|'L2'|'GEODEGREES', vectorType?: 'FLOAT32'|'FLOAT16'|'BFLOAT16',
   *   M?, efConstruction?, mrlScanDim?, doubleFields?: string[] }
   */
  static async create(settings, { initialCapacity = 1024 } = {}) {
    const module = await init();
    return new VectorIndex(module, module.Index.create(settingsToJson(settings), initialCapacity));
  }

  /**
   * Loads an index from the bytes of the server's saved files.
   * `settings` is the parsed (or raw) <name>.json; `data` is <name>.data and may
   * be omitted for an index without metadata.
   */
  static async load({ bin, settings, data = null }) {
    const module = await init();
    const { FS } = module;
    const binPath = tmpPath('bin');
    const dataPath = data ? tmpPath('data') : '';
    try {
      FS.writeFile(binPath, toBytes(bin, 'bin'));
      if (data) FS.writeFile(dataPath, toBytes(data, 'data'));
      return new VectorIndex(module, module.Index.loadFiles(binPath, settingsToJson(settings), dataPath));
    } finally {
      unlinkQuietly(FS, binPath);
      if (dataPath) unlinkQuietly(FS, dataPath);
    }
  }

  /**
   * Fetches and loads `${baseUrl}/${name}.bin|.json|.data`, i.e. a copy of the
   * server's indices/ directory hosted anywhere. A missing .data file (404) is
   * treated as an index without metadata.
   */
  static async fromUrl(baseUrl, name, { fetch: fetchFn = globalThis.fetch, requestInit } = {}) {
    const base = String(baseUrl).replace(/\/+$/, '');
    const enc = encodeURIComponent(name);
    const [bin, settingsBytes, data] = await Promise.all([
      fetchBytes(fetchFn, `${base}/${enc}.bin`, requestInit, false),
      fetchBytes(fetchFn, `${base}/${enc}.json`, requestInit, false),
      fetchBytes(fetchFn, `${base}/${enc}.data`, requestInit, true),
    ]);
    const settings = new TextDecoder().decode(settingsBytes);
    return VectorIndex.load({ bin, settings, data });
  }

  #live() {
    if (!this.#handle) throw new Error('VectorIndex has been disposed');
    return this.#handle;
  }

  get dimension() {
    return this.#settings.dimension;
  }

  /** The index settings, in the server's <name>.json format. */
  get settings() {
    return structuredClone(this.#settings);
  }

  /** { currentElements, maxElements, deletedElements } */
  status() {
    return JSON.parse(this.#live().status());
  }

  /**
   * Adds or replaces documents. Same shape as the server's /add_documents body:
   * { ids: number[], vectors: number[][] | Float32Array (flat, ids.length * dimension),
   *   metadatas?: object[] }
   */
  addDocuments({ ids, vectors, metadatas }) {
    const handle = this.#live();
    const flat = flattenVectors(vectors, this.dimension);
    handle.addDocuments(ids, flat, metadatas ? JSON.stringify(metadatas) : '');
  }

  deleteDocuments(ids) {
    this.#live().deleteDocuments(ids);
  }

  /**
   * Same options as the server's /search body. Returns
   * { hits: number[], distances: number[], metadatas?: object[] }, nearest first.
   */
  search(queryVector, { k = 10, efSearch = 512, filter = '', returnMetadata = false, rerankSize = 0 } = {}) {
    const query = queryVector instanceof Float32Array ? queryVector : Float32Array.from(queryVector);
    const r = this.#live().search(query, k, efSearch, filter, returnMetadata, rerankSize);
    const result = { hits: Array.from(r.hits), distances: Array.from(r.distances) };
    if (returnMetadata) result.metadatas = JSON.parse(r.metadatas);
    return result;
  }

  contains(id) {
    return this.#live().contains(id);
  }

  /** Returns { id, vector: Float32Array, metadata } or null when the id is unknown. */
  getDocument(id) {
    const doc = this.#live().getDocument(id);
    if (doc === null) return null;
    return { id, vector: doc.vector, metadata: JSON.parse(doc.metadata) };
  }

  /**
   * Serialises the index to the server's file format. Write `bin`, `settings`
   * (as JSON) and `data` to indices/<name>.bin|.json|.data to load it into the
   * server with /load_index, or keep the bytes (e.g. in OPFS/IndexedDB) and
   * pass them back to VectorIndex.load().
   */
  save() {
    const handle = this.#live();
    const { FS } = this.#module;
    const binPath = tmpPath('bin');
    const dataPath = tmpPath('data');
    try {
      handle.saveFiles(binPath, dataPath);
      return { bin: FS.readFile(binPath), settings: this.settings, data: FS.readFile(dataPath) };
    } finally {
      unlinkQuietly(FS, binPath);
      unlinkQuietly(FS, dataPath);
    }
  }

  /** Frees the WASM memory held by this index. The instance is unusable afterwards. */
  dispose() {
    if (this.#handle) {
      this.#handle.delete();
      this.#handle = null;
    }
  }

  [Symbol.dispose]() {
    this.dispose();
  }
}

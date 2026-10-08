// Cross-compatibility test between the HTTP server and the WASM build.
//
//   1. builds indexes on the server and saves them
//   2. loads the saved files in WASM and compares search results with the server
//   3. mutates the index in WASM, saves it into the server's indices/ directory,
//      loads it with /load_index and compares again
//
// Usage (server running from the repo root so it writes to ./indices):
//   node wasm/test/compat.mjs [serverUrl] [indicesDir]
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { VectorIndex } from '../dist/hnswlib-edge.mjs';

const SERVER = process.argv[2] ?? 'http://localhost:8685';
const INDICES = process.argv[3] ?? path.resolve(import.meta.dirname, '../../indices');

async function call(method, route, body) {
  const res = await fetch(`${SERVER}${route}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${method} ${route} -> ${res.status}: ${text}`);
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function dropIndex(name) {
  await call('DELETE', '/delete_index', { indexName: name }).catch(() => {});
  await call('DELETE', '/delete_index_from_disk', { indexName: name }).catch(() => {});
}

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomVectors(n, dim, seed) {
  const r = rng(seed);
  return Array.from({ length: n }, () => Array.from({ length: dim }, () => r() * 2 - 1));
}

// prices are always fractional so JSON keeps them floats on both sides
const metadataFor = (i) => ({
  category: ['books', 'music', 'film', 'games'][i % 4],
  price: (i % 50) + 0.25,
  year: 1990 + (i % 30),
  tags: [['new', 'sale', 'popular', 'rare'][i % 4], 'all'],
});

const FILTERS = ['', 'category = "music"', 'year >= 2010 AND price < 20.0', 'tags CONTAINS "rare"', 'category IN ["books", "film"]'];

// Native builds use SSE/NEON distance kernels while WASM uses scalar code, so
// distances can differ in the last bits; compare with a tolerance and require
// near-identical hit sets rather than exact equality.
function assertSimilar(server, wasm, label) {
  const overlap = wasm.hits.filter((h) => server.hits.includes(h)).length;
  assert.ok(overlap >= server.hits.length - 1, `${label}: overlap ${overlap}/${server.hits.length}`);
  assert.equal(wasm.hits.length, server.hits.length, `${label}: hit count`);
  server.hits.forEach((id, i) => {
    const j = wasm.hits.indexOf(id);
    if (j >= 0) assert.ok(Math.abs(server.distances[i] - wasm.distances[j]) < 1e-3, `${label}: distance for ${id}`);
  });
  if (server.metadatas) {
    server.hits.forEach((id, i) => {
      const j = wasm.hits.indexOf(id);
      if (j >= 0) assert.deepEqual(wasm.metadatas[j], server.metadatas[i], `${label}: metadata for ${id}`);
    });
  }
}

async function compareSearches(name, index, queries, extra = {}) {
  for (const filter of FILTERS) {
    for (const [qi, q] of queries.entries()) {
      const opts = { k: 10, efSearch: 200, filter, returnMetadata: true, ...extra };
      const server = await call('POST', '/search', { indexName: name, queryVector: q, ...opts });
      assertSimilar(server, index.search(q, opts), `${name} filter=${JSON.stringify(filter)} q=${qi}`);
    }
  }
}

async function comparePagination(name, index, queries, extra = {}) {
  for (const [qi, q] of queries.slice(0, 3).entries()) {
    for (const offset of [0, 7, 25]) {
      const opts = { k: 10, offset, efSearch: 200, returnMetadata: true, ...extra };
      const server = await call('POST', '/search', { indexName: name, queryVector: q, ...opts });
      assertSimilar(server, index.search(q, opts), `${name} offset=${offset} q=${qi}`);
    }
  }
  for (const docId of [0, 500, 2999]) {
    for (const offset of [0, 10]) {
      const opts = { k: 10, offset, efSearch: 200, filter: 'year >= 2000', returnMetadata: true, ...extra };
      const server = await call('POST', '/similar', { indexName: name, docId, ...opts });
      const wasm = index.similar(docId, opts);
      assert.ok(!wasm.hits.includes(docId) && !server.hits.includes(docId), `${name} similar ${docId}: input excluded`);
      assertSimilar(server, wasm, `${name} similar docId=${docId} offset=${offset}`);
    }
  }
}

const CONFIGS = [
  { spaceType: 'L2', vectorType: 'FLOAT32' },
  { spaceType: 'IP', vectorType: 'FLOAT16' },
  { spaceType: 'L2', vectorType: 'BFLOAT16', mrlScanDim: 8, search: { rerankSize: 100 } },
];

const DIM = 32;
const N = 3000;
const vectors = randomVectors(N, DIM, 11);
const queries = randomVectors(10, DIM, 12);

for (const [ci, config] of CONFIGS.entries()) {
  const { search: searchExtra = {}, ...settings } = config;
  const name = `wasm_compat_${ci}`;
  const roundTrip = `${name}_rt`;
  await dropIndex(name);
  await dropIndex(roundTrip);

  // 1. server builds and saves
  await call('POST', '/create_index', { indexName: name, dimension: DIM, M: 16, efConstruction: 200, ...settings });
  for (let i = 0; i < N; i += 500) {
    const ids = vectors.slice(i, i + 500).map((_, j) => i + j);
    await call('POST', '/add_documents', { indexName: name, ids, vectors: vectors.slice(i, i + 500), metadatas: ids.map(metadataFor) });
  }
  await call('DELETE', '/delete_documents', { indexName: name, ids: [1, 2, 3] });
  await call('POST', '/save_index', { indexName: name });

  // 2. WASM loads the server's files
  const [bin, json, data] = await Promise.all(['bin', 'json', 'data'].map((ext) => readFile(path.join(INDICES, `${name}.${ext}`))));
  const index = await VectorIndex.load({ bin, settings: json.toString(), data });
  assert.equal(index.getDocument(2), null, 'server-side delete survives');
  assert.deepEqual(index.getDocument(10).metadata, metadataFor(10));
  await compareSearches(name, index, queries, searchExtra);
  await comparePagination(name, index, queries, searchExtra);
  console.log(`ok - ${name}: server -> wasm (${JSON.stringify(settings)})`);

  // 3. WASM mutates and saves; server loads the result
  const extra = randomVectors(200, DIM, 13);
  const extraIds = extra.map((_, i) => N + i);
  index.addDocuments({ ids: extraIds, vectors: extra, metadatas: extraIds.map(metadataFor) });
  index.deleteDocuments([10, 11, 12]);
  index.addDocuments({ ids: [20], vectors: [vectors[20]], metadatas: [{ category: 'updated', price: 1.5, year: 2024, tags: ['x'] }] });
  const saved = index.save();
  await writeFile(path.join(INDICES, `${roundTrip}.bin`), saved.bin);
  await writeFile(path.join(INDICES, `${roundTrip}.json`), JSON.stringify({ ...saved.settings, indexName: roundTrip }));
  await writeFile(path.join(INDICES, `${roundTrip}.data`), saved.data);
  await call('POST', '/load_index', { indexName: roundTrip });

  const doc = await call('GET', `/get_document/${roundTrip}/20`);
  assert.equal(doc.metadata.category, 'updated');
  await assert.rejects(call('GET', `/get_document/${roundTrip}/10`), /404/);
  const status = await call('GET', `/index_status/${roundTrip}`);
  assert.equal(status.currentElements, index.status().currentElements);
  await compareSearches(roundTrip, index, [...queries, extra[0], extra[199]], searchExtra);
  console.log(`ok - ${name}: wasm -> server`);

  index.dispose();
  await dropIndex(name);
  await dropIndex(roundTrip);
}

console.log('all compatibility checks passed');

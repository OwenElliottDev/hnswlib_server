// Node tests for the WASM build: node --test wasm/test/test.mjs (after wasm/build.sh)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { VectorIndex, init, version } from '../dist/hnswlib-edge.mjs';

// deterministic PRNG so failures are reproducible
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

const l2 = (a, b) => a.reduce((acc, v, i) => acc + (v - b[i]) ** 2, 0);

function bruteForce(vectors, ids, query, k) {
  return ids
    .map((id) => [l2(vectors[id], query), id])
    .sort((a, b) => a[0] - b[0])
    .slice(0, k)
    .map(([, id]) => id);
}

const CATEGORIES = ['books', 'music', 'film', 'games'];
const TAGS = ['new', 'sale', 'popular', 'rare'];

function makeMetadata(i) {
  return {
    category: CATEGORIES[i % 4],
    price: (i % 50) + (i % 3 === 0 ? 0 : 0.5), // whole and fractional values in one field
    year: 1990 + (i % 30),
    tags: [TAGS[i % 4], TAGS[(i + 1) % 4]],
  };
}

const DIM = 16;
const N = 2000;
const vectors = randomVectors(N, DIM, 1);
const ids = vectors.map((_, i) => i);
const metadatas = ids.map(makeMetadata);
const queries = randomVectors(50, DIM, 2);

async function buildIndex() {
  const index = await VectorIndex.create({ dimension: DIM, spaceType: 'L2', M: 16, efConstruction: 200, doubleFields: ['price'] });
  index.addDocuments({ ids, vectors, metadatas });
  return index;
}

test('approximate search recall against brute force', async () => {
  const index = await buildIndex();
  let found = 0;
  for (const q of queries) {
    const truth = new Set(bruteForce(vectors, ids, q, 10));
    const { hits, distances } = index.search(q, { k: 10, efSearch: 100 });
    assert.equal(hits.length, 10);
    assert.ok(distances.every((d, i) => i === 0 || d >= distances[i - 1]), 'distances sorted ascending');
    found += hits.filter((h) => truth.has(h)).length;
  }
  const recall = found / (queries.length * 10);
  assert.ok(recall >= 0.95, `recall ${recall}`);
  index.dispose();
});

test('filters restrict results and match brute force', async () => {
  const index = await buildIndex();
  const cases = [
    ['category = "music"', (m) => m.category === 'music'],
    ['category = "music" AND year >= 2010', (m) => m.category === 'music' && m.year >= 2010],
    ['price < 10.0', (m) => m.price < 10],
    ['price > 40.0 OR category = "film"', (m) => m.price > 40 || m.category === 'film'],
    ['NOT category = "books"', (m) => m.category !== 'books'],
    ['category IN ["books", "games"]', (m) => m.category === 'books' || m.category === 'games'],
    ['tags CONTAINS "rare"', (m) => m.tags.includes('rare')],
    ['year = 1995', (m) => m.year === 1995],
  ];
  for (const [filter, pred] of cases) {
    const allowed = ids.filter((id) => pred(metadatas[id]));
    assert.ok(allowed.length > 0, filter);
    for (const q of queries.slice(0, 10)) {
      const { hits, metadatas: metas } = index.search(q, { k: 10, efSearch: 200, filter, returnMetadata: true });
      assert.equal(hits.length, Math.min(10, allowed.length), filter);
      metas.forEach((m, i) => assert.ok(pred(m), `${filter}: hit ${hits[i]} has ${JSON.stringify(m)}`));
      const truth = new Set(bruteForce(vectors, allowed, q, 10));
      const overlap = hits.filter((h) => truth.has(h)).length;
      assert.ok(overlap >= 8, `${filter}: overlap ${overlap}/10`);
    }
  }
  index.dispose();
});

test('doubleFields stores whole numbers as floats', async () => {
  const index = await buildIndex();
  // id 0 has price 0 (whole), id 1 has price 1.5; both must be doubles for range filters to work
  assert.deepEqual(index.getDocument(0).metadata.price, 0);
  const { hits } = index.search(vectors[0], { k: N, efSearch: N, filter: 'price < 1.0' });
  const expected = ids.filter((id) => metadatas[id].price < 1);
  assert.deepEqual(new Set(hits), new Set(expected));
  index.dispose();
});

test('getDocument returns the stored vector and metadata', async () => {
  const index = await buildIndex();
  const doc = index.getDocument(42);
  assert.equal(doc.id, 42);
  assert.ok(doc.vector instanceof Float32Array);
  doc.vector.forEach((v, i) => assert.ok(Math.abs(v - vectors[42][i]) < 1e-6));
  assert.deepEqual(doc.metadata, metadatas[42]);
  assert.equal(index.getDocument(N + 10), null);
  assert.equal(index.contains(42), true);
  assert.equal(index.contains(N + 10), false);
  index.dispose();
});

test('upsert replaces vector and metadata, including the filter index', async () => {
  const index = await buildIndex();
  const target = new Array(DIM).fill(5);
  index.addDocuments({ ids: [7], vectors: [target], metadatas: [{ category: 'replaced' }] });
  assert.equal(index.search(target, { k: 1 }).hits[0], 7);
  assert.deepEqual(index.getDocument(7).metadata, { category: 'replaced' });
  const old = index.search(vectors[7], { k: N, efSearch: N, filter: `category = "${metadatas[7].category}"` });
  assert.ok(!old.hits.includes(7), 'stale metadata must not match filters');
  assert.equal(index.status().currentElements, N);
  index.dispose();
});

test('deleted documents disappear from search and lookups', async () => {
  const index = await buildIndex();
  const victims = [0, 1, 2, 3, 4];
  index.deleteDocuments(victims);
  index.deleteDocuments([N + 100]); // unknown ids are a no-op
  for (const id of victims) {
    assert.equal(index.getDocument(id), null);
    assert.ok(!index.search(vectors[id], { k: 10 }).hits.includes(id));
  }
  // remaining docs are still found
  assert.equal(index.search(vectors[10], { k: 1 }).hits[0], 10);
  index.dispose();
});

test('save/load round trip preserves results', async () => {
  const index = await buildIndex();
  index.deleteDocuments([5]);
  const saved = index.save();
  assert.ok(saved.bin.length > 0 && saved.data.length > 0);
  assert.equal(saved.settings.dimension, DIM);

  const loaded = await VectorIndex.load(saved);
  for (const q of queries.slice(0, 10)) {
    const opts = { k: 10, efSearch: 100, filter: 'category != "film"', returnMetadata: true };
    assert.deepEqual(loaded.search(q, opts), index.search(q, opts));
  }
  assert.equal(loaded.getDocument(5), null);
  assert.deepEqual(loaded.settings.doubleFields, ['price']);

  // loaded index stays mutable and grows past its load-time capacity
  const extra = randomVectors(3000, DIM, 3);
  loaded.addDocuments({ ids: extra.map((_, i) => N + i), vectors: extra });
  assert.equal(loaded.search(extra[2999], { k: 1 }).hits[0], N + 2999);
  index.dispose();
  loaded.dispose();
});

test('settings that do not match the index file are rejected', async () => {
  const index = await buildIndex();
  const saved = index.save();
  await assert.rejects(VectorIndex.load({ ...saved, settings: { ...saved.settings, dimension: DIM * 2 } }), /do not match/);
  index.dispose();
});

test('index grows beyond its initial capacity', async () => {
  const index = await VectorIndex.create({ dimension: 8, spaceType: 'L2' }, { initialCapacity: 10 });
  const vs = randomVectors(500, 8, 4);
  for (let i = 0; i < vs.length; i += 50) {
    index.addDocuments({ ids: vs.slice(i, i + 50).map((_, j) => i + j), vectors: vs.slice(i, i + 50) });
  }
  assert.equal(index.status().currentElements, 500);
  assert.equal(index.search(vs[123], { k: 1 }).hits[0], 123);
  index.dispose();
});

test('flat Float32Array input is accepted', async () => {
  const index = await VectorIndex.create({ dimension: 4, spaceType: 'L2' });
  index.addDocuments({ ids: [10, 20], vectors: new Float32Array([0, 0, 0, 0, 1, 1, 1, 1]) });
  assert.deepEqual(index.search(new Float32Array([0.9, 0.9, 0.9, 0.9]), { k: 2 }).hits, [20, 10]);
  index.dispose();
});

for (const vectorType of ['FLOAT16', 'BFLOAT16']) {
  test(`${vectorType} vectors`, async () => {
    const index = await VectorIndex.create({ dimension: DIM, spaceType: 'L2', vectorType });
    index.addDocuments({ ids, vectors });
    let found = 0;
    for (const q of queries.slice(0, 20)) {
      const truth = new Set(bruteForce(vectors, ids, q, 10));
      found += index.search(q, { k: 10, efSearch: 100 }).hits.filter((h) => truth.has(h)).length;
    }
    assert.ok(found / 200 >= 0.85, `recall ${found / 200}`);
    const doc = index.getDocument(3);
    doc.vector.forEach((v, i) => assert.ok(Math.abs(v - vectors[3][i]) < 0.02));

    const loaded = await VectorIndex.load(index.save());
    assert.deepEqual(loaded.search(queries[0], { k: 5 }), index.search(queries[0], { k: 5 }));
    index.dispose();
    loaded.dispose();
  });
}

test('inner product space', async () => {
  const index = await VectorIndex.create({ dimension: 3, spaceType: 'IP' });
  index.addDocuments({ ids: [1, 2, 3], vectors: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] });
  assert.equal(index.search([0.1, 0.9, 0.1], { k: 1 }).hits[0], 2);
  index.dispose();
});

test('geodegrees space returns nearest by great-circle distance', async () => {
  const index = await VectorIndex.create({ dimension: 2, spaceType: 'GEODEGREES' });
  const cities = {
    1: [51.5074, -0.1278], // London
    2: [48.8566, 2.3522], // Paris
    3: [40.7128, -74.006], // New York
    4: [-33.8688, 151.2093], // Sydney
  };
  index.addDocuments({ ids: Object.keys(cities).map(Number), vectors: Object.values(cities) });
  const { hits } = index.search([50.85, 4.35], { k: 4 }); // Brussels
  assert.deepEqual(hits, [2, 1, 3, 4]);
  index.dispose();
});

test('MRL index scans truncated vectors and reranks at full dimension', async () => {
  // Matryoshka-style data: leading dimensions carry most of the variance, so a
  // 4-dim scan is a good proxy for full distance (uniform data would not be)
  const decay = (vs) => vs.map((v) => v.map((x, d) => x * 0.6 ** d));
  const mrlVectors = decay(vectors);
  const mrlQueries = decay(queries);
  const index = await VectorIndex.create({ dimension: DIM, spaceType: 'L2', mrlScanDim: 4 });
  index.addDocuments({ ids, vectors: mrlVectors });
  let found = 0;
  for (const q of mrlQueries.slice(0, 20)) {
    const truth = new Set(bruteForce(mrlVectors, ids, q, 10));
    found += index.search(q, { k: 10, efSearch: 200, rerankSize: 200 }).hits.filter((h) => truth.has(h)).length;
  }
  assert.ok(found / 200 >= 0.95, `recall ${found / 200}`);
  const loaded = await VectorIndex.load(index.save());
  const opts = { k: 10, efSearch: 200, rerankSize: 200 };
  assert.deepEqual(loaded.search(queries[0], opts), index.search(queries[0], opts));
  index.dispose();
  loaded.dispose();
});

test('errors surface as JS exceptions with messages', async () => {
  const index = await buildIndex();
  assert.throws(() => index.search([1, 2, 3]), /dimension 3, expected 16/);
  assert.throws(() => index.search(queries[0], { filter: 'category = ' }), Error);
  assert.throws(() => index.addDocuments({ ids: [1, 2], vectors: [vectors[0]] }), /Expected 2 vectors/);
  assert.throws(() => index.addDocuments({ ids: [-1], vectors: [vectors[0]] }), /non-negative/);
  assert.throws(() => index.addDocuments({ ids: [1], vectors: [vectors[0]], metadatas: [{ big: 2 ** 40 }] }), /32-bit/);
  await assert.rejects(VectorIndex.create({ dimension: 0 }), /dimension/);
  await assert.rejects(VectorIndex.create({ dimension: 4, spaceType: 'COSINE' }), /spaceType/);
  // the index is still usable after errors
  assert.equal(index.search(vectors[9], { k: 1 }).hits[0], 9);
  index.dispose();
  assert.throws(() => index.status(), /disposed/);
});

test('empty index searches return no hits', async () => {
  const index = await VectorIndex.create({ dimension: 4 });
  assert.deepEqual(index.search([1, 0, 0, 0], { k: 5 }), { hits: [], distances: [] });
  const loaded = await VectorIndex.load(index.save());
  assert.deepEqual(loaded.search([1, 0, 0, 0], { k: 5, filter: 'a = 1' }), { hits: [], distances: [] });
  index.dispose();
  loaded.dispose();
});

// points on a line under L2, so neighbour order is fully determined
async function lineIndex(n = 50) {
  const index = await VectorIndex.create({ dimension: 4, spaceType: 'L2' });
  const lineIds = Array.from({ length: n }, (_, i) => i);
  index.addDocuments({
    ids: lineIds,
    vectors: lineIds.map((i) => [i, 0, 0, 0]),
    metadatas: lineIds.map((i) => ({ position: i, parity: i % 2 === 0 ? 'even' : 'odd' })),
  });
  return index;
}

const range = (from, to) => Array.from({ length: to - from }, (_, i) => from + i);
const ORIGIN = [0, 0, 0, 0];

test('search offset pages through results', async () => {
  const index = await lineIndex();
  assert.deepEqual(index.search(ORIGIN, { k: 10, offset: 0 }), index.search(ORIGIN, { k: 10 }));

  const full = index.search(ORIGIN, { k: 30 });
  const pages = [0, 10, 20].map((offset) => index.search(ORIGIN, { k: 10, offset }));
  assert.deepEqual(pages.flatMap((p) => p.hits), full.hits);
  assert.deepEqual(pages.flatMap((p) => p.distances), full.distances);
  assert.deepEqual(full.hits, range(0, 30));

  assert.deepEqual(index.search(ORIGIN, { k: 10, offset: 45 }).hits, range(45, 50), 'partial last page');
  assert.deepEqual(index.search(ORIGIN, { k: 10, offset: 50 }), { hits: [], distances: [] }, 'past the end');
  assert.deepEqual(index.search(ORIGIN, { k: 5, offset: 40, efSearch: 10 }).hits, range(40, 45), 'efSearch raised to cover offset');
  index.dispose();
});

test('search offset with filters and metadata', async () => {
  const index = await lineIndex();
  assert.deepEqual(index.search(ORIGIN, { k: 5, offset: 5, filter: 'parity = "odd"' }).hits, [11, 13, 15, 17, 19]);
  assert.deepEqual(index.search(ORIGIN, { k: 10, offset: 20, filter: 'parity = "odd"' }).hits, [41, 43, 45, 47, 49]);
  const page = index.search(ORIGIN, { k: 4, offset: 8, returnMetadata: true });
  assert.deepEqual(page.metadatas.map((m) => m.position), page.hits);
  assert.throws(() => index.search(ORIGIN, { k: 5, offset: -1 }), /offset must be non-negative/);
  index.dispose();
});

test('similar uses the stored vector and excludes the input document', async () => {
  const index = await lineIndex();
  assert.deepEqual(index.similar(0, { k: 5 }).hits, [1, 2, 3, 4, 5]);
  const withSelf = index.similar(0, { k: 5, excludeInputDocument: false });
  assert.deepEqual(withSelf.hits, [0, 1, 2, 3, 4]);
  assert.equal(withSelf.distances[0], 0);

  const full = index.similar(0, { k: 30 });
  const pages = [0, 10, 20].map((offset) => index.similar(0, { k: 10, offset }));
  assert.deepEqual(pages.flatMap((p) => p.hits), full.hits);
  assert.deepEqual(full.hits, range(1, 31));
  assert.deepEqual(index.similar(0, { k: 10, offset: 45 }).hits, [46, 47, 48, 49], 'partial last page');
  assert.deepEqual(index.similar(0, { k: 10, offset: 50 }).hits, []);

  const filtered = index.similar(20, { k: 4, offset: 2, filter: 'parity = "even"', returnMetadata: true });
  assert.deepEqual(new Set(filtered.hits), new Set([14, 16, 24, 26]));
  assert.deepEqual(filtered.metadatas.map((m) => m.position), filtered.hits);

  assert.throws(() => index.similar(1000, { k: 5 }), /not found/);
  assert.throws(() => index.similar(0, { k: 5, offset: -1 }), /offset must be non-negative/);
  index.dispose();
});

test('similar matches search with the document vector for reduced-precision indexes', async () => {
  for (const vectorType of ['FLOAT16', 'BFLOAT16']) {
    const index = await VectorIndex.create({ dimension: DIM, spaceType: 'L2', vectorType });
    index.addDocuments({ ids, vectors });
    const opts = { k: 10, offset: 3, efSearch: 200, excludeInputDocument: false };
    const stored = index.getDocument(42).vector;
    assert.deepEqual(index.similar(42, opts), index.search(stored, opts), vectorType);
    index.dispose();
  }
});

test('JS wrapper and wasm module carry the VERSION file version', async () => {
  const expected = readFileSync(new URL('../../VERSION', import.meta.url), 'utf8').trim();
  assert.equal(version, expected);
  assert.equal((await init()).version(), expected);
});

# WAL recovery with live search traffic

Demonstrates crash recovery via the write-ahead log (WAL), and that the index
**stays searchable while it is being rebuilt**. When an index is loaded from
disk, the server replays the WAL on a background thread; searches run
concurrently against the partially-rebuilt graph rather than blocking until
replay finishes.

The script drives the whole scenario end to end:

1. **populate** — start the server container, build a large index and add many
   documents. Every add is appended to the WAL on the mounted volume. The index
   is **not** saved.
2. **crash** — `docker kill` the container. The in-memory index is lost; only the
   WAL remains on the volume.
3. **recover** — start a fresh container on the same volume and `POST /load_index`.
   It returns immediately and replays the WAL in the background.
4. **serve** — hammer `/search` from several clients while replay runs, polling
   `/index_status` to watch `currentElements` climb back to the original count.

## Run it

This script starts and stops **its own** server container on port 8685, so make
sure nothing else is using that port. Build the image once from the repo root
(`docker build -t hnswlib_server:local .`), then:

```bash
cd examples/wal-recovery
uv run --with numpy --with requests python run.py
```

Options: `--num-docs 150000`, `--dim 64`, `--batch-size 2000`, `--keep-workdir`.
Set `HNSW_IMAGE` to use a different image, or pass `--binary` to run a local
`./build/bin/server` instead of Docker.

## Sample output

```
[3/3] recover — restart, replay WAL, serve live search ...
  /load_index returned in 3ms (replay running in background)
  live search clients started

    replay    elements   searches ok   errors
       12%      18,300         1,204        0
       41%      61,500         4,118        0
       73%     110,200         7,532        0
      100%     150,000        10,944        0

  replay completed in ~6.2s
  searches served during recovery: 11,210 ok, 0 errors
  final elements: 150,000 (expected 150,000)
  RESULT: index fully recovered with zero failed searches during replay
```

## How it works

- The server records each `add_documents`/`delete_documents` op in
  `indices/<name>.wal` and fsyncs periodically (`WAL_FSYNC_INTERVAL_MS`, set to
  250 ms here so the tail is durable quickly).
- On `/load_index` with a WAL but no snapshot, the server reconstructs the index
  config from the WAL header and replays the entries on a background thread,
  exposing progress through `/index_status` (`walReplayProgress`).
- `/search` only takes a shared read lock, so it runs concurrently with the
  background replay — results simply grow as more documents are re-inserted.

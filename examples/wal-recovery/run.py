"""WAL recovery demo: crash a large index, then serve live search during replay.

Scenario:
  1. populate  - start the server container, build a large index and add many
                 documents (each add is written to the write-ahead log on the
                 mounted volume). The index is NOT saved.
  2. crash     - `docker kill` the container (SIGKILL). The in-memory index is
                 gone; only the WAL survives on the mounted volume.
  3. recover   - start a fresh container on the same volume and call /load_index,
                 which rebuilds the index by replaying the WAL in a background
                 thread.
  4. serve     - hammer /search from several clients WHILE the replay runs,
                 polling /index_status to watch the index fill back up. Searches
                 keep returning (HTTP 200) the whole time and the result set
                 grows as replay progresses.

The server runs in Docker (image set by HNSW_IMAGE, default hnswlib_server:local).
This binds port 8685, so make sure nothing else is using it. Use --binary to run
a local ./build/bin/server instead of a container.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

SERVER = "http://localhost:8685"
INDEX = "recovery_demo"
HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.getenv("HNSW_IMAGE", "hnswlib_server:local")
CONTAINER = "hnsw_recovery"
DEFAULT_BIN = os.path.normpath(os.path.join(HERE, "..", "..", "build", "bin", "server"))


def wait_healthy(proc=None, tries=150):
    for _ in range(tries):
        try:
            if requests.get(f"{SERVER}/health", timeout=0.5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise SystemExit("server did not become healthy")


class DockerServer:
    """Runs the server in a container with ./indices bind-mounted for WAL durability."""

    def __init__(self, workdir):
        self.indices = os.path.join(workdir, "indices")
        os.makedirs(self.indices, exist_ok=True)
        if (
            subprocess.run(
                ["docker", "image", "inspect", IMAGE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            raise SystemExit(
                f"docker image '{IMAGE}' not found.\n"
                "Build it from the repo root:  docker build -t hnswlib_server:local .\n"
                "or set HNSW_IMAGE to a published image, or pass --binary."
            )

    def start(self):
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER,
                "-p",
                "8685:8685",
                "-v",
                f"{self.indices}:/indices",
                "-e",
                "WAL_FSYNC_INTERVAL_MS=250",
                IMAGE,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        wait_healthy()

    def kill(self):
        subprocess.run(
            ["docker", "kill", CONTAINER],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def cleanup(self):
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class BinaryServer:
    """Runs a locally-built ./build/bin/server process (fallback for --binary)."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.binary = os.getenv("HNSW_SERVER_BIN", DEFAULT_BIN)
        self.logfile = os.path.join(workdir, "server.log")
        self.proc = None
        if not os.path.exists(self.binary):
            raise SystemExit(
                f"binary not found at {self.binary}; build it or set HNSW_SERVER_BIN"
            )

    def start(self):
        env = dict(os.environ, WAL_FSYNC_INTERVAL_MS="250")
        log = open(self.logfile, "a")
        self.proc = subprocess.Popen(
            [self.binary], cwd=self.workdir, env=env, stdout=log, stderr=log
        )
        wait_healthy()

    def kill(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def cleanup(self):
        self.kill()


def status():
    return requests.get(f"{SERVER}/index_status/{INDEX}", timeout=5).json()


def populate(num_docs, dim, batch_size):
    requests.post(
        f"{SERVER}/create_index",
        json={
            "indexName": INDEX,
            "dimension": dim,
            "spaceType": "IP",
            "efConstruction": 200,
            "M": 16,
        },
    ).raise_for_status()

    rng = np.random.default_rng(42)
    batches = []
    for start in range(0, num_docs, batch_size):
        end = min(start + batch_size, num_docs)
        vecs = rng.standard_normal((end - start, dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        batches.append((list(range(start, end)), vecs.tolist()))

    def send(batch):
        ids, vecs = batch
        requests.post(
            f"{SERVER}/add_documents",
            json={"indexName": INDEX, "ids": ids, "vectors": vecs},
        ).raise_for_status()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(send, batches))
    while True:
        st = status()
        if not st.get("resizing") and st.get("bufferedWrites", 0) == 0:
            break
        time.sleep(0.05)
    time.sleep(0.6)  # > WAL_FSYNC_INTERVAL_MS so the tail is durable
    elapsed = time.perf_counter() - t0
    count = status()["currentElements"]
    print(f"  added {count} docs in {elapsed:.1f}s ({count / elapsed:,.0f} docs/s)")
    return count


class SearchLoad:
    """Background search clients; counts successes/failures while running."""

    def __init__(self, dim, clients=8):
        self.dim, self.clients = dim, clients
        self.stop = threading.Event()
        self.ok = self.err = 0
        self.lock = threading.Lock()
        self.threads = []

    def _worker(self):
        rng = np.random.default_rng()
        sess = requests.Session()
        while not self.stop.is_set():
            qv = rng.standard_normal(self.dim, dtype=np.float32)
            qv /= np.linalg.norm(qv)
            try:
                r = sess.post(
                    f"{SERVER}/search",
                    json={
                        "indexName": INDEX,
                        "queryVector": qv.tolist(),
                        "k": 10,
                        "efSearch": 64,
                    },
                    timeout=5,
                )
                # availability metric: a 200 is a success even if the still-filling
                # index returns few/no hits. Only transport/HTTP failures are errors.
                with self.lock:
                    if r.ok:
                        self.ok += 1
                    else:
                        self.err += 1
            except requests.RequestException:
                with self.lock:
                    self.err += 1

    def start(self):
        for _ in range(self.clients):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def snapshot(self):
        with self.lock:
            return self.ok, self.err

    def shutdown(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=2)


def recover_and_serve(dim, expected):
    t0 = time.perf_counter()
    requests.post(f"{SERVER}/load_index", json={"indexName": INDEX}).raise_for_status()
    print(
        f"  /load_index returned in {(time.perf_counter() - t0) * 1000:.0f}ms (replay running in background)"
    )

    load = SearchLoad(dim)
    load.start()
    print("  live search clients started\n")
    print(f"  {'replay':>8}  {'elements':>10}  {'searches ok':>12}  {'errors':>7}")

    while True:
        st = status()
        pct = st.get("walReplayProgress", {}).get(
            "percentComplete", 100 if not st.get("replayingWal") else 0
        )
        ok, err = load.snapshot()
        print(f"  {pct:>7}%  {st['currentElements']:>10,}  {ok:>12,}  {err:>7}")
        if not st.get("replayingWal"):
            break
        time.sleep(0.4)

    replay_secs = time.perf_counter() - t0
    time.sleep(0.5)
    ok, err = load.snapshot()
    load.shutdown()
    final = status()["currentElements"]

    print()
    print(f"  replay completed in ~{replay_secs:.1f}s")
    print(f"  searches served during recovery: {ok:,} ok, {err} errors")
    print(f"  final elements: {final:,} (expected {expected:,})")
    if final == expected and err == 0:
        print("  RESULT: index fully recovered with zero failed searches during replay")
    else:
        print("  RESULT: WARNING — see counts above")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=150_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument(
        "--binary", action="store_true", help="run ./build/bin/server instead of Docker"
    )
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    workdir = tempfile.mkdtemp(prefix="hnsw_recovery_")
    print(f"workdir: {workdir}")
    server = BinaryServer(workdir) if args.binary else DockerServer(workdir)
    print(f"server: {'binary' if args.binary else 'docker image ' + IMAGE}")

    try:
        print("\n[1/3] populate — building a large index (no save) ...")
        server.start()
        expected = populate(args.num_docs, args.dim, args.batch_size)

        print("\n[2/3] crash — kill the server (only the WAL survives) ...")
        server.kill()
        print("  server killed")

        print("\n[3/3] recover — restart, replay WAL, serve live search ...")
        server.start()
        recover_and_serve(args.dim, expected)
    finally:
        server.cleanup()
        if args.keep_workdir:
            print(f"\nworkdir kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

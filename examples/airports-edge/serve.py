# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Static file server for the in-browser airport search demo.

Serves this folder's index.html, the WASM build from ../../wasm/dist and the
server's saved index files from ../../indices (or --indices). There is no
search API here: the browser downloads the index once and queries it locally.
"""

import argparse
import functools
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIST = os.path.join(ROOT, "wasm", "dist")


class Handler(SimpleHTTPRequestHandler):
    # module scripts and streaming wasm compilation need exact MIME types
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".bin": "application/octet-stream",
        ".data": "application/octet-stream",
    }

    def __init__(self, *args, indices, **kwargs):
        self.indices = indices
        super().__init__(*args, directory=HERE, **kwargs)

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        name = os.path.basename(path)
        if path.startswith("/indices/"):
            return os.path.join(self.indices, name)
        if name.startswith(("hnswlib-edge", "hnswlib_edge")):
            return os.path.join(DIST, name)
        return super().translate_path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--indices", default=os.path.join(ROOT, "indices"), help="directory holding airports.bin/.json/.data")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(DIST, "hnswlib_edge.wasm")):
        raise SystemExit(f"WASM build not found in {DIST}. Run: ./wasm/build.sh")
    if not os.path.exists(os.path.join(args.indices, "airports.bin")):
        print(f"warning: {args.indices}/airports.bin not found yet; see README.md to export the index")

    handler = functools.partial(Handler, indices=os.path.abspath(args.indices))
    print(f"Open http://localhost:{args.port}  (index files from {args.indices})")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()

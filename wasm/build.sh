#!/usr/bin/env bash
# Builds the WASM module into wasm/dist. Requires an activated Emscripten SDK
# (source /path/to/emsdk/emsdk_env.sh).
set -euo pipefail

WASM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$WASM_DIR")"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-wasm}"

if ! command -v emcmake >/dev/null 2>&1; then
    echo "emcmake not found: install and activate the Emscripten SDK first" >&2
    exit 1
fi

emcmake cmake -S "$WASM_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j

VERSION="$(head -n1 "$ROOT_DIR/VERSION" | tr -d '[:space:]')"
sed "s/__HNSWLIB_VERSION__/$VERSION/" "$WASM_DIR/js/hnswlib-edge.mjs" > "$WASM_DIR/dist/hnswlib-edge.mjs"
cp "$WASM_DIR/js/hnswlib-edge.d.ts" "$WASM_DIR/dist/"
echo "Built hnswlib-edge $VERSION:" $(ls "$WASM_DIR/dist")

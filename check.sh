#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo "Build..."
mkdir -p "$BUILD"
cd "$BUILD"
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
make -j"$JOBS" 2>&1 | grep -E "Built target|error:" || true
echo ""

echo "Tests..."
FAILED=0
for t in test_filters test_data_store test_save_load test_wal; do
    if ./"$t" --gtest_brief=1 2>&1 | tail -1 | grep -q PASSED; then
        echo "  $t: PASSED"
    else
        echo "  $t: FAILED"
        FAILED=1
    fi
done
echo ""

if [ "$FAILED" -eq 1 ]; then
    echo "tests failed, skipping benchmark"
    exit 1
fi

echo "Benchmark..."
./bin/server &
SERVER_PID=$!
sleep 1

cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

for vt in FLOAT32 FLOAT16 BFLOAT16; do
    echo "--- $vt ---"
    uv run "$ROOT/speed_test.py" --vector-type "$vt" 2>&1 | grep -E "Vector type|per document|per second|per query"
    echo ""
done

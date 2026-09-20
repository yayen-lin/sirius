#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-3/prep.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
SRC="$REPO/bench/vector/data/gist1m.duckdb"
OUT="$REPO/bench/vector/data/gist1m_dims.duckdb"

DIMS=(128 256 512 768 960)

rm -f "$OUT"
{
  echo "SET gpu_execution = false;"
  echo "ATTACH '$SRC' AS src (READ_ONLY);"
  for d in "${DIMS[@]}"; do
    echo "CREATE TABLE base_d$d    AS SELECT vec[1:$d]::FLOAT[$d] AS vec FROM src.base;"
    echo "CREATE TABLE queries_d$d AS SELECT vec[1:$d]::FLOAT[$d] AS vec FROM src.queries;"
  done
} | "$CLI" "$OUT"

echo "created in $OUT:"
for d in "${DIMS[@]}"; do
  b=$("$CLI" -csv -noheader "$OUT" -c "SET gpu_execution=false; SELECT count(*) FROM base_d$d;")
  q=$("$CLI" -csv -noheader "$OUT" -c "SET gpu_execution=false; SELECT count(*) FROM queries_d$d;")
  printf "  d%-4s base=%s queries=%s\n" "$d" "$b" "$q"
done

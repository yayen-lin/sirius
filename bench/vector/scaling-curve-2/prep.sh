#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-2/prep.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
SRC="$REPO/bench/vector/data/bigann10m.duckdb"
OUT="$REPO/bench/vector/data/bigann10m_sliced.duckdb"

SIZES=(
  "1:1"
  "10:10"
  "100:100"
  "1k:1000"
  "10k:10000"
  "100k:100000"
  "1m:1000000"
  "10m:10000000"
)

rm -f "$OUT"
{
  echo "SET gpu_execution = false;"
  echo "ATTACH '$SRC' AS src (READ_ONLY);"
  for entry in "${SIZES[@]}"; do
    label="${entry%%:*}"
    n="${entry#*:}"
    echo "CREATE TABLE base_$label AS SELECT vec FROM src.base ORDER BY id LIMIT $n;"
  done
  # scaling curve 6 probes the slices with these queries
  echo "CREATE TABLE queries AS SELECT id, vec FROM src.queries ORDER BY id;"
} | "$CLI" "$OUT"

echo "created in $OUT:"
for entry in "${SIZES[@]}"; do
  label="${entry%%:*}"
  c=$("$CLI" -csv -noheader "$OUT" -c "SET gpu_execution=false; SELECT count(*) FROM base_$label;")
  printf "  %-6s rows=%s\n" "$label" "$c"
done

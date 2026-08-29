#!/usr/bin/env bash
# Build a persistent <name>.duckdb from the parquet files, using the repo's
# own duckdb CLI so the storage format matches the Sirius extension binary.
#
# Usage: bench/vss/build_duckdb.sh <dim> <name>
#   e.g. bench/vss/build_duckdb.sh 960 gist1m
set -euo pipefail

DIM="${1:?dim required, e.g. 960}"
NAME="${2:?name required, e.g. gist1m}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DATA="$REPO/bench/vss/data"
DB="$DATA/$NAME.duckdb"

rm -f "$DB"
"$CLI" "$DB" <<SQL
CREATE TABLE base AS
  SELECT id, vec::FLOAT[$DIM] AS vec FROM read_parquet('$DATA/parquet/base.parquet');
CREATE TABLE queries AS
  SELECT id, vec::FLOAT[$DIM] AS vec FROM read_parquet('$DATA/parquet/queries.parquet');
CREATE TABLE gt AS
  SELECT query_id, rank, neighbor_id, distance FROM read_parquet('$DATA/parquet/gt.parquet');
SQL

# Shipped gt is L2 only. Build the cosine ground truth here (exact top-100 per
# query via DuckDB brute force) so the benchmark can score cosine recall.
echo "building cosine ground truth (brute force over queries x base, a few minutes)..."
"$CLI" "$DB" <<'SQL'
SET gpu_execution=false;
CREATE TABLE gt_cosine AS
SELECT query_id, (row_number() OVER (PARTITION BY query_id ORDER BY d) - 1)::INT AS rank,
       neighbor_id, d AS distance
FROM (
  SELECT q.id AS query_id, t.neighbor_id, t.d
  FROM queries q,
  LATERAL (SELECT b.id AS neighbor_id, array_cosine_distance(b.vec, q.vec) AS d
           FROM base b ORDER BY d LIMIT 100) t
);
SQL

"$CLI" "$DB" <<'SQL'
SELECT 'base' AS t, count(*) AS n FROM base
UNION ALL SELECT 'queries', count(*) FROM queries
UNION ALL SELECT 'gt', count(*) FROM gt
UNION ALL SELECT 'gt_cosine', count(*) FROM gt_cosine;
SQL
echo "built $DB"

# Lance datasets (raw + normalized) and their IVF_FLAT indexes.
"$(dirname "$0")/build_lance.sh" "$NAME"

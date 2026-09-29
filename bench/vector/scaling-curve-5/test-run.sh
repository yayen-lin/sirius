#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-5/test-run.sh

set -euo pipefail

K=10
PROBE=1000

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SIRIUS_CLI="$REPO/build/release/duckdb"
DUCKDB_CLI="$HOME/andy/duckdb-v1.5.5/duckdb"
DB="$REPO/bench/vector/data/sift1m.duckdb"
PY="$REPO/bench/vector/scaling-curve-5/.venv/bin/python"

echo "db=$DB k=$K probe=$PROBE"

# --- Sirius ---
"$SIRIUS_CLI" "$DB" <<SQL
SET gpu_execution = true;

.timer on
SELECT count(*) AS rows_sirius
FROM (SELECT id, vec FROM queries LIMIT $PROBE) l,
LATERAL (SELECT r.id, array_distance(l.vec, r.vec) AS dist
         FROM base r ORDER BY array_distance(l.vec, r.vec) LIMIT $K) rr;
.timer off
SQL

# --- DuckDB brute force ---
"$DUCKDB_CLI" "$DB" <<SQL
.timer on
SELECT count(*) AS rows_duckdb
FROM (SELECT id, vec FROM queries LIMIT $PROBE) l,
LATERAL (SELECT r.id, array_distance(l.vec, r.vec) AS dist
         FROM base r ORDER BY array_distance(l.vec, r.vec) LIMIT $K) rr;
.timer off
SQL

# --- DuckDB HNSW index accelerated lateral join ---
"$DUCKDB_CLI" <<SQL
LOAD vss;
ATTACH '$DB' AS src (READ_ONLY);
CREATE TABLE base AS SELECT id, vec FROM src.base;
CREATE TABLE queries AS SELECT id, vec FROM src.queries LIMIT $PROBE;

CREATE INDEX base_hnsw ON base USING HNSW (vec) WITH (metric='l2sq', ef_construction=128, M=16);

.timer on
SELECT count(*) AS rows_hnsw_join
FROM queries l,
LATERAL (SELECT r.id, array_distance(l.vec, r.vec) AS dist
         FROM base r ORDER BY array_distance(l.vec, r.vec) LIMIT $K) rr;
.timer off
SQL

# --- cuVS brute force ---
"$PY" "$REPO/bench/vector/scaling-curve-5/cuvs_knn.py" --db "$DB" --k "$K" --probe "$PROBE"

# --- FAISS GPU brute force ---
# system CUDA 12.6 on LD_LIBRARY_PATH shadows the venv's newer cuBLAS and breaks the faiss import
LD_LIBRARY_PATH= "$PY" "$REPO/bench/vector/scaling-curve-5/faiss_knn.py" --db "$DB" --k "$K" --probe "$PROBE"

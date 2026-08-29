#!/usr/bin/env bash
# Usage: ./bench/vss/enn.sh

set -euo pipefail

QID="999"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vss/data/gist1m.duckdb"

DIM=$("$CLI" "$DB" -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
QVEC=$("$CLI" "$DB" -noheader -list -c "SELECT vec FROM queries WHERE id=$QID;")
Q="${QVEC}::FLOAT[$DIM]"

# Lance exact search is L2-only, so cosine uses the unit-normalized dataset and a unit-normalized query
LANCE_L2="$REPO/bench/vss/data/gist1m_base.lance"
LANCE_COS="$REPO/bench/vss/data/gist1m_base_norm.lance"
QVEC_NORM=$("$CLI" "$DB" -noheader -list -c "SET gpu_execution=false; SELECT list_transform(v, lambda x: (x/nrm)::FLOAT) FROM (SELECT vec::DOUBLE[] AS v, sqrt(list_sum(list_transform(vec::DOUBLE[], lambda y: y*y))) AS nrm FROM queries WHERE id=$QID);")
QL="${QVEC}::FLOAT[]"        # raw query as LIST(FLOAT) for lance_vector_search
QN="${QVEC_NORM}::FLOAT[]"   # normalized query for the cosine dataset

echo "db=$DB dim=$DIM query_id=$QID"

"$CLI" "$DB" <<SQL

-- ===== Sirius =====
-- Warmup & Setup
SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10, metric => 'l2', use_index => false, output_columns => ['id']);

.timer on
-- l2
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'l2', use_index => false, output_columns => ['id']);
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'l2', use_index => false, output_columns => ['id']);

-- cosine
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'cosine', use_index => false, output_columns => ['id']);
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'cosine', use_index => false, output_columns => ['id']);
.timer off

-- ===== DuckDB =====
-- Warmup & Setup
SET gpu_execution = false;
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);

.timer on
-- l2
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 100);

-- cosine
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 10);
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 100);
.timer off

-- ===== Lance =====
-- Warmup & Setup
LOAD lance;
SELECT count(*) FROM lance_vector_search('$LANCE_L2', 'vec', $QL, k => 10, use_index => false);

.timer on
-- l2
SELECT count(*) FROM lance_vector_search('$LANCE_L2', 'vec', $QL, k => 10,  use_index => false);
SELECT count(*) FROM lance_vector_search('$LANCE_L2', 'vec', $QL, k => 100, use_index => false);

-- cosine
SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 10,  use_index => false);
SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 100, use_index => false);
.timer off
SQL

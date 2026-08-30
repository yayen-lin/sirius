#!/usr/bin/env bash
# Usage: ./bench/vss/ann.sh

set -euo pipefail

export TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

QID="999"
NLIST=1024        # IVF partitions; must match on both engines
NPROBES=32        # lists probed per query; must match on both engines
HNSW_M=16         # DuckDB HNSW graph degree (build-time)
HNSW_EFC=128      # DuckDB HNSW ef_construction (build-time)
HNSW_EFS=64       # DuckDB HNSW ef_search (query-time recall knob, HNSW's nprobes-equivalent)
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vss/data/gist1m.duckdb"

DIM=$("$CLI" "$DB" -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
QVEC=$("$CLI" "$DB" -noheader -list -c "SELECT vec FROM queries WHERE id=$QID;")
Q="${QVEC}::FLOAT[$DIM]"

# lance native cosine returns L2 at 1M scale, so cosine uses the pre-normalized dataset + l2 index + normalized query
LANCE="$REPO/bench/vss/data/gist1m_base.lance"
LANCE_COS="$REPO/bench/vss/data/gist1m_base_norm.lance"
QL="${QVEC}::FLOAT[]"
QVEC_NORM=$("$CLI" "$DB" -noheader -list -c "SET gpu_execution=false; SELECT list_transform(v, lambda x:(x/nrm)::FLOAT) FROM (SELECT vec::DOUBLE[] AS v, sqrt(list_sum(list_transform(vec::DOUBLE[], lambda y:y*y))) AS nrm FROM queries WHERE id=$QID);")
QN="${QVEC_NORM}::FLOAT[]"

echo "db=$DB dim=$DIM query_id=$QID n_lists=$NLIST n_probes=$NPROBES"

"$CLI" "$DB" <<SQL
-- ===== Sirius =====
-- Warmup & Setup
SELECT count(*) FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');
SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'l2',     index_type => 'ivf_flat', n_lists => $NLIST);
SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'cosine', index_type => 'ivf_flat', n_lists => $NLIST);
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10, metric => 'l2', use_index => true, n_probes => $NPROBES, output_columns => ['id']);

.timer on
-- l2
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'l2', use_index => true, n_probes => $NPROBES, output_columns => ['id']);
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'l2', use_index => true, n_probes => $NPROBES, output_columns => ['id']);

-- cosine
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'cosine', use_index => true, n_probes => $NPROBES, output_columns => ['id']);
SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'cosine', use_index => true, n_probes => $NPROBES, output_columns => ['id']);
.timer off

SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'l2');
SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'cosine');
SELECT * FROM unpin_table('base');

-- ===== Lance =====
-- l2 setup
LOAD lance;
CREATE INDEX vec_idx ON '$LANCE' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='l2');
SELECT count(*) FROM lance_vector_search('$LANCE', 'vec', $QL, k => 10, use_index => true, nprobs => $NPROBES);

-- l2 search
.timer on
SELECT count(*) FROM lance_vector_search('$LANCE', 'vec', $QL, k => 10,  use_index => true, nprobs => $NPROBES);
SELECT count(*) FROM lance_vector_search('$LANCE', 'vec', $QL, k => 100, use_index => true, nprobs => $NPROBES);
.timer off

-- l2 cleanup
DROP INDEX vec_idx ON '$LANCE';
SELECT * FROM __lance_cleanup_old_versions('$LANCE', '{"older_than_seconds":0,"delete_unverified":true}');

-- cosine setup
CREATE INDEX vec_idx ON '$LANCE_COS' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='l2');
SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 10, use_index => true, nprobs => $NPROBES);

-- cosine search
.timer on
SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 10,  use_index => true, nprobs => $NPROBES);
SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 100, use_index => true, nprobs => $NPROBES);
.timer off

-- cosine cleanup
DROP INDEX vec_idx ON '$LANCE_COS';
SELECT * FROM __lance_cleanup_old_versions('$LANCE_COS', '{"older_than_seconds":0,"delete_unverified":true}');

-- ===== DuckDB HNSW =====
-- turn off Sirius interception so the ORDER BY runs on plain DuckDB and hits the HNSW index
SET gpu_execution=false;
LOAD vss;
-- persistent (on-disk) DB needs this flag before an HNSW index can be built
SET hnsw_enable_experimental_persistence=true;

-- l2 setup & warmup (native cosine works here, no normalized dataset needed)
CREATE INDEX h_l2  ON base USING HNSW (vec) WITH (metric='l2sq',   ef_construction=$HNSW_EFC, M=$HNSW_M);
CREATE INDEX h_cos ON base USING HNSW (vec) WITH (metric='cosine', ef_construction=$HNSW_EFC, M=$HNSW_M);
SET hnsw_ef_search=$HNSW_EFS;
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);

.timer on
-- l2
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 100);

-- cosine
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 10);
SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 100);
.timer off

-- cleanup
DROP INDEX h_l2;
DROP INDEX h_cos;
SQL

#!/usr/bin/env bash
# Usage: ./bench/vss/ann_bench_latency.sh
set -euo pipefail

export TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

REPS=10
QID="999"
NLIST=1024
NPROBES=32
HNSW_M=16
HNSW_EFC=128
HNSW_EFS=128
CLI=./build/release/duckdb
DB=bench/vss/data/gist1m.duckdb
LANCE=bench/vss/data/gist1m_base.lance
LANCE_COS=bench/vss/data/gist1m_base_norm.lance

DIM=$($CLI $DB -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
QVEC=$($CLI $DB -noheader -list -c "SELECT vec FROM queries WHERE id=$QID;")
Q="${QVEC}::FLOAT[$DIM]"
QL="${QVEC}::FLOAT[]"
QVEC_NORM=$($CLI $DB -noheader -list -c "SET gpu_execution=false; SELECT list_transform(v, lambda x:(x/nrm)::FLOAT) FROM (SELECT vec::DOUBLE[] AS v, sqrt(list_sum(list_transform(vec::DOUBLE[], lambda y:y*y))) AS nrm FROM queries WHERE id=$QID);")
QN="${QVEC_NORM}::FLOAT[]"

echo "db=$DB dim=$DIM query_id=$QID n_lists=$NLIST n_probes=$NPROBES"

# Reads the CLI output on stdin and prints min/mean of the '.timer' real
# seconds, grouped by the '@@label' markers printed before each block.
report() {
  awk '
    /^@@/  { block = substr($0, 3); next }
    /real/ { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
             n[block]++; sum[block]+=t; if (!(block in mn) || t<mn[block]) mn[block]=t }
    END    { for (b in n) { split(b, a, " ")   # b = "engine metric"
               idx = (a[1]=="duckdb") ? "HNSW" : "IVF-FLAT"
               printf "%-8s %-8s %-8s %-9s %8.1f %8.1f\n",
                      a[1], "ann", a[2], idx, 1000*mn[b], 1000*sum[b]/n[b] } }
  ' | sort | { printf "\n%-8s %-8s %-8s %-9s %8s %8s\n" engine search metric index min_ms mean_ms; cat; }
}

{

# Sirius warmup & setup
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
echo "SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'l2',     index_type => 'ivf_flat', n_lists => $NLIST);"
echo "SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'cosine', index_type => 'ivf_flat', n_lists => $NLIST);"
echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10, metric => 'l2', use_index => true, n_probes => $NPROBES, output_columns => ['id']);"
echo ".timer on"

echo ".print @@sirius l2"
for i in $(seq $REPS); do
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'l2', use_index => true, n_probes => $NPROBES, output_columns => ['id']);"
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'l2', use_index => true, n_probes => $NPROBES, output_columns => ['id']);"
done

echo ".print @@sirius cosine"
for i in $(seq $REPS); do
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'cosine', use_index => true, n_probes => $NPROBES, output_columns => ['id']);"
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'cosine', use_index => true, n_probes => $NPROBES, output_columns => ['id']);"
done
echo ".timer off"

# Sirius cleanup
echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'l2');"
echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'cosine');"
echo "SELECT * FROM unpin_table('base');"

# Lance warmup & setup
echo "LOAD lance;"
echo "CREATE INDEX vec_idx ON '$LANCE' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='l2');"
echo "SELECT count(*) FROM lance_vector_search('$LANCE', 'vec', $QL, k => 10, use_index => true, nprobs => $NPROBES);"
echo ".timer on"

# search
echo ".print @@lance l2"
for i in $(seq $REPS); do
  echo "SELECT count(*) FROM lance_vector_search('$LANCE', 'vec', $QL, k => 10,  use_index => true, nprobs => $NPROBES);"
  echo "SELECT count(*) FROM lance_vector_search('$LANCE', 'vec', $QL, k => 100, use_index => true, nprobs => $NPROBES);"
done
echo ".timer off"

# cleanup
echo "DROP INDEX vec_idx ON '$LANCE';"
echo "SELECT * FROM __lance_cleanup_old_versions('$LANCE', '{\"older_than_seconds\":0,\"delete_unverified\":true}');"

# warmup & setup
echo "CREATE INDEX vec_idx ON '$LANCE_COS' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='l2');"
echo "SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 10, use_index => true, nprobs => $NPROBES);"
echo ".timer on"

# search
echo ".print @@lance cosine"
for i in $(seq $REPS); do
  echo "SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 10,  use_index => true, nprobs => $NPROBES);"
  echo "SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 100, use_index => true, nprobs => $NPROBES);"
done
echo ".timer off"

# Lance cleanup
echo "DROP INDEX vec_idx ON '$LANCE_COS';"
echo "SELECT * FROM __lance_cleanup_old_versions('$LANCE_COS', '{\"older_than_seconds\":0,\"delete_unverified\":true}');"

# DuckDB HNSW warmup & setup
echo "SET gpu_execution=false;"
echo "LOAD vss;"
echo "SET hnsw_enable_experimental_persistence=true;"   # persistent DB needs this to build an HNSW index
echo "CREATE INDEX h_l2  ON base USING HNSW (vec) WITH (metric='l2sq',   ef_construction=$HNSW_EFC, M=$HNSW_M);"
echo "CREATE INDEX h_cos ON base USING HNSW (vec) WITH (metric='cosine', ef_construction=$HNSW_EFC, M=$HNSW_M);"
echo "SET hnsw_ef_search=$HNSW_EFS;"
echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);"
echo ".timer on"

echo ".print @@duckdb l2"
for i in $(seq $REPS); do
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);"
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 100);"
done

echo ".print @@duckdb cosine"
for i in $(seq $REPS); do
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 10);"
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 100);"
done
echo ".timer off"

# DuckDB HNSW cleanup
echo "DROP INDEX h_l2;"
echo "DROP INDEX h_cos;"

} | $CLI $DB | report

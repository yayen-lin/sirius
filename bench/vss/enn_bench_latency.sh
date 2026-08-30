#!/usr/bin/env bash
# Usage: ./bench/vss/enn_bench_latency.sh
set -euo pipefail

REPS=10
QID="999"
CLI=./build/release/duckdb
DB=bench/vss/data/gist1m.duckdb

DIM=$($CLI $DB -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
QVEC=$($CLI $DB -noheader -list -c "SELECT vec FROM queries WHERE id=$QID;")
Q="${QVEC}::FLOAT[$DIM]"

# Lance exact search is L2-only, so cosine uses the unit-normalized dataset and a unit-normalized query
LANCE_L2=bench/vss/data/gist1m_base.lance
LANCE_COS=bench/vss/data/gist1m_base_norm.lance
QVEC_NORM=$($CLI $DB -noheader -list -c "SET gpu_execution=false; SELECT list_transform(v, lambda x: (x/nrm)::FLOAT) FROM (SELECT vec::DOUBLE[] AS v, sqrt(list_sum(list_transform(vec::DOUBLE[], lambda y: y*y))) AS nrm FROM queries WHERE id=$QID);")
QL="${QVEC}::FLOAT[]"        # raw query as LIST(FLOAT) for lance_vector_search
QN="${QVEC_NORM}::FLOAT[]"   # normalized query for the cosine dataset

echo "db=$DB dim=$DIM query_id=$QID"

# Reads the CLI output on stdin and prints min/mean of the '.timer' real
# seconds, grouped by the '@@label' markers printed before each block.
report() {
  awk '
    /^@@/  { block = substr($0, 3); next }
    /real/ { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
             n[block]++; sum[block]+=t; if (!(block in mn) || t<mn[block]) mn[block]=t }
    END    { for (b in n) { split(b, a, " ")   # b = "engine metric k"
               printf "%-8s %-8s %-8s %4d %8.1f %8.1f\n",
                      a[1], "enn", a[2], a[3], 1000*mn[b], 1000*sum[b]/n[b] } }
  ' | sort -k1,1 -k3,3 -k4,4n | { printf "\n%-8s %-8s %-8s %4s %8s %8s\n" engine search metric k min_ms mean_ms; cat; }
}

{

# Sirius warmup & setup
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10, metric => 'l2', use_index => false, output_columns => ['id']);"
echo ".timer on"

for i in $(seq $REPS); do
  echo ".print @@sirius l2 10"
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'l2', use_index => false, output_columns => ['id']);"
  echo ".print @@sirius l2 100"
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'l2', use_index => false, output_columns => ['id']);"
done

for i in $(seq $REPS); do
  echo ".print @@sirius cosine 10"
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 10,  metric => 'cosine', use_index => false, output_columns => ['id']);"
  echo ".print @@sirius cosine 100"
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', $Q, k => 100, metric => 'cosine', use_index => false, output_columns => ['id']);"
done
echo ".timer off"

# DuckDB warmup & setup
echo "SET gpu_execution = false;"
echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);"
echo ".timer on"

for i in $(seq $REPS); do
  echo ".print @@duckdb l2 10"
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 10);"
  echo ".print @@duckdb l2 100"
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, $Q) LIMIT 100);"
done

for i in $(seq $REPS); do
  echo ".print @@duckdb cosine 10"
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 10);"
  echo ".print @@duckdb cosine 100"
  echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $Q) LIMIT 100);"
done
echo ".timer off"

# Lance warmup & setup
echo "LOAD lance;"
echo "SELECT count(*) FROM lance_vector_search('$LANCE_L2', 'vec', $QL, k => 10, use_index => false);"
echo ".timer on"

for i in $(seq $REPS); do
  echo ".print @@lance l2 10"
  echo "SELECT count(*) FROM lance_vector_search('$LANCE_L2', 'vec', $QL, k => 10,  use_index => false);"
  echo ".print @@lance l2 100"
  echo "SELECT count(*) FROM lance_vector_search('$LANCE_L2', 'vec', $QL, k => 100, use_index => false);"
done

for i in $(seq $REPS); do
  echo ".print @@lance cosine 10"
  echo "SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 10,  use_index => false);"
  echo ".print @@lance cosine 100"
  echo "SELECT count(*) FROM lance_vector_search('$LANCE_COS', 'vec', $QN, k => 100, use_index => false);"
done
echo ".timer off"
} | $CLI $DB | report

#!/usr/bin/env bash
# Usage: ./bench/vss/enn_bench_throughput.sh
set -euo pipefail

QID=0           # first query id
NQ=100          # num of queries per run
K=10            # top-k
CLI=./build/release/duckdb
DB=bench/vss/data/gist1m.duckdb

DIM=$($CLI $DB -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
# The NQ query vectors as array literals
mapfile -t QVECS < <($CLI $DB -noheader -list -c "SELECT vec FROM queries WHERE id >= $QID ORDER BY id LIMIT $NQ;")

echo "db=$DB dim=$DIM qid=$QID nq=$NQ k=$K"

# Reads the CLI output on stdin and prints queries/sec, grouped by the
# '@@engine metric batch' markers. QPS = sum(batch) / sum(.timer real seconds).
report() {
  awk '
    /^@@/  { split(substr($0, 3), a, " "); key=a[1]" "a[2]; batch=a[3]; next }
    /real/ { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
             q[key]+=batch; s[key]+=t }
    END    { for (b in q) { split(b, a, " ")   # b = "engine metric"
               printf "%-8s %-8s %-8s %8d %10.1f\n", a[1], "enn", a[2], q[b], q[b]/s[b] } }
  ' | sort | { printf "\n%-8s %-8s %-8s %8s %10s\n" engine search metric queries qps; cat; }
}

{

# Sirius warmup & setup
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', ${QVECS[0]}::FLOAT[$DIM], k => $K, metric => 'l2', use_index => false, output_columns => ['id']);"
echo ".timer on"

echo ".print @@sirius l2 1"
for v in "${QVECS[@]}"; do
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', ${v}::FLOAT[$DIM], k => $K, metric => 'l2', use_index => false, output_columns => ['id']);"
done

echo ".print @@sirius cosine 1"
for v in "${QVECS[@]}"; do
  echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', ${v}::FLOAT[$DIM], k => $K, metric => 'cosine', use_index => false, output_columns => ['id']);"
done
echo ".timer off"

# DuckDB warmup & setup
echo "SET gpu_execution = false;"
echo "SELECT count(*) FROM (SELECT q.id FROM (SELECT * FROM queries LIMIT 5) q, LATERAL (SELECT b.id FROM base b ORDER BY array_distance(b.vec, q.vec) LIMIT $K) t);"
echo ".timer on"

echo ".print @@duckdb l2 $NQ"
echo "SELECT count(*) FROM (SELECT q.id FROM (SELECT * FROM queries WHERE id >= $QID ORDER BY id LIMIT $NQ) q, LATERAL (SELECT b.id FROM base b ORDER BY array_distance(b.vec, q.vec) LIMIT $K) t);"

echo ".print @@duckdb cosine $NQ"
echo "SELECT count(*) FROM (SELECT q.id FROM (SELECT * FROM queries WHERE id >= $QID ORDER BY id LIMIT $NQ) q, LATERAL (SELECT b.id FROM base b ORDER BY array_cosine_distance(b.vec, q.vec) LIMIT $K) t);"
echo ".timer off"
} | $CLI $DB | report

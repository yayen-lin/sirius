#!/usr/bin/env bash
# Usage: ./bench/vjoin/scenario2/latency.sh

set -euo pipefail

REPS=10
PROBE=100
EPS_LIST=(250 150)
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vjoin/data/bigann100m.duckdb"

echo "db=$DB probe=$PROBE eps=[${EPS_LIST[*]}] reps=$REPS"

# Reads the CLI output on stdin and prints min/mean/max of the '.timer' real
# seconds, grouped by the '@@engine eps' markers printed before each timed query.
report() {
  awk '
    /^@@/  { block = substr($0, 3); want=1; next }
    /real/ { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
             n[block]++; sum[block]+=t
             if (!(block in mn) || t<mn[block]) mn[block]=t
             if (!(block in mx) || t>mx[block]) mx[block]=t
             next }
    # capture the count(*) row, but only right after a marker so the untimed
    # warmup runs (which have no marker) cannot overwrite a real block
    want && $0 ~ /[0-9]/ && $0 !~ /[A-Za-z]/ { r=$0; gsub(/[^0-9]/,"",r); if (r!="") { rows[block]=r; want=0 } }
    END    { for (b in n) { split(b, a, " ")
               printf "%-8s %6s %4d %11s %11.1f %11.1f %11.1f\n",
                      a[1], a[2], n[b], (b in rows ? rows[b] : "-"),
                      1000*mn[b], 1000*sum[b]/n[b], 1000*mx[b] } }
  ' | sort -k1,1 -k2,2r | { printf "\n%-8s %6s %4s %11s %11s %11s %11s\n" \
                              engine eps reps rows min_ms mean_ms max_ms; cat; }
}

query() { echo "SELECT count(*) FROM (SELECT vec FROM queries LIMIT $PROBE) l JOIN base r ON array_distance(l.vec, r.vec) <= $1;"; }

{
# Sirius warmup and setup
echo "SET gpu_execution = true;"
query "${EPS_LIST[0]}"

# Sirius eval
for eps in "${EPS_LIST[@]}"; do
  echo ".timer on"
  for i in $(seq $REPS); do
    echo ".print @@sirius $eps"
    query "$eps"
  done
  echo ".timer off"
done

# DuckDB warmup and setup
echo "SET gpu_execution = false;"
query "${EPS_LIST[0]}"

# DuckDB eval
for eps in "${EPS_LIST[@]}"; do
  echo ".timer on"
  for i in $(seq $REPS); do
    echo ".print @@duckdb $eps"
    query "$eps"
  done
  echo ".timer off"
done
} | "$CLI" "$DB" | report

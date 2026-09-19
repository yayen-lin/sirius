#!/usr/bin/env bash
# Usage: ./bench/vector/scenario2/latency.sh

set -euo pipefail

REPS=1
PROBE=100
EPS_LIST=(250 150)
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vector/data/bigann100m.duckdb"

echo "db=$DB probe=$PROBE eps=[${EPS_LIST[*]}] reps=$REPS"

DIM=$("$CLI" -csv -noheader "$DB" -c "SELECT len(vec) FROM base LIMIT 1;")
OUTER=$PROBE
INNER=$("$CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM base;")
PAIRS=$((OUTER * INNER))
DIST_OPS=$((PAIRS * DIM))
PAIRS_G=$(printf '%g' "$PAIRS")
DIST_OPS_G=$(printf '%g' "$DIST_OPS")

# Reads the CLI output on stdin and prints min/mean/max of the '.timer' real
# seconds, grouped by the '@@engine eps' markers printed before each timed query.
report() {
  awk -v dim="$DIM" -v probe="$OUTER" -v corpus="$INNER" -v pairs="$PAIRS_G" -v dist_ops="$DIST_OPS_G" '
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
               mean = 1000*sum[b]/n[b]           # mean wall-clock of one query, ms
               dops = dist_ops + 0               # coerce the %g string to a number
               mspo = (dops>0 ? mean/dops : 0)   # ms spent per distance element-op
               gops = (mean>0 ? dops/mean/1e6 : 0) # billion distance element-ops per second
               printf "%-8s %6s %4d %11s %5s %11s %12s %9s %9s %11.1f %11.1f %11.1f %11s %24.2f\n",
                      a[1], a[2], n[b], (b in rows ? rows[b] : "-"),
                      dim, probe, corpus, pairs, dist_ops,
                      1000*mn[b], mean, 1000*mx[b],
                      sprintf("%.3g", mspo), gops } }
  ' | sort -k1,1 -k2,2r | { printf "\n%-8s %6s %4s %11s %5s %11s %12s %9s %9s %11s %11s %11s %11s %24s\n" \
                              engine eps reps rows dim probe_rows corpus_rows pairs dist_ops min_ms mean_ms max_ms ms_per_ops billion_dist_ops_per_sec; cat; }
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

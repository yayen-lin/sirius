#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-1/run.sh
# Title: scaling curve 1, fixed probe, growing corpus, duckdb vs sirius

set -euo pipefail

REPS=10
EPS=250
TIMEOUT=20m
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"

DATASETS=(
  "bigann1m:$REPO/bench/vector/data/sift1m.duckdb"
  "bigann10m:$REPO/bench/vector/data/bigann10m.duckdb"
  "bigann100m:$REPO/bench/vector/data/bigann100m.duckdb"
  "bigann1b:$REPO/bench/vector/data/bigann1b.duckdb"
)

echo "probe=queries eps=$EPS reps=$REPS datasets=[$(for d in "${DATASETS[@]}"; do printf '%s ' "${d%%:*}"; done)]"

query() { echo "SELECT count(*) FROM queries l JOIN base r ON array_distance(l.vec, r.vec) <= $EPS;"; }
warmup() { echo "SELECT count(*) FROM (SELECT vec FROM queries LIMIT 1) l JOIN base r ON array_distance(l.vec, r.vec) <= $EPS;"; }
rows() {
  awk -v dim="$DIM" -v probe="$OUTER" -v corpus="$INNER" -v pairs="$PAIRS" -v dist_ops="$DIST_OPS" '
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
               printf "%-8s %10s %4d %11s %5s %11s %12s %11g %11g %11.1f %11.1f %11.1f %15.10f %24.2f\n",
                      a[1], a[2], n[b], (b in rows ? rows[b] : "-"),
                      dim, probe, corpus, pairs+0, dist_ops+0,
                      1000*mn[b], mean, 1000*mx[b],
                      mspo, gops } }
  '
}

BUF="$(mktemp)"
trap 'rm -f "$BUF"' EXIT

# Emit a placeholder when DuckDB times out
emit_duckdb_row() {
  echo "duckdb $LABEL: $1" >&2
  printf "%-8s %10s %4d %11s %5s %11s %12s %11g %11g %11s %11s %11s %15s %24s\n" \
    duckdb "$LABEL" "$REPS" "$1" "$DIM" "$OUTER" "$INNER" "$PAIRS" "$DIST_OPS" \
    "$1" "$1" "$1" "-" "-" >> "$BUF"
}

DUCKDB_DEAD=0

for entry in "${DATASETS[@]}"; do
  LABEL="${entry%%:*}"
  DB="${entry#*:}"
  echo "running $LABEL ($DB)" >&2

  DIM=$("$CLI" -csv -noheader "$DB" -c "SELECT len(vec) FROM base LIMIT 1;")
  INNER=$("$CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM base;")
  OUTER=$("$CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM queries;")
  PAIRS=$((OUTER * INNER))
  DIST_OPS=$((PAIRS * DIM))

  # --- Sirius ---
  {
    echo "SET gpu_execution = true;"
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@sirius $LABEL"
      query
    done
    echo ".timer off"
  } | "$CLI" "$DB" | rows >> "$BUF"

  # --- DuckDB ---
  before=$(wc -l < "$BUF")
  if [ "$DUCKDB_DEAD" -eq 1 ]; then
    emit_duckdb_row TIMEOUT
  elif timeout "$TIMEOUT" "$CLI" "$DB" <<SQL | rows >> "$BUF"
SET gpu_execution = false;
$(warmup)
.timer on
$(for i in $(seq $REPS); do echo ".print @@duckdb $LABEL"; query; done)
.timer off
SQL
  then :
  else
    code=$?
    if [ "$(wc -l < "$BUF")" -gt "$before" ]; then
      [ "$code" -eq 124 ] && DUCKDB_DEAD=1 || true
    elif [ "$code" -eq 124 ]; then
      DUCKDB_DEAD=1
      emit_duckdb_row TIMEOUT
    else
      emit_duckdb_row "ERROR($code)"
    fi
  fi
done

{ printf "\n%-8s %10s %4s %11s %5s %11s %12s %11s %11s %11s %11s %11s %15s %24s\n" \
    engine corpus reps rows dim probe_rows corpus_rows pairs dist_ops min_ms mean_ms max_ms ms_per_op billion_dist_ops_per_sec
  sort -k1,1 -k7,7n "$BUF"; }

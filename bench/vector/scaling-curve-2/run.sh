#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-2/run.sh
# Title: scaling curve 2, self-join at growing size, duckdb vs sirius

set -euo pipefail

REPS=1
EPS=250
TIMEOUT=20m
SIRIUS_TIMEOUT=6h
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
SRC="$REPO/bench/vector/data/bigann10m.duckdb"

SIZES=(
  "1k:1000"
  "10k:10000"
  "100k:100000"
  "1m:1000000"
#  "10m:10000000"
)

echo "selfjoin(sliced) src=$(basename "$SRC") eps=$EPS reps=$REPS sizes=[$(for s in "${SIZES[@]}"; do printf '%s ' "${s%%:*}"; done)]"

query()  { echo "SELECT count(*) FROM (SELECT vec FROM base LIMIT $1) l JOIN (SELECT vec FROM base LIMIT $1) r ON array_distance(l.vec, r.vec) <= $EPS;"; }
warmup() { echo "SELECT count(*) FROM (SELECT vec FROM base LIMIT 1) l JOIN (SELECT vec FROM base LIMIT $1) r ON array_distance(l.vec, r.vec) <= $EPS;"; }
rows() {
  awk -v dim="$DIM" -v probe="$N" -v corpus="$N" -v pairs="$PAIRS" -v dist_ops="$DIST_OPS" '
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

# Emit a placeholder when an engine times out
emit_row() {
  echo "$1 $LABEL: $2" >&2
  printf "%-8s %10s %4d %11s %5s %11s %12s %11g %11g %11s %11s %11s %15s %24s\n" \
    "$1" "$LABEL" "$REPS" "$2" "$DIM" "$N" "$N" "$PAIRS" "$DIST_OPS" \
    "$2" "$2" "$2" "-" "-" >> "$BUF"
}

DUCKDB_DEAD=0
SIRIUS_DEAD=0

DIM=$("$CLI" -csv -noheader "$SRC" -c "SELECT len(vec) FROM base LIMIT 1;")

for entry in "${SIZES[@]}"; do
  LABEL="${entry%%:*}"
  N="${entry#*:}"
  echo "running $LABEL (N=$N)" >&2

  PAIRS=$((N * N))
  DIST_OPS=$((PAIRS * DIM))

  # --- Sirius ---
  if [ "$SIRIUS_DEAD" -eq 1 ]; then
    emit_row sirius TIMEOUT
  elif {
    echo "SET gpu_execution = true;"
    warmup "$N"
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@sirius $LABEL"
      query "$N"
    done
    echo ".timer off"
  } | timeout "$SIRIUS_TIMEOUT" "$CLI" "$SRC" | rows >> "$BUF"
  then :
  else
    code=$?
    if [ "$code" -eq 124 ]; then SIRIUS_DEAD=1; emit_row sirius TIMEOUT
    else emit_row sirius "ERROR($code)"; fi
  fi

  # --- DuckDB ---
  if [ "$DUCKDB_DEAD" -eq 1 ]; then
    emit_row duckdb TIMEOUT
  elif {
    echo "SET gpu_execution = false;"
    warmup "$N"
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@duckdb $LABEL"
      query "$N"
    done
    echo ".timer off"
  } | timeout "$TIMEOUT" "$CLI" "$SRC" | rows >> "$BUF"
  then :
  else
    code=$?
    if [ "$code" -eq 124 ]; then DUCKDB_DEAD=1; emit_row duckdb TIMEOUT
    else emit_row duckdb "ERROR($code)"; fi
  fi
done

{ printf "\n%-8s %10s %4s %11s %5s %11s %12s %11s %11s %11s %11s %11s %15s %24s\n" \
    engine size reps rows dim n_left n_right pairs dist_ops min_ms mean_ms max_ms ms_per_op billion_dist_ops_per_sec
  sort -k7,7n "$BUF"; }

#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-1/run.sh
# Title: scaling curve 1, fixed probe, growing corpus, duckdb vs sirius

set -euo pipefail

REPS=10
EPS=250
TIMEOUT=1800
SIRIUS_TIMEOUT=7200
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"

DATASETS=(
  "bigann1m:$REPO/bench/vector/data/sift1m.duckdb"
  "bigann10m:$REPO/bench/vector/data/bigann10m.duckdb"
  "bigann100m:$REPO/bench/vector/data/bigann100m.duckdb"
#  "bigann1b:$REPO/bench/vector/data/bigann1b.duckdb"
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

# a watchdog times out a query if it goes over the specified time
watchdog() {
  local limit=$1 pidf rc=0
  shift
  pidf=$(mktemp)
  # line buffered, so a line shows up when its search ends, not when the buffer fills
  ( echo "$BASHPID" > "$pidf"; exec stdbuf -oL "$@" ) | {
    armed=0
    while :; do
      if [ "$armed" -eq 1 ]; then IFS= read -r -t "$limit" line; else IFS= read -r line; fi
      r=$?
      if [ "$r" -gt 128 ]; then kill "$(cat "$pidf")" 2>/dev/null; exit 124; fi
      if [ "$r" -ne 0 ]; then [ -n "$line" ] && printf '%s\n' "$line"; exit 0; fi
      [ "$line" = "##go" ] && armed=1
      printf '%s\n' "$line"
    done
  } || rc=$?
  rm -f "$pidf"
  return "$rc"
}

BUF="$(mktemp)"
trap 'rm -f "$BUF"' EXIT

# Emit a placeholder when an engine times out
emit_row() {
  echo "$1 $LABEL: $2" >&2
  printf "%-8s %10s %4d %11s %5s %11s %12s %11g %11g %11s %11s %11s %15s %24s\n" \
    "$1" "$LABEL" "$REPS" "$2" "$DIM" "$OUTER" "$INNER" "$PAIRS" "$DIST_OPS" \
    "$2" "$2" "$2" "-" "-" >> "$BUF"
}

DUCKDB_DEAD=0
SIRIUS_DEAD=0

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
  before=$(wc -l < "$BUF")
  if [ "$SIRIUS_DEAD" -eq 1 ]; then
    emit_row sirius TIMEOUT
  elif {
    echo "SET gpu_execution = true;"
    echo ".print ##go"
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@sirius $LABEL"
      query
    done
    echo ".timer off"
  } | watchdog "$SIRIUS_TIMEOUT" "$CLI" "$DB" | rows >> "$BUF"
  then :
  else
    code=$?
    if [ "$(wc -l < "$BUF")" -gt "$before" ]; then
      [ "$code" -eq 124 ] && SIRIUS_DEAD=1 || true
    elif [ "$code" -eq 124 ]; then SIRIUS_DEAD=1; emit_row sirius TIMEOUT
    else emit_row sirius "ERROR($code)"; fi
  fi

  # --- DuckDB ---
  before=$(wc -l < "$BUF")
  if [ "$DUCKDB_DEAD" -eq 1 ]; then
    emit_row duckdb TIMEOUT
  elif {
    echo "SET gpu_execution = false;"
    echo ".print ##go"
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@duckdb $LABEL"
      query
    done
    echo ".timer off"
  } | watchdog "$TIMEOUT" "$CLI" "$DB" | rows >> "$BUF"
  then :
  else
    code=$?
    if [ "$(wc -l < "$BUF")" -gt "$before" ]; then
      [ "$code" -eq 124 ] && DUCKDB_DEAD=1 || true
    elif [ "$code" -eq 124 ]; then DUCKDB_DEAD=1; emit_row duckdb TIMEOUT
    else emit_row duckdb "ERROR($code)"; fi
  fi
done

{ printf "\n%-8s %10s %4s %11s %5s %11s %12s %11s %11s %11s %11s %11s %15s %24s\n" \
    engine corpus reps rows dim probe_rows corpus_rows pairs dist_ops min_ms mean_ms max_ms ms_per_op billion_dist_ops_per_sec
  sort -k1,1 -k7,7n "$BUF"; }

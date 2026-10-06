#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-5/run.sh
# Title: scaling curve 5, per-row top-k join with growing k, sirius vs duckdb (brute force, hnsw) vs cuvs vs faiss

set -euo pipefail

REPS=10
TIMEOUT=3600
SIRIUS_TIMEOUT=7200
IALJ_TIMEOUT=3600
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SIRIUS_CLI="$REPO/build/release/duckdb"
DUCKDB_CLI="$HOME/andy/duckdb-v1.5.5/duckdb"
PY="$REPO/bench/vector/scaling-curve-5/.venv/bin/python"
DB="$REPO/bench/vector/data/bigann10m.duckdb"
APPROX=duckdb-ialj

KS=(2 4 8 16 32 64 65 128 256 512 1024)

echo "ksweep db=$(basename "$DB") reps=$REPS ks=[${KS[*]}]"

query()  { echo "SELECT count(*) FROM queries l, LATERAL (SELECT r.id, array_distance(l.vec, r.vec) AS dist FROM base r ORDER BY array_distance(l.vec, r.vec) LIMIT $1) rr;"; }
warmup() { echo "SELECT count(*) FROM (SELECT id, vec FROM queries LIMIT 1) l, LATERAL (SELECT r.id, array_distance(l.vec, r.vec) AS dist FROM base r ORDER BY array_distance(l.vec, r.vec) LIMIT $1) rr;"; }
rows() {
  awk -v dim="$DIM" -v probe="$PROBE" -v corpus="$INNER" -v pairs="$PAIRS" -v dist_ops="$DIST_OPS" -v approx="$APPROX" '
    /^@@/  { block = substr($0, 3); want=1; next }
    /real/ { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
             n[block]++; sum[block]+=t
             if (!(block in mn) || t<mn[block]) mn[block]=t
             if (!(block in mx) || t>mx[block]) mx[block]=t
             next }
    want && $0 ~ /[0-9]/ && $0 !~ /[A-Za-z]/ { r=$0; gsub(/[^0-9]/,"",r); if (r!="") { rows[block]=r; want=0 } }
    END    { for (b in n) { split(b, a, " ")
               mean = 1000*sum[b]/n[b]             # mean wall-clock of one query, ms
               dops = dist_ops + 0                 # coerce the %g string to a number
               mspo = (dops>0 ? mean/dops : 0)     # ms spent per distance element-op
               gops = (mean>0 ? dops/mean/1e6 : 0) # billion distance element-ops per second
               ex = (a[1] != approx)
               printf "%-11s %7s %4d %11s %5s %11s %12s %11s %11s %11.1f %11.1f %11.1f %15s %24s\n",
                      a[1], a[2], n[b], (b in rows ? rows[b] : "-"),
                      dim, probe, corpus,
                      (ex ? sprintf("%g", pairs+0) : "-"), (ex ? sprintf("%g", dops) : "-"),
                      1000*mn[b], mean, 1000*mx[b],
                      (ex ? sprintf("%.10f", mspo) : "-"), (ex ? sprintf("%.2f", gops) : "-") } }
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
LOG="$(mktemp)"
trap 'rm -f "$BUF" "$LOG"' EXIT

# Emit a placeholder with 0 reps when an engine times out, errors, or is unsupported
emit_row() {
  local pairs=- dops=-
  [ "$1" != "$APPROX" ] && pairs=$(printf %g "$PAIRS") && dops=$(printf %g "$DIST_OPS")
  echo "$1 $LABEL: $2" >&2
  printf "%-11s %7s %4d %11s %5s %11s %12s %11s %11s %11s %11s %11s %15s %24s\n" \
    "$1" "$LABEL" "$REPS" "$2" "$DIM" "$PROBE" "$INNER" "$pairs" "$dops" \
    "$2" "$2" "$2" "-" "-" >> "$BUF"
}

sirius_k() {
  {
    echo "SET gpu_execution = true;"
    echo ".print ##go"
    warmup "$K"
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@sirius $K"
      query "$K"
    done
    echo ".timer off"
  } | watchdog "$SIRIUS_TIMEOUT" "$SIRIUS_CLI" "$DB" | rows
}

duckdb_bf_k() {
  {
    echo ".print ##go"
    warmup "$K"
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@duckdb-bf $K"
      query "$K"
    done
    echo ".timer off"
  } | watchdog "$TIMEOUT" "$DUCKDB_CLI" "$DB" | rows
}

cuvs_k() {
  { echo "@@cuvs $K"
    watchdog "$TIMEOUT" "$PY" "$REPO/bench/vector/scaling-curve-5/cuvs_knn.py" --db "$DB" --k "$K" --reps "$REPS"
  } | rows
}

faiss_k() {
  # system CUDA 12.6 on LD_LIBRARY_PATH shadows the venv's newer cuBLAS and breaks the faiss import
  { echo "@@faiss $K"
    LD_LIBRARY_PATH= watchdog "$TIMEOUT" "$PY" "$REPO/bench/vector/scaling-curve-5/faiss_knn.py" --db "$DB" --k "$K" --reps "$REPS"
  } | rows
}

declare -A DEAD=([sirius]=0 [duckdb-bf]=0 [cuvs]=0 [faiss]=0)

# Runs one engine at the current k; after a timeout, the larger k's are marked TIMEOUT without running
run() {
  local engine=$1 before code=0
  before=$(wc -l < "$BUF")
  if [ "${DEAD[$engine]}" -eq 1 ]; then emit_row "$engine" TIMEOUT; return; fi
  "${engine//-/_}_k" >> "$BUF" || code=$?
  if [ "$code" -ne 0 ]; then
    if [ "$(wc -l < "$BUF")" -gt "$before" ]; then
      [ "$code" -eq 124 ] && DEAD[$engine]=1 || true
    elif [ "$code" -eq 124 ]; then DEAD[$engine]=1; emit_row "$engine" TIMEOUT
    else emit_row "$engine" "ERROR($code)"; fi
  fi
}

DIM=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT len(vec) FROM base LIMIT 1;")
INNER=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM base;")
PROBE=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM queries;")
PAIRS=$((PROBE * INNER))
DIST_OPS=$((PAIRS * DIM))

for K in "${KS[@]}"; do
  LABEL="$K"
  echo "running k=$K" >&2
  for engine in sirius duckdb-bf cuvs faiss; do
    run "$engine"
  done
done

# --- DuckDB HNSW index accelerated lateral join ---
echo "running duckdb-ialj" >&2
code=0
{
  echo "LOAD vss;"
  echo "SET disabled_optimizers = 'top_n_window_elimination';"
  echo "ATTACH '$DB' AS src (READ_ONLY);"
  echo "CREATE TABLE base AS SELECT id, vec FROM src.base;"
  echo "CREATE TABLE queries AS SELECT id, vec FROM src.queries;"
  echo "CREATE TABLE warmup AS SELECT id, vec FROM queries LIMIT 1;"
  echo "CREATE INDEX base_hnsw ON base USING HNSW (vec) WITH (metric='l2sq', ef_construction=128, M=8);"
  echo ".print ##go"
  query "${KS[0]}" | sed 's/FROM queries l/FROM warmup l/'
  for K in "${KS[@]}"; do
    # plan goes to the log only ('##' is not a block marker for rows)
    echo ".print ##plan $K"
    echo "EXPLAIN $(query "$K")"
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@duckdb-ialj $K"
      query "$K"
    done
    echo ".timer off"
  done
} | watchdog "$IALJ_TIMEOUT" "$DUCKDB_CLI" | tee "$LOG" | rows >> "$BUF" || code=$?

for K in "${KS[@]}"; do
  LABEL="$K"
  # a plan without HNSW_INDEX_JOIN is just brute force on an in-memory copy: drop its timings
  if awk -v k="$K" '$1=="##plan" { cur=$2; next } /^@@/ { cur="" } cur==k && /HNSW_INDEX_JOIN/ { f=1 } END { exit !f }' "$LOG"; then
    :
  elif grep -q "^##plan $K\$" "$LOG"; then
    awk -v k="$K" '!($1=="duckdb-ialj" && $2==k)' "$BUF" > "$BUF.tmp" && mv "$BUF.tmp" "$BUF"
    emit_row duckdb-ialj NO_INDEX
    continue
  fi
  # k's the session didn't reach get a placeholder
  if ! awk -v k="$K" '$1=="duckdb-ialj" && $2==k { found=1 } END { exit !found }' "$BUF"; then
    [ "$code" -eq 124 ] && emit_row duckdb-ialj TIMEOUT || emit_row duckdb-ialj "ERROR($code)"
  fi
done

{ printf "\n%-11s %7s %4s %11s %5s %11s %12s %11s %11s %11s %11s %11s %15s %24s\n" \
    engine k reps rows dim probe_rows corpus_rows pairs dist_ops min_ms mean_ms max_ms ms_per_op billion_dist_ops_per_sec
  sort -k1,1 -k2,2n "$BUF"; }

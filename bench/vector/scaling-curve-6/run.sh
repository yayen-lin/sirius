#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-6/run.sh
# Title: scaling curve 6, per-row top-k join with growing corpus

set -euo pipefail

REPS=10
K=10
TIMEOUT=3600
SIRIUS_TIMEOUT=7200
IALJ_TIMEOUT=3600
APPROX=duckdb-ialj
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SIRIUS_CLI="$REPO/build/release/duckdb"
DUCKDB_CLI="$HOME/andy/duckdb-v1.5.5/duckdb"
PY="$REPO/bench/vector/scaling-curve-5/.venv/bin/python"

SLICED="$REPO/bench/vector/data/bigann10m_sliced.duckdb"

# label:db:corpus table (1k..10m are scaling curve 2's slices of bigann10m)
DATASETS=(
  "1k:$SLICED:base_1k"
  "10k:$SLICED:base_10k"
  "100k:$SLICED:base_100k"
  "1m:$SLICED:base_1m"
  "10m:$SLICED:base_10m"
  "100m:$REPO/bench/vector/data/bigann100m.duckdb:base"
)

echo "corpussweep k=$K reps=$REPS datasets=[$(for d in "${DATASETS[@]}"; do printf '%s ' "${d%%:*}"; done)]"

query()  { echo "SELECT count(*) FROM queries l, LATERAL (SELECT array_distance(l.vec, r.vec) AS dist FROM $TABLE r ORDER BY array_distance(l.vec, r.vec) LIMIT $K) rr;"; }
warmup() { echo "SELECT count(*) FROM (SELECT id, vec FROM queries LIMIT 1) l, LATERAL (SELECT array_distance(l.vec, r.vec) AS dist FROM $TABLE r ORDER BY array_distance(l.vec, r.vec) LIMIT $K) rr;"; }
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
               mean = 1000*sum[b]/n[b]           # mean wall-clock of one query, ms
               dops = dist_ops + 0               # coerce the %g string to a number
               mspo = (dops>0 ? mean/dops : 0)   # ms spent per distance element-op
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

sirius_d() {
  {
    echo "SET gpu_execution = true;"
    echo ".print ##go"
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@sirius $LABEL"
      query
    done
    echo ".timer off"
  } | watchdog "$SIRIUS_TIMEOUT" "$SIRIUS_CLI" "$DB" | rows
}

duckdb_bf_d() {
  {
    echo ".print ##go"
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@duckdb-bf $LABEL"
      query
    done
    echo ".timer off"
  } | watchdog "$TIMEOUT" "$DUCKDB_CLI" "$DB" | rows
}

duckdb_ialj_d() {
  local out code=0
  out=$({
    echo "LOAD vss;"
    echo "SET disabled_optimizers = 'top_n_window_elimination';"
    echo "ATTACH '$DB' AS src (READ_ONLY);"
    echo "CREATE TABLE $TABLE AS SELECT vec FROM src.$TABLE;"
    echo "CREATE TABLE queries AS SELECT id, vec FROM src.queries;"
    echo "CREATE TABLE warmup AS SELECT id, vec FROM queries LIMIT 1;"
    echo "CREATE INDEX base_hnsw ON $TABLE USING HNSW (vec) WITH (metric='l2sq', ef_construction=128, M=16);"
    echo ".print ##go"
    echo "EXPLAIN $(query)"
    query | sed 's/FROM queries l/FROM warmup l/'
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@duckdb-ialj $LABEL"
      query
    done
    echo ".timer off"
  } | watchdog "$IALJ_TIMEOUT" "$DUCKDB_CLI" | tee "$LOG" | rows) || code=$?
  if grep -q "^@@duckdb-ialj" "$LOG" && ! grep -q "HNSW_INDEX_JOIN" "$LOG"; then return 4; fi
  [ -n "$out" ] && echo "$out"
  return "$code"
}

cuvs_d() {
  { echo "@@cuvs $LABEL"
    watchdog "$TIMEOUT" "$PY" "$REPO/bench/vector/scaling-curve-5/cuvs_knn.py" --db "$DB" --k "$K" --reps "$REPS" --table "$TABLE"
  } | rows
}

faiss_d() {
  # system CUDA 12.6 on LD_LIBRARY_PATH shadows the venv's newer cuBLAS and breaks the faiss import
  { echo "@@faiss $LABEL"
    LD_LIBRARY_PATH= watchdog "$TIMEOUT" "$PY" "$REPO/bench/vector/scaling-curve-5/faiss_knn.py" --db "$DB" --k "$K" --reps "$REPS" --table "$TABLE"
  } | rows
}

declare -A DEAD=([sirius]=0 [duckdb-bf]=0 [duckdb-ialj]=0 [cuvs]=0 [faiss]=0)

# Runs one engine on the current dataset; after a timeout, the larger datasets are marked TIMEOUT without running
run() {
  local engine=$1 before code=0
  before=$(wc -l < "$BUF")
  if [ "${DEAD[$engine]}" -eq 1 ]; then emit_row "$engine" TIMEOUT; return; fi
  # skip 100m for duckdb-ialj so it doesn't drain all the memory
  if [ "$engine" = duckdb-ialj ] && [ "$LABEL" = 100m ]; then emit_row "$engine" SKIPPED; return; fi
  "${engine//-/_}_d" >> "$BUF" || code=$?
  if [ "$code" -ne 0 ]; then
    if [ "$(wc -l < "$BUF")" -gt "$before" ]; then
      [ "$code" -eq 124 ] && DEAD[$engine]=1 || true
    elif [ "$code" -eq 124 ]; then DEAD[$engine]=1; emit_row "$engine" TIMEOUT
    elif [ "$code" -eq 3 ]; then emit_row "$engine" UNSUPPORTED # cuvs: corpus doesn't fit on the GPU
    elif [ "$code" -eq 4 ]; then emit_row "$engine" NO_INDEX    # duckdb-ialj: plan fell back to brute force
    else emit_row "$engine" "ERROR($code)"; fi
  fi
}

for entry in "${DATASETS[@]}"; do
  IFS=: read -r LABEL DB TABLE <<< "$entry"
  echo "running $LABEL ($(basename "$DB") $TABLE)" >&2

  DIM=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT len(vec) FROM $TABLE LIMIT 1;")
  INNER=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM $TABLE;")
  PROBE=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM queries;")
  PAIRS=$((PROBE * INNER))
  DIST_OPS=$((PAIRS * DIM))

  for engine in sirius duckdb-bf duckdb-ialj cuvs faiss; do
    run "$engine"
  done
done

{ printf "\n%-11s %7s %4s %11s %5s %11s %12s %11s %11s %11s %11s %11s %15s %24s\n" \
    engine corpus reps rows dim probe_rows corpus_rows pairs dist_ops min_ms mean_ms max_ms ms_per_op billion_dist_ops_per_sec
  sort -k1,1 -k7,7n "$BUF"; }

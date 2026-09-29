#!/usr/bin/env bash
# Usage: ./bench/vector/scaling-curve-6/run.sh
# Title: scaling curve 6, per-row top-k join with growing corpus, sirius vs duckdb (brute force, hnsw) vs cuvs vs faiss

set -euo pipefail

REPS=10
PROBE=10000
K=10
TIMEOUT=1h
SIRIUS_TIMEOUT=2h
HNSW_TIMEOUT=1h
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

echo "corpussweep probe=$PROBE k=$K reps=$REPS datasets=[$(for d in "${DATASETS[@]}"; do printf '%s ' "${d%%:*}"; done)]"

query()  { echo "SELECT count(*) FROM (SELECT id, vec FROM queries LIMIT $PROBE) l, LATERAL (SELECT array_distance(l.vec, r.vec) AS dist FROM $TABLE r ORDER BY array_distance(l.vec, r.vec) LIMIT $K) rr;"; }
warmup() { echo "SELECT count(*) FROM (SELECT id, vec FROM queries LIMIT 1) l, LATERAL (SELECT array_distance(l.vec, r.vec) AS dist FROM $TABLE r ORDER BY array_distance(l.vec, r.vec) LIMIT $K) rr;"; }
rows() {
  awk -v dim="$DIM" -v probe="$PROBE" -v corpus="$INNER" -v pairs="$PAIRS" -v dist_ops="$DIST_OPS" '
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

# Emit a placeholder with 0 reps when an engine times out, errors, or is unsupported
emit_row() {
  echo "$1 $LABEL: $2" >&2
  printf "%-8s %10s %4d %11s %5s %11s %12s %11g %11g %11s %11s %11s %15s %24s\n" \
    "$1" "$LABEL" 0 "$2" "$DIM" "$PROBE" "$INNER" "$PAIRS" "$DIST_OPS" \
    "$2" "$2" "$2" "-" "-" >> "$BUF"
}

# One engine on one dataset: warmup, then REPS timed queries tagged with '@@engine dataset'
sirius_d() {
  {
    echo "SET gpu_execution = true;"
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@sirius $LABEL"
      query
    done
    echo ".timer off"
  } | timeout "$SIRIUS_TIMEOUT" "$SIRIUS_CLI" "$DB" | rows
}
duckdb_d() {
  {
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@duckdb $LABEL"
      query
    done
    echo ".timer off"
  } | timeout "$TIMEOUT" "$DUCKDB_CLI" "$DB" | rows
}
# HNSW indexes only work on in-memory tables, so copy the data in and build the index untimed
hnsw_d() {
  {
    echo "LOAD vss;"
    echo "ATTACH '$DB' AS src (READ_ONLY);"
    echo "CREATE TABLE $TABLE AS SELECT vec FROM src.$TABLE;"
    echo "CREATE TABLE queries AS SELECT id, vec FROM src.queries;"
    echo "CREATE INDEX base_hnsw ON $TABLE USING HNSW (vec) WITH (metric='l2sq', ef_construction=128, M=16);"
    warmup
    echo ".timer on"
    for i in $(seq $REPS); do
      echo ".print @@hnsw $LABEL"
      query
    done
    echo ".timer off"
  } | timeout "$HNSW_TIMEOUT" "$DUCKDB_CLI" | rows
}
cuvs_d() {
  { echo "@@cuvs $LABEL"
    timeout "$TIMEOUT" "$PY" "$REPO/bench/vector/scaling-curve-5/cuvs_knn.py" --db "$DB" --k "$K" --probe "$PROBE" --reps "$REPS" --table "$TABLE"
  } | rows
}
faiss_d() {
  # system CUDA 12.6 on LD_LIBRARY_PATH shadows the venv's newer cuBLAS and breaks the faiss import
  { echo "@@faiss $LABEL"
    LD_LIBRARY_PATH= timeout "$TIMEOUT" "$PY" "$REPO/bench/vector/scaling-curve-5/faiss_knn.py" --db "$DB" --k "$K" --probe "$PROBE" --reps "$REPS" --table "$TABLE"
  } | rows
}

declare -A DEAD=([sirius]=0 [duckdb]=0 [hnsw]=0 [cuvs]=0 [faiss]=0)

# Runs one engine on the current dataset; after a timeout, the larger datasets are marked TIMEOUT without running
run() {
  local engine=$1 before code=0
  before=$(wc -l < "$BUF")
  if [ "${DEAD[$engine]}" -eq 1 ]; then emit_row "$engine" TIMEOUT; return; fi
  "${engine}_d" >> "$BUF" || code=$?
  if [ "$code" -ne 0 ]; then
    if [ "$(wc -l < "$BUF")" -gt "$before" ]; then
      [ "$code" -eq 124 ] && DEAD[$engine]=1 || true
    elif [ "$code" -eq 124 ]; then DEAD[$engine]=1; emit_row "$engine" TIMEOUT
    elif [ "$code" -eq 3 ]; then emit_row "$engine" UNSUPPORTED # cuvs: corpus doesn't fit on the GPU
    else emit_row "$engine" "ERROR($code)"; fi
  fi
}

for entry in "${DATASETS[@]}"; do
  IFS=: read -r LABEL DB TABLE <<< "$entry"
  echo "running $LABEL ($(basename "$DB") $TABLE)" >&2

  DIM=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT len(vec) FROM $TABLE LIMIT 1;")
  INNER=$("$DUCKDB_CLI" -csv -noheader "$DB" -c "SELECT count(*) FROM $TABLE;")
  PAIRS=$((PROBE * INNER))
  DIST_OPS=$((PAIRS * DIM))

  for engine in sirius duckdb hnsw cuvs faiss; do
    run "$engine"
  done
done

{ printf "\n%-8s %10s %4s %11s %5s %11s %12s %11s %11s %11s %11s %11s %15s %24s\n" \
    engine corpus reps rows dim probe_rows corpus_rows pairs dist_ops min_ms mean_ms max_ms ms_per_op billion_dist_ops_per_sec
  sort -k1,1 -k7,7n "$BUF"; }

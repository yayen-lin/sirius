#!/usr/bin/env bash
# End-to-end: download every raw dataset, convert each to a .duckdb in
# bench/vector/data/, and delete the intermediates as it goes.
#
# Produces:
#   gist1m.duckdb     base + queries + gt + gt_cosine   (dim 960)
#   sift1m.duckdb     base + queries + gt + gt_cosine   (dim 128)
#   bigann10m.duckdb  base + queries                    (dim 128)
#   bigann100m.duckdb base + queries                    (dim 128)
#   bigann1b.duckdb   base + queries                    (dim 128)
#
# The bigann builds carry no ground truth on purpose (brute force is intractable
# at that scale); recall is evaluated on gist1m/sift1m.
#
# Run it on the big SSD, from anywhere:  bench/vector/data/build_all.sh
set -euo pipefail

DATA="$(cd "$(dirname "$0")" && pwd)"              # .../bench/vector/data
REPO="$(cd "$DATA/../../.." && pwd)"               # repo root
RAW="${RAW:-$DATA/raw}"                            # raw downloads
WORK="${WORK:-$DATA/_parquet}"                     # per-dataset parquet staging
# duckdb CLI: the repo's own build matches the Sirius storage format. Override
# with DUCKDB=/path/to/duckdb (same duckdb version) if it isn't built yet.
CLI="${DUCKDB:-$REPO/build/release/duckdb}"

echo "REPO=$REPO"; echo "DATA=$DATA"; echo "CLI=$CLI"
[ -x "$CLI" ] || { echo "duckdb CLI not found at $CLI -- build sirius first, or set DUCKDB=" >&2; exit 1; }

# --- python env with numpy/pyarrow/h5py for the converters -------------------
PY="$WORK/.venv/bin/python"
ensure_py () {
  if [ ! -x "$PY" ]; then
    echo "creating converter venv..."
    python3 -m venv "$WORK/.venv"
    "$PY" -m ensurepip --upgrade >/dev/null
    "$PY" -m pip install --quiet --upgrade pip numpy pyarrow h5py
  fi
  "$PY" -c "import numpy,pyarrow,h5py" 2>/dev/null || {
    "$PY" -m pip install --quiet numpy pyarrow h5py; }
}

# gpu_execution only exists on the Sirius-loaded CLI; probe once so gt_cosine
# runs on the CPU there and the script still works with a plain duckdb.
GPUOFF=""
if "$CLI" -c "SET gpu_execution=false;" >/dev/null 2>&1; then GPUOFF="SET gpu_execution=false;"; fi

mkdir -p "$WORK"

# --- helpers -----------------------------------------------------------------
build_hdf5 () {   # <hdf5-file> <dim> <name>
  local hdf5="$1" dim="$2" name="$3"
  local out="$WORK/$name" db="$DATA/$name.duckdb"
  echo "=== [$name] hdf5 -> parquet ==="
  "$PY" "$DATA/hdf5-to-duckdb/hdf5_to_parquet.py" --hdf5 "$RAW/$hdf5" --out "$out"
  echo "=== [$name] parquet -> duckdb (base, queries, gt) ==="
  rm -f "$db"
  "$CLI" "$db" <<SQL
CREATE TABLE base    AS SELECT id, vec::FLOAT[$dim] AS vec FROM read_parquet('$out/base.parquet');
CREATE TABLE queries AS SELECT id, vec::FLOAT[$dim] AS vec FROM read_parquet('$out/queries.parquet');
CREATE TABLE gt      AS SELECT query_id, rank, neighbor_id, distance FROM read_parquet('$out/gt.parquet');
SQL
  echo "=== [$name] gt_cosine (exact brute force, cpu) ==="
  "$CLI" "$db" <<SQL
$GPUOFF
CREATE TABLE gt_cosine AS
SELECT query_id, (row_number() OVER (PARTITION BY query_id ORDER BY d) - 1)::INT AS rank,
       neighbor_id, d AS distance
FROM (SELECT q.id AS query_id, t.neighbor_id, t.d
      FROM queries q,
      LATERAL (SELECT b.id AS neighbor_id, array_cosine_distance(b.vec, q.vec) AS d
               FROM base b ORDER BY d LIMIT 100) t);
SQL
  "$CLI" "$db" "SELECT 'base' t,count(*) n FROM base UNION ALL SELECT 'queries',count(*) FROM queries UNION ALL SELECT 'gt',count(*) FROM gt UNION ALL SELECT 'gt_cosine',count(*) FROM gt_cosine;"
  rm -rf "$out"
  echo "built $db"
}

build_bigann () {  # <base-file> <name>
  local base="$1" name="$2"
  local out="$WORK/$name" db="$DATA/$name.duckdb"
  echo "=== [$name] u8bin -> parquet ==="
  "$PY" "$DATA/u8bin-to-duckdb/u8bin_to_parquet.py" \
        --base "$RAW/$base" --queries "$RAW/query.public.10K.u8bin" --out "$out"
  echo "=== [$name] parquet -> duckdb (base, queries) ==="
  rm -f "$db"
  "$CLI" "$db" <<SQL
CREATE TABLE base    AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$out/base.parquet');
CREATE TABLE queries AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$out/queries.parquet');
SQL
  "$CLI" "$db" "SELECT 'base' t,count(*) n FROM base UNION ALL SELECT 'queries',count(*) FROM queries;"
  rm -rf "$out"
  echo "built $db"
}

# --- run ---------------------------------------------------------------------
"$DATA/download_raw.sh" "$RAW"
ensure_py

build_hdf5   gist-960-euclidean.hdf5 960 gist1m
build_hdf5   sift-128-euclidean.hdf5 128 sift1m
build_bigann base.10M.u8bin              bigann10m
build_bigann base.100M.u8bin             bigann100m
build_bigann base.1B.u8bin               bigann1b

echo "=== cleanup: removing raw downloads and parquet staging ==="
rm -rf "$RAW" "$WORK"

echo
echo "done. datasets in $DATA:"
ls -lh "$DATA"/*.duckdb

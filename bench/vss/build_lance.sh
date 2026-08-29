#!/usr/bin/env bash
# Write Lance datasets for the benchmark from an existing <name>.duckdb base table.
#   <name>_base.lance       raw vectors     -> Lance L2 ENN + L2 ANN
#   <name>_base_norm.lance  unit-normalized -> Lance cosine ENN + cosine ANN
# Lance exact search is L2-only, so cosine is done on unit vectors (L2 order == cosine
# order). Each dataset gets one IVF_FLAT (l2) index, matching Sirius n_lists.
#
# Run from anywhere; needs the GPU free (repo CLI loads Sirius) and the .duckdb
# not open in another session.
#   bench/vss/build_lance.sh <name>
set -euo pipefail
NAME="${1:?name required, e.g. gist1m}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DATA="$REPO/bench/vss/data"
DB="$DATA/$NAME.duckdb"
BASE="$DATA/${NAME}_base.lance"
NORM="$DATA/${NAME}_base_norm.lance"

DIM=$("$CLI" "$DB" -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
echo "writing lance datasets (dim=$DIM)..."
rm -rf "$BASE" "$NORM"

"$CLI" "$DB" <<SQL
LOAD lance;
SET gpu_execution=false;
-- raw vectors
COPY (SELECT id, vec FROM base) TO '$BASE' (FORMAT lance, mode 'overwrite');
-- unit-normalized vectors so an L2 index/search reproduces cosine order
COPY (
  SELECT id, list_transform(v, x -> (x/nrm)::FLOAT)::FLOAT[$DIM] AS vec
  FROM (SELECT id, vec::DOUBLE[] AS v,
               sqrt(list_sum(list_transform(vec::DOUBLE[], y -> y*y))) AS nrm
        FROM base)
) TO '$NORM' (FORMAT lance, mode 'overwrite');
SQL
echo "built $BASE and $NORM (indexes are built by bench.sh per n_lists)"

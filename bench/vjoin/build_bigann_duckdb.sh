#!/usr/bin/env bash
# Build bigann100m.duckdb from parquet, using the repo's own duckdb CLI so the
# storage format matches the Sirius extension binary.
# Unlike build_duckdb.sh this skips the gt / gt_cosine brute force, which is
# intractable over a 100M base.
#
# Usage: bench/vjoin/build_bigann_duckdb.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DATA="$REPO/bench/vjoin/data"
DB="$DATA/bigann100m.duckdb"

rm -f "$DB"
"$CLI" "$DB" <<SQL
CREATE TABLE base AS
  SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/parquet/base.parquet');
CREATE TABLE queries AS
  SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/parquet/queries.parquet');
SQL

"$CLI" "$DB" <<'SQL'
SELECT 'base' AS t, count(*) AS n FROM base
UNION ALL SELECT 'queries', count(*) FROM queries;
SQL
echo "built $DB"

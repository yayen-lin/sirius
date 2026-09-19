#!/usr/bin/env bash
# Usage: ./bench/vector/scenario3/run.sh

set -euo pipefail

EPS=0.4
TIMEOUT=1h # DuckDB is expected to time out

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vector/data/gist1m.duckdb"

echo "db=$DB eps=$EPS timeout=$TIMEOUT"

"$CLI" "$DB" <<SQL
SET gpu_execution = true;

.timer on
SELECT count(*) AS pairs_sirius
FROM base l JOIN base r
  ON array_distance(l.vec, r.vec) <= $EPS;
.timer off
SQL

set +e
timeout "$TIMEOUT" "$CLI" "$DB" <<SQL
SET gpu_execution = false;

.timer on
SELECT count(*) AS pairs_duckdb
FROM base l JOIN base r
  ON array_distance(l.vec, r.vec) <= $EPS;
.timer off
SQL
code=$?
set -e
[ "$code" -eq 124 ] && echo "duckdb: TIMED OUT after $TIMEOUT"

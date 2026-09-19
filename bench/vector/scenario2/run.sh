#!/usr/bin/env bash
# Usage: ./bench/vector/scenario2/run.sh

set -euo pipefail

PROBE=100
EPS=250

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vector/data/bigann100m.duckdb"

echo "db=$DB probe=$PROBE eps=$EPS"

"$CLI" "$DB" <<SQL
SET gpu_execution = true;

.timer on
SELECT count(*) AS pairs_sirius
FROM (SELECT vec FROM queries LIMIT $PROBE) l JOIN base r
  ON array_distance(l.vec, r.vec) <= $EPS;
.timer off

SET gpu_execution = false;

.timer on
SELECT count(*) AS pairs_duckdb
FROM (SELECT vec FROM queries LIMIT $PROBE) l JOIN base r
  ON array_distance(l.vec, r.vec) <= $EPS;
.timer off
SQL

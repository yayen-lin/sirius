#!/usr/bin/env bash
# Usage: ./bench/vector/scenario1/run.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vector/data/gist1m.duckdb"

"$CLI" "$DB" <<'SQL'
SET gpu_execution = true;

.timer on
SELECT count(*) AS pairs_sirius
FROM queries l JOIN base r
  ON array_distance(l.vec, r.vec) <= 1.0;
.timer off

SET gpu_execution = false;

.timer on
SELECT count(*) AS pairs_duckdb
FROM queries l JOIN base r
  ON array_distance(l.vec, r.vec) <= 1.0;
.timer off
SQL

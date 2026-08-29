#!/usr/bin/env bash
# Usage: ./bench/vss/index_creation.sh
set -euo pipefail

export TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

NLIST=1024
REPEATS=5
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO/build/release/duckdb"
DB="$REPO/bench/vss/data/gist1m.duckdb"
LANCE="$REPO/bench/vss/data/gist1m_base.lance"

echo "db=$DB n_lists=$NLIST REPEATS=$REPEATS"

# Reads the CLI output on stdin and prints min/mean of the '.timer' real
# seconds, grouped by the '@@engine metric' markers printed before each block.
report() {
  awk '
    /^@@/  { block = substr($0, 3); next }
    /real/ { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
             n[block]++; sum[block]+=t; if (!(block in mn) || t<mn[block]) mn[block]=t }
    END    { for (b in n) { split(b, a, " ")   # b = "engine metric"
               printf "%-8s %-8s %-8s %8.2f %8.2f\n",
                      a[1], "index", a[2], mn[b], sum[b]/n[b] } }
  ' | sort | { printf "\n%-8s %-8s %-8s %8s %8s\n" engine op metric min_s mean_s; cat; }
}

{

# ===== Sirius IVF-Flat =====
# Warmup & Setup
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
echo "SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'l2', index_type => 'ivf_flat', n_lists => $NLIST);"
echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'l2');"

echo ".print @@sirius l2"
for i in $(seq $REPEATS); do
  echo ".timer on"
  echo "SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'l2', index_type => 'ivf_flat', n_lists => $NLIST);"
  echo ".timer off"
  echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'l2');"
done

echo ".print @@sirius cosine"
for i in $(seq $REPEATS); do
  echo ".timer on"
  echo "SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'cosine', index_type => 'ivf_flat', n_lists => $NLIST);"
  echo ".timer off"
  echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'cosine');"
done

# Cleanup
echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'l2');"
echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'cosine');"
echo "SELECT * FROM unpin_table('base');"

# ===== Lance IVF-Flat =====
# Warmup & Setup
echo "LOAD lance;"
echo "CREATE INDEX vec_idx ON '$LANCE' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='l2');"
echo "DROP INDEX vec_idx ON '$LANCE';"

echo ".print @@lance l2"
for i in $(seq $REPEATS); do
  echo ".timer on"
  echo "CREATE INDEX vec_idx ON '$LANCE' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='l2');"
  echo ".timer off"
  echo "DROP INDEX vec_idx ON '$LANCE';"
  echo "SELECT * FROM __lance_cleanup_old_versions('$LANCE', '{\"older_than_seconds\":0,\"delete_unverified\":true}');"
done

echo ".print @@lance cosine"
for i in $(seq $REPEATS); do
  echo ".timer on"
  echo "CREATE INDEX vec_idx ON '$LANCE' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='cosine');"
  echo ".timer off"
  echo "DROP INDEX vec_idx ON '$LANCE';"
  echo "SELECT * FROM __lance_cleanup_old_versions('$LANCE', '{\"older_than_seconds\":0,\"delete_unverified\":true}');"
done

} | "$CLI" "$DB" | report

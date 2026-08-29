#!/usr/bin/env bash
# Usage: ./bench/vss/ann_bench_recall.sh
set -euo pipefail

export TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

K=10
NQ=20 # number of query (max 1000)
NLIST=1024
PROBES=(8 16 32 64 128 256)
CLI=./build/release/duckdb
DB=bench/vss/data/gist1m.duckdb
LANCE=bench/vss/data/gist1m_base.lance
LANCE_COS=bench/vss/data/gist1m_base_norm.lance

DIM=$($CLI $DB -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
mapfile -t QIDS      < <($CLI $DB -noheader -list -c "SELECT id  FROM queries ORDER BY id LIMIT $NQ;")
mapfile -t QVECS     < <($CLI $DB -noheader -list -c "SELECT vec FROM queries ORDER BY id LIMIT $NQ;")
mapfile -t QVECS_NORM < <($CLI $DB -noheader -list -c "SET gpu_execution=false; SELECT list_transform(v, lambda x:(x/nrm)::FLOAT) FROM (SELECT vec::DOUBLE[] AS v, sqrt(list_sum(list_transform(vec::DOUBLE[], lambda y:y*y))) AS nrm FROM queries ORDER BY id LIMIT $NQ);")

echo "db=$DB dim=$DIM k=$K nq=$NQ n_lists=$NLIST n_probes=${PROBES[*]}"

report() {
  awk -v k=$K -v nl=$NLIST '
    /^@@/     { block = substr($0, 3); next }
    /^RECALL/ { rc[$2" "$3" "$4] = $5; next }
    /real/    { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
                n[block]++; sum[block]+=t; if (!(block in mn) || t<mn[block]) mn[block]=t }
    END { for (b in n) { split(b, a, " ")
            r = (b in rc) ? rc[b] : "0"
            printf "%-8s %-8s %-8s %4d %8d %8d %6.2f%% %9.1f %9.1f %8s\n",
                   a[1], a[2], "ann", k, a[3], nl, 100*a[3]/nl,
                   1000*sum[b]/n[b], 1000*mn[b], r } }
  ' | sort -k1,1 -k2,2 -k5,5n | { printf "\n%-8s %-8s %-8s %4s %8s %8s %7s %9s %9s %8s\n" \
                 engine metric search k nprobes nlists "scan%" mean_ms min_ms recall; cat; }
}

{

echo "CREATE TEMP TABLE retrieved(engine VARCHAR, metric VARCHAR, nprobes INTEGER, query_id BIGINT, neighbor_id BIGINT);"

# ===== Sirius =====
# Setup
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
echo "SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'l2',     index_type => 'ivf_flat', n_lists => $NLIST);"
echo "SELECT * FROM sirius_create_ann_index('base', 'vec', metric => 'cosine', index_type => 'ivf_flat', n_lists => $NLIST);"
# Warmup
echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', ${QVECS[0]}::FLOAT[$DIM], k => $K, metric => 'l2',     use_index => true, n_probes => ${PROBES[0]}, output_columns => ['id']);"
echo "SELECT count(*) FROM sirius_knn_search('base', 'vec', ${QVECS[0]}::FLOAT[$DIM], k => $K, metric => 'cosine', use_index => true, n_probes => ${PROBES[0]}, output_columns => ['id']);"

# Search
echo ".timer on"
for idx in "${!QIDS[@]}"; do
  qid=${QIDS[$idx]}; v="${QVECS[$idx]}::FLOAT[$DIM]"
  for np in "${PROBES[@]}"; do
    echo ".print @@sirius l2 $np"
    echo "INSERT INTO retrieved SELECT 'sirius', 'l2',     $np, $qid, id FROM sirius_knn_search('base', 'vec', $v, k => $K, metric => 'l2',     use_index => true, n_probes => $np, output_columns => ['id']);"
    echo ".print @@sirius cosine $np"
    echo "INSERT INTO retrieved SELECT 'sirius', 'cosine', $np, $qid, id FROM sirius_knn_search('base', 'vec', $v, k => $K, metric => 'cosine', use_index => true, n_probes => $np, output_columns => ['id']);"
  done
done
echo ".timer off"

# Cleanup
echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'l2');"
echo "SELECT * FROM sirius_drop_ann_index('base', 'vec', metric => 'cosine');"
echo "SELECT * FROM unpin_table('base');"

# ===== Lance =====
for metric in l2 cosine; do
  if [ "$metric" = cosine ]; then LDS=$LANCE_COS; else LDS=$LANCE; fi
  # Setup
  echo "LOAD lance;"
  echo "CREATE INDEX vec_idx ON '$LDS' (vec) USING IVF_FLAT WITH (num_partitions=$NLIST, metric_type='l2');"
  # Warmup
  if [ "$metric" = cosine ]; then wv="${QVECS_NORM[0]}::FLOAT[]"; else wv="${QVECS[0]}::FLOAT[]"; fi
  echo "SELECT count(*) FROM lance_vector_search('$LDS', 'vec', $wv, k => $K, use_index => true, nprobs => ${PROBES[0]});"

  # Search
  echo ".timer on"
  for np in "${PROBES[@]}"; do
    for idx in "${!QIDS[@]}"; do
      qid=${QIDS[$idx]}
      if [ "$metric" = cosine ]; then vl="${QVECS_NORM[$idx]}::FLOAT[]"; else vl="${QVECS[$idx]}::FLOAT[]"; fi
      echo ".print @@lance $metric $np"
      echo "INSERT INTO retrieved SELECT 'lance', '$metric', $np, $qid, id FROM lance_vector_search('$LDS', 'vec', $vl, k => $K, use_index => true, nprobs => $np);"
    done
  done
  echo ".timer off"

  # Cleanup
  echo "DROP INDEX vec_idx ON '$LDS';"
  echo "SELECT * FROM __lance_cleanup_old_versions('$LDS', '{\"older_than_seconds\":0,\"delete_unverified\":true}');"
done

# ===== Recall =====
echo "SET gpu_execution = false;"
echo ".headers off"
echo ".mode list"
echo "SELECT 'RECALL ' || r.engine || ' ' || r.metric || ' ' || r.nprobes || ' ' || round(count(*)::DOUBLE / ($NQ * $K), 4)
      FROM retrieved r JOIN gt g
        ON g.query_id = r.query_id AND g.rank < $K AND g.neighbor_id = r.neighbor_id
      WHERE r.metric = 'l2'
      GROUP BY r.engine, r.metric, r.nprobes
      UNION ALL
      SELECT 'RECALL ' || r.engine || ' ' || r.metric || ' ' || r.nprobes || ' ' || round(count(*)::DOUBLE / ($NQ * $K), 4)
      FROM retrieved r JOIN gt_cosine g
        ON g.query_id = r.query_id AND g.rank < $K AND g.neighbor_id = r.neighbor_id
      WHERE r.metric = 'cosine'
      GROUP BY r.engine, r.metric, r.nprobes;"

} | $CLI $DB | report

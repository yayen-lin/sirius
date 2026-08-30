#!/usr/bin/env bash
# Usage: ./bench/vss/ann_bench_recall_qps.sh
set -euo pipefail

export TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

K=10
NQ=1000 # number of query (max 1000)
NLIST=1024
PROBES=(8 16 32 64 128 256)
EFS=(10 20 40 80 160 320)
HNSW_M=16
HNSW_EFC=128
TARGET=0.95  # QPS @ recall >= TARGET
CLI=./build/release/duckdb
DB=bench/vss/data/gist1m.duckdb
LANCE=bench/vss/data/gist1m_base.lance
LANCE_COS=bench/vss/data/gist1m_base_norm.lance

DIM=$($CLI $DB -noheader -list -c "SELECT len(vec) FROM base LIMIT 1;")
mapfile -t QIDS      < <($CLI $DB -noheader -list -c "SELECT id  FROM queries ORDER BY id LIMIT $NQ;")
mapfile -t QVECS     < <($CLI $DB -noheader -list -c "SELECT vec FROM queries ORDER BY id LIMIT $NQ;")
mapfile -t QVECS_NORM < <($CLI $DB -noheader -list -c "SET gpu_execution=false; SELECT list_transform(v, lambda x:(x/nrm)::FLOAT) FROM (SELECT vec::DOUBLE[] AS v, sqrt(list_sum(list_transform(vec::DOUBLE[], lambda y:y*y))) AS nrm FROM queries ORDER BY id LIMIT $NQ);")

echo "db=$DB dim=$DIM k=$K nq=$NQ n_lists=$NLIST target=$TARGET n_probes=${PROBES[*]} ef_search=${EFS[*]}"

report() {
  local d; d=$(cat)
  echo "$d" | awk -v k=$K -v nl=$NLIST '
    /^@@/     { block = substr($0, 3); next }
    /^RECALL/ { rc[$2" "$3" "$4] = $5; next }
    /real/    { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1)
                n[block]++; sum[block]+=t; if (!(block in mn) || t<mn[block]) mn[block]=t }
    END { for (b in n) { split(b, a, " ")
            r = (b in rc) ? rc[b] : "0"
            if (a[1]=="duckdb") { idx="HNSW";     nls="-"; scan="-" }
            else                { idx="IVF-FLAT"; nls=nl;  scan=sprintf("%.2f%%", 100*a[3]/nl) }
            printf "%-8s %-8s %-8s %-9s %4d %8d %8s %8s %9.1f %9.1f %8s %9.1f\n",
                   a[1], a[2], "ann", idx, k, a[3], nls, scan,
                   1000*sum[b]/n[b], 1000*mn[b], r, n[b]/sum[b] } }
  ' | sort -k1,1 -k2,2 -k6,6n | { printf "\n%-8s %-8s %-8s %-9s %4s %8s %8s %8s %9s %9s %8s %9s\n" \
                 engine metric search index k "probe/ef" nlists "scan%" mean_ms min_ms recall qps; cat; }

  echo "$d" | awk -v target=$TARGET -v probes="${PROBES[*]}" -v efs="${EFS[*]}" '
    /^@@/     { block = substr($0, 3); next }
    /^RECALL/ { rc[$2" "$3" "$4] = $5; next }
    /real/    { for (i=1;i<=NF;i++) if ($i=="real") t=$(i+1); n[block]++; s[block]+=t }
    END { npp = split(probes, P, " "); npe = split(efs, EF, " ")
          ne = split("sirius lance duckdb", E, " "); nm = split("l2 cosine", M, " ")
          printf "\nQPS @ recall >= %s:\n", target
          for (ei=1; ei<=ne; ei++) for (mi=1; mi<=nm; mi++) {
            e=E[ei]; m=M[mi]; hit=0
            if (e=="duckdb") { ns=npe; knob="ef";      for (i=1;i<=ns;i++) SW[i]=EF[i] }
            else             { ns=npp; knob="nprobes"; for (i=1;i<=ns;i++) SW[i]=P[i]  }
            for (i=1; i<=ns; i++) { key = e" "m" "SW[i]
              if ((key in rc) && rc[key]+0 >= target) {
                printf "  %-8s %-8s %s=%-5s recall=%-7s qps=%.1f\n", e, m, knob, SW[i], rc[key], n[key]/s[key]
                hit=1; break } }
            if (!hit) printf "  %-8s %-8s not reached in sweep\n", e, m } }
  '
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

# ===== DuckDB HNSW =====
# Setup
echo "SET gpu_execution = false;"
echo "LOAD vss;"
echo "SET hnsw_enable_experimental_persistence=true;"   # persistent DB needs this to build an HNSW index
echo "CREATE INDEX h_l2  ON base USING HNSW (vec) WITH (metric='l2sq',   ef_construction=$HNSW_EFC, M=$HNSW_M);"
echo "CREATE INDEX h_cos ON base USING HNSW (vec) WITH (metric='cosine', ef_construction=$HNSW_EFC, M=$HNSW_M);"
# Warmup
echo "SET hnsw_ef_search=${EFS[0]};"
echo "SELECT count(*) FROM (SELECT id FROM base ORDER BY array_distance(vec, ${QVECS[0]}::FLOAT[$DIM]) LIMIT $K);"

# Search
for ef in "${EFS[@]}"; do
  echo ".timer off"
  echo "SET hnsw_ef_search=$ef;"
  echo ".timer on"
  for idx in "${!QIDS[@]}"; do
    qid=${QIDS[$idx]}; v="${QVECS[$idx]}::FLOAT[$DIM]"
    echo ".print @@duckdb l2 $ef"
    echo "INSERT INTO retrieved SELECT 'duckdb', 'l2',     $ef, $qid, id FROM (SELECT id FROM base ORDER BY array_distance(vec, $v)        LIMIT $K);"
    echo ".print @@duckdb cosine $ef"
    echo "INSERT INTO retrieved SELECT 'duckdb', 'cosine', $ef, $qid, id FROM (SELECT id FROM base ORDER BY array_cosine_distance(vec, $v) LIMIT $K);"
  done
done
echo ".timer off"

# Cleanup
echo "DROP INDEX h_l2;"
echo "DROP INDEX h_cos;"

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

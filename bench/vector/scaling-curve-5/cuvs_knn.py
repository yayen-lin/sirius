"""Per-row exact top-k with cuVS brute force; times only the search call.

Usage: bench/vector/scaling-curve-5/.venv/bin/python cuvs_knn.py --db <file.duckdb> --k 10 --probe 1000 [--reps 10] [--table base]
"""
import argparse
import sys
import time

import cupy as cp
import duckdb
from cuvs.neighbors import brute_force


def load(con, table, sql):
    # unnest gives one flat float32 column, which reshapes to (rows, dim) without per-row copies
    dim = con.sql(f"SELECT len(vec) FROM {table} LIMIT 1").fetchone()[0]
    return con.sql(f"SELECT unnest(vec) AS x FROM ({sql})").fetchnumpy()["x"].reshape(-1, dim)


ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--probe", type=int, required=True)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--table", default="base")
args = ap.parse_args()

con = duckdb.connect(args.db, read_only=True)

# cuVS brute force needs the whole corpus on the GPU; leave half the free memory for workspace
corpus_rows = con.sql(f"SELECT count(*) FROM {args.table}").fetchone()[0]
dim = con.sql(f"SELECT len(vec) FROM {args.table} LIMIT 1").fetchone()[0]
if corpus_rows * dim * 4 >cp.cuda.Device().mem_info[0] // 2:
    # exit code 3 tells run.sh to record UNSUPPORTED instead of an error
    print(f"cuvs  k={args.k} probe={args.probe} corpus={corpus_rows} unsupported (corpus doesn't fit on the GPU)", file=sys.stderr)
    raise SystemExit(3)

base = cp.asarray(load(con, args.table, f"SELECT vec FROM {args.table}"))
queries = cp.asarray(load(con, args.table, f"SELECT vec FROM queries LIMIT {args.probe}"))

index = brute_force.build(base, metric="sqeuclidean")

# warmup, so CUDA context and kernel setup aren't in the timed search
brute_force.search(index, queries, args.k)
cp.cuda.Device().synchronize()

# prints the row count, then one duckdb-CLI style timer line per rep so run.sh can parse it
print(f"cuvs  k={args.k} probe={args.probe} corpus={base.shape[0]}", file=sys.stderr)
for rep in range(args.reps):
    t0 = time.perf_counter()
    distances, neighbors = brute_force.search(index, queries, args.k)
    cp.cuda.Device().synchronize()
    if rep == 0:
        print("num results: ", cp.asarray(neighbors).size)
    print(f"Run Time (s): real {time.perf_counter() - t0:.6f}", flush=True)

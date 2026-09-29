"""Per-row exact top-k with FAISS GPU; times only the search call.

Corpus fits on the GPU: GpuIndexFlatL2, data copied to the GPU before timing.
Corpus doesn't fit: knn_gpu streams the host corpus to the GPU in chunks, so the time includes the copies.

Usage: LD_LIBRARY_PATH= bench/vector/scaling-curve-5/.venv/bin/python faiss_knn.py --db <file.duckdb> --k 10 --probe 1000 [--reps 10] [--table base]
"""
import argparse
import sys
import time

import cupy as cp
import duckdb
import faiss


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
base = load(con, args.table, f"SELECT vec FROM {args.table}")
queries = load(con, args.table, f"SELECT vec FROM queries LIMIT {args.probe}")

res = faiss.StandardGpuResources()
# leave half the free GPU memory for distance and top-k workspace
budget = cp.cuda.Device().mem_info[0] // 2

if base.nbytes <= budget:
    mode = "resident"
    index = faiss.GpuIndexFlatL2(res, base.shape[1])
    index.add(base)

    def search():
        return index.search(queries, args.k)

    # warmup, so CUDA context and kernel setup aren't in the timed search
    search()
else:
    mode = "chunked"

    def search():
        return faiss.knn_gpu(res, queries, base, args.k, vectorsMemoryLimit=budget)

    # warmup on a small slice, so it doesn't stream the whole corpus an extra time
    faiss.knn_gpu(res, queries, base[:100_000], args.k)

# prints the row count, then one duckdb-CLI style timer line per rep so run.sh can parse it
print(f"faiss k={args.k} probe={args.probe} corpus={base.shape[0]} mode={mode}", file=sys.stderr)
for rep in range(args.reps):
    # search copies results back to host, so it has finished when it returns
    t0 = time.perf_counter()
    distances, neighbors = search()
    if rep == 0:
        print(neighbors.size)
    print(f"Run Time (s): real {time.perf_counter() - t0:.6f}", flush=True)

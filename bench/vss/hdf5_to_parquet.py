#!/usr/bin/env python3
"""Convert an ann-benchmarks HDF5 file into parquet tables.

The parquet files are then loaded into a persistent .duckdb by build_duckdb.sh,
using the repo's own duckdb CLI so the storage format matches the Sirius binary.

Emits three parquet files next to the output dir:
  base.parquet     id BIGINT, vec list<float32>[dim]   -- the pinned table
  queries.parquet  id BIGINT, vec list<float32>[dim]   -- query vectors
  gt.parquet       query_id BIGINT, rank INT, neighbor_id BIGINT, distance FLOAT
"""
import argparse
import os
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def vec_table(vectors: np.ndarray) -> pa.Table:
    n, dim = vectors.shape
    ids = pa.array(np.arange(n, dtype=np.int64))
    flat = pa.array(vectors.reshape(-1).astype(np.float32))
    vec = pa.FixedSizeListArray.from_arrays(flat, dim)
    return pa.table({"id": ids, "vec": vec})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True, help="output directory for parquet files")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only first N base rows (0 = all); ground truth is NOT "
                         "recomputed, so only use with the full base set")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        base = f["train"][:]
        queries = f["test"][:]
        neighbors = f["neighbors"][:]
        distances = f["distances"][:]
        metric = dict(f.attrs).get("distance", "unknown")

    if args.limit and args.limit < len(base):
        base = base[: args.limit]
        print(f"WARNING: truncated base to {args.limit}; shipped ground truth no longer valid")

    print(f"metric={metric} base={base.shape} queries={queries.shape} gt={neighbors.shape}")

    pq.write_table(vec_table(base), os.path.join(args.out, "base.parquet"))
    pq.write_table(vec_table(queries), os.path.join(args.out, "queries.parquet"))

    nq, k = neighbors.shape
    gt = pa.table({
        "query_id": pa.array(np.repeat(np.arange(nq, dtype=np.int64), k)),
        "rank": pa.array(np.tile(np.arange(k, dtype=np.int32), nq)),
        "neighbor_id": pa.array(neighbors.reshape(-1).astype(np.int64)),
        "distance": pa.array(distances.reshape(-1).astype(np.float32)),
    })
    pq.write_table(gt, os.path.join(args.out, "gt.parquet"))
    print("wrote base.parquet, queries.parquet, gt.parquet to", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert big-ann-benchmarks .u8bin files into parquet tables.

Mirrors hdf5_to_parquet.py so build_duckdb.sh can load them unchanged.
u8bin layout: uint32 count, uint32 dim, then count*dim uint8 values.
A range-downloaded prefix keeps the original 1B count in its header, so the
real row count is derived from the file size, not the header.

Emits:
  base.parquet     id BIGINT, vec fixed_size_list<float32>[dim]
  queries.parquet  id BIGINT, vec fixed_size_list<float32>[dim]
"""
import argparse
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HEADER = 8  # two uint32: count, dim


def real_rows(path: str, dim: int) -> int:
    return (os.path.getsize(path) - HEADER) // dim


def convert(src: str, dst: str, dim: int, chunk: int) -> None:
    n = real_rows(src, dim)
    data = np.memmap(src, dtype=np.uint8, mode="r", offset=HEADER, shape=(n, dim))
    schema = pa.schema([
        ("id", pa.int64()),
        ("vec", pa.list_(pa.float32(), dim)),
    ])
    with pq.ParquetWriter(dst, schema) as w:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = np.asarray(data[start:end], dtype=np.float32)
            ids = pa.array(np.arange(start, end, dtype=np.int64))
            flat = pa.array(block.reshape(-1))
            vec = pa.FixedSizeListArray.from_arrays(flat, dim)
            w.write_table(pa.table({"id": ids, "vec": vec}, schema=schema))
            print(f"  {end}/{n}")
    print(f"wrote {dst} ({n} rows, dim {dim})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base .u8bin (possibly a prefix)")
    ap.add_argument("--queries", required=True, help="queries .u8bin")
    ap.add_argument("--out", required=True, help="output dir for parquet files")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2_000_000,
                    help="rows per parquet row group / conversion batch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    convert(args.base, os.path.join(args.out, "base.parquet"), args.dim, args.chunk)
    convert(args.queries, os.path.join(args.out, "queries.parquet"), args.dim, args.chunk)


if __name__ == "__main__":
    main()

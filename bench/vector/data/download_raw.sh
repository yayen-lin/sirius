#!/usr/bin/env bash
# Download the raw source files for all five vector datasets, then slice the
# bigann prefixes locally so nothing is downloaded twice.
#
# Datasets and where each duckdb comes from:
#   gist1m      <- gist-960-euclidean.hdf5   (hdf5-to-duckdb/build_duckdb.sh -> gt + gt_cosine)
#   sift1m      <- sift-128-euclidean.hdf5   (hdf5-to-duckdb/build_duckdb.sh -> gt + gt_cosine)
#   bigann10m   <- first  10M rows of base.1B.u8bin + shared queries (u8bin-to-duckdb, no gt)
#   bigann100m  <- first 100M rows of base.1B.u8bin + shared queries (u8bin-to-duckdb, no gt)
#   bigann1b    <- full        base.1B.u8bin        + shared queries (u8bin-to-duckdb, no gt)
#
# Usage: ./download_raw.sh [DEST_DIR]     (default: ./raw)
# Run this on a disk with room for the full bigann base: ~142 GB of raw files
# (128 GB full + 12.8 GB 100M slice + 1.28 GB 10M slice), plus headroom for the
# parquet/duckdb builds that follow. The H100 box's 750 GB ephemeral drive is
# the place for it -- not the 100 GB root disk.
set -euo pipefail

DEST="${1:-raw}"
mkdir -p "$DEST"
cd "$DEST"

ANN="http://ann-benchmarks.com"
BIGANN="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann"

# byte size = 8-byte header + rows * 128 (uint8, dim 128)
BYTES_1B=128000000008     # full base
BYTES_100M=12800000008    # 8 + 100000000*128
BYTES_10M=1280000008      # 8 + 10000000*128

verify () {  # verify <file> <expected-bytes>
  local f="$1" want="$2" got
  got=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")   # linux || macOS
  if [ "$got" = "$want" ]; then echo "  OK   $f ($got bytes)"
  else echo "  FAIL $f: got $got, expected $want" >&2; return 1; fi
}

echo "=== [1/5] gist-960-euclidean.hdf5 (~3.6 GiB) ==="
curl -C - -o gist-960-euclidean.hdf5 "$ANN/gist-960-euclidean.hdf5"
verify gist-960-euclidean.hdf5 3844648288

echo "=== [2/5] sift-128-euclidean.hdf5 (~500 MiB) ==="
curl -C - -o sift-128-euclidean.hdf5 "$ANN/sift-128-euclidean.hdf5"
verify sift-128-euclidean.hdf5 525128288

echo "=== [3/5] bigann shared queries (1.3 MB) ==="
curl -C - -o query.public.10K.u8bin "$BIGANN/query.public.10K.u8bin"
verify query.public.10K.u8bin 1280008

echo "=== [4/5] bigann base.1B.u8bin (128 GB -- this is the long one) ==="
curl -C - -o base.1B.u8bin "$BIGANN/base.1B.u8bin"
verify base.1B.u8bin "$BYTES_1B"

echo "=== [5/5] slice 100M and 10M prefixes locally (no re-download) ==="
head -c "$BYTES_100M" base.1B.u8bin > base.100M.u8bin
head -c "$BYTES_10M"  base.1B.u8bin > base.10M.u8bin
verify base.100M.u8bin "$BYTES_100M"
verify base.10M.u8bin  "$BYTES_10M"

cat <<EOF

All raw files are in: $(pwd)
  gist-960-euclidean.hdf5   sift-128-euclidean.hdf5
  base.1B.u8bin  base.100M.u8bin  base.10M.u8bin  query.public.10K.u8bin

Next (convert each to a .duckdb with your existing scripts):

  # gist1m / sift1m -- HDF5 path, builds gt + gt_cosine
  bench/vector/data/hdf5-to-duckdb/hdf5_to_parquet.py --hdf5 $(pwd)/gist-960-euclidean.hdf5 --out <dir>/parquet
  bench/vector/data/hdf5-to-duckdb/build_duckdb.sh 960 gist1m
  #   ...same for sift (dim 128, name sift1m) using sift-128-euclidean.hdf5

  # bigann 10m / 100m / 1b -- u8bin path, base+queries only
  bench/vector/data/u8bin-to-duckdb/u8bin_to_parquet.py --base $(pwd)/base.10M.u8bin  --queries $(pwd)/query.public.10K.u8bin --out <dir>/parquet
  bench/vector/data/u8bin-to-duckdb/u8bin_to_parquet.py --base $(pwd)/base.100M.u8bin --queries $(pwd)/query.public.10K.u8bin --out <dir>/parquet
  bench/vector/data/u8bin-to-duckdb/u8bin_to_parquet.py --base $(pwd)/base.1B.u8bin   --queries $(pwd)/query.public.10K.u8bin --out <dir>/parquet
EOF

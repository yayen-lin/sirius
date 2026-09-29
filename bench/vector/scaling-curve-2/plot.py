"""Scaling curve 2: exact N x N self join as N grows (dim 128)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_style import dual_plot, parse  # noqa: E402

DATA = """
engine         size reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb           1k    1        1374   128        1000         1000       1e+06    1.28e+08       123.0       123.0       123.0    0.0000009609                     1.04
duckdb          10k    1       77468   128       10000        10000       1e+08    1.28e+10     12923.0     12923.0     12923.0    0.0000010096                     0.99
duckdb         100k    1     5069692   128      100000       100000       1e+10    1.28e+12   1235557.0   1235557.0   1235557.0    0.0000009653                     1.04
duckdb           1m    1     TIMEOUT   128     1000000      1000000       1e+12    1.28e+14     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb          10m    1     TIMEOUT   128    10000000     10000000       1e+14    1.28e+16     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius           1k    1        1374   128        1000         1000       1e+06    1.28e+08        29.0        29.0        29.0    0.0000002266                     4.41
sirius          10k    1       77468   128       10000        10000       1e+08    1.28e+10        34.0        34.0        34.0    0.0000000027                   376.47
sirius         100k    1     5069692   128      100000       100000       1e+10    1.28e+12       460.0       460.0       460.0    0.0000000004                  2782.61
sirius           1m    1   524865066   128     1000000      1000000       1e+12    1.28e+14     36808.0     36808.0     36808.0    0.0000000003                  3477.50
sirius          10m    1 56234686976   128    10000000     10000000       1e+14    1.28e+16   3868678.0   3868678.0   3868678.0    0.0000000003                  3308.62
"""

if __name__ == "__main__":
    dual_plot(parse(DATA), ["sirius", "duckdb"], "corpus_rows", "N × N self join",
              HERE / "scaling-curve-2", xticks=[1e3, 1e4, 1e5, 1e6, 1e7])

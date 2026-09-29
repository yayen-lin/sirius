"""Scaling curve 1: exact vector join as the corpus grows (10K probes, dim 128)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_style import dual_plot, parse  # noqa: E402

DATA = """
engine       corpus reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb     bigann1m   10    12404959   128       10000      1000000       1e+10    1.28e+12    167292.0    167573.3    167978.0    0.0000001309                     7.64
duckdb    bigann10m    4    52974965   128       10000     10000000       1e+11    1.28e+13    415759.0    416586.8    417446.0    0.0000000325                    30.73
duckdb   bigann100m   10     TIMEOUT   128       10000    100000000       1e+12    1.28e+14     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius     bigann1m   10    12404959   128       10000      1000000       1e+10    1.28e+12       520.0       528.8       571.0    0.0000000004                  2420.57
sirius    bigann10m   10    52974965   128       10000     10000000       1e+11    1.28e+13      4220.0      4273.9      4326.0    0.0000000003                  2994.92
sirius   bigann100m   10   520018995   128       10000    100000000       1e+12    1.28e+14     53089.0     53724.7     55092.0    0.0000000004                  2382.52
"""

if __name__ == "__main__":
    dual_plot(parse(DATA), ["sirius", "duckdb"], "corpus_rows", "Corpus size",
              HERE / "scaling-curve-1", xticks=[1e6, 1e7, 1e8])

"""Scaling curve 4: exact vector join as the probe side grows (10M corpus, dim 128)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_style import dual_plot, parse  # noqa: E402

DATA = """
engine        probe reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb            1   10          85   128           1     10000000       1e+07    1.28e+09       153.0       165.4       182.0    0.0000001292                     7.74
duckdb           10   10         701   128          10     10000000       1e+08    1.28e+10       537.0       546.0       558.0    0.0000000427                    23.44
duckdb          100   10       65161   128         100     10000000       1e+09    1.28e+11      4174.0      4219.6      4336.0    0.0000000330                    30.33
duckdb           1k   10     1674893   128        1000     10000000       1e+10    1.28e+12     40450.0     40665.5     40869.0    0.0000000318                    31.48
duckdb          10k    8    58920227   128       10000     10000000       1e+11    1.28e+13    404026.0    405190.5    407182.0    0.0000000317                    31.59
duckdb         100k   10     TIMEOUT   128      100000     10000000       1e+12    1.28e+14     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb           1m   10     TIMEOUT   128     1000000     10000000       1e+13    1.28e+15     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb          10m   10     TIMEOUT   128    10000000     10000000       1e+14    1.28e+16     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius            1   10          85   128           1     10000000       1e+07    1.28e+09       804.0       851.4       917.0    0.0000006652                     1.50
sirius           10   10         701   128          10     10000000       1e+08    1.28e+10       822.0       841.5       889.0    0.0000000657                    15.21
sirius          100   10       65161   128         100     10000000       1e+09    1.28e+11       844.0       893.2       940.0    0.0000000070                   143.30
sirius           1k   10     1674893   128        1000     10000000       1e+10    1.28e+12      1003.0      1022.4      1078.0    0.0000000008                  1251.96
sirius          10k   10    58920227   128       10000     10000000       1e+11    1.28e+13      4272.0      4349.1      4482.0    0.0000000003                  2943.14
sirius         100k   10   510250913   128      100000     10000000       1e+12    1.28e+14     37769.0     39011.0     39346.0    0.0000000003                  3281.13
sirius           1m   10  5410900720   128     1000000     10000000       1e+13    1.28e+15    384446.0    384557.9    384697.0    0.0000000003                  3328.50
sirius          10m    1 56234686976   128    10000000     10000000       1e+14    1.28e+16   3907974.0   3907974.0   3907974.0    0.0000000003                  3275.35
"""

if __name__ == "__main__":
    dual_plot(parse(DATA), ["sirius", "duckdb"], "probe_rows", "Probe size",
              HERE / "scaling-curve-4", xticks=[1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

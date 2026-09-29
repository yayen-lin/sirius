"""Scaling curve 3: exact vector join as dimensionality grows (1K probes x 1M corpus)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_style import dual_plot, parse  # noqa: E402

DATA = """
engine          dim reps        rows        probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb          128   10   230428916              1000      1000000       1e+09    1.28e+11     16863.0     16971.9     17169.0    0.0000001326                     7.54
duckdb          256   10    13597287              1000      1000000       1e+09    2.56e+11     35429.0     35646.3     35908.0    0.0000001392                     7.18
duckdb          512   10     1058406              1000      1000000       1e+09    5.12e+11     73322.0     73546.7     73937.0    0.0000001436                     6.96
duckdb          768   10      211676              1000      1000000       1e+09    7.68e+11    112895.0    113663.0    114874.0    0.0000001480                     6.76
duckdb          960   10       81443              1000      1000000       1e+09     9.6e+11    141948.0    142591.1    143435.0    0.0000001485                     6.73
sirius          128   10   230428935              1000      1000000       1e+09    1.28e+11       256.0       277.3       312.0    0.0000000022                   461.59
sirius          256   10    13597294              1000      1000000       1e+09    2.56e+11       289.0       301.1       312.0    0.0000000012                   850.22
sirius          512   10     1058404              1000      1000000       1e+09    5.12e+11       487.0       571.1       614.0    0.0000000011                   896.52
sirius          768   10      211676              1000      1000000       1e+09    7.68e+11       707.0       810.6       904.0    0.0000000011                   947.45
sirius          960   10       81443              1000      1000000       1e+09     9.6e+11       803.0       858.5       905.0    0.0000000009                  1118.23
"""

if __name__ == "__main__":
    dual_plot(parse(DATA), ["sirius", "duckdb"], "dim", "Vector dimension",
              HERE / "scaling-curve-3", xfmt=lambda x, _p=None: f"{x:g}", categorical=True)

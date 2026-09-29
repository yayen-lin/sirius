"""Scaling curve 5: per-row top-k join as k grows (10K probes x 1M corpus, dim 128)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from plot_style import (COLORS, MARKERS, MUTED, engine_handles, figure, headroom, human, log_axis,  # noqa: E402
                        mark_timeouts, parse, save, series)

DATA = """
engine            k reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
cuvs              1   10       10000   128       10000      1000000       1e+10    1.28e+12       730.5       741.3       746.6    0.0000000006                  1726.74
cuvs             16   10      160000   128       10000      1000000       1e+10    1.28e+12       719.1       723.6       726.8    0.0000000006                  1768.94
cuvs             32   10      320000   128       10000      1000000       1e+10    1.28e+12       723.4       727.4       733.2    0.0000000006                  1759.57
cuvs             64   10      640000   128       10000      1000000       1e+10    1.28e+12       806.7       810.6       814.7    0.0000000006                  1579.14
cuvs             65   10      650000   128       10000      1000000       1e+10    1.28e+12       455.3       457.5       461.6    0.0000000004                  2797.97
cuvs            128   10     1280000   128       10000      1000000       1e+10    1.28e+12       459.1       461.2       464.2    0.0000000004                  2775.53
cuvs            256   10     2560000   128       10000      1000000       1e+10    1.28e+12       488.1       489.6       496.1    0.0000000004                  2614.43
cuvs            512   10     5120000   128       10000      1000000       1e+10    1.28e+12      2167.4      2183.9      2200.4    0.0000000017                   586.12
cuvs           1024   10    10240000   128       10000      1000000       1e+10    1.28e+12      2204.9      2221.5      2238.0    0.0000000017                   576.19
duckdb            1   10       10000   128       10000      1000000       1e+10    1.28e+12     12947.0     13093.2     13200.0    0.0000000102                    97.76
duckdb           16   10      160000   128       10000      1000000       1e+10    1.28e+12    175349.0    176199.1    176633.0    0.0000001377                     7.26
duckdb           32   10      320000   128       10000      1000000       1e+10    1.28e+12    174625.0    176214.9    177186.0    0.0000001377                     7.26
duckdb           64   10      640000   128       10000      1000000       1e+10    1.28e+12    174465.0    175330.0    176422.0    0.0000001370                     7.30
duckdb           65   10      650000   128       10000      1000000       1e+10    1.28e+12    175209.0    176059.6    177183.0    0.0000001375                     7.27
duckdb          128   10     1280000   128       10000      1000000       1e+10    1.28e+12    175319.0    176984.0    177742.0    0.0000001383                     7.23
duckdb          256   10     2560000   128       10000      1000000       1e+10    1.28e+12    176800.0    177867.1    178530.0    0.0000001390                     7.20
duckdb          512   10     5120000   128       10000      1000000       1e+10    1.28e+12    179078.0    180532.4    181651.0    0.0000001410                     7.09
duckdb         1024   10    10240000   128       10000      1000000       1e+10    1.28e+12    184461.0    185306.8    186528.0    0.0000001448                     6.91
faiss             1   10       10000   128       10000      1000000       1e+10    1.28e+12       284.8       286.3       287.6    0.0000000002                  4470.21
faiss            16   10      160000   128       10000      1000000       1e+10    1.28e+12       302.2       303.7       305.6    0.0000000002                  4214.43
faiss            32   10      320000   128       10000      1000000       1e+10    1.28e+12       306.0       307.7       310.1    0.0000000002                  4160.28
faiss            64   10      640000   128       10000      1000000       1e+10    1.28e+12       324.4       326.6       332.5    0.0000000003                  3919.10
faiss            65   10      650000   128       10000      1000000       1e+10    1.28e+12       330.0       332.1       337.6    0.0000000003                  3854.61
faiss           128   10     1280000   128       10000      1000000       1e+10    1.28e+12       335.9       341.4       348.7    0.0000000003                  3749.26
faiss           256   10     2560000   128       10000      1000000       1e+10    1.28e+12       373.4       384.3       396.1    0.0000000003                  3331.12
faiss           512   10     5120000   128       10000      1000000       1e+10    1.28e+12       472.6       478.8       485.9    0.0000000004                  2673.41
faiss          1024   10    10240000   128       10000      1000000       1e+10    1.28e+12       830.7       839.1       843.3    0.0000000007                  1525.52
hnsw              1   10       10000   128       10000      1000000       1e+10    1.28e+12     12912.0     13049.9     13271.0    0.0000000102                    98.09
hnsw             16   10      160000   128       10000      1000000       1e+10    1.28e+12    174558.0    175739.2    176854.0    0.0000001373                     7.28
hnsw             32    9      320000   128       10000      1000000       1e+10    1.28e+12    174477.0    175372.3    176408.0    0.0000001370                     7.30
hnsw             64    0     TIMEOUT   128       10000      1000000       1e+10    1.28e+12     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
hnsw             65    0     TIMEOUT   128       10000      1000000       1e+10    1.28e+12     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
hnsw            128    0     TIMEOUT   128       10000      1000000       1e+10    1.28e+12     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
hnsw            256    0     TIMEOUT   128       10000      1000000       1e+10    1.28e+12     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
hnsw            512    0     TIMEOUT   128       10000      1000000       1e+10    1.28e+12     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
hnsw           1024    0     TIMEOUT   128       10000      1000000       1e+10    1.28e+12     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius            1   10       10000   128       10000      1000000       1e+10    1.28e+12     13293.0     13582.3     14083.0    0.0000000106                    94.24
sirius           16   10      160000   128       10000      1000000       1e+10    1.28e+12       821.0       826.0       831.0    0.0000000006                  1549.64
sirius           32   10      320000   128       10000      1000000       1e+10    1.28e+12       822.0       830.8       842.0    0.0000000006                  1540.68
sirius           64   10      640000   128       10000      1000000       1e+10    1.28e+12       920.0       932.2       959.0    0.0000000007                  1373.10
sirius           65   10      650000   128       10000      1000000       1e+10    1.28e+12       569.0       576.9       609.0    0.0000000005                  2218.76
sirius          128   10     1280000   128       10000      1000000       1e+10    1.28e+12       584.0       598.8       625.0    0.0000000005                  2137.61
sirius          256   10     2560000   128       10000      1000000       1e+10    1.28e+12       673.0       681.1       723.0    0.0000000005                  1879.31
sirius          512   10     5120000   128       10000      1000000       1e+10    1.28e+12      3153.0      3179.7      3198.0    0.0000000025                   402.55
sirius         1024   10    10240000   128       10000      1000000       1e+10    1.28e+12      3427.0      3440.7      3466.0    0.0000000027                   372.02
"""

ENGINES = ["sirius", "duckdb", "hnsw", "cuvs", "faiss"]

if __name__ == "__main__":
    rows = parse(DATA)
    fig, ax = figure()
    for e in ENGINES:
        x, y = series(rows, e, "k", "mean_ms")
        # HNSW lands on the brute-force DuckDB times, so dash it to keep both lines visible
        ls = (0, (3, 1.5)) if e == "hnsw" else "-"
        ax.plot(x, y, color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.4, ls=ls, zorder=4 if e == "hnsw" else 3)
    any_to = mark_timeouts(ax, rows, ENGINES, "k")

    ax.set_xscale("log", base=2)
    # 65 is plotted but not labeled: on a log axis it sits on top of 64
    ax.set_xticks([1, 16, 32, 64, 128, 256, 512, 1024])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelsize=6.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g}"))
    ax.set_yscale("log")
    if any_to:
        headroom(ax)
    log_axis(ax.yaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(human))
    ax.set_xlabel("k")
    ax.set_ylabel("Mean time (ms)")

    # every k does the same 1.28e12 distance ops, so throughput is just 1280 / ms: a relabeled time axis
    ops_per_ms = rows[0]["dist_ops"] / 1e9 * 1e3
    inv = lambda v: ops_per_ms / np.maximum(v, 1e-12)  # noqa: E731
    ax2 = ax.secondary_yaxis("right", functions=(inv, inv))
    log_axis(ax2.yaxis)
    ax2.yaxis.set_major_formatter(FuncFormatter(human))
    ax2.set_ylabel("Throughput (billion ops/s)")
    ax2.spines["right"].set_color(MUTED)

    handles = engine_handles(ENGINES)
    handles[ENGINES.index("hnsw")].set_linestyle((0, (3, 1.5)))
    if any_to:
        handles.append(Line2D([], [], color=MUTED, ls="none", marker="x", ms=5, mew=1.2, label="Timeout"))
    save(fig, handles, HERE / "scaling-curve-5", ncol=3)

"""In-memory scaling curve 4: exact threshold join as the probe side grows (10M corpus, dim 128, tables pinned)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from plot_style import (COLORS, MARKERS, MUTED, engine_handles, figure, human, log_axis, parse, save,  # noqa: E402
                        timeouts)

# sirius rows: pinned run; duckdb rows: earlier scaling-curve-4 run (duckdb reads from host memory)
DATA = """
engine        probe reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb            1   10          85   128           1     10000000       1e+07    1.28e+09       151.0       162.4       180.0    0.0000001269                     7.88
duckdb           10   10         701   128          10     10000000       1e+08    1.28e+10       524.0       536.5       563.0    0.0000000419                    23.86
duckdb          100   10       65161   128         100     10000000       1e+09    1.28e+11      4142.0      4169.5      4194.0    0.0000000326                    30.70
duckdb           1k   10     1674893   128        1000     10000000       1e+10    1.28e+12     40166.0     40225.3     40274.0    0.0000000314                    31.82
duckdb          10k   10    58920227   128       10000     10000000       1e+11    1.28e+13    400533.0    401827.2    402429.0    0.0000000314                    31.85
duckdb         100k   10     TIMEOUT   128      100000     10000000       1e+12    1.28e+14     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb           1m   10     TIMEOUT   128     1000000     10000000       1e+13    1.28e+15     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb          10m   10     TIMEOUT   128    10000000     10000000       1e+14    1.28e+16     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius            1   10          85   128           1     10000000       1e+07    1.28e+09        44.0        45.6        47.0    0.0000000356                    28.07
sirius           10   10         701   128          10     10000000       1e+08    1.28e+10        49.0        51.4        60.0    0.0000000040                   249.03
sirius          100   10       65161   128         100     10000000       1e+09    1.28e+11        75.0        76.8        84.0    0.0000000006                  1666.67
sirius           1k   10     1674893   128        1000     10000000       1e+10    1.28e+12       392.0       394.4       396.0    0.0000000003                  3245.44
sirius          10k   10    58920227   128       10000     10000000       1e+11    1.28e+13      3738.0      3785.2      3829.0    0.0000000003                  3381.59
sirius         100k   10   510250913   128      100000     10000000       1e+12    1.28e+14     38259.0     38989.2     39170.0    0.0000000003                  3282.96
sirius           1m   10  5410900720   128     1000000     10000000       1e+13    1.28e+15    390443.0    391101.9    392584.0    0.0000000003                  3272.80
sirius          10m   10 56234686976   128    10000000     10000000       1e+14    1.28e+16   3901989.0   3909989.6   3930612.0    0.0000000003                  3273.67
"""

ENGINES = ["sirius", "duckdb"]


def dist_per_sec(rows, engine):
    """Distance computations per second: pairs / mean time."""
    pts = [(r["probe_rows"], r["pairs"] / (r["mean_ms"] / 1e3)) for r in rows
           if r["engine"] == engine and r["mean_ms"] is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def speedup_label(s):
    return f"{s:.2g}×" if s < 10 else f"{s:.0f}×"


if __name__ == "__main__":
    rows = parse(DATA)
    fig, ax = figure()
    ax.spines["right"].set_visible(False)
    for e in ENGINES:
        x, y = dist_per_sec(rows, e)
        ax.plot(x, y, color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.4, zorder=3)
        # timeouts sit on the last measured throughput, joined by a dotted line (projected from the plateau)
        to = timeouts(rows, e, "probe_rows")
        if to:
            ax.plot([x[-1]] + to, [y[-1]] * (len(to) + 1), color=COLORS[e], lw=1.2, ls=":", zorder=2)
            ax.plot(to, [y[-1]] * len(to), ls="none", marker="x", ms=5, mew=1.2, color=COLORS[e], zorder=5)
    any_to = any(timeouts(rows, e, "probe_rows") for e in ENGINES)

    # Sirius speedup over DuckDB at each probe size both engines finish
    duck = {r["probe_rows"]: r["mean_ms"] for r in rows if r["engine"] == "duckdb" and r["mean_ms"] is not None}
    sx, sy = dist_per_sec(rows, "sirius")
    for x, y in zip(sx, sy):
        if x not in duck:
            continue
        s = duck[x] / next(r["mean_ms"] for r in rows if r["engine"] == "sirius" and r["probe_rows"] == x)
        # right of the point when Sirius is the lower line, so the label stays off the DuckDB line and the axis
        if s < 1:
            xytext, ha, va = (6, 0), "left", "center"
        else:
            xytext, ha, va = (0, 5), "center", "bottom"
        ax.annotate(speedup_label(s), (x, y), xytext=xytext, textcoords="offset points",
                    ha=ha, va=va, fontsize=6.5, color=COLORS["sirius"])

    ax.set_xscale("log")
    ax.set_xticks([1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(human))
    ax.set_yscale("log")
    # end on a labeled decade above the Sirius plateau
    ax.set_ylim(top=1e11)
    log_axis(ax.yaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(human))
    ax.set_xlabel(r"Probe rows ($m$)")
    ax.set_ylabel("Throughput")

    handles = engine_handles(ENGINES)
    if any_to:
        handles.append(Line2D([], [], color=MUTED, ls="none", marker="x", ms=5, mew=1.2, label="Timeout"))
    save(fig, handles, HERE / "scaling-curve-4", ncol=3)

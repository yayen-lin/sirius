"""Shared table parsing and styling for the scaling-curve plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# fixed engine -> style so an engine looks the same in every figure
COLORS = {"sirius": "#2a78d6", "duckdb": "#eb6834", "cuvs": "#1baf7a", "faiss": "#eda100", "hnsw": "#e87ba4"}
MARKERS = {"sirius": "o", "duckdb": "s", "cuvs": "^", "faiss": "D", "hnsw": "v"}
LABELS = {"sirius": "Sirius", "duckdb": "DuckDB", "cuvs": "cuVS", "faiss": "FAISS", "hnsw": "DuckDB HNSW"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "pdf.fonttype": 42,  # embed TrueType, not Type 3 (camera-ready checks reject Type 3)
    "ps.fonttype": 42,
})


def parse(table):
    """Parse the run.sh result table; TIMEOUT cells become None."""
    lines = [l.split() for l in table.strip().splitlines() if l.strip() and not l.startswith("#")]
    header, rows = lines[0], []
    for cells in lines[1:]:
        row = {}
        for k, v in zip(header, cells):
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = None if v in ("TIMEOUT", "-") else v
        rows.append(row)
    return rows


def human(x, _pos=None):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{x / div:g}{suf}"
    return f"{x:g}"


def series(rows, engine, xcol, ycol):
    pts = [(r[xcol], r[ycol]) for r in rows if r["engine"] == engine and r[ycol] is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def timeouts(rows, engine, xcol):
    return [r[xcol] for r in rows if r["engine"] == engine and r["mean_ms"] is None]


def log_axis(axis):
    axis.set_major_locator(LogLocator(base=10))
    axis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    axis.set_minor_formatter(NullFormatter())


def headroom(ax, factor=4.0):
    """Stretch the log y-range upward so timeout marks at the top edge don't sit on data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * factor)


def mark_timeouts(ax, rows, engines, xcol):
    """Engines that timed out get an x pinned to the top edge of the plot."""
    any_to = False
    for e in engines:
        xs = timeouts(rows, e, xcol)
        if xs:
            any_to = True
            ax.plot(xs, [0.95] * len(xs), transform=ax.get_xaxis_transform(), ls="none",
                    marker="x", ms=5, mew=1.2, color=COLORS[e], clip_on=False, zorder=5)
    return any_to


def engine_handles(engines):
    return [Line2D([], [], color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.4, label=LABELS[e]) for e in engines]


def figure():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    return fig, ax


def dual_plot(rows, engines, xcol, xlabel, out, xticks=None, xfmt=human, categorical=False):
    """Mean time (solid, left axis) and throughput (dashed, right axis) per engine."""
    if categorical:
        # place the x values evenly by rank instead of by magnitude
        xvals = sorted({r[xcol] for r in rows})
        rows = [{**r, xcol: xvals.index(r[xcol])} for r in rows]
    fig, ax = figure()
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    for e in engines:
        x, y = series(rows, e, xcol, "mean_ms")
        ax.plot(x, y, color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.4, zorder=3)
        x, y = series(rows, e, xcol, "billion_dist_ops_per_sec")
        ax2.plot(x, y, color=COLORS[e], marker=MARKERS[e], ms=4, lw=1.2, ls=(0, (3, 1.5)),
                 mfc="white", mew=1.0, zorder=3)
    any_to = mark_timeouts(ax, rows, engines, xcol)

    for a in (ax, ax2):
        a.set_yscale("log")
        if any_to:
            headroom(a)
        log_axis(a.yaxis)
        a.yaxis.set_major_formatter(FuncFormatter(human))
    if categorical:
        ax.set_xticks(range(len(xvals)), [xfmt(v) for v in xvals])
        ax.set_xlim(-0.4, len(xvals) - 0.6)
    elif xticks is not None:
        ax.set_xscale("log")
        ax.set_xticks(xticks)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(xfmt))
    else:
        ax.set_xscale("log")
        log_axis(ax.xaxis)
        ax.xaxis.set_major_formatter(FuncFormatter(xfmt))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean time (ms)")
    ax2.set_ylabel("Throughput (billion ops/s)")
    ax2.spines["right"].set_color(MUTED)

    handles = engine_handles(engines) + [
        Line2D([], [], color=MUTED, lw=1.4, label="Time"),
        Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 1.5)), marker="o", ms=4, mfc="white", label="Throughput"),
    ]
    if any_to:
        handles.append(Line2D([], [], color=MUTED, ls="none", marker="x", ms=5, mew=1.2, label="Timeout"))
    save(fig, handles, out)


def save(fig, handles, out, ncol=5):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles), ncol),
               frameon=False, handlelength=2.2, columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out.with_suffix('.pdf')}")

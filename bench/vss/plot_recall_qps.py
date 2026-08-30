#!/usr/bin/env python3
"""Plot the recall vs QPS tradeoff (Sirius vs Lance vs DuckDB) from a CSV.

Usage:
  python bench/vss/plot_recall_qps.py [--csv recall_qps.csv] [--out recall_qps.png] [--target 0.95]
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# color = engine; metric = line style + marker shape (l2 = solid circle,
# cosine = dashed X) so the two metrics read clearly even when curves overlap
COLOR = {"sirius": "#2a78d6", "lance": "#eb6834", "duckdb": "#3c9e58"}
DASH = {"l2": "-", "cosine": "--"}
MARKER = {"l2": "o", "cosine": "X"}
MSIZE = {"l2": 5, "cosine": 6.5}
LABEL = {"sirius": "Sirius", "lance": "Lance", "duckdb": "DuckDB"}


def load(path):
    series = defaultdict(list)  # (engine, metric) -> list of (nprobes, recall, qps)
    with open(path) as f:
        for row in csv.DictReader(f):
            series[(row["engine"], row["metric"])].append(
                (int(row["nprobes"]), float(row["recall"]), float(row["qps"]))
            )
    for k in series:
        series[k].sort()  # by nprobes
    return series


def crossover(points, target):
    """First (nprobes, recall, qps) whose recall >= target, or None."""
    for np_, r, q in points:
        if r >= target:
            return np_, r, q
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bench/vss/recall_qps.csv")
    ap.add_argument("--out", default="bench/vss/recall_qps.png")
    ap.add_argument("--target", type=float, default=0.95)
    args = ap.parse_args()

    series = load(args.csv)

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    fig.patch.set_facecolor("white")

    # recall >= target reference band + line
    ax.axvspan(args.target, 1.0, color="#000000", alpha=0.035, zorder=0)
    ax.axvline(args.target, color="#c3c2b7", lw=1, ls=(0, (3, 4)), zorder=1)
    ax.text(args.target + 0.003, 0.97, f"recall ≥ {args.target:g}",
            transform=ax.get_xaxis_transform(), va="top", ha="left",
            fontsize=9, color="#898781")

    # one line per (engine, metric); collect where each first clears the target.
    # color = engine, and metric gets both a line style AND a marker shape so the
    # two metrics are distinguishable even where the curves sit on top of one another
    crossings = []  # (engine, metric, nprobes, recall, qps) -- cleared the target
    missed = []     # (engine, metric, best_recall)          -- never cleared it
    for (engine, metric), pts in sorted(series.items()):
        recs = [p[1] for p in pts]
        qpss = [p[2] for p in pts]
        ax.plot(recs, qpss, DASH[metric], color=COLOR[engine], lw=2,
                marker=MARKER[metric], ms=MSIZE[metric], mec="white", mew=0.8,
                zorder=3, label=f"{LABEL[engine]} · {metric}")

        # hollow ring at the first point with recall >= target
        c = crossover(pts, args.target)
        if c:
            np_, r, q = c
            ax.plot([r], [q], marker="o", ms=13, mfc="none",
                    mec=COLOR[engine], mew=2.0, zorder=4)
            crossings.append((engine, metric, np_, r, q))
        else:
            missed.append((engine, metric, max(recs)))

    # single boxed panel in the empty upper-left band, so the per-series crossover
    # numbers never overlap each other or the curves
    if crossings:
        rows = sorted(crossings, key=lambda c: -c[4])  # fastest first
        missed_rows = sorted(missed, key=lambda m: -m[2])
        x0, ytop, dy = 0.075, 0.70, 0.058
        n = len(rows) + len(missed_rows)
        panel = FancyBboxPatch(
            (x0 - 0.035, ytop - (n - 1) * dy - 0.045), 0.40, n * dy + 0.085,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            transform=ax.transAxes, facecolor="#faf9f5", edgecolor="#d4d3ca",
            lw=1, zorder=3.4)
        ax.add_patch(panel)
        ax.text(x0, ytop + dy, f"QPS @ recall ≥ {args.target:g}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="#52514e", va="bottom", ha="left", family="monospace",
                zorder=4)
        i = 0
        for engine, metric, np_, r, q in rows:
            ax.text(x0, ytop - i * dy,
                    f"{LABEL[engine]:6s} {metric:6s} {q:5.0f} qps",
                    transform=ax.transAxes, fontsize=9.5, color=COLOR[engine],
                    va="top", ha="left", family="monospace", zorder=4)
            i += 1
        for engine, metric, best in missed_rows:
            ax.text(x0, ytop - i * dy,
                    f"{LABEL[engine]:6s} {metric:6s}   —  max {best:.2f}",
                    transform=ax.transAxes, fontsize=9.5, color="#a9a89f",
                    va="top", ha="left", family="monospace", zorder=4)
            i += 1

    ax.set_yscale("log")
    ax.set_xlim(0.35, 1.005)
    ax.set_ylim(3.5, 260)
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries / sec  (log)")
    ax.set_title("Recall vs QPS  ·  gist-960 (1M×960, k=10)  ·  "
                 "Sirius/Lance IVF-Flat, DuckDB HNSW",
                 fontsize=12, loc="left", pad=12)

    ax.grid(True, which="major", color="#e1e0d9", lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=3)

    fig.text(0.5, -0.02,
             "QPS is serial (1 / mean latency); the single-query call can't batch, so Sirius's GPU number is a floor.",
             ha="center", fontsize=8, color="#898781")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Turn run_nuscenes.py CSVs into the report table and figures.

  python scripts/make_figures.py results/dense_ped.csv results/dense_car.csv ... --threshold 0

Produces (in --outdir, default results/figures):
  table.md               FPR / TPR per run and per object type (like paper Table 1)
  range_<run>.png        TPR vs distance of the fake object, per object type
  metric_hist_<run>.png  CD vs DCD vs ours, benign vs attacked (like paper Fig. 6)
  roc_<run>.png          TPR vs FPR when sweeping the score threshold
  timing.txt             average runtime per frame
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.metrics import rates  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENIGN, ATTACKED = "#6b6b6b", "#e8913a"


KIND_COLORS = {"pedestrian": "#3b6fb6", "cyclist": "#4c9a5a", "car": "#e8913a"}


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def col(variant, key, kind=None):
        return np.array([float(r[key]) for r in rows if r["variant"] == variant
                         and (kind is None or r.get("kind") == kind) and r.get(key, "") != ""])
    return rows, col


def range_figure(rows, name, outdir, threshold, bin_width=5.0):
    att = [r for r in rows if r["variant"] == "attacked" and r.get("fake_range", "") != ""]
    if not att:
        return
    kinds = [k for k in ("pedestrian", "cyclist", "car") if any(r["kind"] == k for r in att)]
    edges = np.arange(0, max(float(r["fake_range"]) for r in att) + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for k in kinds:
        rr = np.array([float(r["fake_range"]) for r in att if r["kind"] == k])
        hit = np.array([float(r["score"]) > threshold for r in att if r["kind"] == k])
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rr >= lo) & (rr < hi)
            if m.sum() >= 3:                       # skip bins with too few frames to mean anything
                xs.append((lo + hi) / 2); ys.append(100 * hit[m].mean()); ns.append(int(m.sum()))
        if xs:
            ax.plot(xs, ys, "-o", ms=4, color=KIND_COLORS[k], label=k)
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=KIND_COLORS[k])
    ax.set_ylim(-3, 105)
    ax.set_xlabel("distance of fake object (m)")
    ax.set_ylabel("TPR (%)")
    ax.set_title(f"Detection vs distance: {name}  (numbers = frames)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"range_{name}.png"), dpi=150)
    plt.close(fig)


def hist_figure(col, name, outdir, threshold):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, title in [(axes[0], "cd", "(a) Chamfer Distance"),
                           (axes[1], "dcd", "(b) Density-aware CD"),
                           (axes[2], "score", "(c) Ours: points without history")]:
        b, a = col("benign", key), col("attacked", key)
        b, a = b[np.isfinite(b)], a[np.isfinite(a)]
        if len(b) + len(a) == 0:
            continue
        lo, hi = min(b.min(initial=np.inf), a.min(initial=np.inf)), max(b.max(initial=-np.inf), a.max(initial=-np.inf))
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 40)
        ax.hist(b, bins, color=BENIGN, alpha=0.75, label=f"Benign (mean {b.mean():.3g})")
        ax.hist(a, bins, color=ATTACKED, alpha=0.75, label=f"Poisoned (mean {a.mean():.3g})")
        if key == "score":
            ax.axvline(threshold + 0.5, color="#d1342f", ls="--", lw=1, label="threshold")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("value")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("# frames")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"metric_hist_{name}.png"), dpi=150)
    plt.close(fig)


def roc_figure(col, name, outdir):
    b, a = col("benign", "score"), col("attacked", "score")
    ths = np.unique(np.r_[-1, b, a])
    pts = np.array([rates(b, a, t) for t in ths]) / 100.0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=2.5, color="#3b6fb6")
    ax.plot([0, 1], [0, 1], ls=":", color="#999999")
    ax.scatter([0], [1], marker="s", color="#d1342f", zorder=5, label="ideal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Score-threshold sweep: {name}", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"roc_{name}.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--threshold", type=int, default=0, help="frame attacked if score > threshold")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    table = ["| run | object | frames (benign / attacked) | FPR % ↓ | TPR % ↑ | fake points caught % |",
             "|---|---|---|---|---|---|"]
    timing = []
    for path in args.csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        rows, col = load(path)
        b = col("benign", "score")
        kinds = [k for k in ("pedestrian", "cyclist", "car") if any(r.get("kind") == k for r in rows)]
        for kind in kinds + (["all"] if len(kinds) > 1 else []):
            kk = None if kind == "all" else kind
            a = col("attacked", "score", kk)
            fpr, tpr = rates(b, a, args.threshold)
            caught = col("attacked", "fake_points_caught", kk).sum() / max(col("attacked", "n_fake", kk).sum(), 1) * 100
            table.append(f"| {name} | {kind} | {len(b)} / {len(a)} | {fpr:.1f} | {tpr:.1f} | {caught:.1f} |")
        range_figure(rows, name, args.outdir, args.threshold)
        t_syn, t_chk = col("benign", "t_synth"), col("benign", "t_check")
        timing.append(f"{name}: synthesis {t_syn.mean():.2f}s (includes new pair fits), "
                      f"check {t_chk.mean():.2f}s per frame")
        hist_figure(col, name, args.outdir, args.threshold)
        roc_figure(col, name, args.outdir)

    md = "\n".join(table)
    open(os.path.join(args.outdir, "table.md"), "w", encoding="utf-8").write(md + "\n")
    open(os.path.join(args.outdir, "timing.txt"), "w", encoding="utf-8").write("\n".join(timing) + "\n")
    print(md)
    print("\n".join(timing))
    print("figures ->", args.outdir)


if __name__ == "__main__":
    main()

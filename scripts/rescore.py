"""Recompute the detection results of a finished run from its saved clouds.

The scene flow is the slow part; its output (warped synthesis + new frame) is
saved by `run_nuscenes.py --save-clouds`. This script re-runs only the cheap
clustering step with new detection settings and writes a new CSV that
make_figures.py can read. No GPU needed.

  python scripts/rescore.py results/full/dense.csv                       # paper DBSCAN + 0.3 m height filter
  python scripts/rescore.py results/full/dense.csv --min-height 0 --min-range 0   # paper only, no extra filters
  python scripts/rescore.py results/full/sparse.csv --eps 0.75 --min-samples 9

Output: <csv>_rescored.csv next to the input (or --out).
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.detector import cluster_consistency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--clouds", default=None, help="cloud folder (default: <csv>_clouds)")
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--min-samples", type=int, default=17)
    ap.add_argument("--min-height", type=float, default=0.3)
    ap.add_argument("--min-range", type=float, default=4.0, help="ignore history-free clusters closer than this (m)")
    ap.add_argument("--threshold", type=int, default=0, help="frame attacked if score > threshold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = os.path.splitext(args.csv)[0]
    clouds = args.clouds or base + "_clouds"
    out = args.out or base + "_rescored.csv"

    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)

    changed = missing = 0
    for r in rows:
        name = f"{r['sample']}_benign" if r["variant"] == "benign" else f"{r['sample']}_attacked_{r['kind']}"
        path = os.path.join(clouds, name + ".npz")
        if not os.path.exists(path):
            missing += 1
            continue
        d = np.load(path)
        res = cluster_consistency(d["synth"], d["new"], args.eps, args.min_samples, args.min_height, args.min_range)
        fake = d["fake"].astype(bool)
        new_vals = {
            "score": res["score"],
            "attacked": int(res["score"] > args.threshold),
            "n_clusters_before": res["n_clusters_before"],
            "n_clusters_after": res["n_clusters_after"],
            "largest_leftover": res["largest_leftover"],
            "fake_points_caught": int((res["leftover_mask"] & fake).sum()) if fake.any() else 0,
        }
        if str(new_vals["attacked"]) != str(r["attacked"]):
            changed += 1
        r.update({k: v for k, v in new_vals.items() if k in r})

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"settings: eps={args.eps}, min_samples={args.min_samples}, min_height={args.min_height} m, "
          f"min_range={args.min_range} m")
    print(f"{len(rows)} rows, {changed} decisions changed, {missing} without saved clouds (kept as they were)")
    print("->", out)


if __name__ == "__main__":
    main()

"""DBSCAN threshold search (paper App. B, Fig. 12) on saved clouds.

Needs a run made with --save-clouds. Re-runs only the clustering step for
every (min_samples, eps) pair, so it is fast compared with scene flow.

  python scripts/sweep_dbscan.py results/dense_ped_clouds --name dense_ped
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.detector import cluster_consistency  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud_dir")
    ap.add_argument("--name", default="run")
    ap.add_argument("--min-samples", type=int, nargs="+", default=[5, 9, 13, 17])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    ap.add_argument("--threshold", type=int, default=0)
    ap.add_argument("--min-height", type=float, default=0.3, help="ignore flatter history-free clusters (m)")
    ap.add_argument("--min-range", type=float, default=4.0, help="ignore history-free clusters closer than this (m)")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.cloud_dir, "*.npz")))
    data = [(("_attacked" in f), np.load(f)) for f in files]
    print(f"{len(data)} clouds")

    results = []   # (min_samples, eps, fpr, tpr)
    for ms in args.min_samples:
        for eps in args.eps:
            flags = {True: [], False: []}
            for is_att, d in data:
                r = cluster_consistency(d["synth"], d["new"], eps, ms, args.min_height, args.min_range)
                flags[is_att].append(r["score"] > args.threshold)
            fpr, tpr = np.mean(flags[False]), np.mean(flags[True])
            results.append((ms, eps, fpr, tpr))
            print(f"min_samples={ms:2d} eps={eps:.2f}  FPR={100 * fpr:5.1f}%  TPR={100 * tpr:5.1f}%", flush=True)

    res = np.array(results)
    dist = np.hypot(res[:, 2], 1 - res[:, 3])           # distance to the ideal point (0, 1)
    best = res[dist.argmin()]
    print(f"\nbest (closest to FPR=0, TPR=1): min_samples={int(best[0])} eps={best[1]:.2f} "
          f"FPR={100 * best[2]:.1f}% TPR={100 * best[3]:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 4.3))
    sc = None
    for ms in args.min_samples:
        sub = res[res[:, 0] == ms]
        ax.plot(sub[:, 2], sub[:, 3], lw=1, label=f"min_samples={ms}")
        sc = ax.scatter(sub[:, 2], sub[:, 3], c=sub[:, 1], cmap="viridis", s=18, zorder=3,
                        vmin=min(args.eps), vmax=max(args.eps))
    ax.scatter([0], [1], marker="s", color="#d1342f", s=40, zorder=4, label="ideal")
    ax.plot([0, 1], [0, 1], ls=":", color="#999999")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"DBSCAN threshold search: {args.name}", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.colorbar(sc, ax=ax, label="eps (m)")
    fig.tight_layout()
    out = os.path.join(args.outdir, f"dbscan_sweep_{args.name}.png")
    fig.savefig(out, dpi=150)
    np.savetxt(out.replace(".png", ".csv"), res, delimiter=",", header="min_samples,eps,fpr,tpr", comments="")
    print("saved", out)


if __name__ == "__main__":
    main()

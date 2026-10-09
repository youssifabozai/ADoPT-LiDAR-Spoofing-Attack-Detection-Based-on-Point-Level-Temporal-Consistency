"""Explain why each injected fake object was caught or missed.

Works on clouds saved with --save-clouds. For every attacked frame it reports:
  * how many fake points ended up flagged
  * whether the fake alone is dense enough to form a DBSCAN cluster
  * whether it touches the history synthesis (-> cluster gets deleted),
    both after the final scene flow (hist<eps) and before it (raw<eps).
    hist > 0 but raw = 0 means the final scene flow PULLED history onto the fake (paper Fig. 7a)
  * whether it touches a real object in the new frame (-> merged, App. C failure case)
and saves a zoomed top-down picture of every missed frame.

It then explains every FALSE ALARM (benign frame with score > 0): for each
history-free cluster it reports its distance from the car, its size, and how far
the nearest history point is before and after the final scene flow:
  * nothing within 2 m before the flow  -> newly visible object (was hidden or out of range)
  * history close before, but not after -> the final scene flow moved history away (flow error)
  * history close both times (< 2 m)    -> object moved more than the flow explained

  python scripts/inspect_misses.py results/debug_clouds
  python scripts/inspect_misses.py results/debug_clouds --eps 0.25 --min-samples 17 --all
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.detector import cluster_consistency  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reason(caught, self_cluster, touches_hist, touches_real, touches_raw=None):
    if caught:
        return "caught"
    if self_cluster == 0:
        return "too sparse (no cluster on its own)"
    if touches_hist and touches_raw is False:
        return "history PULLED onto fake by final scene flow"
    if touches_hist:
        return "touches history points"
    if touches_real:
        return "merged with a real object"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud_dir")
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--min-samples", type=int, default=17)
    ap.add_argument("--min-height", type=float, default=0.3, help="ignore flatter history-free clusters (m)")
    ap.add_argument("--min-range", type=float, default=4.0, help="ignore history-free clusters closer than this (m)")
    ap.add_argument("--all", action="store_true", help="also plot caught frames")
    ap.add_argument("--outdir", default="results/figures/misses")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.cloud_dir, "*_attacked*.npz")))
    print(f"{len(files)} attacked frames   (eps={args.eps}, min_samples={args.min_samples})\n")
    print(f"{'frame':<20} {'range':>6} {'fake':>5} {'flagged':>7} {'own clus':>8} "
          f"{'hist<eps':>8} {'raw<eps':>8} {'real<eps':>8}  reason")
    print("-" * 106)

    counts = {}
    for f in files:
        d = np.load(f)
        synth, new, fake = d["synth"], d["new"], d["fake"].astype(bool)
        name = os.path.basename(f).replace(".npz", "").replace("_attacked", "")
        tok, _, kind = name.rpartition("_")
        if kind not in ("pedestrian", "cyclist", "car"):     # older runs: no object type in the file name
            tok, kind = name, ""
        label = tok[:8] + (f" {kind}" if kind else "")
        if fake.sum() == 0:
            print(f"{label:<20} fake object outside detection range")
            continue
        fp, real = new[fake], new[~fake]
        res = cluster_consistency(synth, new, args.eps, args.min_samples, args.min_height, args.min_range)
        flagged = int(res["leftover_mask"][fake].sum())

        lab = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(fp)
        own = int(np.bincount(lab[lab >= 0]).max()) if (lab >= 0).any() else 0
        n_hist = int(sum(len(x) for x in cKDTree(synth).query_ball_point(fp, args.eps))) if len(synth) else 0
        n_real = int(sum(len(x) for x in cKDTree(real).query_ball_point(fp, args.eps))) if len(real) else 0
        raw = d["synth_raw"] if "synth_raw" in d.files else None
        n_raw = int(sum(len(x) for x in cKDTree(raw).query_ball_point(fp, args.eps))) \
            if raw is not None and len(raw) else None
        why = reason(flagged > 0, own, n_hist > 0, n_real > 0, None if n_raw is None else n_raw > 0)
        counts[why] = counts.get(why, 0) + 1
        rng_m = np.linalg.norm(fp[:, :2].mean(0))
        print(f"{label:<20} {rng_m:>5.1f}m {len(fp):>5} {flagged:>7} {own:>8} {n_hist:>8} {('-' if n_raw is None else n_raw):>8} {n_real:>8}  {why}")

        if why != "caught" or args.all:
            c = fp[:, :2].mean(0)
            win = lambda p: p[np.all(np.abs(p[:, :2] - c) < 6, axis=1)]
            fig, ax = plt.subplots(figsize=(5, 5))
            s, n = win(synth), win(real)
            if raw is not None:
                r = win(raw)
                ax.scatter(r[:, 0], r[:, 1], s=6, marker="x", lw=0.6, c="#4c9a5a", label="history BEFORE final flow")
            ax.scatter(s[:, 0], s[:, 1], s=2, c="#b8b8b8", label="history AFTER final flow")
            ax.scatter(n[:, 0], n[:, 1], s=3, c="#3b6fb6", label="new frame (real)")
            ax.scatter(fp[:, 0], fp[:, 1], s=8, c="#d1342f" if flagged else "#e8913a",
                       label="fake (flagged)" if flagged else "fake (missed)")
            ax.set_title(f"{label}  {why}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlim(c[0] - 6, c[0] + 6)
            ax.set_ylim(c[1] - 6, c[1] + 6)
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"{name}.png"), dpi=130)
            plt.close(fig)

    print("\nsummary:", ", ".join(f"{k}: {v}" for k, v in counts.items()))
    explain_false_alarms(args)
    print("pictures ->", args.outdir)


def fa_reason(d_raw, d_warp, near=2.0):
    if d_raw is None:
        return "?"
    if d_raw > near:
        return "newly visible (no history within 2 m)"
    if d_warp > near:
        return "flow moved history away"
    return "moved more than the flow explained"


def explain_false_alarms(args):
    files = sorted(glob.glob(os.path.join(args.cloud_dir, "*_benign.npz")))
    alarms = []
    for f in files:
        d = np.load(f)
        res = cluster_consistency(d["synth"], d["new"], args.eps, args.min_samples, args.min_height, args.min_range)
        if res["score"] > 0:
            alarms.append((f, d, res))
    print(f"\nfalse alarms: {len(alarms)} of {len(files)} benign frames")
    if not alarms:
        return
    print(f"{'frame':<10} {'range':>6} {'points':>6} {'size(m)':>8} {'height':>7} {'hist before':>11} "
          f"{'hist after':>10}  likely cause")
    print("-" * 100)
    for f, d, res in alarms:
        name = os.path.basename(f).replace("_benign.npz", "")[:8]
        new, synth = d["new"], d["synth"]
        raw = d["synth_raw"] if "synth_raw" in d.files else None
        left = new[res["leftover_mask"]]
        lab = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(left)
        for l in np.unique(lab[lab >= 0]):
            c = left[lab == l]
            ctr = c.mean(0)
            d_warp = float(cKDTree(synth).query(c)[0].min()) if len(synth) else np.inf
            d_raw = float(cKDTree(raw).query(c)[0].min()) if raw is not None and len(raw) else None
            ext = np.ptp(c[:, :2], axis=0)
            print(f"{name:<10} {np.linalg.norm(ctr[:2]):>5.1f}m {len(c):>6} {ext[0]:>3.1f}x{ext[1]:<4.1f} "
                  f"{np.ptp(c[:, 2]):>6.1f}m {('-' if d_raw is None else f'{d_raw:.2f}m'):>11} "
                  f"{d_warp:>9.2f}m  {fa_reason(d_raw, d_warp)}")
            fig, ax = plt.subplots(figsize=(5, 5))
            win = lambda p: p[np.all(np.abs(p[:, :2] - ctr[:2]) < 8, axis=1)]
            if raw is not None:
                r = win(raw)
                ax.scatter(r[:, 0], r[:, 1], s=6, marker="x", lw=0.6, c="#4c9a5a", label="history BEFORE final flow")
            sw, nw = win(synth), win(new)
            ax.scatter(sw[:, 0], sw[:, 1], s=2, c="#b8b8b8", label="history AFTER final flow")
            ax.scatter(nw[:, 0], nw[:, 1], s=3, c="#3b6fb6", label="new frame")
            ax.scatter(c[:, 0], c[:, 1], s=8, c="#d1342f", label="false alarm cluster")
            ax.scatter([0], [0], marker="^", s=60, c="black", label="ego car")
            ax.set_title(f"false alarm {name}: {fa_reason(d_raw, d_warp)}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlim(ctr[0] - 8, ctr[0] + 8)
            ax.set_ylim(ctr[1] - 8, ctr[1] + 8)
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"false_alarm_{name}_{l}.png"), dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()

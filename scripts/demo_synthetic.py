"""Run ADoPT on a synthetic street scene (no dataset needed).

  python scripts/demo_synthetic.py            # neural scene flow (needs PyTorch, GPU recommended)
  python scripts/demo_synthetic.py --oracle   # ground-truth flow: tests synthesis + detector only

Prints the detection score for a benign and an attacked frame and saves a
top-down plot to results/demo_synthetic.png.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adopt.config import ADoPTConfig                                       # noqa: E402
from adopt.pipeline import ADoPT                                           # noqa: E402
from adopt.synthetic import OracleFlow, fake_pedestrian, make_sequence     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="use ground-truth flow instead of the neural SFE")
    ap.add_argument("--iters", type=int, default=None, help="override SFE iterations")
    ap.add_argument("--out", default="results/demo_synthetic.png")
    args = ap.parse_args()

    cfg = ADoPTConfig.dense(remove_ground=False)          # the synthetic scene has no ground
    if args.iters:
        cfg.iters = args.iters
    dt = 0.1
    seq = make_sequence(cfg.L + 1, dt)
    vox = lambda p: ADoPT(cfg, fit_fn=lambda s, t: None).preprocess(p)
    frames = [vox(p) for p, _ in seq]
    history, new = frames[:-1], frames[-1]
    fake = fake_pedestrian()
    attacked = np.vstack([new, fake])

    if args.oracle:
        adopt = ADoPT(cfg, fit_fn=lambda s, t: None)
        flows = [OracleFlow(p, v, dt) for p, v in seq[:-2]]
        final = OracleFlow(*seq[-2], dt)
        t0 = time.perf_counter()
        synth = adopt.synthesize(history, flows)
        t_syn = time.perf_counter() - t0
        res_b = adopt.check(synth, new, final)
        res_a = adopt.check(synth, attacked, final)
    else:
        adopt = ADoPT(cfg)
        t0 = time.perf_counter()
        synth = adopt.synthesize(history)
        t_syn = time.perf_counter() - t0
        res_b = adopt.check(synth, new)
        res_a = adopt.check(synth, attacked)

    n_fake_in = len(fake)
    print(f"mode           : {'oracle flow' if args.oracle else 'neural SFE'}")
    print(f"points/frame   : {len(new)}   synthesis: {len(synth)}   synthesis time: {t_syn:.2f}s")
    print(f"benign   frame : score={res_b['score']:4d}  clusters {res_b['n_clusters_before']} -> "
          f"{res_b['n_clusters_after']}  attacked={res_b['attacked']}")
    print(f"attacked frame : score={res_a['score']:4d}  clusters {res_a['n_clusters_before']} -> "
          f"{res_a['n_clusters_after']}  attacked={res_a['attacked']}  "
          f"(fake points caught: {res_a['leftover_mask'][-n_fake_in:].sum()}/{n_fake_in})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, res, title in [(axes[0], res_b, "Benign frame"), (axes[1], res_a, "Attacked frame")]:
        s, n, m = res["warped_synth"], res["new_det"], res["leftover_mask"]
        ax.scatter(s[:, 0], s[:, 1], s=0.2, c="#b8b8b8", label="warped synthesis (history)")
        ax.scatter(n[~m, 0], n[~m, 1], s=0.4, c="#3b6fb6", label="new frame")
        ax.scatter(n[m, 0], n[m, 1], s=6, c="#d1342f", label="no history -> flagged")
        ax.set_title(f"{title}: score = {res['score']}")
        ax.set_aspect("equal")
        ax.set_xlim(-30, 45)
        ax.set_ylim(-14, 14)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    handles, labels = axes[1].get_legend_handles_labels()
    for h in handles:
        h.set_sizes([20])
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()

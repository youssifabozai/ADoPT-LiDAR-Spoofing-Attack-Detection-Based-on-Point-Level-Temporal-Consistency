"""Scene-flow accuracy on the synthetic street, where the true motion is known.

Fits the flow between two consecutive frames (0.1 s apart) for several
settings and reports End-Point Error (EPE = distance between predicted and
true displacement), separately for moving and static points.

  python scripts/diag_sfe.py                                   # speed/accuracy check (default grid)
  python scripts/diag_sfe.py --iters 30 100 300 1000 --lrs 8e-4 3e-3 --betas 0 2 --nn-every 1

Use the result to choose `iters` / `lr` / `nn_every` for nuScenes, and as an
extra table in the report (the paper says 30 iterations; check if that holds).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch                                                   # noqa: E402
from adopt.config import ADoPTConfig                           # noqa: E402
from adopt.sfe import fit_flow                                 # noqa: E402
from adopt.synthetic import make_sequence                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, nargs="+", default=[100, 300])
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-3])
    ap.add_argument("--betas", type=float, nargs="+", default=[2.0])
    ap.add_argument("--nn-every", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--cd-trunc", type=float, default=2.0, help="Chamfer cutoff used for this table (m)")
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--csv", default="results/diag_sfe.csv")
    args = ap.parse_args()

    dt = 0.1
    (p0, v0), (p1, _) = make_sequence(2, dt, seed=3)
    rng = np.random.default_rng(0)
    if len(p0) > args.max_points:
        keep = rng.choice(len(p0), args.max_points, replace=False)
        p0, v0 = p0[keep], v0[keep]
    gt = v0 * dt
    moving = np.linalg.norm(gt, axis=1) > 1e-6
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}   points: {len(p0)} "
          f"({moving.sum()} moving)   true motion of moving points: "
          f"{np.linalg.norm(gt[moving], axis=1).min():.2f}-{np.linalg.norm(gt[moving], axis=1).max():.2f} m")
    print(f"zero-flow baseline (predict no motion): EPE moving = "
          f"{np.linalg.norm(gt[moving], axis=1).mean():.3f} m\n")

    # warm-up so the first row's time is not inflated by CUDA start-up
    fit_flow(p0[:2000], p1[:2000], ADoPTConfig(iters=2, beta=0.0))

    head = (f"{'iters':>6} {'lr':>8} {'beta':>5} {'nn_every':>8} | {'EPE moving':>10} {'acc<0.1m':>9} | "
            f"{'EPE static':>10} | {'time':>6}")
    print(head)
    print("-" * len(head))
    rows = []
    for lr in args.lrs:
        for beta in args.betas:
            for k in args.nn_every:
                for it in args.iters:
                    cfg = ADoPTConfig(iters=it, lr=lr, beta=beta, nn_every=k, cd_trunc=args.cd_trunc)
                    t0 = time.perf_counter()
                    field = fit_flow(p0, p1, cfg)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t = time.perf_counter() - t0
                    err = np.linalg.norm(field(p0) - gt, axis=1)
                    em, acc, es = err[moving].mean(), 100 * (err[moving] < 0.1).mean(), err[~moving].mean()
                    rows.append((it, lr, beta, k, em, acc, es, t))
                    print(f"{it:>6} {lr:>8.0e} {beta:>5.1f} {k:>8} | {em:>9.3f}m {acc:>8.1f}% | "
                          f"{es:>9.3f}m | {t:>5.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    np.savetxt(args.csv, np.array(rows), delimiter=",", comments="",
               header="iters,lr,beta,nn_every,epe_moving,acc_moving,epe_static,time_s")
    print("\nsaved", args.csv)


if __name__ == "__main__":
    main()

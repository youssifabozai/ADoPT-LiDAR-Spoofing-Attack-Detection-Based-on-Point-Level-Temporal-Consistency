"""Accuracy of every trained model on clean data and under FGSM, PGD and both, for several budgets.

  python scripts/adv_eval.py                     # uses results/adv/models/*.pt and objects.npz
  python scripts/adv_eval.py --smoke             # after adv_train.py --smoke

Writes results/adv/eval.csv and prints two tables:
  1. without defense  (standard model)  vs  with defense (adversarially trained models)
  2. accuracy against eps for every model and attack
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.adversarial import MODES, PointNetCls, evaluate  # noqa: E402
from adopt.objects import CLASSES, ObjectSet, synthetic_objects  # noqa: E402

LABEL = {"standard": "no defense", "fgsm": "AT-FGSM", "pgd": "AT-PGD", "both": "AT-FGSM+PGD"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/adv/objects.npz")
    ap.add_argument("--models", default="results/adv/models")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--main-eps", type=float, default=0.05, help="budget shown in the summary table")
    ap.add_argument("--pgd-steps", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--out", default="results/adv/eval.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        data, args.models, args.out = synthetic_objects(150), os.path.join(args.models, "smoke"), "results/adv/smoke_eval.csv"
    else:
        data = ObjectSet.load(args.data)
    te = data.idx(1)
    x = torch.from_numpy(data.batch(te, args.n_points, np.random.default_rng(1234))).to(device)
    y = torch.from_numpy(data.labels[te]).to(device)
    print(f"test objects: {len(te)}  " + "  ".join(f"{c}={int((data.labels[te] == i).sum())}" for i, c in enumerate(CLASSES)))
    if args.main_eps not in args.eps:
        args.eps = sorted(args.eps + [args.main_eps])

    rows = []
    for mode in MODES:
        path = os.path.join(args.models, f"{mode}.pt")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        model = PointNetCls().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        for r in evaluate(model, x, y, args.eps, pgd_steps=args.pgd_steps):
            rows.append({"model": mode, **r})
        print(f"evaluated {mode}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    get = lambda m, a, e: next((r for r in rows if r["model"] == m and r["attack"] == a
                                and (a == "clean" or abs(r["eps"] - e) < 1e-9)), None)
    models = [m for m in MODES if any(r["model"] == m for r in rows)]
    e = args.main_eps
    print(f"\n## Accuracy (%) at eps = {e} m (higher = more robust)\n")
    print("| model | defense | clean | FGSM | PGD | FGSM+PGD |")
    print("|---|---|---|---|---|---|")
    for m in models:
        cells = [get(m, a, e) for a in ("clean", "fgsm", "pgd", "both")]
        print(f"| {m} | {LABEL[m]} | " + " | ".join(f"{100 * c['acc']:.1f}" for c in cells) + " |")

    print(f"\n## Attack success rate (%) at eps = {e} m (share of correct predictions the attack flips)\n")
    print("| model | FGSM | PGD | FGSM+PGD |")
    print("|---|---|---|---|")
    for m in models:
        print(f"| {m} | " + " | ".join(f"{100 * get(m, a, e)['asr']:.1f}" for a in ("fgsm", "pgd", "both")) + " |")

    print("\n## Accuracy (%) against eps\n")
    print("| model | attack | " + " | ".join(f"{v:g} m" for v in args.eps) + " |")
    print("|---|---|" + "---|" * len(args.eps))
    for m in models:
        print(f"| {m} | clean | {100 * get(m, 'clean', 0)['acc']:.1f} |" + " |" * (len(args.eps) - 1))
        for a in ("fgsm", "pgd", "both"):
            print(f"| {m} | {a} | " + " | ".join(f"{100 * get(m, a, v)['acc']:.1f}" for v in args.eps) + " |")

    print(f"\n## Per class, PGD at eps = {e} m\n")
    print("| model | clean " + " ".join(f"| {c} " for c in CLASSES) + "| PGD " + " ".join(f"| {c} " for c in CLASSES) + "|")
    print("|---|---|" + "---|" * (2 * len(CLASSES) + 0) + "---|")
    for m in models:
        c, p = get(m, "clean", 0), get(m, "pgd", e)
        print(f"| {m} | {100 * c['acc']:.1f} " + " ".join(f"| {100 * c['acc_' + n]:.1f} " for n in CLASSES)
              + f"| {100 * p['acc']:.1f} " + " ".join(f"| {100 * p['acc_' + n]:.1f} " for n in CLASSES) + "|")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

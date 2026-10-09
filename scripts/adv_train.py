"""Train the PointNet classifier without and with adversarial training.

  python scripts/adv_train.py --smoke                 # 1-minute check on toy shapes, no nuScenes needed
  python scripts/adv_train.py                         # all four models on results/adv/objects.npz

Models (saved to results/adv/models/<mode>.pt):
  standard  no defense
  fgsm      adversarial training on FGSM examples
  pgd       adversarial training on PGD examples
  both      adversarial training on a mix of FGSM and PGD examples
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.adversarial import MODES, train_model  # noqa: E402
from adopt.objects import CLASSES, ObjectSet, synthetic_objects  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/adv/objects.npz")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--eps", type=float, default=0.05, help="training budget (m per axis)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--pgd-steps", type=int, default=7, help="PGD steps used during training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/adv/models")
    ap.add_argument("--smoke", action="store_true", help="toy data, 2 epochs, just checks that everything runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        data, args.epochs, args.out = synthetic_objects(150), 2, os.path.join(args.out, "smoke")
    else:
        data = ObjectSet.load(args.data)
    os.makedirs(args.out, exist_ok=True)
    n_tr, n_te = (data.split == 0).sum(), (data.split == 1).sum()
    print(f"device={device}  objects: {n_tr} train / {n_te} test  classes={CLASSES}  eps_train={args.eps} m")

    log_rows = []
    for mode in args.modes:
        print(f"\n=== training '{mode}' ===")
        t0 = time.time()
        model, hist = train_model(data, mode, args.eps, epochs=args.epochs, lr=args.lr, batch=args.batch,
                                  n_points=args.n_points, pgd_steps=args.pgd_steps, device=device, seed=args.seed)
        dt = time.time() - t0
        torch.save(model.state_dict(), os.path.join(args.out, f"{mode}.pt"))
        for r in hist:
            r["minutes"] = round(dt / 60, 2)
        log_rows += hist
        print(f"saved {mode}.pt  ({dt / 60:.1f} min)")

    with open(os.path.join(args.out, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["mode", "epoch", "loss", "test_clean_acc", "test_pgd_acc", "minutes"]
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(log_rows)
    json.dump(vars(args), open(os.path.join(args.out, "train_config.json"), "w"), indent=2)
    print("\nfinal test accuracy (clean / PGD-10 at training eps):")
    for mode in args.modes:
        last = [r for r in log_rows if r["mode"] == mode][-1]
        print(f"  {mode:<9} {100 * last['test_clean_acc']:5.1f}% / {100 * last.get('test_pgd_acc', float('nan')):5.1f}%")
    if args.smoke:
        print("\nsmoke test OK - now run the real training without --smoke")


if __name__ == "__main__":
    main()

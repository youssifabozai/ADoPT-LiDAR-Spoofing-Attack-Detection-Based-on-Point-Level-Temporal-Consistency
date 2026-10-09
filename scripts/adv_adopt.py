"""ADoPT as a defense: does it flag objects perturbed by FGSM / PGD?

Uses the frames saved by the full dense run (results/full/dense_clouds/<token>_benign.npz: the
warped history and the new frame) and the test-split objects of results/adv/objects.npz. For each
frame and model, every test object is attacked on its own (FGSM, PGD, and 'both' = PGD if it fools
the model, else FGSM), the perturbed points are put back into the frame, and ADoPT's cluster check
is rerun.

  python scripts/adv_adopt.py --max-frames 3                     # quick check (about a minute)
  python scripts/adv_adopt.py                                     # all frames (about 1-1.5 h)
  python scripts/adv_adopt.py --summary-only                      # reprint the table from the CSV

Writes results/adv/adopt.csv (one row per object, model, attack, eps) and adopt_summary.md.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.adv_check import check_frame, frame_objects, summarize  # noqa: E402
from adopt.config import ADoPTConfig  # noqa: E402
from adopt.objects import ObjectSet, from_canonical, to_canonical  # noqa: E402

FIELDS = ["sample", "obj", "label", "n_pts", "range", "model", "attack", "eps", "pred_clean", "pred_adv",
          "flagged", "frame_score", "max_disp"]


def print_summary(rows, out_md):
    s = summarize(rows)
    lines = ["| model | attack | eps (m) | objects | fooled, no ADoPT % | flagged by ADoPT % | fooled and not flagged % | mean max shift (m) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in s:
        lines.append(f"| {r['model']} | {r['attack']} | {r['eps']:g} | {r['objects']} | {100 * r['fooled']:.1f} | "
                     f"{100 * r['flagged']:.1f} | {100 * r['evaded']:.1f} | {r['mean_max_disp']:.3f} |")
    md = "\n".join(lines)
    print("\n" + md)
    open(out_md, "w", encoding="utf-8").write(md + "\n")
    print(f"\nsaved {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/adv/objects.npz")
    ap.add_argument("--clouds", default="results/full/dense_clouds")
    ap.add_argument("--models-dir", default="results/adv/models")
    ap.add_argument("--models", nargs="+", default=["standard", "both"])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--pgd-steps", type=int, default=20)
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="results/adv/adopt.csv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out_md = os.path.splitext(args.out)[0] + "_summary.md"

    if args.summary_only:
        print_summary(list(csv.DictReader(open(args.out, encoding="utf-8"))), out_md)
        return

    import torch                                               # only needed from here on
    from adopt.adversarial import PointNetCls, fgsm, pgd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ADoPTConfig.dense()
    data = ObjectSet.load(args.data)
    models = {}
    for name in args.models:
        m = PointNetCls().to(device)
        m.load_state_dict(torch.load(os.path.join(args.models_dir, f"{name}.pt"), map_location=device))
        models[name] = m.eval()

    by_sample = {}
    for i in data.idx(1):
        by_sample.setdefault(str(data.sample[i]), []).append(int(i))
    tokens = [t for t in dict.fromkeys(str(s) for s in data.sample)     # keyframe order of the dataset
              if t in by_sample and os.path.exists(os.path.join(args.clouds, f"{t}_benign.npz"))]
    tokens = tokens[::args.sample_every]
    if args.max_frames:
        tokens = tokens[:args.max_frames]
    print(f"{len(tokens)} frames with test objects and saved clouds; models={list(models)}; eps={args.eps}")

    done = set()
    if args.resume and os.path.exists(args.out):
        done = {r["sample"] for r in csv.DictReader(open(args.out, encoding="utf-8"))}
        print(f"resuming: {len(done)} frames already done")
    f = open(args.out, "a" if done else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not done:
        writer.writeheader()

    def classify(model, pts):
        with torch.no_grad():
            return int(model(torch.from_numpy(pts)[None].to(device)).argmax(1).item())

    torch.manual_seed(0)
    for k, tok in enumerate(tokens):
        if tok in done:
            continue
        t0 = time.time()
        z = np.load(os.path.join(args.clouds, f"{tok}_benign.npz"))
        synth, new = z["synth"], z["new"]
        boxes = [(i, data.center[i], data.wlh[i], float(data.yaw[i])) for i in by_sample[tok]]
        objs = frame_objects(new, boxes, min_points=args.min_points)
        if not objs:
            continue
        canon = {i: to_canonical(new[idx], data.center[i], float(data.yaw[i])) for i, idx in objs}
        flagged0, score0 = check_frame(synth, new, objs, {}, cfg)
        rows = []

        def emit(model_name, attack, eps, preds_clean, preds_adv, flagged, score, adv):
            for i, idx in objs:
                disp = np.linalg.norm(adv[i] - canon[i], axis=1).max() if adv else 0.0
                rows.append({"sample": tok, "obj": i, "label": int(data.labels[i]), "n_pts": len(idx),
                             "range": round(float(data.range[i]), 2), "model": model_name, "attack": attack,
                             "eps": eps, "pred_clean": preds_clean[i], "pred_adv": preds_adv[i],
                             "flagged": int(flagged[i]), "frame_score": score, "max_disp": round(float(disp), 4)})

        for name, model in models.items():
            pc = {i: classify(model, canon[i]) for i, _ in objs}
            emit(name, "none", 0.0, pc, pc, flagged0, score0, None)
            for eps in args.eps:
                adv_f, adv_p = {}, {}
                for i, _ in objs:
                    x = torch.from_numpy(canon[i])[None].to(device)
                    y = torch.tensor([int(data.labels[i])], device=device)
                    adv_f[i] = fgsm(model, x, y, eps)[0].cpu().numpy()
                    adv_p[i] = pgd(model, x, y, eps, steps=args.pgd_steps)[0].cpu().numpy()
                pf = {i: classify(model, adv_f[i]) for i in adv_f}
                pp = {i: classify(model, adv_p[i]) for i in adv_p}
                # 'both': the attacker tries both and keeps PGD if it works, otherwise FGSM
                y_of = {i: int(data.labels[i]) for i, _ in objs}
                adv_b = {i: adv_p[i] if pp[i] != y_of[i] or pf[i] == y_of[i] else adv_f[i] for i in adv_p}
                pb = {i: pp[i] if adv_b[i] is adv_p[i] else pf[i] for i in adv_b}
                for attack, adv, pa in (("fgsm", adv_f, pf), ("pgd", adv_p, pp), ("both", adv_b, pb)):
                    ego = {i: from_canonical(adv[i], data.center[i], float(data.yaw[i])) for i in adv}
                    flagged, score = check_frame(synth, new, objs, ego, cfg)
                    emit(name, attack, eps, pc, pa, flagged, score, adv)
        writer.writerows(rows)
        f.flush()
        print(f"[{k + 1}/{len(tokens)}] {tok[:8]}  objects={len(objs)}  clean flagged={sum(flagged0.values())}  "
              f"{time.time() - t0:.1f} s")
    f.close()
    print_summary(list(csv.DictReader(open(args.out, encoding="utf-8"))), out_md)


if __name__ == "__main__":
    main()

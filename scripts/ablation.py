"""Compare several runs on exactly the same keyframes (ablations: paper Fig. 7, 8, 9).

Every run's CSV is reduced to the keyframes present in ALL of them, so the
comparison is fair (fakes are seeded per keyframe, so they are identical too).

  python scripts/ablation.py "L=10 (default)=results/full/dense_rescored.csv" \
                             "L=2=results/abl/L2.csv" "L=5=results/abl/L5.csv" "L=15=results/abl/L15.csv" \
                             --name history_length --x 10 2 5 15 --xlabel "history length L (frames)"

Each argument is  label=path.csv  (the label may itself contain '=' signs; the
path is the part after the last '='). Outputs <outdir>/<name>.md and, with --x,
<outdir>/<name>.png (FPR and TPR against the x values).
"""

import argparse
import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path.csv")
    ap.add_argument("--name", default="ablation")
    ap.add_argument("--x", type=float, nargs="*", default=None, help="numeric x value per run, for the plot")
    ap.add_argument("--xlabel", default="")
    ap.add_argument("--outdir", default="results/ablation")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = []
    for spec in args.runs:
        label, _, path = spec.rpartition("=")
        runs.append((label or os.path.basename(path), load(path)))

    common = set.intersection(*[{r["sample"] for r in rows if r["variant"] == "benign"} for _, rows in runs])
    print(f"{len(common)} keyframes present in all {len(runs)} runs")

    kinds = sorted({r["kind"] for _, rows in runs for r in rows if r["variant"] == "attacked"})
    head = "| run | frames | FPR % ↓ | " + " | ".join(f"TPR {k} % ↑" for k in kinds) + " | TPR all % ↑ | s / keyframe |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    stats = []
    for label, rows in runs:
        rows = [r for r in rows if r["sample"] in common]
        ben = [r for r in rows if r["variant"] == "benign"]
        att = [r for r in rows if r["variant"] == "attacked"]
        fpr = 100 * np.mean([int(r["score"]) > 0 for r in ben])
        tprs = [100 * np.mean([int(r["score"]) > 0 for r in att if r["kind"] == k] or [np.nan]) for k in kinds]
        tpr = 100 * np.mean([int(r["score"]) > 0 for r in att])
        # time per keyframe = synthesis + every check of that keyframe
        t = np.mean([float(b["t_synth"]) + sum(float(r["t_check"]) for r in rows if r["sample"] == b["sample"])
                     for b in ben])
        stats.append((fpr, tpr, t))
        lines.append(f"| {label} | {len(ben)} | {fpr:.1f} | " + " | ".join(f"{x:.1f}" for x in tprs)
                     + f" | {tpr:.1f} | {t:.0f} |")
    md = "\n".join(lines)
    print(md)
    open(os.path.join(args.outdir, args.name + ".md"), "w", encoding="utf-8").write(md + "\n")

    if args.x:
        assert len(args.x) == len(runs), "--x needs one value per run"
        order = np.argsort(args.x)
        xs = np.array(args.x)[order]
        st = np.array(stats)[order]
        fig, ax1 = plt.subplots(figsize=(5, 3.6))
        ax1.plot(xs, st[:, 1], "-o", color="#3b6fb6", label="TPR (all fakes)")
        ax1.plot(xs, st[:, 0], "-s", color="#e8913a", label="FPR")
        ax1.set_ylim(-3, 105)
        ax1.set_xlabel(args.xlabel)
        ax1.set_ylabel("%")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(xs, st[:, 2], ":", color="#6b6b6b", label="seconds / keyframe")
        ax2.set_ylabel("seconds / keyframe", color="#6b6b6b")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
        ax1.set_title(args.name.replace("_", " "), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, args.name + ".png"), dpi=150)
        print("saved", os.path.join(args.outdir, args.name + ".png"))


if __name__ == "__main__":
    main()

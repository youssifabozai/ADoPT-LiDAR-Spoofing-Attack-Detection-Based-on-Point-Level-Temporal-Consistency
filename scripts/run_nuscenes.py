"""Evaluate ADoPT on nuScenes with simulated spoofing attacks.

For every keyframe that has L earlier sweeps (at 10 Hz):
  * build the synthesis from the L history sweeps (benign),
  * check the clean newest sweep                    -> "benign" row   (used for FPR)
  * for each object type: inject a fake and check   -> "attacked" row (used for TPR)

Examples:
  # one pass, all three object types (they share the expensive synthesis)
  python scripts/run_nuscenes.py --dataroot data/nuscenes --attack dense --out results/full/dense.csv --save-clouds
  # sparse attack (cars only, as in the paper)
  python scripts/run_nuscenes.py --dataroot data/nuscenes --attack sparse --out results/full/sparse.csv --save-clouds

Every keyframe gives one benign row plus one attacked row per object type.
Fakes are copied from other scenes at their original distance; by default the
distance is balanced over 5 m bins between --range-min and --range-max.

Outputs a CSV (one row per frame & variant) + config JSON, and optionally
the point clouds (--save-clouds) so that scripts/sweep_dbscan.py can redo
the DBSCAN threshold search (paper Fig. 12) without re-running scene flow.
"""

import argparse
import csv
import zlib
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adopt.attack import inject, pick_object                      # noqa: E402
from adopt.config import ADoPTConfig                               # noqa: E402
from adopt.geometry import crop, transform, voxel_downsample       # noqa: E402
from adopt.metrics import chamfer_np, dcd_np                       # noqa: E402
from adopt.nuscenes_loader import (NuScenes, build_object_bank,    # noqa: E402
                                   ego_a_to_ego_b, history_tokens, load_sweep_ego)
from adopt.pipeline import ADoPT                                   # noqa: E402


class TransformedFlow:
    """Re-use a flow fitted in ego frame A for points given in ego frame C."""

    def __init__(self, field, T_c_to_a):
        self.field, self.T = field, T_c_to_a

    def __call__(self, p):
        fa = self.field(transform(p, self.T))
        return fa @ self.T[:3, :3]          # rotate the displacement back into frame C


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--attack", choices=["dense", "sparse"], default="dense")
    ap.add_argument("--kinds", "--kind", dest="kinds", default="pedestrian,cyclist,car",
                    help="comma-separated object types to inject (one attacked variant each)")
    ap.add_argument("--range-min", type=float, default=5.0, help="min distance (m) of injected fakes")
    ap.add_argument("--range-max", type=float, default=30.0, help="max distance (m) of injected fakes")
    ap.add_argument("--range-mode", choices=["auto", "shift", "natural"], default="auto",
                    help="shift: take a dense object and move it to a random distance in [range-min, range-max], "
                         "keeping its points (attacker-controlled density). natural: use objects seen at that "
                         "distance, with their real density (balanced over 5 m bins). auto: shift for dense, "
                         "natural for sparse")
    ap.add_argument("--source-max-range", type=float, default=12.0,
                    help="with --range-mode shift: only copy objects originally seen within this distance (m), "
                         "so their point spacing is dense")
    ap.add_argument("--natural-range", action="store_true",
                    help="with --range-mode natural: do not balance over distance bins (mostly close objects)")
    ap.add_argument("--out", default="results/run.csv")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="use every N-th keyframe of every scene (e.g. 6 -> ~64 keyframes spread over all "
                         "10 mini scenes; for quick ablations). Fakes per keyframe are seeded, so the same "
                         "keyframe gets the same fakes in every run")
    ap.add_argument("--scenes", type=str, default="", help="comma-separated scene names (default: all)")
    ap.add_argument("--save-clouds", action="store_true")
    ap.add_argument("--inject-anywhere", action="store_true",
                    help="old behaviour: fake only has to avoid the new frame's points, not the history synthesis")
    ap.add_argument("--min-fake-points", type=int, default=None,
                    help="only inject bank objects with at least this many raw points "
                         "(default: 100 with shift = clearly visible object, 20 with natural)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep the rows already in --out and skip those keyframes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. --set iters=50 beta=0")
    return ap.parse_args()


def apply_overrides(cfg, pairs):
    for kv in pairs:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        elif cur is None:
            v = None if v.lower() == "none" else float(v)
        else:
            v = type(cur)(v) if v.lower() != "none" else None
        setattr(cfg, k, v)
    return cfg


def main():
    args = parse()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    if args.attack == "sparse" and kinds != ["car"]:
        print("[note] the paper's sparse attack only injects cars; using --kinds car")
        kinds = ["car"]
    bad = set(kinds) - {"pedestrian", "cyclist", "car"}
    if bad:
        raise SystemExit(f"unknown kinds: {bad}")
    if args.range_mode == "auto":
        args.range_mode = "shift" if args.attack == "dense" else "natural"
    if args.min_fake_points is None:
        args.min_fake_points = 100 if args.range_mode == "shift" else 20

    cfg = ADoPTConfig.dense() if args.attack == "dense" else ADoPTConfig.sparse()
    cfg = apply_overrides(cfg, args.set)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + "_config.json", "w", encoding="utf-8") as f:
        json.dump({**cfg.to_dict(), **vars(args), "kinds": kinds}, f, indent=2)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    rng = np.random.default_rng(args.seed)

    bank_path = os.path.join("results", f"bank_{args.version}_v2.pkl")   # shared by all runs
    if os.path.exists(bank_path):
        bank = pickle.load(open(bank_path, "rb"))
    else:
        print("building object bank ...")
        bank = build_object_bank(nusc)
        os.makedirs(os.path.dirname(bank_path), exist_ok=True)
        pickle.dump(bank, open(bank_path, "wb"))
    print({k: sum(o.kind == k for o in bank) for k in ("pedestrian", "cyclist", "car")}, "objects in bank")
    print(f"fake placement: {args.range_mode}, {args.range_min:.0f}-{args.range_max:.0f} m, "
          f">= {args.min_fake_points} points")
    for k in kinds:
        if args.range_mode == "shift":
            usable = [o for o in bank if o.kind == k and len(o.points) >= args.min_fake_points
                      and o.range < args.source_max_range]
            print(f"  usable {k:<10}: {len(usable):4d}   (seen within {args.source_max_range:.0f} m, "
                  f"compact chunk, moved to a random distance)")
        else:
            usable = [o for o in bank if o.kind == k and len(o.points) >= args.min_fake_points
                      and args.range_min <= o.range < args.range_max]
            hist = np.histogram([o.range for o in usable], bins=np.arange(args.range_min, args.range_max + 5, 5))[0]
            print(f"  usable {k:<10}: {len(usable):4d}   per 5 m bin from {args.range_min:.0f} m: {hist.tolist()}")

    adopt = ADoPT(cfg)
    cloud_dir = os.path.splitext(args.out)[0] + "_clouds"
    if args.save_clouds:
        os.makedirs(cloud_dir, exist_ok=True)

    wanted = set(args.scenes.split(",")) if args.scenes else None
    fields = ["scene", "sample", "variant", "kind", "n_fake", "fake_range", "score", "attacked", "n_clusters_before",
              "n_clusters_after", "largest_leftover", "fake_points_caught", "cd", "dcd", "n_new", "n_synth",
              "t_synth", "t_check"]
    n_done = 0
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # a keyframe counts as done only if its benign row and all its attacked rows were written
        by_sample = {}
        for r in rows:
            by_sample.setdefault(r["sample"], []).append(r)
        done = set(by_sample)        # rows of a keyframe are written together, so present = complete
        print(f"resuming: {len(done)} keyframes already done")
    mode = "a" if done else "w"
    with open(args.out, mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        if not done:
            writer.writeheader()
        for scene in nusc.scene:
            if wanted and scene["name"] not in wanted:
                continue
            prep_cache, flow_cache = {}, {}           # per scene
            sample_token = scene["first_sample_token"]
            sample_idx = -1
            while sample_token:
                sample = nusc.get("sample", sample_token)
                sample_token = sample["next"]
                sample_idx += 1                      # position in the scene, independent of L
                if sample_idx % args.sample_every:
                    continue
                if sample["token"] in done:
                    n_done += 1
                    continue
                # fixed random stream per keyframe: same fakes whether the run was resumed or not
                rng = np.random.default_rng([args.seed, zlib.crc32(sample["token"].encode())])
                toks = history_tokens(nusc, sample, cfg.L, cfg.stride)
                if toks is None:
                    continue
                ref = toks[-1]

                # ---- load + preprocess each sweep once (in its own ego frame), then move to ref frame
                def frame(tok, voxel="default"):
                    key = (tok, voxel)
                    if key not in prep_cache:
                        prep_cache[key] = adopt.preprocess(load_sweep_ego(nusc, tok, cfg.min_dist),
                                                           radius=cfg.roi_synth + 20, voxel=voxel)
                    p = transform(prep_cache[key], ego_a_to_ego_b(nusc, tok, ref))
                    return crop(p, cfg.roi_synth, cfg.z_min, cfg.z_max)

                frames = [frame(t) for t in toks]
                history, new = frames[:-1], frames[-1]
                new_fine = frame(ref, cfg.det_voxel)       # finer copy of the newest frame, for detection

                # ---- synthesis, re-using pair flows fitted for earlier keyframes
                t0 = time.perf_counter()
                flow_fns = []
                for k, (a, b) in enumerate(zip(toks[:-2], toks[1:-1])):
                    if (a, b) not in flow_cache:
                        flow_cache[(a, b)] = (adopt.fit(history[k], history[k + 1]), ref)
                    field, fit_ref = flow_cache[(a, b)]
                    flow_fns.append(field if fit_ref == ref else
                                    TransformedFlow(field, ego_a_to_ego_b(nusc, ref, fit_ref)))
                synth = adopt.synthesize(history, flow_fns)
                t_synth = time.perf_counter() - t0

                # ---- benign + attacked variants
                # each variant: (name, detection cloud at det_voxel, fake mask, info)
                variants = [("benign", new_fine, None, {"kind": "", "n_fake": 0, "range": ""})]
                for kind in kinds:
                    shift = args.range_mode == "shift"
                    obj = pick_object(bank, kind, rng, exclude_scene=scene["token"],
                                      min_points=args.min_fake_points,
                                      r_min=0.0 if shift else args.range_min,
                                      r_max=args.source_max_range if shift else args.range_max,
                                      uniform_range=not (shift or args.natural_range))
                    if obj is None:
                        continue
                    target = float(rng.uniform(args.range_min, args.range_max)) if shift else None
                    poisoned, mask, info = inject(new_fine, obj, args.attack, rng, voxel=cfg.det_voxel,
                                                  occupied=None if args.inject_anywhere else synth,
                                                  target_range=target, compact=shift)
                    if poisoned is not None:
                        variants.append(("attacked", poisoned, mask, info))

                scores, rows_out = [], []
                for name, cloud, fake_mask, info in variants:
                    t1 = time.perf_counter()
                    coarse = new if fake_mask is None else voxel_downsample(cloud, cfg.input_voxel)
                    res = adopt.check(synth, coarse, new_frame_det=cloud)
                    t_check = time.perf_counter() - t1
                    caught = 0
                    if fake_mask is not None:
                        in_det = np.linalg.norm(cloud[:, :2], axis=1) <= cfg.roi_det
                        caught = int((res["leftover_mask"] & fake_mask[in_det]).sum())
                    rows_out.append({
                        "scene": scene["name"], "sample": sample["token"], "variant": name,
                        "kind": info["kind"], "n_fake": info["n_fake"],
                        "fake_range": round(info["range"], 1) if info["range"] != "" else "",
                        "score": res["score"],
                        "attacked": int(res["attacked"]), "n_clusters_before": res["n_clusters_before"],
                        "n_clusters_after": res["n_clusters_after"], "largest_leftover": res["largest_leftover"],
                        "fake_points_caught": caught,
                        "cd": round(chamfer_np(res["warped_synth"], res["new_det"]), 5),
                        "dcd": round(dcd_np(res["warped_synth"], res["new_det"]), 5),
                        "n_new": len(res["new_det"]), "n_synth": len(res["warped_synth"]),
                        "t_synth": round(t_synth, 3), "t_check": round(t_check, 3)})
                    tag = name if name == "benign" else f"{info['kind'][:3]}@{info['range']:.0f}m"
                    scores.append(f"{tag}={res['score']}")
                    if args.save_clouds:
                        fname = f"{sample['token']}_{name}" + ("" if name == "benign" else f"_{info['kind']}")
                        np.savez_compressed(os.path.join(cloud_dir, fname + ".npz"),
                                            synth=res["warped_synth"].astype(np.float32),
                                            synth_raw=crop(synth, cfg.roi_det + 2.0).astype(np.float32),
                                            new=res["new_det"].astype(np.float32),
                                            fake=(fake_mask[np.linalg.norm(cloud[:, :2], axis=1) <= cfg.roi_det]
                                                  if fake_mask is not None else np.zeros(len(res["new_det"]), bool)))
                writer.writerows(rows_out)          # all rows of this keyframe at once
                fcsv.flush()
                n_done += 1
                print(f"[{n_done}] {scene['name']}  synth {t_synth:.2f}s  " + "  ".join(scores), flush=True)
                if args.max_samples and n_done >= args.max_samples:
                    print("done ->", args.out)
                    return
    print("done ->", args.out)


if __name__ == "__main__":
    main()

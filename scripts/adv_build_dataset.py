"""Cut every annotated car / pedestrian / cyclist out of the nuScenes keyframes (classifier dataset).

  python scripts/adv_build_dataset.py --dataroot data/nuscenes

Each object keeps the points inside its box, thinned to 5 cm voxels (the same resolution ADoPT's
detection step sees), and is stored in its own box frame. Writes results/adv/objects.npz.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.geometry import transform, voxel_downsample  # noqa: E402
from adopt.objects import CLASSES, ObjectSet, class_of, in_box_mask, is_test_instance, to_canonical  # noqa: E402

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install the nuScenes devkit: pip install nuscenes-devkit") from e

from adopt.nuscenes_loader import sensor_to_ego  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--min-points", type=int, default=20, help="after 5 cm voxels")
    ap.add_argument("--range-min", type=float, default=4.0)
    ap.add_argument("--range-max", type=float, default=50.0)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--out", default="results/adv/objects.npz")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    items = []
    for k, sample in enumerate(nusc.sample):
        sd_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", sd_token)
        pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
        pts = transform(pc.points[:3].T, sensor_to_ego(nusc, sd_token)).astype(np.float32)
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        scene = nusc.get("scene", sample["scene_token"])["name"]
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            label = class_of(ann["category_name"])
            if label is None or ann["num_lidar_pts"] < args.min_points:
                continue
            box = nusc.get_box(ann_token)                       # global frame
            box.translate(-np.array(pose["translation"]))       # -> ego frame of this sweep
            box.rotate(Quaternion(pose["rotation"]).inverse)
            center, wlh = box.center.astype(np.float32), box.wlh.astype(np.float32)
            yaw = float(box.orientation.yaw_pitch_roll[0])
            rng = float(np.linalg.norm(center[:2]))
            if not (args.range_min <= rng <= args.range_max):
                continue
            obj = voxel_downsample(pts[in_box_mask(pts, center, wlh, yaw)], args.voxel)
            if len(obj) < args.min_points:
                continue
            items.append(dict(points=to_canonical(obj, center, yaw), label=label,
                              split=int(is_test_instance(ann["instance_token"])), sample=sample["token"],
                              instance=ann["instance_token"], scene=scene, center=center, wlh=wlh, yaw=yaw,
                              range=rng))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(nusc.sample)} keyframes, {len(items)} objects")

    data = ObjectSet.from_list(items)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data.save(args.out)
    print(f"saved {len(data)} objects -> {args.out}")
    print(f"{'class':<12}{'train':>7}{'test':>7}   points (median)   instances train/test")
    for c, name in enumerate(CLASSES):
        m = data.labels == c
        tr, te = m & (data.split == 0), m & (data.split == 1)
        print(f"{name:<12}{tr.sum():7d}{te.sum():7d}   {np.median(data.n_points[m]) if m.any() else 0:9.0f}"
              f"         {len(set(data.instance[tr]))}/{len(set(data.instance[te]))}")


if __name__ == "__main__":
    main()

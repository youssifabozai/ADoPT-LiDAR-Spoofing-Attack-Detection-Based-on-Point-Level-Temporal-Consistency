"""nuScenes helpers: sweep sequences, ego-motion compensation, object bank.

Requires `nuscenes-devkit` (pip install nuscenes-devkit).

Coordinate convention: every cloud is returned in the *ego frame* of a
reference sweep (x forward, y left, z up, origin on the road under the rear
axle). Expressing all history frames in the same ego frame removes the ego
vehicle's own motion, so the scene flow only has to explain moving objects.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from .attack import BankObject
from .geometry import transform

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import points_in_box, transform_matrix
    from pyquaternion import Quaternion
except ImportError as e:  # pragma: no cover
    raise ImportError("Install the nuScenes devkit: pip install nuscenes-devkit") from e


# ------------------------------------------------------------------ transforms
def sensor_to_ego(nusc: "NuScenes", sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return transform_matrix(cs["translation"], Quaternion(cs["rotation"]))


def ego_to_global(nusc: "NuScenes", sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return transform_matrix(pose["translation"], Quaternion(pose["rotation"]))


def ego_a_to_ego_b(nusc: "NuScenes", sd_a: str, sd_b: str) -> np.ndarray:
    """4x4 transform mapping points in ego frame of sweep a into ego frame of sweep b."""
    return np.linalg.inv(ego_to_global(nusc, sd_b)) @ ego_to_global(nusc, sd_a)


# ------------------------------------------------------------------ loading
def load_sweep_ego(nusc: "NuScenes", sd_token: str, min_dist: float = 3.0) -> np.ndarray:
    """Load one LiDAR sweep, drop points near the sensor, return (N,3) in its own ego frame."""
    sd = nusc.get("sample_data", sd_token)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
    pc.remove_close(min_dist)
    return transform(pc.points[:3].T, sensor_to_ego(nusc, sd_token)).astype(np.float32)


def history_tokens(nusc: "NuScenes", sample: dict, L: int, stride: int) -> Optional[List[str]]:
    """L+1 LIDAR_TOP sample_data tokens ending at this keyframe, oldest first.

    nuScenes LiDAR runs at 20 Hz; stride=2 gives 10 Hz like the paper.
    Returns None if the scene does not have enough earlier sweeps.
    """
    chain = [sample["data"]["LIDAR_TOP"]]
    while len(chain) < L * stride + 1:
        prev = nusc.get("sample_data", chain[-1])["prev"]
        if not prev:
            return None
        chain.append(prev)
    return chain[::stride][: L + 1][::-1]


# ------------------------------------------------------------------ object bank
def _kind(nusc: "NuScenes", ann: dict) -> Optional[str]:
    name = ann["category_name"]
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    if name == "vehicle.car":
        return "car"
    if name in ("vehicle.bicycle", "vehicle.motorcycle"):
        # [assumed] nuScenes-mini has very few ridden bicycles, so ridden motorcycles
        # (similar size and shape: a person on a two-wheeler) also count as "cyclist".
        attrs = {nusc.get("attribute", a)["name"] for a in ann["attribute_tokens"]}
        if "cycle.with_rider" in attrs:
            return "cyclist"
    return None


def build_object_bank(nusc: "NuScenes", min_points: Optional[Dict[str, int]] = None,
                      min_range: float = 5.0, max_range: float = 40.0, max_per_kind: int = 600) -> List[BankObject]:
    """Collect real LiDAR points of annotated pedestrians, cyclists and cars (keyframes only).

    Objects keep the range they were seen at, so far objects are naturally sparser.
    """
    min_points = min_points or {"pedestrian": 20, "cyclist": 20, "car": 30}
    bank: List[BankObject] = []
    counts = {k: 0 for k in min_points}
    for sample in nusc.sample:
        if all(counts[k] >= max_per_kind for k in counts):
            break
        sd_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", sd_token)
        pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
        pts = pc.points[:3]
        _, boxes, _ = nusc.get_sample_data(sd_token)          # boxes in the sensor frame
        T = sensor_to_ego(nusc, sd_token)
        for box in boxes:
            ann = nusc.get("sample_annotation", box.token)
            kind = _kind(nusc, ann)
            if kind is None or counts[kind] >= max_per_kind:
                continue
            rng_xy = np.linalg.norm(box.center[:2])
            if not (min_range <= rng_xy <= max_range):
                continue
            mask = points_in_box(box, pts)
            mask &= pts[2] > box.corners()[2].min() + 0.15        # drop ground hits under the object
            if mask.sum() < min_points[kind]:
                continue
            obj = transform(pts[:, mask].T, T).astype(np.float32)
            scene = nusc.get("scene", sample["scene_token"])["token"]
            bank.append(BankObject(obj, kind, scene, sample["token"]))
            counts[kind] += 1
    return bank

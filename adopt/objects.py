"""Object point clouds for the adversarial-robustness experiments (no PyTorch needed here).

Every annotated car / pedestrian / cyclist is stored in its *canonical* frame: the box centre is
the origin and the box heading is the +x axis. A classifier then sees the object's shape the same
way regardless of where it stood, and an attack perturbation defined in this frame can be mapped
back into the LiDAR frame (`from_canonical`) to test whether ADoPT notices it.
"""

import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

CLASSES = ["car", "pedestrian", "cyclist"]


def class_of(category_name: str) -> Optional[int]:
    """nuScenes category -> class index (None = not used).

    [assumed] 'cyclist' = every bicycle and motorcycle, with or without rider: nuScenes-mini has
    only a handful of ridden two-wheelers, far too few to train a classifier.
    """
    if category_name == "vehicle.car":
        return 0
    if category_name.startswith("human.pedestrian"):
        return 1
    if category_name in ("vehicle.bicycle", "vehicle.motorcycle"):
        return 2
    return None


def is_test_instance(instance_token: str, test_fraction: int = 5) -> bool:
    """Deterministic split by object instance (~1 in 5 instances is test).

    Splitting by instance (not by frame) keeps every view of one physical object on the same side,
    so the test set never contains an object the model was trained on.
    """
    return zlib.crc32(instance_token.encode()) % test_fraction == 0


# ------------------------------------------------------------------ box geometry
def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_canonical(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    """LiDAR/ego frame -> box frame (origin at box centre, x along the heading)."""
    return ((points - center) @ _rot(yaw)).astype(np.float32)          # row-vector form of R^T (p - c)


def from_canonical(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    """Box frame -> LiDAR/ego frame (inverse of `to_canonical`)."""
    return (points @ _rot(yaw).T + center).astype(np.float32)


def in_box_mask(points: np.ndarray, center: np.ndarray, wlh: np.ndarray, yaw: float,
                margin: float = 0.0, floor: float = 0.15) -> np.ndarray:
    """Points inside an oriented nuScenes box (wlh = width, length, height).

    `floor` drops points in the lowest `floor` metres of the box (ground hits under the object).
    """
    if len(points) == 0:
        return np.zeros(0, bool)
    c = to_canonical(points, center, yaw)
    w, l, h = wlh
    return ((np.abs(c[:, 0]) <= l / 2 + margin) & (np.abs(c[:, 1]) <= w / 2 + margin)
            & (c[:, 2] >= -h / 2 + floor) & (c[:, 2] <= h / 2 + margin))


# ------------------------------------------------------------------ dataset container
@dataclass
class ObjectSet:
    points: np.ndarray        # (M, 3) float32, all objects concatenated, canonical frame
    offsets: np.ndarray       # (K + 1,) int64, object i = points[offsets[i]:offsets[i + 1]]
    labels: np.ndarray        # (K,) int64, index into CLASSES
    split: np.ndarray         # (K,) int8, 0 = train, 1 = test
    sample: np.ndarray        # (K,) str, nuScenes keyframe (sample) token
    instance: np.ndarray      # (K,) str, nuScenes instance token
    scene: np.ndarray         # (K,) str, scene name
    center: np.ndarray        # (K, 3) box centre in the keyframe's ego frame
    wlh: np.ndarray           # (K, 3) box size
    yaw: np.ndarray           # (K,) box heading in the ego frame
    range: np.ndarray         # (K,) xy distance from the ego origin

    def __len__(self):
        return len(self.labels)

    def obj(self, i: int) -> np.ndarray:
        return self.points[self.offsets[i]:self.offsets[i + 1]]

    @property
    def n_points(self) -> np.ndarray:
        return np.diff(self.offsets)

    def idx(self, split: int) -> np.ndarray:
        return np.where(self.split == split)[0]

    def batch(self, idx, n: int, rng: np.random.Generator, augment: bool = False) -> np.ndarray:
        """(B, n, 3) float32: every object resampled to exactly n points.

        Objects with >= n points are subsampled without replacement, smaller ones are padded by
        repeating points (max-pooling in PointNet is unaffected by duplicates).
        Augmentation: random left-right mirror (y -> -y) and 1 cm jitter.
        """
        out = np.empty((len(idx), n, 3), np.float32)
        for b, i in enumerate(idx):
            p = self.obj(i)
            sel = rng.choice(len(p), n, replace=len(p) < n)
            q = p[sel].copy()
            if augment:
                if rng.random() < 0.5:
                    q[:, 1] = -q[:, 1]
                q += np.clip(rng.normal(0, 0.01, q.shape), -0.02, 0.02).astype(np.float32)
            out[b] = q
        return out

    # -------------------------------------------------------------- io
    def save(self, path: str):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "ObjectSet":
        z = np.load(path)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})

    @classmethod
    def from_list(cls, items) -> "ObjectSet":
        """items: dicts with keys points (canonical), label, split, sample, instance, scene,
        center, wlh, yaw, range."""
        pts = [it["points"].astype(np.float32) for it in items]
        offsets = np.r_[0, np.cumsum([len(p) for p in pts])].astype(np.int64)
        col = lambda k, dt=None: np.array([it[k] for it in items], dtype=dt)
        return cls(points=np.concatenate(pts) if pts else np.zeros((0, 3), np.float32), offsets=offsets,
                   labels=col("label", np.int64), split=col("split", np.int8), sample=col("sample", str),
                   instance=col("instance", str), scene=col("scene", str),
                   center=col("center", np.float32).reshape(-1, 3), wlh=col("wlh", np.float32).reshape(-1, 3),
                   yaw=col("yaw", np.float32), range=col("range", np.float32))


# ------------------------------------------------------------------ synthetic data (smoke test)
def _surface_box(n, l, w, h, rng):
    """Points on the two LiDAR-facing sides + top of a box (rough car / cyclist shape)."""
    face = rng.integers(0, 3, n)
    u, v = rng.random(n), rng.random(n)
    x = np.where(face == 0, -l / 2, (u - 0.5) * l)
    y = np.where(face == 1, -w / 2, np.where(face == 0, (u - 0.5) * w, (v - 0.5) * w))
    z = np.where(face == 2, h / 2, (v - 0.5) * h)
    return np.c_[x, y, z]


def _cylinder(n, r, h, rng):
    a = rng.uniform(-np.pi / 2, np.pi / 2, n) + np.pi           # half facing the sensor
    return np.c_[r * np.cos(a), r * np.sin(a), (rng.random(n) - 0.5) * h]


def synthetic_objects(n_per_class: int = 200, seed: int = 0) -> ObjectSet:
    """Three toy shape classes, only for testing the code path without nuScenes."""
    rng = np.random.default_rng(seed)
    items = []
    for label in range(3):
        for k in range(n_per_class):
            n = int(rng.integers(30, 300))
            if label == 0:
                wlh = np.array([1.9, 4.5, 1.5]) * rng.uniform(0.9, 1.1, 3)
                p = _surface_box(n, wlh[1], wlh[0], wlh[2], rng)
            elif label == 1:
                wlh = np.array([0.6, 0.6, 1.75]) * rng.uniform(0.9, 1.1, 3)
                p = _cylinder(n, 0.25, wlh[2], rng)
            else:
                wlh = np.array([0.7, 1.8, 1.5]) * rng.uniform(0.9, 1.1, 3)
                p = _surface_box(n, wlh[1], wlh[0] * 0.5, wlh[2], rng)
            p = p + rng.normal(0, 0.02, p.shape)
            center = np.array([rng.uniform(5, 40), rng.uniform(-10, 10), 0.8])
            items.append(dict(points=p, label=label, split=int(k % 5 == 0), sample=f"s{k % 40}",
                              instance=f"i{label}_{k}", scene="synthetic", center=center, wlh=wlh,
                              yaw=float(rng.uniform(-np.pi, np.pi)), range=float(np.linalg.norm(center[:2]))))
    return ObjectSet.from_list(items)

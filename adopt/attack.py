"""Simulated point-injection (spoofing) attacks.

The paper's poisoned data is not public (dense data came privately from the
3D-TC2 authors; sparse data was generated following CARLO). We approximate
both by copying *real* LiDAR points of a real object (pedestrian / cyclist /
car) from another scene and pasting them into the newest frame:

  dense  : up to 200 points  (paper: "up to 200 points")
  sparse : up to 64 points   (paper: "up to 64 points", cars only)

To keep the object looking like what a LiDAR would see, the object is
rotated around the *sensor* (z axis) by a random angle. This keeps its range
and the side facing the sensor, so occlusion/shape stay plausible.
This is a data-level simulation for evaluating a defense only.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import rot_z, voxel_downsample

MAX_POINTS = {"dense": 200, "sparse": 64}


@dataclass
class BankObject:
    points: np.ndarray      # (N, 3) in the ego frame of the sample it came from
    kind: str               # pedestrian | cyclist | car
    scene_token: str = ""
    sample_token: str = ""

    @property
    def range(self) -> float:
        """Distance (m, xy) of the object's centre from the ego vehicle."""
        return float(np.linalg.norm(self.points[:, :2].mean(0)))


def pick_object(bank, kind, rng, exclude_scene="", min_points=0, r_min=0.0, r_max=np.inf,
                bin_width=5.0, uniform_range=True):
    """Pick one bank object of `kind`.

    uniform_range=True first picks a distance bin uniformly, then an object inside it.
    Without this, close objects (which are far more common in the bank) would dominate.
    """
    pool = [o for o in bank if o.kind == kind and o.scene_token != exclude_scene
            and len(o.points) >= min_points and r_min <= o.range < r_max]
    if not pool:
        return None
    if not uniform_range:
        return pool[rng.integers(len(pool))]
    bins = {}
    for o in pool:
        bins.setdefault(int(o.range // bin_width), []).append(o)
    keys = sorted(bins)
    chosen = bins[keys[rng.integers(len(keys))]]
    return chosen[rng.integers(len(chosen))]


def inject(frame: np.ndarray, obj: BankObject, mode: str = "dense",
           rng: Optional[np.random.Generator] = None,
           azimuth_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
           clearance: float = 0.5, tries: int = 100, voxel: Optional[float] = None,
           occupied: Optional[np.ndarray] = None, target_range: Optional[float] = None,
           compact: bool = False):
    """Paste `obj` into `frame` at a free location.

    `frame` should already be preprocessed (ground removed), otherwise ground
    points block every location. Pass the same `voxel` used for preprocessing
    so the fake points get the same downsampling as real ones.
    `occupied` (e.g. the synthesis of history frames) adds extra points that
    the fake must also stay away from, so we never paste it onto something
    that was visible a moment ago (that would test the injector, not ADoPT).
    `target_range` moves the object along its viewing ray to that distance while
    keeping ALL its points: the attacker controls the laser, so an injected
    object can be denser than a real one at that distance (dense attack).
    `compact=True`: when the object has more points than allowed, keep the
    contiguous chunk around its centre instead of thinning it randomly. A
    spoofing laser covers a narrow angle, so injected points form a dense patch.

    Returns (poisoned_frame, fake_mask, info) or (None, None, None) if no free spot was found.
    fake_mask marks the injected points inside poisoned_frame.
    """
    rng = rng or np.random.default_rng()
    pts = obj.points
    max_pts = MAX_POINTS[mode]
    if len(pts) > max_pts:
        if compact:
            anchor = pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))]
            pts = pts[np.argsort(np.linalg.norm(pts - anchor, axis=1))[:max_pts]]
        else:
            pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    if target_range is not None:
        c = pts[:, :2].mean(0)
        r = float(np.linalg.norm(c))
        pts = pts.copy()
        pts[:, :2] += c / max(r, 1e-6) * (target_range - r)
    blockers = frame[:, :2] if occupied is None or len(occupied) == 0 else \
        np.vstack([frame[:, :2], occupied[:, :2]])

    theta0 = np.arctan2(pts[:, 1].mean(), pts[:, 0].mean())
    for _ in range(tries):
        theta = rng.uniform(*azimuth_range)
        cand = pts @ rot_z(theta - theta0).T
        lo, hi = cand[:, :2].min(0) - clearance, cand[:, :2].max(0) + clearance
        inside = np.all((blockers >= lo) & (blockers <= hi), axis=1)
        if not inside.any():
            if voxel:
                cand = voxel_downsample(cand, voxel)
            poisoned = np.vstack([frame, cand.astype(frame.dtype)])
            mask = np.r_[np.zeros(len(frame), bool), np.ones(len(cand), bool)]
            info = {"kind": obj.kind, "n_fake": len(cand), "azimuth": float(theta),
                    "range": float(np.linalg.norm(cand[:, :2].mean(0)))}
            return poisoned, mask, info
    return None, None, None

"""A tiny synthetic street scene with known motion, for testing without nuScenes.

Objects are boxes; each frame re-samples random points on their surfaces
(like a LiDAR never hitting exactly the same spot twice). Every point carries
its true velocity, so we can build an *oracle* scene flow and check the
detector independently of the neural scene-flow estimator.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import rot_z


def default_objects():
    # (center_x, center_y, length, width, height, yaw, vx, vy, kind)
    return [
        (0.0, 12.0, 80.0, 0.3, 3.0, 0.0, 0.0, 0.0, "wall"),
        (0.0, -12.0, 80.0, 0.3, 3.0, 0.0, 0.0, 0.0, "wall"),
        (15.0, 7.5, 4.5, 1.9, 1.6, 0.0, 0.0, 0.0, "parked_car"),
        (-10.0, -7.5, 4.5, 1.9, 1.6, 0.0, 0.0, 0.0, "parked_car"),
        (8.0, 3.5, 4.5, 1.9, 1.6, 0.0, 8.0, 0.0, "car"),
        (25.0, -3.5, 4.5, 1.9, 1.6, np.pi, -10.0, 0.0, "car"),
        (20.0, 9.5, 0.6, 0.6, 1.75, 0.0, 0.0, -1.3, "pedestrian"),
        (-5.0, 6.0, 1.8, 0.6, 1.7, 0.0, 4.0, 0.0, "cyclist"),
        (30.0, 11.0, 0.3, 0.3, 4.0, 0.0, 0.0, 0.0, "pole"),
    ]


def _box_surface(cx, cy, l, w, h, yaw, rng, density_fn):
    faces = [  # (origin, u-axis, v-axis) in the box frame, z from 0..h
        (np.array([-l / 2, -w / 2, 0]), np.array([l, 0, 0]), np.array([0, 0, h])),
        (np.array([-l / 2, w / 2, 0]), np.array([l, 0, 0]), np.array([0, 0, h])),
        (np.array([-l / 2, -w / 2, 0]), np.array([0, w, 0]), np.array([0, 0, h])),
        (np.array([l / 2, -w / 2, 0]), np.array([0, w, 0]), np.array([0, 0, h])),
        (np.array([-l / 2, -w / 2, h]), np.array([l, 0, 0]), np.array([0, w, 0])),
    ]
    R = rot_z(yaw)
    out = []
    for o, u, v in faces:
        area = np.linalg.norm(u) * np.linalg.norm(v)
        center = (o + 0.5 * u + 0.5 * v) @ R.T + [cx, cy, 0]
        n = rng.poisson(area * density_fn(np.linalg.norm(center[:2])))
        a, b = rng.random((n, 1)), rng.random((n, 1))
        p = (o + a * u + b * v) @ R.T + [cx, cy, 0]
        out.append(p)
    return np.vstack(out)


def render_frame(t, objects=None, rng=None, density=60.0, noise=0.02):
    """Points and per-point velocities of the scene at time t (seconds)."""
    rng = rng or np.random.default_rng()
    objects = objects or default_objects()
    dens = lambda r: density * (10.0 / max(r, 5.0)) ** 2
    pts, vel = [], []
    for cx, cy, l, w, h, yaw, vx, vy, _ in objects:
        p = _box_surface(cx + vx * t, cy + vy * t, l, w, h, yaw, rng, dens)
        pts.append(p)
        vel.append(np.tile([vx, vy, 0.0], (len(p), 1)))
    pts = np.vstack(pts) + rng.normal(0, noise, (sum(len(p) for p in pts), 3))
    return pts.astype(np.float32), np.vstack(vel).astype(np.float32)


def make_sequence(n_frames=11, dt=0.1, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [render_frame(k * dt, rng=rng, **kw) for k in range(n_frames)]


def fake_pedestrian(center_xy=(12.0, -7.0), n=150, seed=1):
    """A cylinder of points shaped like a pedestrian (radius 0.3 m, 1.7 m tall)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi / 2, np.pi / 2, n) + np.arctan2(-center_xy[1], -center_xy[0])  # side facing the sensor
    z = rng.uniform(0.05, 1.7, n)
    return np.c_[center_xy[0] + 0.3 * np.cos(a), center_xy[1] + 0.3 * np.sin(a), z].astype(np.float32)


class OracleFlow:
    """Ground-truth flow: displacement of the nearest point of the frame at time k, over dt."""

    def __init__(self, points, velocities, dt):
        self.tree = cKDTree(points)
        self.vel = velocities
        self.dt = dt

    def __call__(self, p):
        _, idx = self.tree.query(p)
        return self.vel[idx] * self.dt

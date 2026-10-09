"""Point-cloud utilities: cropping, voxel downsampling, ground removal, transforms."""

import numpy as np


def crop(points: np.ndarray, radius: float, z_min: float = -np.inf, z_max: float = np.inf) -> np.ndarray:
    """Keep points within `radius` (xy) of the origin and inside [z_min, z_max]."""
    if len(points) == 0:
        return points
    r = np.linalg.norm(points[:, :2], axis=1)
    keep = (r <= radius) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[keep]


def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    """Replace all points inside each cubic voxel of edge `size` by their mean."""
    if size is None or size <= 0 or len(points) == 0:
        return points
    idx = np.floor(points / size).astype(np.int64)
    _, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.stack([np.bincount(inv, weights=points[:, d], minlength=len(counts)) for d in range(3)], axis=1)
    return (out / counts[:, None]).astype(points.dtype)


def fit_ground_plane(points: np.ndarray, iters: int = 150, inlier_thresh: float = 0.15,
                     z_window: float = 1.0, r_max: float = 30.0, seed: int = 0):
    """RANSAC fit of a near-horizontal plane n.p + d = 0 (n points up).

    Only points near z=0 (ego frame: origin is on the ground under the car)
    and within r_max are used as candidates.
    """
    rng = np.random.default_rng(seed)
    r = np.linalg.norm(points[:, :2], axis=1)
    cand = points[(np.abs(points[:, 2]) < z_window) & (r < r_max)]
    default = (np.array([0.0, 0.0, 1.0]), 0.0)
    if len(cand) < 50:
        return default
    best, best_count = default, -1
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        if n[2] < 0:
            n = -n
        if n[2] < 0.95:          # reject steep planes (walls, car sides)
            continue
        d = -float(n @ s[0])
        count = int((np.abs(cand @ n + d) < inlier_thresh).sum())
        if count > best_count:
            best, best_count = (n, d), count
    return best


def remove_ground(points: np.ndarray, thresh: float = 0.25, **kw):
    """Remove points within `thresh` metres above (or anywhere below) the fitted ground plane."""
    if len(points) == 0:
        return points, (np.array([0.0, 0.0, 1.0]), 0.0)
    n, d = fit_ground_plane(points, **kw)
    height = points @ n + d
    return points[height > thresh], (n, d)


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    return points @ T[:3, :3].T + T[:3, 3]


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

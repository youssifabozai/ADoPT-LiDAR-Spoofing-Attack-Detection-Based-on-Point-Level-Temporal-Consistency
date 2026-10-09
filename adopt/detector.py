"""Cluster-based consistency measurement (paper Sec. 5.2, App. A).

1. Merge the warped synthesis (history) and the incoming frame, tagging each point.
2. Run DBSCAN on the merged cloud.
3. Drop every cluster that contains at least one synthesis point (it has history -> real).
4. Clusters made only of incoming-frame points have no history -> suspected fake objects.
5. [assumed, not in paper] Clusters flatter than `min_height` are ignored: a single laser ring
   hitting a curb or road bump re-samples at a slightly different spot every sweep and can look
   history-free, while every object an attacker would fake (pedestrian, cyclist, car) is tall.
6. [assumed, not in paper] Clusters centred closer than `min_range` to the sensor are ignored.
   Every sweep drops its points within cfg.min_dist (3 m) of its own position (car body); as the
   car drives, the area just outside that circle has no history in older sweeps, so real objects
   there look history-free. Coordinates are ego-centred, so the range is the xy distance to (0, 0).
"""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_consistency(warped_synth: np.ndarray, new_frame: np.ndarray,
                        eps: float, min_samples: int, min_height: float = 0.0,
                        min_range: float = 0.0) -> dict:
    n_s = len(warped_synth)
    pts = np.vstack([warped_synth, new_frame]) if n_s else new_frame
    is_new = np.r_[np.zeros(n_s, bool), np.ones(len(new_frame), bool)]

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(pts)
    valid = labels >= 0
    n_before = int(len(np.unique(labels[valid]))) if valid.any() else 0

    leftover = np.zeros(len(pts), bool)
    if valid.any():
        has_history = np.zeros(labels.max() + 1, bool)
        has_history[np.unique(labels[valid & ~is_new])] = True
        leftover = valid & is_new & ~has_history[np.clip(labels, 0, None)]
        if min_height > 0 or min_range > 0:
            for l in np.unique(labels[leftover]):
                m = labels == l
                too_flat = np.ptp(pts[m, 2]) < min_height
                too_close = np.linalg.norm(pts[m, :2].mean(0)) < min_range
                if too_flat or too_close:
                    leftover &= ~m

    left_labels = np.unique(labels[leftover])
    new_mask = leftover[n_s:]
    sizes = [int((labels == l).sum()) for l in left_labels]
    return {
        "score": int(new_mask.sum()),            # points of the new frame without history
        "n_clusters_before": n_before,
        "n_clusters_after": int(len(left_labels)),
        "largest_leftover": max(sizes) if sizes else 0,
        "leftover_mask": new_mask,               # (len(new_frame),) bool
    }

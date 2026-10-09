"""Baseline point-cloud distances used in the paper's metric comparison (Fig. 6)."""

import numpy as np
from scipy.spatial import cKDTree


def chamfer_np(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance: mean squared NN distance a->b plus b->a."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    return float((d_ab ** 2).mean() + (d_ba ** 2).mean())


def dcd_np(a: np.ndarray, b: np.ndarray, alpha: float = 1.0) -> float:
    """Density-aware Chamfer Distance (Wu et al., NeurIPS 2021), in [0, 1].

    Each NN match is down-weighted by how many points share the same
    neighbour, so many-to-one matches are penalised. `alpha` sets the
    distance scale (1/metres here; the original paper uses normalised shapes).
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    def one_side(x, y):
        d, idx = cKDTree(y).query(x)
        n_shared = np.bincount(idx, minlength=len(y))[idx]
        return float(np.mean(1.0 - np.exp(-alpha * d) / n_shared))

    return 0.5 * (one_side(a, b) + one_side(b, a))


def rates(scores_benign, scores_attacked, threshold):
    """FPR / TPR (in %) when a frame is flagged if score > threshold."""
    sb, sa = np.asarray(scores_benign), np.asarray(scores_attacked)
    fpr = 100.0 * (sb > threshold).mean() if len(sb) else float("nan")
    tpr = 100.0 * (sa > threshold).mean() if len(sa) else float("nan")
    return fpr, tpr

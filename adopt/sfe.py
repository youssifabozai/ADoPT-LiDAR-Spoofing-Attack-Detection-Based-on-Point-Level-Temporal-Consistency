"""Coherence-enhanced Scene Flow Estimation (paper Sec. 5.1).

Scene flow is represented by a small MLP  f_theta: R^3 -> R^3  (a "neural
prior", as in NSFP [11]). For every pair of frames (F1, F2) we create a
fresh MLP and optimise it *at runtime* (no training dataset) with

    L = alpha * L_chamfer(F1 + f(F1), F2) + beta * L_coherence(F1)

L_coherence penalises points of the same DBSCAN cluster moving differently:

    L_coh = 1/N^2 * sum_{i,j} M_ij * w_ij * ||f(p_i) - f(p_j)||^2

With w_ij = 1 (as in the paper) the O(N^2) pair sum has an exact O(N) form:

    sum_{i,j in c} ||f_i - f_j||^2 = 2 * n_c * sum_{i in c} ||f_i - mean_c||^2
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from .config import ADoPTConfig

FlowFn = Callable[[np.ndarray], np.ndarray]   # (N,3) points -> (N,3) displacement


# --------------------------------------------------------------------------- model
class NeuralFlow(nn.Module):
    """MLP neural prior for scene flow: point (x, y, z) -> displacement (dx, dy, dz)."""

    def __init__(self, n_layers: int = 6, hidden: int = 128):
        super().__init__()
        layers, d = [], 3
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        last = nn.Linear(d, 3)
        # Start close to zero flow: after ego-motion compensation most points are static.
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- losses
@torch.no_grad()
def nn_index(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """For each row of a, index of its nearest neighbour in b (brute force, chunked)."""
    b2 = (b * b).sum(1)
    out = []
    for i in range(0, a.shape[0], chunk):
        ai = a[i:i + chunk]
        d = (ai * ai).sum(1, keepdim=True) + b2[None, :] - 2.0 * ai @ b.T
        out.append(d.argmin(1))
    return torch.cat(out)


def chamfer_loss(a: torch.Tensor, b: torch.Tensor, trunc: Optional[float] = None,
                 idx_ab: Optional[torch.Tensor] = None, idx_ba: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Symmetric Chamfer distance (mean of squared NN distances in both directions).

    Neighbour search runs without gradients; distances to the chosen
    neighbours are then recomputed with gradients (memory-friendly).
    `trunc` ignores matches farther than `trunc` metres (occlusions, new objects).
    `idx_ab` / `idx_ba` let the caller reuse neighbours from an earlier iteration.
    """
    if idx_ab is None:
        idx_ab = nn_index(a, b)
    if idx_ba is None:
        idx_ba = nn_index(b, a)
    d_ab = ((a - b[idx_ab]) ** 2).sum(1)
    d_ba = ((b - a[idx_ba]) ** 2).sum(1)
    if trunc is not None:
        t2 = trunc ** 2
        d_ab = d_ab[d_ab < t2] if (d_ab < t2).any() else d_ab
        d_ba = d_ba[d_ba < t2] if (d_ba < t2).any() else d_ba
    return d_ab.mean() + d_ba.mean()


def coherence_loss(flow: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Paper's coherence loss with w_ij = 1, computed exactly in O(N).

    flow:   (N, 3) predicted flow for the source points
    labels: (N,)   DBSCAN cluster id per point, -1 = outlier (excluded)
    """
    valid = labels >= 0
    if valid.sum() < 2:
        return flow.sum() * 0.0
    f, lab = flow[valid], labels[valid]
    n = f.shape[0]
    k = int(lab.max().item()) + 1
    counts = torch.bincount(lab, minlength=k).to(f.dtype)
    sums = torch.zeros(k, 3, device=f.device, dtype=f.dtype).index_add_(0, lab, f)
    means = sums / counts.clamp_min(1.0)[:, None]
    sq = ((f - means[lab]) ** 2).sum(1)
    per_cluster = torch.zeros(k, device=f.device, dtype=f.dtype).index_add_(0, lab, sq)
    return (2.0 * counts * per_cluster).sum() / (n * n)


def coherence_loss_bruteforce(flow: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Literal O(N^2) version of the paper's formula. Only for testing."""
    valid = labels >= 0
    f, lab = flow[valid], labels[valid]
    n = f.shape[0]
    same = (lab[:, None] == lab[None, :]).to(f.dtype)
    diff = ((f[:, None, :] - f[None, :, :]) ** 2).sum(-1)
    return (same * diff).sum() / (n * n)


# --------------------------------------------------------------------------- fitting
class NeuralFlowField:
    """Callable wrapper: numpy points -> numpy flow, using a fitted NeuralFlow."""

    def __init__(self, model: NeuralFlow, device: torch.device, history: Optional[list] = None):
        self.model = model.eval()
        self.device = device
        self.history = history or []

    @torch.no_grad()
    def __call__(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        out = []
        for i in range(0, len(points), 65536):
            x = torch.as_tensor(points[i:i + 65536], dtype=torch.float32, device=self.device)
            out.append(self.model(x).cpu().numpy())
        return np.concatenate(out, 0)


def cluster_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(points).astype(np.int64)


def fit_flow(src: np.ndarray, tgt: np.ndarray, cfg: ADoPTConfig,
             device: Optional[torch.device] = None, seed: int = 0) -> NeuralFlowField:
    """Optimise a fresh MLP so that src + f(src) matches tgt (Sec. 5.1)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    s = torch.as_tensor(src, dtype=torch.float32, device=device)
    t = torch.as_tensor(tgt, dtype=torch.float32, device=device)

    model = NeuralFlow(cfg.n_layers, cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    labels = None
    if cfg.beta > 0:
        labels = torch.as_tensor(cluster_labels(src, cfg.coh_eps, cfg.coh_min_samples), device=device)

    history = []
    idx_ab = idx_ba = None
    nn_every = max(1, int(getattr(cfg, "nn_every", 1)))
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        flow = model(s)
        warped = s + flow
        if it % nn_every == 0:          # neighbour search is the slow part; correspondences change slowly
            with torch.no_grad():
                idx_ab, idx_ba = nn_index(warped, t), nn_index(t, warped)
        l_cd = chamfer_loss(warped, t, cfg.cd_trunc, idx_ab, idx_ba)
        l_coh = coherence_loss(flow, labels) if labels is not None else torch.zeros((), device=device)
        loss = cfg.alpha * l_cd + cfg.beta * l_coh
        loss.backward()
        opt.step()
        history.append((float(l_cd), float(l_coh)))
    return NeuralFlowField(model, device, history)

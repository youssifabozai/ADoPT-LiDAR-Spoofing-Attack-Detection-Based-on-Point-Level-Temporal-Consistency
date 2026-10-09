"""Tests that need only numpy / scipy / scikit-learn.   Run:  python -m pytest tests -q"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adopt.attack import BankObject, inject                       # noqa: E402
from adopt.config import ADoPTConfig                              # noqa: E402
from adopt.detector import cluster_consistency                    # noqa: E402
from adopt.geometry import remove_ground, voxel_downsample        # noqa: E402
from adopt.metrics import chamfer_np, dcd_np, rates               # noqa: E402
from adopt.pipeline import ADoPT, build_synthesis                 # noqa: E402
from adopt.synthetic import OracleFlow, fake_pedestrian, make_sequence  # noqa: E402


def blob(center, n=200, s=0.15, seed=0):
    return np.random.default_rng(seed).normal(center, s, (n, 3)).astype(np.float32)


def test_detector_benign_has_no_leftover():
    synth = np.vstack([blob([5, 0, 1]), blob([10, 3, 1], seed=1)])
    new = np.vstack([blob([5.05, 0, 1], seed=2), blob([10, 3.05, 1], seed=3)])
    r = cluster_consistency(synth, new, eps=0.25, min_samples=10)
    assert r["score"] == 0 and r["n_clusters_after"] == 0


def test_detector_finds_object_without_history():
    synth = blob([5, 0, 1])
    new = np.vstack([blob([5, 0, 1], seed=2), blob([12, -4, 1], n=80, seed=3)])
    r = cluster_consistency(synth, new, eps=0.25, min_samples=10)
    assert r["n_clusters_after"] == 1
    assert r["leftover_mask"][200:].mean() > 0.9 and r["leftover_mask"][:200].sum() == 0


def test_voxel_downsample_merges_points():
    p = np.array([[0.01, 0.01, 0.01], [0.02, 0.03, 0.04], [1.0, 1.0, 1.0]])
    out = voxel_downsample(p, 0.1)
    assert len(out) == 2


def test_ground_removal():
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(-20, 20, (3000, 2)), rng.normal(0, 0.03, 3000)]
    obj = blob([8, 2, 1.0])
    kept, (n, d) = remove_ground(np.vstack([ground, obj]), 0.25)
    assert abs(n[2]) > 0.99 and len(kept) <= len(obj) + 30 and len(kept) >= len(obj) - 30


def test_inject_places_object_in_free_space():
    frame = np.vstack([blob([10, 0, 1], n=500), blob([0, 10, 1], n=500, seed=1)])
    obj = BankObject(blob([8, 0, 0.9], n=300, s=0.2, seed=5), "pedestrian")
    poisoned, mask, info = inject(frame, obj, "dense", np.random.default_rng(0))
    assert poisoned is not None and mask.sum() == 200 and info["n_fake"] == 200
    fake = poisoned[mask]
    assert np.min(np.linalg.norm(fake[:, None, :2] - frame[None, :, :2], axis=-1)) > 0.4


def test_metrics():
    a = blob([0, 0, 0])
    assert chamfer_np(a, a) == 0.0
    assert dcd_np(a, a) < dcd_np(a, a + 1.0)
    assert np.allclose(rates([0, 0, 5], [10, 0], 0), (100 / 3, 50.0))


def test_end_to_end_oracle_flow():
    """Synthesis + detector with ground-truth flow on the synthetic street."""
    cfg = ADoPTConfig.dense(remove_ground=False)
    dt, seq = 0.1, make_sequence(cfg.L + 1, 0.1)
    adopt = ADoPT(cfg, fit_fn=lambda s, t: None)
    history = [p for p, _ in seq[:-1]]
    synth = build_synthesis(history, [OracleFlow(p, v, dt) for p, v in seq[:-2]], cfg.synth_voxel)
    final = OracleFlow(*seq[-2], dt)
    new = seq[-1][0]
    benign = adopt.check(synth, new, final)
    fake = fake_pedestrian()
    attacked = adopt.check(synth, np.vstack([new, fake]), final)
    assert benign["score"] == 0
    assert attacked["attacked"] and attacked["leftover_mask"][-len(fake):].mean() > 0.9

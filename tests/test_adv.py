"""Tests for the adversarial-robustness extension (FGSM / PGD / adversarial training / ADoPT check)."""

import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adopt.adv_check import check_frame, frame_objects, summarize  # noqa: E402
from adopt.config import ADoPTConfig  # noqa: E402
from adopt.objects import ObjectSet, from_canonical, in_box_mask, synthetic_objects, to_canonical  # noqa: E402


# --------------------------------------------------------------------------- numpy parts
def test_canonical_roundtrip_and_box():
    rng = np.random.default_rng(0)
    center, yaw, wlh = np.array([10.0, -4.0, 0.9]), 0.7, np.array([1.9, 4.5, 1.6])
    local = rng.uniform(-0.5, 0.5, (200, 3)) * [4.5, 1.9, 1.6]          # inside the box (x = length)
    ego = from_canonical(local.astype(np.float32), center, yaw)
    assert np.allclose(to_canonical(ego, center, yaw), local, atol=1e-4)
    assert in_box_mask(ego, center, wlh, yaw, floor=0.0).all()
    outside = ego + np.array([0.0, 0.0, 5.0])
    assert not in_box_mask(outside, center, wlh, yaw).any()


def test_objectset_batch_and_io(tmp_path):
    data = synthetic_objects(20)
    assert len(data) == 60 and data.offsets[-1] == len(data.points)
    b = data.batch(data.idx(0)[:7], 64, np.random.default_rng(0), augment=True)
    assert b.shape == (7, 64, 3) and b.dtype == np.float32
    path = str(tmp_path / "o.npz")
    data.save(path)
    back = ObjectSet.load(path)
    assert np.array_equal(back.points, data.points) and list(back.sample) == list(data.sample)


def _scene(rng):
    """Static wall + one box-shaped object; history = same points with small noise."""
    wall = np.c_[rng.uniform(5, 30, 3000), np.full(3000, 8.0) + rng.normal(0, 0.02, 3000), rng.uniform(0.3, 2.5, 3000)]
    center, wlh, yaw = np.array([15.0, 0.0, 0.9]), np.array([1.0, 2.0, 1.6]), 0.3
    local = np.c_[rng.uniform(-1, 1, 1500), rng.uniform(-0.5, 0.5, 1500), rng.uniform(-0.6, 0.8, 1500)]
    obj = from_canonical(local.astype(np.float32), center, yaw)
    new = np.vstack([wall, obj]).astype(np.float32)
    synth = (new + rng.normal(0, 0.02, new.shape)).astype(np.float32)
    return synth, new, (7, center, wlh, yaw)


def test_adopt_check_small_vs_large_shift():
    rng = np.random.default_rng(1)
    synth, new, box = _scene(rng)
    cfg = ADoPTConfig.dense()
    objs = frame_objects(new, [box])
    assert len(objs) == 1 and len(objs[0][1]) > 1000
    oid, idx = objs[0]
    flagged, _ = check_frame(synth, new, objs, {}, cfg)
    assert flagged[oid] is False                                       # real object has history
    small = new[idx] + rng.uniform(-0.05, 0.05, (len(idx), 3)).astype(np.float32)
    flagged, _ = check_frame(synth, new, objs, {oid: small}, cfg)
    assert flagged[oid] is False                                       # 5 cm nudges stay in the cluster
    moved = new[idx] + np.array([0.0, 3.0, 0.0], np.float32)           # 3 m away: no history there
    flagged, score = check_frame(synth, new, objs, {oid: moved}, cfg)
    assert flagged[oid] is True and score > 0


def test_summarize():
    rows = [dict(model="standard", attack="pgd", eps=0.05, label=0, pred_clean=0, pred_adv=1, flagged=0, max_disp=0.08),
            dict(model="standard", attack="pgd", eps=0.05, label=1, pred_clean=1, pred_adv=1, flagged=1, max_disp=0.08),
            dict(model="standard", attack="pgd", eps=0.05, label=2, pred_clean=0, pred_adv=0, flagged=0, max_disp=0.08)]
    (s,) = summarize(rows)
    assert s["objects"] == 2 and s["fooled"] == 0.5 and s["flagged"] == 0.5 and s["evaded"] == 0.5


# --------------------------------------------------------------------------- torch parts
needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")


def _tiny_model():
    import torch
    from adopt.adversarial import train_model
    data = synthetic_objects(60)
    model, _ = train_model(data, "standard", 0.05, epochs=3, batch=32, n_points=64, log=lambda *a: None)
    x = torch.from_numpy(data.batch(data.idx(1), 64, np.random.default_rng(0)))
    y = torch.from_numpy(data.labels[data.idx(1)])
    return model, x, y


@needs_torch
def test_attacks_stay_in_budget():
    import torch
    from adopt.adversarial import fgsm, pgd
    model, x, y = _tiny_model()
    for eps in (0.01, 0.1):
        assert (fgsm(model, x, y, eps) - x).abs().max() <= eps + 1e-6
        assert (pgd(model, x, y, eps, steps=5) - x).abs().max() <= eps + 1e-6
    assert torch.equal(fgsm(model, x, y, 0.0), x)


@needs_torch
def test_pgd_at_least_as_strong_as_fgsm_on_average():
    import torch
    from adopt.adversarial import fgsm, pgd
    import torch.nn.functional as F
    model, x, y = _tiny_model()
    model.eval()
    with torch.no_grad():
        l0 = F.cross_entropy(model(x), y)
        lf = F.cross_entropy(model(fgsm(model, x, y, 0.1)), y)
        lp = F.cross_entropy(model(pgd(model, x, y, 0.1, steps=10)), y)
    assert lf > l0 and lp > l0 and lp >= 0.9 * lf


@needs_torch
def test_variable_size_input_and_modes():
    import torch
    from adopt.adversarial import PointNetCls, train_model
    m = PointNetCls().eval()
    assert m(torch.randn(1, 37, 3)).shape == (1, 3)
    data = synthetic_objects(20)
    for mode in ("fgsm", "pgd", "both"):
        model, hist = train_model(data, mode, 0.05, epochs=1, batch=16, n_points=32, pgd_steps=2, log=lambda *a: None)
        assert np.isfinite(hist[-1]["loss"])

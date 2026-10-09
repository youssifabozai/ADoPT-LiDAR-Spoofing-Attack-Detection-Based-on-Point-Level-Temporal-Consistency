"""Tests for the neural scene flow (need PyTorch).   Run:  python -m pytest tests -q"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adopt.config import ADoPTConfig                                            # noqa: E402
from adopt.sfe import chamfer_loss, coherence_loss, coherence_loss_bruteforce, fit_flow  # noqa: E402


def test_coherence_fast_equals_paper_formula():
    """The O(N) implementation must equal the literal O(N^2) formula from the paper."""
    g = torch.Generator().manual_seed(0)
    flow = torch.randn(300, 3, generator=g)
    labels = torch.randint(-1, 5, (300,), generator=g)
    assert torch.allclose(coherence_loss(flow, labels), coherence_loss_bruteforce(flow, labels), rtol=1e-4)


def test_coherence_zero_for_rigid_motion():
    labels = torch.tensor([0] * 50 + [1] * 50)
    flow = torch.cat([torch.tensor([[1.0, 0, 0]]).repeat(50, 1), torch.tensor([[0, 2.0, 0]]).repeat(50, 1)])
    assert coherence_loss(flow, labels).item() < 1e-10


def test_chamfer_zero_on_identical_clouds():
    a = torch.randn(500, 3)
    assert chamfer_loss(a, a).item() < 1e-10


def test_chamfer_gradient_flows():
    a = torch.randn(200, 3, requires_grad=True)
    chamfer_loss(a, torch.randn(250, 3)).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_fit_flow_recovers_translation():
    """A compact object (0.5 m cube) moved 0.6 m: larger than its size, so Chamfer gives a clear signal.

    If this fails, the optimisation / loss plumbing is broken (not just slow).
    """
    rng = np.random.default_rng(0)
    cube = rng.uniform(-0.25, 0.25, (800, 3)).astype(np.float32) + np.float32([8, 3, 1])
    shift = np.float32([0.6, 0.0, 0.0])
    cfg = ADoPTConfig(iters=300, lr=3e-3, cd_trunc=None, nn_every=1)
    field = fit_flow(cube, cube + shift, cfg)
    f = field(cube).mean(0)
    assert abs(f[0] - 0.6) < 0.1 and abs(f[1]) < 0.1 and abs(f[2]) < 0.1, f

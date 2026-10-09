"""PointNet object classifier, FGSM / PGD attacks and adversarial training.

Threat model: the attacker may move every point of an object by at most `eps` metres along each
axis of the object's own frame (an L-infinity ball, the standard setting for FGSM and PGD). The
goal is to make the classifier output a wrong class.

  FGSM  (Goodfellow et al., 2015): one step  x' = x + eps * sign(grad_x loss)
  PGD   (Madry et al., 2018):      many small steps of size alpha, each followed by projecting back
                                   into the eps-ball; starts from a random point inside the ball.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .objects import CLASSES, ObjectSet


# --------------------------------------------------------------------------- model
class PointNetCls(nn.Module):
    """Vanilla PointNet classifier (Qi et al., 2017) without the T-Nets.

    Shared per-point MLP 3 -> 64 -> 128 -> 256, max-pool over points, then 256 -> 128 -> classes.
    Works for any number of points, so a full object (not resampled) can be classified as well.
    """

    def __init__(self, n_classes: int = len(CLASSES), dims=(64, 128, 256), fc: int = 128, dropout: float = 0.3):
        super().__init__()
        layers, d = [], 3
        for h in dims:
            layers += [nn.Conv1d(d, h, 1), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        self.feat = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(d, fc), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, N, 3)
        f = self.feat(x.transpose(1, 2))                          # (B, C, N)
        return self.head(f.max(dim=2).values)


# --------------------------------------------------------------------------- attacks
def _input_grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Gradient of the classification loss with respect to the perturbation only."""
    delta = delta.detach().requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(model(x + delta), y)
        (g,) = torch.autograd.grad(loss, delta)
    return g


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
         rand_init: bool = False, step: Optional[float] = None) -> torch.Tensor:
    """Fast Gradient Sign Method. With rand_init it is the 'FGSM + random start' variant used for
    FGSM adversarial training (Wong et al., 2020), which avoids catastrophic overfitting."""
    x = x.detach()
    if eps <= 0:
        return x.clone()
    delta = torch.empty_like(x).uniform_(-eps, eps) if rand_init else torch.zeros_like(x)
    step = step if step is not None else (1.25 * eps if rand_init else eps)
    g = _input_grad(model, x, y, delta)
    delta = (delta + step * g.sign()).clamp(-eps, eps)
    return (x + delta).detach()


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 10,
        alpha: Optional[float] = None, rand_init: bool = True) -> torch.Tensor:
    """Projected Gradient Descent (L-infinity). Default step alpha = 2.5 * eps / steps (Madry et al.)."""
    x = x.detach()
    if eps <= 0:
        return x.clone()
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    delta = torch.empty_like(x).uniform_(-eps, eps) if rand_init else torch.zeros_like(x)
    for _ in range(steps):
        g = _input_grad(model, x, y, delta)
        delta = (delta + alpha * g.sign()).clamp(-eps, eps)
    return (x + delta).detach()


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 512) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)])


def attack_batched(fn: Callable, model, x, y, eps, batch: int = 256, **kw) -> torch.Tensor:
    model.eval()
    return torch.cat([fn(model, x[i:i + batch], y[i:i + batch], eps, **kw) for i in range(0, len(x), batch)])


# --------------------------------------------------------------------------- training
MODES = ("standard", "fgsm", "pgd", "both")


def train_model(data: ObjectSet, mode: str, eps: float, epochs: int = 40, lr: float = 1e-3, batch: int = 64,
                n_points: int = 128, pgd_steps: int = 7, device: str = "cpu", seed: int = 0,
                log: Callable = print) -> Tuple[nn.Module, List[dict]]:
    """Train one classifier.

    mode = standard : clean examples only (no defense)
           fgsm     : adversarial training with FGSM (+ random start) examples
           pgd      : adversarial training with PGD examples
           both     : adversarial training where each batch is half FGSM, half PGD examples
    Adversarial modes use loss = 0.5 * CE(clean) + 0.5 * CE(adversarial), which keeps clean accuracy
    from collapsing on this small dataset. Classes are weighted by inverse frequency.
    """
    assert mode in MODES, mode
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PointNetCls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr, te = data.idx(0), data.idx(1)
    counts = np.bincount(data.labels[tr], minlength=len(CLASSES)).astype(np.float64)
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    x_te = torch.from_numpy(data.batch(te, n_points, np.random.default_rng(1234))).to(device)
    y_te = torch.from_numpy(data.labels[te]).to(device)
    history = []
    for ep in range(epochs):
        perm = rng.permutation(tr)
        tot, nb = 0.0, 0
        for b in range(0, len(perm), batch):
            idx = perm[b:b + batch]
            if len(idx) < 2:
                continue
            x = torch.from_numpy(data.batch(idx, n_points, rng, augment=True)).to(device)
            y = torch.from_numpy(data.labels[idx]).to(device)
            if mode == "standard":
                model.train()
                loss = F.cross_entropy(model(x), y, weight=w)
            else:
                model.eval()                       # craft the examples without touching BatchNorm statistics
                if mode == "fgsm":
                    xa = fgsm(model, x, y, eps, rand_init=True)
                elif mode == "pgd":
                    xa = pgd(model, x, y, eps, steps=pgd_steps)
                else:
                    h = len(idx) // 2
                    xa = torch.cat([fgsm(model, x[:h], y[:h], eps, rand_init=True),
                                    pgd(model, x[h:], y[h:], eps, steps=pgd_steps)])
                model.train()
                loss = 0.5 * F.cross_entropy(model(x), y, weight=w) + 0.5 * F.cross_entropy(model(xa), y, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        acc = float((predict(model, x_te) == y_te).float().mean())
        row = {"mode": mode, "epoch": ep + 1, "loss": tot / max(nb, 1), "test_clean_acc": acc}
        if (ep + 1) % 10 == 0 or ep + 1 == epochs:
            xa = attack_batched(pgd, model, x_te, y_te, eps, steps=10)
            row["test_pgd_acc"] = float((predict(model, xa) == y_te).float().mean())
        history.append(row)
        log("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    model.eval()
    return model, history


# --------------------------------------------------------------------------- evaluation
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps_list, pgd_steps: int = 20,
             seed: int = 0) -> List[Dict]:
    """Accuracy on clean data and under FGSM, PGD and both (per-sample worst case).

    'both' counts a sample as correct only if it survives FGSM *and* PGD, i.e. an attacker who
    tries both and keeps whichever works.
    """
    torch.manual_seed(seed)
    model.eval()
    rows = []
    clean_ok = predict(model, x) == y

    def row(attack, eps, ok):
        r = {"attack": attack, "eps": eps, "acc": float(ok.float().mean()),
             # attack success rate: share of correctly classified samples that the attack flips
             "asr": float((clean_ok & ~ok).sum() / max(int(clean_ok.sum()), 1))}
        for c, name in enumerate(CLASSES):
            m = y == c
            r[f"acc_{name}"] = float(ok[m].float().mean()) if m.any() else float("nan")
        return r

    rows.append(row("clean", 0.0, clean_ok))
    for eps in eps_list:
        ok_f = predict(model, attack_batched(fgsm, model, x, y, eps)) == y
        ok_p = predict(model, attack_batched(pgd, model, x, y, eps, steps=pgd_steps)) == y
        rows += [row("fgsm", eps, ok_f), row("pgd", eps, ok_p), row("both", eps, ok_f & ok_p)]
    return rows

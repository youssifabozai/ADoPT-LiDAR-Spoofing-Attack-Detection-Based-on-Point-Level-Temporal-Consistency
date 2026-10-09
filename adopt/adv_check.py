"""Does ADoPT notice an adversarially perturbed object? (numpy only, no PyTorch)

ADoPT was built against *injected* points: a fake object that has no history. FGSM and PGD instead
*move existing points* of a real object by at most eps. These helpers put the perturbed points back
into a saved LiDAR frame and rerun ADoPT's cluster check, so we can count how often the perturbed
object ends up in a cluster without history (= flagged).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ADoPTConfig
from .detector import cluster_consistency
from .objects import in_box_mask


def frame_objects(new: np.ndarray, boxes: Sequence[Tuple[int, np.ndarray, np.ndarray, float]],
                  min_points: int = 20, margin: float = 0.1, floor: float = 0.15) -> List[Tuple[int, np.ndarray]]:
    """Points of the (detection-resolution) frame that fall inside each annotated box.

    boxes: (object id, centre, wlh, yaw). A point is given to the first box that contains it.
    Returns (object id, indices into `new`) for every box with at least `min_points` points.
    """
    taken = np.zeros(len(new), bool)
    out = []
    for oid, center, wlh, yaw in boxes:
        m = in_box_mask(new, center, wlh, yaw, margin=margin, floor=floor) & ~taken
        if m.sum() >= min_points:
            idx = np.where(m)[0]
            taken[idx] = True
            out.append((oid, idx))
    return out


def check_frame(synth: np.ndarray, new: np.ndarray, objs: List[Tuple[int, np.ndarray]],
                replaced: Dict[int, np.ndarray], cfg: ADoPTConfig) -> Tuple[Dict[int, bool], int]:
    """Replace the points of the objects in `replaced` (object id -> new ego-frame points, same
    count and order as the object's indices), run ADoPT's cluster check, and report per object
    whether any of its points was left without history. Also returns the frame score."""
    frame = new.copy()
    for oid, idx in objs:
        if oid in replaced:
            frame[idx] = replaced[oid]
    res = cluster_consistency(synth, frame, cfg.det_eps, cfg.det_min_samples,
                              cfg.min_cluster_height, cfg.det_min_range)
    leftover = res["leftover_mask"]
    return {oid: bool(leftover[idx].any()) for oid, idx in objs}, res["score"]


_MODEL_ORDER = ["standard", "fgsm", "pgd", "both"]


def summarize(rows: List[dict]) -> List[dict]:
    """Aggregate per-object rows into the ADoPT-as-defense table.

    For every (model, attack, eps):
      objects       objects the model classified correctly before the attack
      fooled        share of them misclassified after the attack          (attack success, no ADoPT)
      flagged       share of them that ADoPT flags after the attack
      evaded        share fooled AND not flagged                          (attack success, with ADoPT)
    """
    out = []
    keys = sorted({(r["model"], r["attack"], float(r["eps"])) for r in rows},
                  key=lambda k: (_MODEL_ORDER.index(k[0]) if k[0] in _MODEL_ORDER else 9, k[0],
                                 ["none", "fgsm", "pgd", "both"].index(k[1]), k[2]))
    for model, attack, eps in keys:
        g = [r for r in rows if r["model"] == model and r["attack"] == attack and float(r["eps"]) == eps]
        ok = [r for r in g if int(r["pred_clean"]) == int(r["label"])]
        n = max(len(ok), 1)
        fooled = [int(r["pred_adv"]) != int(r["label"]) for r in ok]
        flagged = [str(r["flagged"]) in ("1", "True") for r in ok]
        out.append({"model": model, "attack": attack, "eps": eps, "objects": len(ok),
                    "clean_acc": len(ok) / max(len(g), 1),
                    "fooled": sum(fooled) / n, "flagged": sum(flagged) / n,
                    "evaded": sum(f and not fl for f, fl in zip(fooled, flagged)) / n,
                    "mean_max_disp": float(np.mean([float(r["max_disp"]) for r in ok])) if ok else 0.0})
    return out

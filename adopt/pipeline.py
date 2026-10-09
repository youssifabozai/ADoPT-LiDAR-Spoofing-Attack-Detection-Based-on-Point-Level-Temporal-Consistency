"""End-to-end ADoPT pipeline (paper Fig. 3).

    F_1 ... F_L  --SFE-->  synthesis (all history warped to time L)
    synthesis, F_{L+1}  --SFE-->  warped synthesis at time L+1
    warped synthesis + F_{L+1}  --cluster consistency-->  genuine / fake

Core version: the synthesis is built by *chaining* the per-pair flow MLPs
(each frame k is pushed through f_k, f_{k+1}, ..., f_{L-1}). In a streaming
setting every pair flow is fitted only once and reused, so the number of
optimisations per new frame is O(1) (O(L) for a cold start). The paper's
voxel-based Scene Flow Approximation (Sec. 5.3) is an alternative way of
reusing flows and is part of the "full" stage of this project.
"""

import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from .config import ADoPTConfig
from .detector import cluster_consistency
from .geometry import crop, voxel_downsample, remove_ground

FlowFn = Callable[[np.ndarray], np.ndarray]
FitFn = Callable[[np.ndarray, np.ndarray], FlowFn]


def build_synthesis(history: Sequence[np.ndarray], flow_fns: Sequence[FlowFn], voxel: float) -> np.ndarray:
    """Warp every historical frame to the time of the last one and merge them.

    history:  [F_1, ..., F_L] (same coordinate frame)
    flow_fns: [f_1, ..., f_{L-1}], f_k maps points at time k to their displacement until k+1
    """
    assert len(flow_fns) == len(history) - 1
    merged = []
    for k, pts in enumerate(history):
        p = pts
        for j in range(k, len(history) - 1):
            p = p + flow_fns[j](p)
        merged.append(p)
    return voxel_downsample(np.vstack(merged), voxel)


class ADoPT:
    def __init__(self, cfg: ADoPTConfig, fit_fn: Optional[FitFn] = None, device=None):
        self.cfg = cfg
        if fit_fn is None:
            from .sfe import fit_flow          # lazy: torch only needed here
            fit_fn = lambda src, tgt: fit_flow(src, tgt, cfg, device)
        self.fit = fit_fn

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, points: np.ndarray, radius: Optional[float] = None, voxel="default") -> np.ndarray:
        """Crop, remove ground, voxel-downsample. Points must be in an ego frame (z up, origin on the road).

        voxel="default" uses cfg.input_voxel; pass cfg.det_voxel (or None for raw) for the detection copy.
        """
        c = self.cfg
        p = crop(points, radius or c.roi_synth, c.z_min, c.z_max)
        if c.remove_ground:
            p, _ = remove_ground(p, c.ground_thresh)
        return voxel_downsample(p, c.input_voxel if voxel == "default" else voxel)

    # ---------------------------------------------------------------- stages
    def synthesize(self, history: Sequence[np.ndarray], flow_fns: Optional[List[FlowFn]] = None) -> np.ndarray:
        if flow_fns is None:
            flow_fns = [self.fit(history[k], history[k + 1]) for k in range(len(history) - 1)]
        return build_synthesis(history, flow_fns, self.cfg.synth_voxel)

    def check(self, synth: np.ndarray, new_frame: np.ndarray, final_flow: Optional[FlowFn] = None,
              new_frame_det: Optional[np.ndarray] = None) -> dict:
        """Warp the synthesis to the new frame's time and run the cluster check.

        new_frame (coarse, cfg.input_voxel) is used to fit the final scene flow;
        new_frame_det (finer, cfg.det_voxel) is used for clustering, if given.
        """
        c = self.cfg
        t0 = time.perf_counter()
        h = final_flow if final_flow is not None else self.fit(synth, new_frame)
        warped = synth + h(synth)
        t_sfe = time.perf_counter() - t0

        new_det = crop(new_frame if new_frame_det is None else new_frame_det, c.roi_det)
        synth_det = crop(warped, c.roi_det + 2.0)       # small margin so border points have history
        res = cluster_consistency(synth_det, new_det, c.det_eps, c.det_min_samples,
                                  c.min_cluster_height, c.det_min_range)
        res["attacked"] = res["score"] > c.score_threshold
        res["warped_synth"] = synth_det
        res["new_det"] = new_det
        res["t_final_sfe"] = t_sfe
        res["t_total"] = time.perf_counter() - t0
        return res

    def __call__(self, history: Sequence[np.ndarray], new_frame: np.ndarray) -> dict:
        t0 = time.perf_counter()
        synth = self.synthesize(history)
        res = self.check(synth, new_frame)
        res["t_total"] = time.perf_counter() - t0
        return res

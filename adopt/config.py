"""All hyper-parameters in one place.

Values marked [paper] come straight from the paper (Sec. 6, App. B).
Values marked [assumed] are NOT given in the paper; they are our choices
and should be listed as assumptions in your report.
"""

from dataclasses import dataclass, asdict, replace
from typing import Optional


@dataclass
class ADoPTConfig:
    # ---------------- history ----------------
    L: int = 10                 # [paper] number of historical frames
    stride: int = 2             # [assumed] nuScenes sweeps are 20 Hz -> every 2nd = 10 Hz [paper uses 10 Hz]

    # ---------------- preprocessing ----------------
    roi_synth: float = 55.0     # [assumed] radius (m) kept for synthesis / scene flow
    roi_det: float = 50.0       # [assumed] radius (m) where we look for fake objects
    z_min: float = -3.0         # [assumed] height crop in ego frame (m)
    z_max: float = 4.0
    remove_ground: bool = True  # [assumed] paper does not mention it, but the ground would form one huge cluster
    ground_thresh: float = 0.25 # [assumed] points closer than this to the ground plane are removed
    min_dist: float = 3.0       # [assumed] drop points this close to the sensor (ego-vehicle body)
    input_voxel: float = 0.1    # [paper, Fig. 9] voxel edge (m) for scene flow / synthesis (paper sweeps 0.05-0.25)
    det_voxel: Optional[float] = 0.05  # [assumed] voxel edge of the NEW frame used for detection (None = raw).
                                       #   With 0.1 m, flat surfaces (car sides) never reach 17 points in 0.25 m
    synth_voxel: float = 0.1    # [assumed] voxel edge used to thin the merged synthesis

    # ---------------- scene flow estimation (Sec. 5.1) ----------------
    n_layers: int = 6           # [paper] MLP depth (number of Linear layers)
    hidden: int = 128           # [paper] hidden units
    iters: int = 300            # [tuned] paper: 30. Did not converge in our setup (scripts/diag_sfe.py)
    lr: float = 3e-3            # [tuned] paper: 8e-4
    alpha: float = 1.0          # [paper] Chamfer weight
    beta: float = 2.0           # [paper] coherence weight
    cd_trunc: Optional[float] = 0.5   # [tuned] ignore Chamfer matches farther than this (m); None = off.
                                      #   2.0 let the final flow pull history onto fakes (TPR 70% -> 100% at 0.5)
    nn_every: int = 10          # [tuned] recompute Chamfer nearest neighbours every k iterations (2.4x faster)
    coh_eps: float = 0.4        # [assumed] DBSCAN eps for the coherence mask M(p_i, p_j)
    coh_min_samples: int = 10   # [assumed] DBSCAN min_samples for the coherence mask

    # ---------------- detection (Sec. 5.2, App. B) ----------------
    det_eps: float = 0.25       # [paper] dense: 0.25, sparse: 0.75
    det_min_samples: int = 17   # [paper] dense: 17,  sparse: 9
    min_cluster_height: float = 0.3   # [assumed] ignore history-free clusters flatter than this (m)
    det_min_range: float = 4.0  # [assumed] ignore history-free clusters centred closer than this (m):
                                #   just outside the 3 m self-crop (min_dist) there is no history
    score_threshold: int = 0    # [assumed] frame is "attacked" if score > threshold
                                #           score = #new-frame points in clusters with no history

    @classmethod
    def dense(cls, **overrides) -> "ADoPTConfig":
        return replace(cls(det_eps=0.25, det_min_samples=17), **overrides)

    @classmethod
    def sparse(cls, **overrides) -> "ADoPTConfig":
        return replace(cls(det_eps=0.75, det_min_samples=9), **overrides)

    def to_dict(self) -> dict:
        return asdict(self)

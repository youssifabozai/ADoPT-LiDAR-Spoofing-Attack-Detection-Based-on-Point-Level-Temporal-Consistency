"""ADoPT replication: LiDAR spoofing detection via point-level temporal consistency.

Paper: Cho et al., "ADoPT: LiDAR Spoofing Attack Detection Based on
Point-Level Temporal Consistency", BMVC 2023 (arXiv:2310.14504).

Torch-dependent pieces (scene flow) are imported lazily so that the
geometry / detector / attack modules can be used without PyTorch.
"""

import os
import warnings

# Harmless scikit-learn / joblib version-mismatch warning, printed on every multi-core DBSCAN call.
# It does not change any result but floods the console and hides the progress lines.
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")
# The warning can also come from joblib worker processes, which do not inherit the filter above
# but do inherit environment variables.
_rule = "ignore::UserWarning:sklearn.utils.parallel"
if _rule not in os.environ.get("PYTHONWARNINGS", ""):
    os.environ["PYTHONWARNINGS"] = ",".join(x for x in (os.environ.get("PYTHONWARNINGS"), _rule) if x)

from .config import ADoPTConfig  # noqa: E402

__all__ = ["ADoPTConfig"]

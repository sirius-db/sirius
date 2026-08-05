"""Clock-domain alignment: nsys session ns <-> Quent unix-epoch ns.

Procedure per nsys-extraction.md section 3.2:
1. first order: quent_ns ~= nsys_ns + utcEpochNs (TARGET_INFO_SESSION_START_TIME);
2. robust refinement: least-squares fit quent_t = a + b * nsys_t over matched
   event pairs (Quent Computing transitions vs NVTX op-range starts), with
   >3-sigma outlier rejection and one refit. Expect b ~= 1 +- 1e-5 and
   sub-100 us rms when the sqlite and the trace come from the *same* run.

Alignment is diagnostic-grade only: the physics join itself is by structural
key (query, pipeline, operator, task ordinal), never by timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class ClockFit:
    offset_ns: float  # a
    slope: float  # b
    rms_ns: float
    n_pairs: int
    n_rejected: int

    def nsys_to_epoch(self, nsys_ns: float) -> float:
        return self.offset_ns + self.slope * nsys_ns

    def to_dict(self) -> dict:
        return {
            "offset_ns": self.offset_ns,
            "slope": self.slope,
            "rms_ns": self.rms_ns,
            "n_pairs": self.n_pairs,
            "n_rejected": self.n_rejected,
        }


def _lsq(pairs: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(pairs)
    mean_x = sum(p[0] for p in pairs) / n
    mean_y = sum(p[1] for p in pairs) / n
    var = sum((x - mean_x) ** 2 for x, _ in pairs)
    if var == 0:
        return mean_y - mean_x, 1.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    b = cov / var
    a = mean_y - b * mean_x
    return a, b


def fit_linear(
    pairs: Sequence[Tuple[float, float]],  # (nsys_ns, quent_epoch_ns)
    sigma_reject: float = 3.0,
) -> Optional[ClockFit]:
    """Robust linear fit with one round of >k-sigma outlier rejection.

    The fit is done on coordinates centered at the first pair — epoch-ns
    values are ~1.7e18, where float64 resolution is ~256 ns, so fitting raw
    values would bury the sub-microsecond residuals we care about.
    """
    if len(pairs) < 2:
        return None
    x0, y0 = pairs[0]
    rel = [(x - x0, y - y0) for x, y in pairs]
    a, b = _lsq(rel)
    resid = [y - (a + b * x) for x, y in rel]
    n = len(rel)
    mean_r = sum(resid) / n
    sigma = (sum((r - mean_r) ** 2 for r in resid) / n) ** 0.5
    # sub-ns sigma == numerically perfect fit; 3-sigma rejection on float
    # noise would discard arbitrary points.
    kept: List[Tuple[float, float]] = rel if sigma < 1.0 else [
        p for p, r in zip(rel, resid) if abs(r - mean_r) <= sigma_reject * sigma
    ]
    n_rej = len(rel) - len(kept)
    if len(kept) >= 2 and n_rej:
        a, b = _lsq(kept)
    resid = [y - (a + b * x) for x, y in kept]
    rms = (sum(r * r for r in resid) / len(kept)) ** 0.5 if kept else 0.0
    return ClockFit(
        offset_ns=y0 + a - b * x0,
        slope=b,
        rms_ns=rms,
        n_pairs=len(kept),
        n_rejected=n_rej,
    )


def first_order_offset(utc_epoch_ns: Optional[int]) -> Optional[ClockFit]:
    """Anchor-only alignment when no matched pairs are available."""
    if utc_epoch_ns is None:
        return None
    return ClockFit(
        offset_ns=float(utc_epoch_ns), slope=1.0, rms_ns=0.0, n_pairs=0, n_rejected=0
    )

"""Per-size transfer bandwidth curves fitted from memcpy samples.

Per nsys-extraction.md section 2.2c: fit ``t = alpha + bytes / beta`` per
(direction, src_kind, dst_kind) channel and per log2(bytes) bucket — measured
D2D behavior differs regime-by-regime (the memcpy-flag cap on large buffers vs
small-buffer behavior), so one pooled number is wrong by construction.

Knob application: bandwidth knobs scale beta only; alpha (per-copy latency /
launch cost) does not scale with link speed. ``scale_factor`` therefore
returns time(mult)/time(1) which -> 1 for tiny copies and -> 1/mult for large
ones — the honest size-dependent gain curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class BucketFit:
    log2_bytes: int
    n: int
    alpha_ns: float  # per-copy fixed cost
    beta_gbps: float  # marginal bandwidth, bytes/ns == GB/s
    pooled_gbps: float  # sum(bytes)/sum(ns), for diagnostics

    def to_dict(self) -> dict:
        return {
            "log2_bytes": self.log2_bytes,
            "n": self.n,
            "alpha_ns": self.alpha_ns,
            "beta_gbps": self.beta_gbps,
            "pooled_gbps": self.pooled_gbps,
        }

    @staticmethod
    def from_dict(d: dict) -> "BucketFit":
        return BucketFit(
            int(d["log2_bytes"]),
            int(d["n"]),
            float(d["alpha_ns"]),
            float(d["beta_gbps"]),
            float(d["pooled_gbps"]),
        )


@dataclass
class BandwidthCurve:
    channel: str  # "Host-to-Device|Pinned|Device" etc.
    buckets: Dict[int, BucketFit] = field(default_factory=dict)

    def _bucket_for(self, nbytes: int) -> Optional[BucketFit]:
        if not self.buckets:
            return None
        b = int(math.log2(nbytes)) if nbytes > 0 else 0
        if b in self.buckets:
            return self.buckets[b]
        # nearest available bucket
        key = min(self.buckets, key=lambda k: abs(k - b))
        return self.buckets[key]

    def predict_ns(self, nbytes: int, mult: float = 1.0) -> Optional[float]:
        fit = self._bucket_for(nbytes)
        if fit is None or fit.beta_gbps <= 0:
            return None
        return fit.alpha_ns + nbytes / (fit.beta_gbps * mult)

    def scale_factor(self, copies: List[Tuple[int, float]], mult: float) -> float:
        """time(mult)/time(1) over a set of (bytes, measured_dur_ns) copies.

        Falls back to 1/mult (pure linear scaling) when the curve cannot
        predict — degrading to the v0 behavior rather than failing.
        """
        if mult == 1.0:
            return 1.0
        base = scaled = 0.0
        for nbytes, _dur in copies:
            t1 = self.predict_ns(nbytes, 1.0)
            tm = self.predict_ns(nbytes, mult)
            if t1 is None or tm is None:
                return 1.0 / mult
            base += t1
            scaled += tm
        if base <= 0:
            return 1.0 / mult
        return scaled / base

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "buckets": {str(k): v.to_dict() for k, v in self.buckets.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "BandwidthCurve":
        c = BandwidthCurve(channel=d["channel"])
        c.buckets = {
            int(k): BucketFit.from_dict(v) for k, v in d.get("buckets", {}).items()
        }
        return c


def _fit_bucket(log2b: int, points: List[Tuple[int, float]]) -> BucketFit:
    """Least-squares t = alpha + bytes/beta within one bucket; degenerate
    inputs fall back to the pooled rate with alpha = 0."""
    n = len(points)
    sum_b = sum(p[0] for p in points)
    sum_t = sum(p[1] for p in points)
    pooled = sum_b / sum_t if sum_t > 0 else 0.0
    if n >= 3:
        mean_b = sum_b / n
        mean_t = sum_t / n
        var = sum((b - mean_b) ** 2 for b, _ in points)
        cov = sum((b - mean_b) * (t - mean_t) for b, t in points)
        if var > 0 and cov > 0:
            slope = cov / var  # ns per byte
            alpha = mean_t - slope * mean_b
            if alpha >= 0 and slope > 0:
                return BucketFit(log2b, n, alpha, 1.0 / slope, pooled)
    return BucketFit(log2b, n, 0.0, pooled, pooled)


def fit_curves(
    memcpys: Iterable,  # objects with .bytes, .dur_ns, .channel
) -> Dict[str, BandwidthCurve]:
    by_key: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    for m in memcpys:
        if m.bytes <= 0 or m.dur_ns <= 0:
            continue
        log2b = int(math.log2(m.bytes))
        by_key.setdefault((m.channel, log2b), []).append((m.bytes, m.dur_ns))
    curves: Dict[str, BandwidthCurve] = {}
    for (channel, log2b), pts in sorted(by_key.items()):
        curve = curves.setdefault(channel, BandwidthCurve(channel=channel))
        curve.buckets[log2b] = _fit_bucket(log2b, pts)
    return curves

"""Hardware knobs: continuous multipliers relative to the traced machine.

Every knob defaults to 1.0 (= the traced hardware). Values > 1 mean "more /
faster", < 1 mean "less / slower". Fidelity caveats per knob are emitted as
warnings whenever a knob with a known telemetry gap is moved off 1.0 — see
tools/hwsim/docs/simulator-design.md for the honest-limitations section.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional

KNOWN_KNOBS = (
    "c2c_bandwidth",
    "gpu_mem_capacity",
    "gpu_compute",
    "gpu_mem_bandwidth",
    "io_bandwidth",
    "cpu_mem_capacity",
    "cpu_mem_bandwidth",
    "cpu_compute",
)


@dataclass
class Knobs:
    c2c_bandwidth: float = 1.0
    gpu_mem_capacity: float = 1.0
    gpu_compute: float = 1.0
    # None => tracks gpu_compute (v0 cannot separate compute from HBM until the
    # nsys physics join, gap G4). If set explicitly, operator spans scale by
    # 1/min(gpu_compute, gpu_mem_bandwidth) — a pessimistic roofline bound.
    gpu_mem_bandwidth: Optional[float] = None
    io_bandwidth: float = 1.0
    # Accepted but NOT modeled in v0 (no host-pool admission / host-bandwidth
    # events in the trace) — moving them only produces a warning.
    cpu_mem_capacity: float = 1.0
    cpu_mem_bandwidth: float = 1.0
    cpu_compute: float = 1.0

    @property
    def gpu_speed(self) -> float:
        """Effective multiplier applied to operator Computing spans."""
        if self.gpu_mem_bandwidth is None:
            return self.gpu_compute
        return min(self.gpu_compute, self.gpu_mem_bandwidth)

    def op_scale(self, op_name: str) -> float:
        """Duration divisor for one operator Computing span."""
        s = self.gpu_speed
        if op_name.startswith("GPU_SCAN"):
            s *= self.io_bandwidth
        return s

    def is_baseline(self) -> bool:
        return (
            self.c2c_bandwidth == 1.0
            and self.gpu_mem_capacity == 1.0
            and self.gpu_speed == 1.0
            and self.io_bandwidth == 1.0
        )

    def warnings(self) -> List[str]:
        w: List[str] = []
        if self.io_bandwidth != 1.0:
            w.append(
                "io_bandwidth: DEGRADED FIDELITY (gap G1) — the trace has no "
                "I/O events; disk read time is fused into GPU_SCAN Computing "
                "spans. v0 scales whole GPU_SCAN spans by 1/io_bandwidth, which "
                "also scales the GPU decode work and cannot reproduce the "
                "'faster I/O -> memory back-pressure at the scan' scenario. "
                "Treat results as a rough bound until G1 instrumentation lands."
            )
        if self.gpu_mem_bandwidth is not None:
            w.append(
                "gpu_mem_bandwidth: PLACEHOLDER (gap G4) — quent Computing "
                "spans do not separate SM-bound from HBM-bound time. v0 scales "
                "operator spans by 1/min(gpu_compute, gpu_mem_bandwidth) "
                "(pessimistic roofline). Use the nsys physics join for a real "
                "split."
            )
        if self.gpu_compute != 1.0:
            w.append(
                "gpu_compute: v0 scales whole operator Computing spans — this "
                "conflates SM throughput, HBM bandwidth, kernel-launch overhead "
                "and host-side glue (gap G4). Fixed per-launch overhead does "
                "not really scale with clocks, so speedups are optimistic."
            )
        if self.c2c_bandwidth != 1.0:
            w.append(
                "c2c_bandwidth: Preparing spans include GPU-side decompression "
                "which is SM-bound on GB300, not link-bound. v0 re-times the "
                "whole span from bytes at the scaled effective rate, so gains "
                "from c2c > 1 are optimistic; the channel capacity is the peak "
                "aggregate rate *observed in the trace* (a lower bound on the "
                "real link)."
            )
        if self.gpu_mem_capacity != 1.0:
            w.append(
                "gpu_mem_capacity: shrinking capacity makes tasks WAIT for "
                "reservation admission. The real engine would instead spill "
                "(downgrade) — the sample trace has zero Downgrading events, so "
                "spill cost cannot be calibrated yet; simulated slowdowns are a "
                "back-pressure-only approximation (see v1 roadmap)."
            )
        for name in ("cpu_mem_capacity", "cpu_mem_bandwidth", "cpu_compute"):
            if getattr(self, name) != 1.0:
                w.append(
                    f"{name}: NOT MODELED in v0 — knob accepted but has no "
                    "effect (no host-side pool/bandwidth events in quent)."
                )
        return w

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def describe(self) -> str:
        parts = []
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and v != 1.0:
                parts.append(f"{f.name}={v:g}")
        return ", ".join(parts) if parts else "baseline (all 1.0)"


def parse_knob_args(pairs: List[str]) -> Knobs:
    """Parse repeated ``--knob name=value`` CLI arguments."""
    k = Knobs()
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"bad --knob {pair!r}; expected name=value")
        name, _, val = pair.partition("=")
        name = name.strip()
        if name not in KNOWN_KNOBS:
            raise ValueError(f"unknown knob {name!r}; known: {', '.join(KNOWN_KNOBS)}")
        try:
            fval = float(val)
        except ValueError:
            raise ValueError(f"bad value for knob {name}: {val!r}")
        if fval <= 0:
            raise ValueError(f"knob {name} must be > 0, got {fval}")
        setattr(k, name, fval)
    return k

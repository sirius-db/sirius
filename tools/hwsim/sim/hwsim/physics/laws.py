"""Measured platform coupling laws (GB300 / Grace-Blackwell).

These constants and functions encode physics measured by WS4/WS5 on this box
(tools/hwsim/docs/compute-throttle.md, membw-throttle.md). They are model
*constraints*, not free parameters — do not tune them to fit a trace.

1. SM-issue cap on memory bandwidth (compute-throttle.md, cross-talk section):
   a free SM sustains only ~40 GB/s of streaming issue, so achievable HBM
   bandwidth is capped at ``free_SMs x ~40 GB/s`` (152 x 40 ~= 6.1 TB/s at
   full SM count vs ~4.8 TB/s HBM). Interpreting ``gpu_compute`` as an SM
   count/throughput multiplier, membw-bound portions therefore scale by
   ``min(gpu_mem_bandwidth, gpu_compute * SM_BW_HEADROOM)``: shrinking compute
   below ~1/1.27 of the membw knob starts throttling memory-bound kernels too,
   while *raising* gpu_compute alone never speeds a membw-bound kernel.

2. Grace C2C <-> host DRAM co-limit (membw-throttle.md section 5): a C2C H2D
   transfer reads host DRAM at line rate (and D2H writes it), so C2C-link
   transfers are co-limited: effective multiplier
   ``min(c2c_bandwidth, cpu_mem_bandwidth)``. A C2C experiment is always a
   joint (c2c, cpu_mem) experiment on this machine.

3. Decompress kernels are SM-bound on this box (nsys-extraction.md section
   5.1, memory note "decompress is SM-bound") — encoded as a name-prior in
   ``classify.py``, so the Preparing phase's kernel share scales with
   ``gpu_compute``, not with the link.
"""

from __future__ import annotations

from ..knobs import Knobs

# (152 SMs x ~40 GB/s streaming issue) / ~4.8 TB/s achievable HBM ~= 1.27.
SM_BW_HEADROOM = 1.27


def resolved_gpu_mem_bandwidth(knobs: Knobs) -> float:
    """gpu_mem_bandwidth knob value under split semantics.

    Under the physics join, ``None`` means "traced HBM" (1.0) — NOT "tracks
    gpu_compute" as in v0, because the whole point of the split is that faster
    SMs do not add HBM bandwidth.
    """
    return knobs.gpu_mem_bandwidth if knobs.gpu_mem_bandwidth is not None else 1.0


def effective_membw_mult(knobs: Knobs) -> float:
    """Multiplier applied to membw-bound kernel-busy time (law 1)."""
    return min(resolved_gpu_mem_bandwidth(knobs), knobs.gpu_compute * SM_BW_HEADROOM)


def conflated_mult(knobs: Knobs) -> float:
    """v0 conflated multiplier — used for *unclassified / unmatched* GPU time
    so unmatched time degrades to exactly the v0 behavior."""
    return knobs.gpu_speed


def transfer_mult(direction: str, knobs: Knobs) -> float:
    """Multiplier for explicit-transfer (memcpy) time by direction (law 2).

    direction: "h2d" | "d2h" | "d2d".
    """
    if direction == "d2d":
        # Device-to-device copies contend for HBM (CE path; the memcpy-flag
        # cap is part of the traced rate already).
        return resolved_gpu_mem_bandwidth(knobs)
    # C2C link co-limited by host DRAM on Grace.
    return min(knobs.c2c_bandwidth, knobs.cpu_mem_bandwidth)


def channel_transfer_mult(origin_tier: str, target_tier: str, knobs: Knobs) -> float:
    """Transfer multiplier for a Preparing-materialization channel keyed by
    quent tier names (e.g. "HOST" -> "GPU-0")."""
    if origin_tier.startswith("GPU") and target_tier.startswith("GPU"):
        return transfer_mult("d2d", knobs)
    return transfer_mult("h2d", knobs)


def host_mult(knobs: Knobs) -> float:
    """Host-side glue (launch overhead, gaps, orchestration) scales only with
    cpu_compute (nsys-extraction.md section 5.4) — never with GPU knobs."""
    return knobs.cpu_compute

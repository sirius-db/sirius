"""Channel-capacity sanity checks and nsys wire-rate correction (WS12).

Two measured failure modes of trace-derived channel capacity, both found in
the validation campaign (validation-results.md):

1. **Unphysical capacity on coherent-C2C lanes** (E5): fresh Quent traces on
   GB300 carry ~zero-length SOURCE->GPU Preparing spans (median 22 us for
   ~455 MB), so the line-sweep "observed peak aggregate" comes out at
   ~165,000 GB/s — 430x the physical C2C line. Simulating such a trace with
   ``c2c_bandwidth``/``cpu_mem_bandwidth`` != 1 silently returns a no-op
   prediction. :func:`channel_capacity_warnings` makes that loud.

2. **Overlap-inflated capacity on staged-transfer lanes** (R2/R3, WS12):
   the physics split places each task's memcpy share as one contiguous
   sub-window at the traced span start, so concurrent tasks' sub-windows
   overlap in ways the real (CE-serialized) copies did not — measured
   3,433 GB/s derived vs 382 GB/s of true wire-side aggregate on the
   host-pinned lane, which made a c2c=0.5 what-if predict x1.15 where
   reality is x1.83. :func:`corrected_link_capacities` replaces the derived
   value with the nsys wire-side peak aggregate converted into quent-byte
   units via the measured quent/wire byte ratio of the *matched* transfers
   (compressed payloads make the units differ; the ratio is 1.0 on
   uncompressed lanes). Correction applies only downward and only when
   enough transfer bytes are annotation-covered.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

ChannelKey = Tuple[str, str, int]  # (origin_tier, target_tier, device)

# Physical ceiling for host<->GPU link capacity in quent-byte units.
# Wire line is ~340-380 GB/s (membw-throttle.md section 2); compressed
# payloads legitimately inflate quent-unit rates ~2x and line-sweep overlap
# adds transient spikes (709.6 GB/s observed on the WS6 sample trace, real).
# Anything above 1 TB/s cannot be wire time on this link.
C2C_PHYSICAL_MAX_GBPS = 1000.0

# Correction gating: at least this fraction of the channel's quent transfer
# bytes must come from tasks with a matched nsys transfer annotation.
MIN_COVERAGE = 0.5
# ... and the matched wire volume must be non-trivial (guards against the
# coherent-C2C lane where explicit copies are ~absent).
MIN_WIRE_BYTES = 1e9


def _is_link(key: ChannelKey) -> bool:
    origin, target, _ = key
    return not (origin.startswith("GPU") and target.startswith("GPU"))


def channel_capacity_warnings(
    channel_caps: Dict[ChannelKey, float],
    threshold: float = C2C_PHYSICAL_MAX_GBPS,
) -> List[str]:
    """Warn for host<->GPU channels whose trace-derived capacity is above the
    physical ceiling: on such traces the Preparing spans do not carry wire
    time and the c2c/cpu_mem knobs are silently inert."""
    w: List[str] = []
    for key, cap in sorted(channel_caps.items()):
        if not _is_link(key) or cap is None:
            continue
        if cap > threshold:
            origin, target, dev = key
            w.append(
                f"channel {origin}->{target}@gpu{dev}: trace-derived capacity "
                f"{cap:,.0f} GB/s exceeds the physical C2C ceiling "
                f"(~{threshold:,.0f} GB/s incl. compressed-payload headroom). "
                "Preparing spans in this trace do not carry wire time "
                "(coherent-C2C / zero-copy lane), so c2c_bandwidth / "
                "cpu_mem_bandwidth predictions are INERT on this trace. "
                "Re-capture a staged-transfer lane (e.g. host-pinned tables) "
                "or add wire-time instrumentation to the transfer path."
            )
    return w


def corrected_link_capacities(
    graph,
    annotations,
    profile,
) -> Dict[ChannelKey, float]:
    """nsys-corrected capacity per link channel, in quent-byte units.

    corrected = wire_peak_aggregate(dominant nsys channel) x
                (quent transfer bytes / nsys wire bytes over matched tasks)

    Returns only channels where the correction is applicable (coverage and
    volume gates); callers apply it as an upper bound (downward only).
    """
    wire_peaks = (profile.diagnostics or {}).get("channel_peak_gbps", {})
    if not wire_peaks:
        return {}
    # accumulate per quent channel key
    quent_bytes_all: Dict[ChannelKey, float] = {}
    quent_bytes_matched: Dict[ChannelKey, float] = {}
    wire_bytes_matched: Dict[ChannelKey, float] = {}
    chan_bytes: Dict[ChannelKey, Dict[str, float]] = {}
    for task in graph.tasks.values():
        if not (task.is_transfer_prep and task.prep_bytes > 0):
            continue
        key = (task.prep_origin, task.prep_target, task.device)
        if not _is_link(key):
            continue
        quent_bytes_all[key] = quent_bytes_all.get(key, 0.0) + task.prep_bytes
        ann = annotations.get(task.tid)
        prep = ann.prep if ann is not None else None
        if prep is None or prep.xfer_bytes <= 0:
            continue
        quent_bytes_matched[key] = quent_bytes_matched.get(key, 0.0) + task.prep_bytes
        wire_bytes_matched[key] = wire_bytes_matched.get(key, 0.0) + prep.xfer_bytes
        if prep.dominant_channel:
            d = chan_bytes.setdefault(key, {})
            d[prep.dominant_channel] = (
                d.get(prep.dominant_channel, 0.0) + prep.xfer_bytes
            )
    out: Dict[ChannelKey, float] = {}
    for key, total in quent_bytes_all.items():
        matched = quent_bytes_matched.get(key, 0.0)
        wire = wire_bytes_matched.get(key, 0.0)
        if total <= 0 or matched / total < MIN_COVERAGE or wire < MIN_WIRE_BYTES:
            continue
        by_chan = chan_bytes.get(key)
        if not by_chan:
            continue
        dominant = max(by_chan.items(), key=lambda kv: kv[1])[0]
        wire_peak = wire_peaks.get(dominant)
        if not wire_peak:
            continue
        out[key] = wire_peak * (matched / wire)
    return out

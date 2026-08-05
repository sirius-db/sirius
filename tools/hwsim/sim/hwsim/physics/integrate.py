"""Knob-split re-timing: apply physics annotations to a QueryGraph and run
the (unmodified) v0 discrete-event engine on the transformed graph.

Integration contract with the engine (no engine changes):

- per-operator Computing spans are **pre-scaled** here using the physics
  fractions and the coupling laws; the engine then runs with *neutralized*
  GPU knobs (gpu_compute = gpu_mem_bandwidth = c2c = 1) so nothing is scaled
  twice. ``io_bandwidth`` (gap G1) and ``gpu_mem_capacity`` (pool admission)
  keep their v0 engine-level semantics.
- the Preparing phase is split: the explicit-transfer share stays a fluid-
  channel transfer (re-timed via the measured per-size bandwidth curve, so
  contention stays emergent), while the kernel share (SM-bound decompress
  etc.) + host share move into a pseudo-operator ``PHYS::PREP`` executed after
  the transfer — this is the "split Preparing into decompress (SM) + copy
  (link)" item of the v1 roadmap (simulator-design.md section 10, G4).
- channel capacities are re-derived by line-sweeping the *transfer-only*
  sub-windows (traced full-span rates understate the link when prep contained
  decompress), then scaled by the co-limited transfer multiplier
  min(c2c_bandwidth, cpu_mem_bandwidth) from laws.py.

Fallback rule: any task/op without a physics annotation keeps exactly the v0
conflated scaling (spans / min(gpu_compute, gpu_mem_bandwidth); transfers /
link multiplier) and is counted + warned about — degraded, never dropped.

Device contention model (gap G4b, validation-results.md §8): when a GPU
kernel knob moves (gpu_compute != 1 or gpu_mem_bandwidth set) AND the
capture's measured kernel-serialization fraction (union/sum of kernel time
on the device timeline) shows the lane's kernels serialize (>= 0.9 — the
capacity premise; low-occupancy co-running lanes keep §7 semantics), each
task's knob-scaled kernel time additionally flows through a per-device fluid
compute resource in the engine:

- demand: task compute-phase kernel work ``W`` (kernel shares × spans ×
  per-class multipliers) at natural rate ``W / D`` (D = the phase's scaled
  duration) — at baseline this is exactly the traced achieved kernel rate;
- capacity: ``max(1.0, baseline device-busy fraction)`` kernel-ns per
  wall-ns, derived from the BASELINE trace + profile only (one GPU serializes
  kernel time at ~1 s/s; a busy fraction > 1 is measured multi-stream
  overlap). No degraded-run data is used.

Uncontended, every phase runs at its natural duration (the §7 behavior, which
is what the old host-dominated lane needs); on a device-saturated lane the
aggregate demand exceeds capacity and queue-wait EMERGES, re-deriving the
span "host" time that is really device wait (the §8 failure: both paths
under-predicted MPS throttling by up to −57% because that wait was held
invariant).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..engine import ChannelKey, Engine, SimResult
from ..knobs import Knobs
from ..model import QueryGraph
from . import laws, sanity
from .curves import BandwidthCurve
from .join import JoinStats, TaskAnnotation, join_graph
from .schema import OpPhysics, PhysicsProfile, PrepPhysics

# Below this memcpy share of the Preparing window, the phase is treated as
# non-transfer work (no fluid-channel service; avoids near-zero-duration
# transfers with unbounded demand rates).
MIN_XFER_FRAC = 0.005

PREP_PSEUDO_OP = "PHYS::PREP"

# Baseline device-busy fraction above which a lane is considered
# device-saturated: gpu_compute predictions from a model that cannot
# represent device contention are lower bounds there (validation-results.md
# §8.6 item 3; the L-lane measured 82-100% busy with -41/-47% median
# under-prediction before G4b).
DEVICE_SATURATION_WARN = 0.7

# Minimum measured kernel-serialization fraction (device-timeline union/sum
# of kernel time in the matched capture window) for the fluid device model
# to engage. The capacity model's premise is that kernels fill the machine
# and serialize, so aggregate throughput scales with the SM fraction; the
# L-lane measures 0.91-0.99 (valid), the NVMe lane 0.60-0.86 (low-occupancy
# kernels co-run: neither the 1/f per-kernel stretch nor the f-scaled
# capacity holds, and imposing them measured +56/+96% errors at f=0.25).
# Between the two measured populations, with headroom for the L-lane's ~10%
# stream-overlap slack.
DEVICE_SERIAL_MIN = 0.9


@dataclass
class RetimeStats:
    tasks_total: int = 0
    tasks_annotated: int = 0
    ops_annotated: int = 0
    conflated_op_ns: float = 0.0  # traced op time re-timed with v0 rule
    conflated_prep_ns: float = 0.0
    # traced-span-weighted class totals over annotated spans (baseline shares)
    class_ns: Dict[str, float] = field(
        default_factory=lambda: {
            "compute": 0.0,
            "membw": 0.0,
            "unknown": 0.0,
            "xfer": 0.0,
            "host": 0.0,
        }
    )
    effective_multipliers: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # G4b device contention model (per GPU device id)
    device_model_active: bool = False
    device_busy_frac: Dict[int, float] = field(default_factory=dict)
    device_capacity: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _op_inv_factor(op: OpPhysics, knobs: Knobs) -> float:
    """scaled_duration / traced_span for one annotated operator span."""
    k_c = knobs.gpu_compute
    k_m = laws.effective_membw_mult(knobs)
    k_u = laws.conflated_mult(knobs)
    k_h2d = laws.transfer_mult("h2d", knobs)
    k_d2h = laws.transfer_mult("d2h", knobs)
    k_d2d = laws.transfer_mult("d2d", knobs)
    k_host = laws.host_mult(knobs)
    return (
        op.f_comp / k_c
        + op.f_membw / k_m
        + op.f_unknown / k_u
        + op.f_h2d / k_h2d
        + op.f_d2h / k_d2h
        + op.f_d2d / k_d2d
        + op.f_host / k_host
    )


def _prep_nonxfer_factor(prep: PrepPhysics, knobs: Knobs) -> float:
    """scaled/traced factor for the non-transfer share of a Preparing span
    (kernel time — decompress is SM-bound — plus host glue)."""
    return (
        prep.f_comp / knobs.gpu_compute
        + prep.f_membw / laws.effective_membw_mult(knobs)
        + prep.f_unknown / laws.conflated_mult(knobs)
        + prep.f_host / laws.host_mult(knobs)
    )


def _curve_factor(
    prep: PrepPhysics, curves: Dict[str, BandwidthCurve], mult: float
) -> float:
    """time(mult)/time(1) for this task's transfer, per-size curve if known."""
    if mult == 1.0:
        return 1.0
    curve = curves.get(prep.dominant_channel)
    if curve is not None and prep.copies:
        return curve.scale_factor(prep.copies, mult)
    return 1.0 / mult


def _op_kernel_frac(op: OpPhysics) -> float:
    """Kernel share of one operator span (device work, not memcpy/host)."""
    return op.f_comp + op.f_membw + op.f_unknown


def _op_kernel_inv_factor(op: OpPhysics, knobs: Knobs) -> float:
    """scaled_kernel_time / traced_span for one operator span."""
    return (
        op.f_comp / knobs.gpu_compute
        + op.f_membw / laws.effective_membw_mult(knobs)
        + op.f_unknown / laws.conflated_mult(knobs)
    )


def _prep_kernel_frac(prep: PrepPhysics) -> float:
    return prep.f_comp + prep.f_membw + prep.f_unknown


def _prep_kernel_inv_factor(prep: PrepPhysics, knobs: Knobs) -> float:
    return (
        prep.f_comp / knobs.gpu_compute
        + prep.f_membw / laws.effective_membw_mult(knobs)
        + prep.f_unknown / laws.conflated_mult(knobs)
    )


def _device_model_requested(knobs: Knobs) -> bool:
    """The G4b device resource engages when a GPU kernel knob moves. At
    knobs=1 it is a guaranteed no-op (demands equal traced achieved rates,
    which never exceed the derived capacity), so gating it keeps baseline
    runs byte-identical to the pre-G4b code path."""
    return knobs.gpu_compute != 1.0 or knobs.gpu_mem_bandwidth is not None


def retime_graph(
    graph: QueryGraph,
    annotations: Dict[int, TaskAnnotation],
    knobs: Knobs,
    curves: Dict[str, BandwidthCurve],
    serial_frac: Optional[float] = None,
) -> Tuple[QueryGraph, RetimeStats]:
    stats = RetimeStats()
    stats.effective_multipliers = {
        "gpu_compute": knobs.gpu_compute,
        "gpu_mem_bandwidth_effective": laws.effective_membw_mult(knobs),
        "conflated_v0": laws.conflated_mult(knobs),
        "transfer_h2d": laws.transfer_mult("h2d", knobs),
        "transfer_d2d": laws.transfer_mult("d2d", knobs),
        "host": laws.host_mult(knobs),
    }
    device_requested = _device_model_requested(knobs)
    serial_ok = serial_frac is None or serial_frac >= DEVICE_SERIAL_MIN
    device_on = device_requested and serial_ok
    stats.device_model_active = device_on
    if device_requested and serial_frac is None:
        stats.warnings.append(
            "G4b: the physics profile carries no kernel-serialization "
            "diagnostic (pre-G4b ingest) -- assuming serialized kernels and "
            "engaging the fluid device model; re-run ingest-nsys to verify "
            "the capacity premise on this lane."
        )
    dev_base_kernel: Dict[int, float] = {}  # device -> baseline kernel ns
    g = copy.deepcopy(graph)
    for tid, task in g.tasks.items():
        stats.tasks_total += 1
        ann = annotations.get(tid)
        new_ops: List[Tuple[str, int, int, int]] = []
        prepend: List[Tuple[str, int, int, int]] = []
        base_kernel_ns = 0.0  # traced kernel time of this task's compute phase
        scaled_kernel_ns = 0.0  # same, after the per-class knob multipliers

        # ---- Preparing phase ------------------------------------------
        prep_ann = ann.prep if ann is not None else None
        if prep_ann is not None and task.prep_ns > 0:
            base_kernel_ns += _prep_kernel_frac(prep_ann) * task.prep_ns
            scaled_kernel_ns += _prep_kernel_inv_factor(prep_ann, knobs) * task.prep_ns
            for cls in ("compute", "membw", "unknown", "host"):
                stats.class_ns[cls] += (
                    getattr(prep_ann, f"f_{cls}" if cls != "compute" else "f_comp")
                    * task.prep_ns
                )
            stats.class_ns["xfer"] += prep_ann.f_xfer * task.prep_ns
            nonxfer_scaled = task.prep_ns * _prep_nonxfer_factor(prep_ann, knobs)
            is_link = task.is_transfer_prep
            m_chan = laws.channel_transfer_mult(
                task.prep_origin, task.prep_target, knobs
            )
            if is_link and prep_ann.f_xfer >= MIN_XFER_FRAC and prep_ann.xfer_bytes > 0:
                xfer_base = task.prep_ns * prep_ann.f_xfer
                xfer_scaled = xfer_base * _curve_factor(prep_ann, curves, m_chan)
                task.prep_ns = max(1, round(xfer_scaled))
            else:
                # transfer share negligible (or same-tier prep): fold the
                # (linearly scaled) transfer share in and skip the channel.
                xfer_scaled = (
                    task.prep_ns
                    * prep_ann.f_xfer
                    * _curve_factor(prep_ann, curves, m_chan)
                )
                nonxfer_scaled += xfer_scaled
                task.prep_origin = task.prep_target  # disable channel service
                task.prep_ns = 0
            if nonxfer_scaled >= 1:
                prepend.append((PREP_PSEUDO_OP, -1, round(nonxfer_scaled), 0))
        elif task.prep_ns > 0:
            # unannotated: v0 behavior (whole span is link time), with the
            # physics-mode co-limited multiplier instead of pure c2c.
            stats.conflated_prep_ns += task.prep_ns
            if task.is_transfer_prep:
                m_chan = laws.channel_transfer_mult(
                    task.prep_origin, task.prep_target, knobs
                )
                task.prep_ns = max(1, round(task.prep_ns / m_chan))
            # same-tier prep: v0 replays it unchanged — keep that.

        # ---- Computing spans ------------------------------------------
        for j, (name, op_id, dur, in_bytes) in enumerate(task.ops):
            op_ann = ann.ops[j] if ann is not None and j < len(ann.ops) else None
            if op_ann is not None:
                stats.ops_annotated += 1
                stats.class_ns["compute"] += op_ann.f_comp * dur
                stats.class_ns["membw"] += op_ann.f_membw * dur
                stats.class_ns["unknown"] += op_ann.f_unknown * dur
                stats.class_ns["xfer"] += (
                    op_ann.f_h2d + op_ann.f_d2h + op_ann.f_d2d
                ) * dur
                stats.class_ns["host"] += op_ann.f_host * dur
                scaled = dur * _op_inv_factor(op_ann, knobs)
                base_kernel_ns += _op_kernel_frac(op_ann) * dur
                scaled_kernel_ns += _op_kernel_inv_factor(op_ann, knobs) * dur
            else:
                stats.conflated_op_ns += dur
                scaled = dur / laws.conflated_mult(knobs)
            new_ops.append((name, op_id, max(0, round(scaled)), in_bytes))
        task.ops = prepend + new_ops
        if ann is not None:
            stats.tasks_annotated += 1
        if base_kernel_ns > 0.0:
            dev_base_kernel[task.device] = (
                dev_base_kernel.get(task.device, 0.0) + base_kernel_ns
            )
            if device_on:
                # Kernel work of the compute phase, in knob-scaled ns; the
                # engine serves it through the fluid device resource.
                task.dev_work_ns = scaled_kernel_ns

    conflated = stats.conflated_op_ns + stats.conflated_prep_ns
    total = conflated + sum(stats.class_ns.values())
    if total > 0 and conflated > 0:
        stats.warnings.append(
            f"{100.0 * conflated / total:.1f}% of traced busy time "
            f"({conflated / 1e6:.1f} ms) had no physics annotation and was "
            "re-timed with the v0 conflated rule."
        )

    # ---- G4b device contention model: capacity from BASELINE observables --
    wall = float(graph.traced_exec_wall_ns)
    if wall > 0:
        for dev, kern in sorted(dev_base_kernel.items()):
            busy = kern / wall
            stats.device_busy_frac[dev] = busy
            if device_on:
                # One GPU serializes kernel time at ~1 kernel-ns per wall-ns;
                # a measured busy fraction > 1 is real multi-stream overlap
                # and raises the demonstrated capacity. Derived from the
                # baseline capture only -- never fit to degraded runs.
                stats.device_capacity[dev] = max(1.0, busy)
    if device_on:
        for dev, busy in sorted(stats.device_busy_frac.items()):
            if busy > DEVICE_SATURATION_WARN:
                stats.warnings.append(
                    f"gpu{dev}: baseline device-busy {100.0 * busy:.0f}% of "
                    "wall (device-saturated lane) -- G4b fluid device model "
                    f"ACTIVE, capacity {stats.device_capacity[dev]:.3f} "
                    "kernel-ns/ns; emergent queue-wait replaces the invariant"
                    "-host assumption. Partition co-residency under deep SM "
                    "throttling is not modeled, so predictions are mildly "
                    "pessimistic (measured +5-15% at 25% SM)."
                )
        if total > 0 and conflated > total * 0.5:
            stats.warnings.append(
                "G4b device model is blind on the conflated (unannotated) "
                f"share ({100.0 * conflated / total:.1f}% of busy time): that "
                "time carries no kernel demand, so device contention is "
                "under-represented; on saturated lanes predictions are "
                "lower bounds."
            )
    elif device_requested and not serial_ok:
        busy = max(stats.device_busy_frac.values(), default=0.0)
        msg = (
            f"G4b device model DISENGAGED: measured kernel serialization "
            f"{serial_frac:.2f} < {DEVICE_SERIAL_MIN} -- this lane's kernels "
            "co-run (low occupancy), so the f-scaled capacity premise does "
            "not hold; spans keep the section-7 split scaling."
        )
        if busy > DEVICE_SATURATION_WARN:
            msg += (
                f" Baseline device-busy is {100.0 * busy:.0f}% of wall: "
                "device contention cannot be represented on this lane and "
                "gpu_compute predictions are LOWER BOUNDS."
            )
        stats.warnings.append(msg)
    elif (
        not knobs.is_baseline()
        and stats.device_busy_frac
        and max(stats.device_busy_frac.values()) > DEVICE_SATURATION_WARN
    ):
        stats.warnings.append(
            "baseline device-busy exceeds "
            f"{100 * DEVICE_SATURATION_WARN:.0f}% of wall but no GPU kernel "
            "knob moved: any knob that changes span overlap on this "
            "device-saturated lane may contend for the device in ways this "
            "run does not model."
        )
    return g, stats


def physics_channel_capacity(
    graph: QueryGraph,
    annotations: Dict[int, TaskAnnotation],
    knobs: Knobs,
    session_peak: Optional[Dict[ChannelKey, float]] = None,
    corrected: Optional[Dict[ChannelKey, float]] = None,
) -> Dict[ChannelKey, float]:
    """Channel capacity: line-sweep peak aggregate rate over *transfer-only*
    sub-windows of the traced Preparing spans (annotated tasks contribute
    their memcpy share of the window at the correspondingly higher rate),
    floored by the session-wide v0 peak, capped by the nsys wire-side
    corrected capacity where available (``sanity.corrected_link_capacities``
    — the sub-window line-sweep overlap-inflates the peak; measured 3,433
    vs 382 GB/s wire on the host-pinned lane), then scaled by the transfer
    knob."""
    events: Dict[ChannelKey, List[Tuple[float, float]]] = {}
    for task in graph.tasks.values():
        if not (task.is_transfer_prep and task.prep_ns > 1000 and task.prep_bytes > 0):
            continue
        key = (task.prep_origin, task.prep_target, task.device)
        ann = annotations.get(task.tid)
        frac = 1.0
        if ann is not None and ann.prep is not None and ann.prep.f_xfer > 0:
            frac = max(MIN_XFER_FRAC, min(1.0, ann.prep.f_xfer))
        dur = task.prep_ns * frac
        rate = task.prep_bytes / dur
        t0 = float(task.t_preparing)
        events.setdefault(key, []).append((t0, +rate))
        events.setdefault(key, []).append((t0 + dur, -rate))
    caps: Dict[ChannelKey, float] = {}
    for key, evs in events.items():
        evs.sort()
        cur = peak = 0.0
        for _, d in evs:
            cur += d
            peak = max(peak, cur)
        if session_peak and key in session_peak:
            peak = max(peak, session_peak[key])
        m = laws.channel_transfer_mult(key[0], key[1], knobs)
        # The nsys wire-side correction applies only when the link knob
        # moves: at m == 1 the traced spans already contain the real wire
        # serialization (identity by construction), while the line-sweep
        # peak is overlap-inflated (measured 3,433 vs 382 GB/s wire) and
        # would let a degraded link go unpriced.
        if m != 1.0 and corrected and key in corrected:
            peak = min(peak, corrected[key])
        caps[key] = peak * m
    if session_peak:
        for key, base in session_peak.items():
            if key not in caps:
                m = laws.channel_transfer_mult(key[0], key[1], knobs)
                if m != 1.0 and corrected and key in corrected:
                    base = min(base, corrected[key])
                caps[key] = base * m
    return caps


def _engine_knobs(knobs: Knobs) -> Knobs:
    """Neutralize everything the physics layer already applied; keep the
    engine-level semantics of pool capacity and the G1 io knob."""
    return Knobs(
        gpu_mem_capacity=knobs.gpu_mem_capacity,
        io_bandwidth=knobs.io_bandwidth,
    )


def simulate_with_physics(
    model,
    graph: QueryGraph,
    knobs: Knobs,
    profile: PhysicsProfile,
    queue_order: str = "traced",
    return_graph: bool = False,
):
    """Run the physics-retimed simulation.

    Returns ``(result, jstats, rstats)``; with ``return_graph=True`` also
    returns the retimed graph the engine actually executed as a 4th element —
    span durations there are the knob-scaled ones (the engine ran with
    neutralized GPU knobs), which is what the quent exporter needs to lay out
    per-operator boundaries of a physics run.
    """
    annotations, jstats = join_graph(profile, graph)
    g2, rstats = retime_graph(
        graph,
        annotations,
        knobs,
        profile.curves,
        serial_frac=jstats.kernel_serial_frac,
    )
    rstats.warnings = jstats.warnings() + rstats.warnings
    pool = {
        ms.device_id: ms.capacity_bytes
        for ms in model.memory_spaces.values()
        if ms.tier == "GPU"
    }
    corrected = sanity.corrected_link_capacities(graph, annotations, profile)
    caps = physics_channel_capacity(
        graph,
        annotations,
        knobs,
        session_peak=dict(model.channel_peak_rate),
        corrected=corrected,
    )
    if knobs.c2c_bandwidth != 1.0 or knobs.cpu_mem_bandwidth != 1.0:
        rstats.warnings += sanity.channel_capacity_warnings(caps)
    host = getattr(model, "host_pool_capacity", 0)
    result = Engine(
        g2,
        _engine_knobs(knobs),
        n_threads=model.n_executor_threads,
        pool_capacity=pool,
        channel_capacity=caps,
        queue_order=queue_order,
        host_capacity=host if host else None,
        device_capacity=rstats.device_capacity or None,
    ).run()
    if return_graph:
        return result, jstats, rstats, g2
    return result, jstats, rstats


def physics_knob_warnings(knobs: Knobs, jstats: JoinStats) -> List[str]:
    """Warnings under split semantics (replaces the v0 conflation warnings
    for the physics path; G1/spill caveats still hold)."""
    w: List[str] = []
    if knobs.io_bandwidth != 1.0:
        w.append(
            "io_bandwidth: still gap G1 — no I/O events; scales whole "
            "GPU_SCAN spans (including their GPU work) at the engine level."
        )
    if knobs.gpu_mem_capacity != 1.0:
        w.append(
            "gpu_mem_capacity: engine-level calibrated downgrade model "
            "(docs/spill-model.md) — sub-knee predictions are "
            "order-of-magnitude with ~±40% bands."
        )
    if knobs.gpu_compute != 1.0 or knobs.gpu_mem_bandwidth is not None:
        w.append(
            "split semantics: gpu_compute scales only compute-classified "
            "kernel-busy time; gpu_mem_bandwidth scales membw-classified "
            "time, coupled as min(gpu_mem_bandwidth, gpu_compute x "
            f"{laws.SM_BW_HEADROOM}) per the measured SM-issue cap "
            "(compute-throttle.md). Unclassified/unmatched time uses the v0 "
            "conflated rule. Scaled kernel time also flows through the G4b "
            "fluid device resource, so queue-wait on a saturated device "
            "emerges instead of holding span host time invariant."
        )
    if knobs.c2c_bandwidth != 1.0 or knobs.cpu_mem_bandwidth != 1.0:
        w.append(
            "transfers: effective link multiplier is min(c2c_bandwidth, "
            "cpu_mem_bandwidth) — on Grace, C2C H2D reads host DRAM at line "
            "rate (membw-throttle.md section 5); per-size alpha+beta curve "
            "keeps small copies from speeding up with the link."
        )
    if knobs.cpu_compute != 1.0:
        w.append(
            "cpu_compute: scales only host-side (non-GPU-busy) shares of "
            "annotated spans; unannotated spans are unaffected."
        )
    return w

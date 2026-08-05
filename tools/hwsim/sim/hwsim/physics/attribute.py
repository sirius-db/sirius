"""NVTX-based attribution + per-task physics decomposition.

The attribution chain (nsys-extraction.md section 3.1):

    CUPTI kernel/memcpy row
      --correlationId-->  RUNTIME launch row (globalTid + launch timestamp)
      --same-thread interval containment-->  `Pipeline P Task T [...]` range
      --innermost containing op range-->     operator id

Attribution is by **launch time on the launching thread**; the *execution*
interval is the GPU-resource demand. A task that re-executes (downgrade)
appears as multiple ranges with the same task_id — each becomes a separate
attempt (nsys-extraction.md section 3.1 caveat).

Decomposition per window (op invocation or prep phase), all clipped to the
window: kernel-busy = union of kernel execution intervals (multi-stream
overlap counted once), split across classes proportional to per-kernel
clipped durations; memcpy-busy = memcpy union minus kernel union, split per
direction; host = remainder. Kernels with no enclosing task range are counted
and reported — never dropped silently.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import intervals as iv
from .classify import CLASS_COMPUTE, CLASS_MEMBW, CLASS_MIXED, Classifier
from .reader import (
    HostSpanRow,
    KernelRow,
    MemcpyRow,
    NsysData,
    OpRangeRow,
    TaskRangeRow,
)
from .schema import (
    MAX_COPIES_PER_TASK,
    OpPhysics,
    PrepPhysics,
    QueryPhysics,
    TaskPhysics,
)


# --------------------------------------------------------------------------
# Interval containment index (per launching thread)
# --------------------------------------------------------------------------


class ContainmentIndex:
    """Innermost-containing-range lookup among possibly-nested ranges on one
    thread. Uses the prefix-max-end trick so lookups are O(log n + depth)."""

    def __init__(self, ranges: List) -> None:  # items expose .start/.end
        self.ranges = sorted(ranges, key=lambda r: (r.start, -r.end))
        self.starts = [r.start for r in self.ranges]
        self.prefix_max_end: List[int] = []
        m = float("-inf")
        for r in self.ranges:
            m = max(m, r.end)
            self.prefix_max_end.append(m)

    def find(self, t: int):
        """Innermost range with start <= t < end, or None."""
        i = bisect_right(self.starts, t) - 1
        while i >= 0 and self.prefix_max_end[i] > t:
            r = self.ranges[i]
            if r.start <= t < r.end:
                return r
            i -= 1
        return None


def _by_thread(rows) -> Dict[int, List]:
    out: Dict[int, List] = {}
    for r in rows:
        out.setdefault(r.global_tid, []).append(r)
    return out


# --------------------------------------------------------------------------
# Attribution result containers
# --------------------------------------------------------------------------


@dataclass
class AttributionStats:
    kernels_total: int = 0
    kernels_attributed: int = 0
    kernel_ns_total: float = 0.0
    kernel_ns_attributed: float = 0.0
    kernels_no_runtime: int = 0
    memcpys_total: int = 0
    memcpys_attributed: int = 0
    memcpy_ns_total: float = 0.0
    memcpy_ns_attributed: float = 0.0
    kernel_ns_outside_op_window: float = 0.0
    unattributed_kernel_names: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["pct_kernels_attributed"] = (
            100.0 * self.kernels_attributed / self.kernels_total
            if self.kernels_total
            else 0.0
        )
        d["pct_kernel_ns_attributed"] = (
            100.0 * self.kernel_ns_attributed / self.kernel_ns_total
            if self.kernel_ns_total
            else 0.0
        )
        d["pct_memcpy_ns_attributed"] = (
            100.0 * self.memcpy_ns_attributed / self.memcpy_ns_total
            if self.memcpy_ns_total
            else 0.0
        )
        top = sorted(
            self.unattributed_kernel_names.items(), key=lambda kv: -kv[1]
        )[:10]
        d["unattributed_kernel_names"] = {k: v for k, v in top}
        return d


@dataclass
class _TaskAttempt:
    rng: TaskRangeRow
    attempt: int
    ops: List[OpRangeRow] = field(default_factory=list)
    kernels_by_op: Dict[int, List[KernelRow]] = field(default_factory=dict)
    memcpys_by_op: Dict[int, List[MemcpyRow]] = field(default_factory=dict)
    prep_kernels: List[KernelRow] = field(default_factory=list)
    prep_memcpys: List[MemcpyRow] = field(default_factory=list)


# --------------------------------------------------------------------------
# Main attribution + decomposition
# --------------------------------------------------------------------------


def _class_split(
    kernels: List[KernelRow], lo: int, hi: int, classifier: Classifier
) -> Tuple[float, Dict[str, float], float]:
    """Returns (union_ns, per-class ns shares of the union, out_of_window_ns).

    The union counts multi-stream overlap once; per-class shares distribute it
    proportionally to per-kernel clipped durations (mixed splits 50/50)."""
    clipped = iv.clip([(k.start, k.end) for k in kernels], lo, hi)
    union = iv.total(iv.merge(clipped))
    out_of_window = sum(
        (k.end - k.start) for k in kernels
    ) - sum(e - s for s, e in clipped)
    sums = {CLASS_COMPUTE: 0.0, CLASS_MEMBW: 0.0, "unknown": 0.0}
    for k in kernels:
        d = min(k.end, hi) - max(k.start, lo)
        if d <= 0:
            continue
        cls = classifier.classify(k.name)
        if cls == CLASS_MIXED:
            sums[CLASS_COMPUTE] += d / 2
            sums[CLASS_MEMBW] += d / 2
        elif cls in (CLASS_COMPUTE, CLASS_MEMBW):
            sums[cls] += d
        else:
            sums["unknown"] += d
    total = sum(sums.values())
    if total > 0:
        shares = {c: union * v / total for c, v in sums.items()}
    else:
        shares = {c: 0.0 for c in sums}
    return union, shares, max(0.0, out_of_window)


def _memcpy_split(
    memcpys: List[MemcpyRow], lo: int, hi: int, kernel_union: List[Tuple[float, float]]
) -> Tuple[float, Dict[str, float]]:
    """Memcpy-only busy time (memcpy union minus kernel union), split by
    direction proportional to per-copy clipped durations."""
    clipped = iv.clip([(m.start, m.end) for m in memcpys], lo, hi)
    mc_union = iv.merge(clipped)
    only = iv.total(iv.subtract(mc_union, kernel_union))
    sums = {"h2d": 0.0, "d2h": 0.0, "d2d": 0.0}
    for m in memcpys:
        d = min(m.end, hi) - max(m.start, lo)
        if d <= 0:
            continue
        if m.direction.startswith("Host-to-Device"):
            sums["h2d"] += d
        elif m.direction.startswith("Device-to-Host"):
            sums["d2h"] += d
        else:
            sums["d2d"] += d
    total = sum(sums.values())
    if total > 0:
        shares = {c: only * v / total for c, v in sums.items()}
    else:
        shares = {c: 0.0 for c in sums}
    return only, shares


def _host_ns(spans: List[HostSpanRow], gtid: int, lo: int, hi: int, kind: str) -> float:
    return sum(
        min(s.end, hi) - max(s.start, lo)
        for s in spans
        if s.kind == kind
        and s.global_tid == gtid
        and min(s.end, hi) > max(s.start, lo)
    )


def attribute_and_decompose(
    data: NsysData, classifier: Optional[Classifier] = None
) -> Tuple[List[QueryPhysics], AttributionStats]:
    classifier = classifier or Classifier()
    stats = AttributionStats()

    # Tier B calibration when metrics exist (pct-of-peak semantics; see
    # classify.py note — verify on first real Tier B capture).
    dram = [(t, v) for (t, name, v) in data.gpu_metrics]
    if dram:
        classifier.calibrate_from_metrics(data.kernels, dram)

    task_idx = {g: ContainmentIndex(rs) for g, rs in _by_thread(data.task_ranges).items()}
    op_idx = {g: ContainmentIndex(rs) for g, rs in _by_thread(data.op_ranges).items()}

    # task attempts: same task_id may re-execute (downgrade) — key by range.
    attempts: Dict[Tuple[int, int, int], _TaskAttempt] = {}
    per_tid_count: Dict[int, int] = {}
    for rng in sorted(data.task_ranges, key=lambda r: r.start):
        n = per_tid_count.get(rng.task_id, 0)
        per_tid_count[rng.task_id] = n + 1
        attempts[(rng.task_id, rng.start, rng.global_tid)] = _TaskAttempt(rng, n)

    def _find_attempt(gtid: int, t: int) -> Optional[_TaskAttempt]:
        idx = task_idx.get(gtid)
        if idx is None:
            return None
        rng = idx.find(t)
        if rng is None:
            return None
        return attempts.get((rng.task_id, rng.start, rng.global_tid))

    # assign op ranges to their task attempt
    for op in sorted(data.op_ranges, key=lambda r: r.start):
        att = _find_attempt(op.global_tid, op.start)
        if att is not None and att.rng.pipeline_id == op.pipeline_id:
            att.ops.append(op)

    # attribute kernels / memcpys by launch time
    for k in data.kernels:
        stats.kernels_total += 1
        stats.kernel_ns_total += k.dur_ns
        if k.launch_start < 0 or k.global_tid < 0:
            stats.kernels_no_runtime += 1
            stats.unattributed_kernel_names[k.name] = (
                stats.unattributed_kernel_names.get(k.name, 0.0) + k.dur_ns
            )
            continue
        att = _find_attempt(k.global_tid, k.launch_start)
        if att is None:
            stats.unattributed_kernel_names[k.name] = (
                stats.unattributed_kernel_names.get(k.name, 0.0) + k.dur_ns
            )
            continue
        stats.kernels_attributed += 1
        stats.kernel_ns_attributed += k.dur_ns
        oi = op_idx.get(k.global_tid)
        op = oi.find(k.launch_start) if oi else None
        if op is not None and op.start >= att.rng.start:
            att.kernels_by_op.setdefault(id(op), []).append(k)
        else:
            att.prep_kernels.append(k)

    for m in data.memcpys:
        stats.memcpys_total += 1
        stats.memcpy_ns_total += m.dur_ns
        if m.launch_start < 0 or m.global_tid < 0:
            continue
        att = _find_attempt(m.global_tid, m.launch_start)
        if att is None:
            continue
        stats.memcpys_attributed += 1
        stats.memcpy_ns_attributed += m.dur_ns
        oi = op_idx.get(m.global_tid)
        op = oi.find(m.launch_start) if oi else None
        if op is not None and op.start >= att.rng.start:
            att.memcpys_by_op.setdefault(id(op), []).append(m)
        else:
            att.prep_memcpys.append(m)

    # ---- decompose each attempt into TaskPhysics --------------------------
    windows = data.query_windows or [
        (
            min((r.start for r in data.task_ranges), default=0),
            max((r.end for r in data.task_ranges), default=0) + 1,
        )
    ]
    queries = [QueryPhysics(window=(float(s), float(e))) for s, e in windows]

    def _query_for(t: int) -> Optional[QueryPhysics]:
        for q in queries:
            if q.window[0] <= t < q.window[1]:
                return q
        return None

    for att in sorted(attempts.values(), key=lambda a: a.rng.start):
        rng = att.rng
        qp = _query_for(rng.start)
        if qp is None:
            continue
        tp = TaskPhysics(
            pipeline_id=rng.pipeline_id,
            nsys_task_id=rng.task_id,
            attempt=att.attempt,
            start_ns=float(rng.start),
            end_ns=float(rng.end),
        )
        # prep window: task start -> first op start (whole range when no ops)
        prep_end = att.ops[0].start if att.ops else rng.end
        tp.prep = _decompose_prep(att, rng.start, prep_end, classifier, stats)
        for op in att.ops:
            tp.ops.append(
                _decompose_op(
                    op,
                    att.kernels_by_op.get(id(op), []),
                    att.memcpys_by_op.get(id(op), []),
                    data.host_spans,
                    classifier,
                    stats,
                )
            )
        qp.pipelines.setdefault(rng.pipeline_id, []).append(tp)

    return queries, stats


def _decompose_prep(
    att: _TaskAttempt, lo: int, hi: int, classifier: Classifier, stats: AttributionStats
) -> Optional[PrepPhysics]:
    span = hi - lo
    if span <= 0:
        return None
    union, shares, out_w = _class_split(att.prep_kernels, lo, hi, classifier)
    stats.kernel_ns_outside_op_window += out_w
    k_union_merged = iv.merge(
        iv.clip([(k.start, k.end) for k in att.prep_kernels], lo, hi)
    )
    mc_only, _dir_shares = _memcpy_split(att.prep_memcpys, lo, hi, k_union_merged)
    host = max(0.0, span - union - mc_only)
    by_chan: Dict[str, float] = {}
    copies: List[Tuple[int, float]] = []
    xfer_bytes = 0
    for m in sorted(att.prep_memcpys, key=lambda m: -(m.end - m.start)):
        by_chan[m.channel] = by_chan.get(m.channel, 0.0) + m.dur_ns
        xfer_bytes += m.bytes
        if len(copies) < MAX_COPIES_PER_TASK:
            copies.append((m.bytes, float(m.dur_ns)))
    dominant = max(by_chan, key=by_chan.get) if by_chan else ""
    return PrepPhysics(
        span_ns=float(span),
        f_xfer=mc_only / span,
        f_comp=shares[CLASS_COMPUTE] / span,
        f_membw=shares[CLASS_MEMBW] / span,
        f_unknown=shares["unknown"] / span,
        f_host=host / span,
        xfer_bytes=xfer_bytes,
        dominant_channel=dominant,
        copies=copies,
    )


def _decompose_op(
    op: OpRangeRow,
    kernels: List[KernelRow],
    memcpys: List[MemcpyRow],
    host_spans: List[HostSpanRow],
    classifier: Classifier,
    stats: AttributionStats,
) -> OpPhysics:
    lo, hi = op.start, op.end
    span = max(1, hi - lo)
    union, shares, out_w = _class_split(kernels, lo, hi, classifier)
    stats.kernel_ns_outside_op_window += out_w
    k_union_merged = iv.merge(iv.clip([(k.start, k.end) for k in kernels], lo, hi))
    mc_only, dir_shares = _memcpy_split(memcpys, lo, hi, k_union_merged)
    host = max(0.0, span - union - mc_only)
    return OpPhysics(
        op_id=op.op_id,
        op_name=op.op_name + (" sink" if op.is_sink else ""),
        span_ns=float(span),
        f_comp=shares[CLASS_COMPUTE] / span,
        f_membw=shares[CLASS_MEMBW] / span,
        f_unknown=shares["unknown"] / span,
        f_h2d=dir_shares["h2d"] / span,
        f_d2h=dir_shares["d2h"] / span,
        f_d2d=dir_shares["d2d"] / span,
        f_host=host / span,
        kernel_ns=union,
        memcpy_ns=mc_only,
        launch_ns=_host_ns(host_spans, op.global_tid, lo, hi, "launch"),
        sync_ns=_host_ns(host_spans, op.global_tid, lo, hi, "sync"),
    )

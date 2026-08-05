"""Ingest orchestrator: nsys sqlite (+ optional paired Quent trace) -> PhysicsProfile.

Steps:
1. read the sqlite defensively (reader.py),
2. attribute kernels/memcpys to (pipeline, task, operator) via NVTX + decompose
   each task attempt into physics fractions (attribute.py),
3. fit per-size transfer bandwidth curves from ALL memcpys in the capture
   (curves.py) and line-sweep the peak aggregate concurrent rate per channel,
4. if a Quent trace is supplied: structurally match each nsys query window to
   a trace query (pipeline-id/task-count signature) and, when the sqlite and
   the trace came from the *same run*, do the clock-domain alignment fit
   (clock.py) — both purely diagnostic,
5. bundle everything with match-rate diagnostics into a PhysicsProfile.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

from .attribute import attribute_and_decompose
from .classify import Classifier
from .clock import fit_linear
from .curves import fit_curves
from .reader import read_nsys
from .schema import PhysicsProfile, QueryPhysics

# Same-run detection: first-order residual (quent_ts - (nsys_ts + utcEpochNs))
# must be under this for the capture to be considered the same run as the trace.
SAME_RUN_TOLERANCE_NS = 1_000_000_000  # 1 s


def load_overrides(path: Optional[str]) -> Dict[str, str]:
    """Kernel-classification override table: JSON {kernel_name: class}."""
    if not path:
        return {}
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: override table must be a JSON object")
    return {str(k): str(v) for k, v in d.items()}


def _channel_peaks(memcpys) -> Dict[str, float]:
    """Peak aggregate concurrent achieved rate per channel (GB/s), from a
    line sweep over overlapping memcpy intervals — the measured lower bound
    on each link's capacity (diagnostic; integrate.py keeps quent-byte units)."""
    events: Dict[str, List[Tuple[int, float]]] = {}
    for m in memcpys:
        if m.dur_ns <= 0 or m.bytes <= 0:
            continue
        rate = m.bytes / m.dur_ns
        events.setdefault(m.channel, []).append((m.start, +rate))
        events.setdefault(m.channel, []).append((m.end, -rate))
    peaks: Dict[str, float] = {}
    for chan, evs in events.items():
        evs.sort()
        cur = peak = 0.0
        for _, d in evs:
            cur += d
            peak = max(peak, cur)
        peaks[chan] = peak
    return peaks


def _window_kernel_occupancy(kernels, w0: float, w1: float) -> Tuple[float, float]:
    """(sum, union) of kernel durations clipped to [w0, w1), per device then
    summed — the G4b serialization diagnostic. sum/union ~= 1 means the
    window's kernels serialize on the device (full-machine kernels: the fluid
    device-capacity model is valid); sum >> union means low-occupancy kernels
    co-ran and the model must stand down."""
    by_dev: Dict[int, List[Tuple[float, float]]] = {}
    for k in kernels:
        if k.end <= w0 or k.start >= w1:
            continue
        s, e = max(float(k.start), w0), min(float(k.end), w1)
        if e > s:
            by_dev.setdefault(k.device, []).append((s, e))
    total_sum = total_union = 0.0
    for ivals in by_dev.values():
        ivals.sort()
        cs, ce = ivals[0]
        for s, e in ivals:
            total_sum += e - s
            if s <= ce:
                ce = max(ce, e)
            else:
                total_union += ce - cs
                cs, ce = s, e
        total_union += ce - cs
    return total_sum, total_union


def _pipeline_signature(qp: QueryPhysics) -> Dict[int, int]:
    return {pid: len(tasks) for pid, tasks in qp.pipelines.items()}


def _graph_signature(graph) -> Dict[int, int]:
    sig: Dict[int, int] = {}
    for t in graph.tasks.values():
        p = graph.pipelines.get(t.pipeline_uuid)
        if p is not None:
            sig[p.ordinal] = sig.get(p.ordinal, 0) + 1
    return sig


def signature_score(a: Dict[int, int], b: Dict[int, int]) -> float:
    """Overlap score in [0,1]: matched task count / max total task count."""
    if not a or not b:
        return 0.0
    matched = sum(min(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))
    return matched / max(sum(a.values()), sum(b.values()))


def _op_starts(task) -> List[Tuple[int, int]]:
    """Reconstruct (op_id, rel_start_ns) for a quent TaskSpec: the first
    Computing transition is t_first_computing; each op span abuts the next."""
    out = []
    t = task.t_first_computing
    if t < 0:
        return out
    for _name, op_id, dur, _b in task.ops:
        out.append((op_id, t))
        t += dur
    return out


def _clock_pairs(qp: QueryPhysics, graph, t0_abs: int):
    """Matched (nsys_ns, quent_epoch_ns) pairs: NVTX op-range starts are not
    stored per-op in the profile, so pair on task-range starts vs quent
    Preparing timestamps by (pipeline, ordinal) — same-thread adjacent events."""
    by_ord: Dict[int, List] = {}
    for t in graph.tasks.values():
        p = graph.pipelines.get(t.pipeline_uuid)
        if p is not None and t.t_preparing >= 0:
            by_ord.setdefault(p.ordinal, []).append(t)
    for lst in by_ord.values():
        lst.sort(key=lambda t: (t.t_preparing, t.tid))
    pairs = []
    for pid, tasks in qp.pipelines.items():
        qtasks = by_ord.get(pid, [])
        for i, tp in enumerate(tasks):
            if i < len(qtasks):
                pairs.append((tp.start_ns, float(t0_abs + qtasks[i].t_preparing)))
    return pairs


def _match_trace(profile: PhysicsProfile, trace_model, utc_epoch_ns) -> None:
    """Fill matched_trace_label + clock diagnostics against a paired trace."""
    diags = []
    for qp in profile.queries:
        best_label, best_score, best_graph = "", 0.0, None
        sig = _pipeline_signature(qp)
        for q in trace_model.queries:
            g = trace_model.graphs[q.uuid]
            s = signature_score(sig, _graph_signature(g))
            if s > best_score:
                best_label, best_score, best_graph = q.label, s, g
        qp.matched_trace_label = best_label
        entry = {
            "window": list(qp.window),
            "tasks": qp.n_tasks(),
            "best_trace_label": best_label,
            "structure_score": round(best_score, 4),
            "kernel_serial_frac": qp.kernel_serial_frac,
            "clock": None,
        }
        if best_graph is not None and utc_epoch_ns is not None:
            t0 = best_graph.info.t_executing
            if t0 is not None:
                pairs = _clock_pairs(qp, best_graph, t0)
                if pairs:
                    resid = [y - (x + utc_epoch_ns) for x, y in pairs]
                    resid.sort()
                    med = resid[len(resid) // 2]
                    if abs(med) <= SAME_RUN_TOLERANCE_NS:
                        fit = fit_linear(pairs)
                        if fit:
                            entry["clock"] = {"same_run": True, **fit.to_dict()}
                    else:
                        entry["clock"] = {
                            "same_run": False,
                            "median_first_order_residual_ns": med,
                            "note": "capture is a different run than the trace "
                            "(expected: physics joins by structural key)",
                        }
        diags.append(entry)
    profile.diagnostics["trace_match"] = diags


def ingest_nsys(
    nsys_path: str,
    trace_model=None,
    overrides: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> PhysicsProfile:
    data = read_nsys(nsys_path)
    classifier = Classifier(overrides=overrides or {})
    queries, stats = attribute_and_decompose(data, classifier)

    profile = PhysicsProfile()
    profile.queries = queries
    for qp in queries:
        qp.kernel_sum_ns, qp.kernel_union_ns = _window_kernel_occupancy(
            data.kernels, qp.window[0], qp.window[1]
        )
    profile.curves = fit_curves(data.memcpys)
    profile.source = {
        "nsys_sqlite": os.path.abspath(nsys_path),
        "trace_dir": getattr(trace_model, "session_dir", None),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session_start_utc_ns": data.session_start_utc_ns,
        "overrides": overrides or {},
    }

    # classified share of attributed kernel time (unknown degrades to v0)
    class_ns = {"compute": 0.0, "membw": 0.0, "unknown": 0.0}
    for qp in queries:
        for tasks in qp.pipelines.values():
            for tp in tasks:
                if tp.prep:
                    class_ns["compute"] += tp.prep.f_comp * tp.prep.span_ns
                    class_ns["membw"] += tp.prep.f_membw * tp.prep.span_ns
                    class_ns["unknown"] += tp.prep.f_unknown * tp.prep.span_ns
                for op in tp.ops:
                    class_ns["compute"] += op.f_comp * op.span_ns
                    class_ns["membw"] += op.f_membw * op.span_ns
                    class_ns["unknown"] += op.f_unknown * op.span_ns
    ktot = sum(class_ns.values())
    profile.diagnostics = {
        "reader_notes": data.notes,
        "counts": data.counts,
        "attribution": stats.to_dict(),
        "pct_kernel_time_classified": (
            100.0 * (ktot - class_ns["unknown"]) / ktot if ktot else 0.0
        ),
        "kernel_class_ns": class_ns,
        "channel_peak_gbps": _channel_peaks(data.memcpys),
        "clock_beacons": len(data.clock_beacons),
    }
    if trace_model is not None:
        _match_trace(profile, trace_model, data.session_start_utc_ns)
    return profile


def summarize(profile: PhysicsProfile) -> str:
    d = profile.diagnostics
    att = d.get("attribution", {})
    lines = [
        f"nsys physics profile: {len(profile.queries)} query window(s), "
        f"{sum(q.n_tasks() for q in profile.queries)} task attempts",
        "kernels attributed : "
        f"{att.get('kernels_attributed', 0)}/{att.get('kernels_total', 0)} "
        f"({att.get('pct_kernels_attributed', 0.0):.1f}% count, "
        f"{att.get('pct_kernel_ns_attributed', 0.0):.1f}% of kernel time)",
        "memcpy time attrib : " f"{att.get('pct_memcpy_ns_attributed', 0.0):.1f}%",
        "kernel time classed: "
        f"{d.get('pct_kernel_time_classified', 0.0):.1f}% "
        "(unclassified time falls back to v0 conflated scaling)",
        "bandwidth curves   : "
        + (
            ", ".join(
                f"{c} ({len(v.buckets)} size buckets)"
                for c, v in sorted(profile.curves.items())
            )
            or "none (no memcpys in capture)"
        ),
    ]
    for chan, peak in sorted(d.get("channel_peak_gbps", {}).items()):
        lines.append(f"peak aggregate rate: {chan}: {peak:.1f} GB/s")
    for note in d.get("reader_notes", []):
        lines.append(f"NOTE: {note}")
    for entry in d.get("trace_match", []):
        clock = entry.get("clock")
        cdesc = "no clock fit"
        if clock:
            if clock.get("same_run"):
                cdesc = (
                    f"same run, clock fit rms {clock['rms_ns'] / 1e3:.1f} us "
                    f"over {clock['n_pairs']} pairs (slope {clock['slope']:.8f})"
                )
            else:
                cdesc = "different run (structural join only)"
        serial = entry.get("kernel_serial_frac")
        sdesc = (
            f"; kernel serialization {serial:.2f}"
            if serial is not None
            else "; kernel serialization n/a"
        )
        lines.append(
            f"window {entry['window'][0] / 1e6:.0f}..{entry['window'][1] / 1e6:.0f} ms: "
            f"{entry['tasks']} tasks, best trace match "
            f"{entry['best_trace_label'] or '?'} "
            f"(structure score {entry['structure_score']:.2f}); {cdesc}{sdesc}"
        )
    return "\n".join(lines)

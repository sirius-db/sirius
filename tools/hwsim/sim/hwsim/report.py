"""Reporting: per-query human-readable tables + machine-readable JSON/CSV."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .engine import SimResult
from .knobs import Knobs
from .model import QueryGraph, SessionModel


def _ms(ns: float) -> float:
    return ns / 1e6


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(b) < 1024 or unit == "GiB":
            return f"{b:.1f} {unit}" if unit != "B" else f"{b:.0f} B"
        b /= 1024
    return f"{b:.1f} GiB"


def _downsample(points: List[Tuple], max_points: int = 2000) -> List[Tuple]:
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    out = [points[int(i * step)] for i in range(max_points)]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def per_pipeline_breakdown(
    graph: QueryGraph, result: SimResult, knobs: Knobs
) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for tid, t in graph.tasks.items():
        p = graph.pipelines.get(t.pipeline_uuid)
        key = t.pipeline_uuid
        row = rows.get(key)
        if row is None:
            row = rows[key] = {
                "pipeline": p.ordinal if p else -1,
                "chain": p.chain if p else "?",
                "tasks": 0,
                "traced_busy_ms": 0.0,
                "sim_busy_ms": 0.0,
                "traced_prep_ms": 0.0,
                "sim_prep_ms": 0.0,
                "traced_compute_ms": 0.0,
                "sim_compute_ms": 0.0,
                "sim_queue_wait_ms": 0.0,
                "sim_mem_wait_ms": 0.0,
                "transfer_bytes": 0,
            }
        rec = result.task_times[tid]
        sim_prep = max(0.0, rec.prep_end - rec.prep_start) if rec.prep_end >= 0 else 0.0
        sim_compute = sum(d / knobs.op_scale(name) for (name, _, d, _) in t.ops)
        row["tasks"] += 1
        row["traced_busy_ms"] += _ms(t.service_ns)
        row["sim_busy_ms"] += _ms(sim_prep + sim_compute + t.tail_ns)
        row["traced_prep_ms"] += _ms(t.prep_ns)
        row["sim_prep_ms"] += _ms(sim_prep)
        row["traced_compute_ms"] += _ms(t.compute_ns)
        row["sim_compute_ms"] += _ms(sim_compute)
        row["sim_queue_wait_ms"] += _ms(rec.queue_wait_ns)
        row["sim_mem_wait_ms"] += _ms(rec.mem_wait_ns)
        if t.is_transfer_prep:
            row["transfer_bytes"] += t.prep_bytes
    return sorted(rows.values(), key=lambda r: r["pipeline"])


def per_operator_breakdown(graph: QueryGraph, knobs: Knobs) -> List[Dict[str, Any]]:
    agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "traced_ms": 0.0, "sim_ms": 0.0, "input_bytes": 0.0}
    )
    for t in graph.tasks.values():
        for name, _oid, dur, in_bytes in t.ops:
            a = agg[name.split("(")[0]]
            a["calls"] += 1
            a["traced_ms"] += _ms(dur)
            a["sim_ms"] += _ms(dur / knobs.op_scale(name))
            a["input_bytes"] += in_bytes
    rows = [{"operator": k, **v} for k, v in agg.items()]
    return sorted(rows, key=lambda r: -r["traced_ms"])


def query_report(
    model: SessionModel,
    graph: QueryGraph,
    knobs: Knobs,
    result: SimResult,
    baseline_result: Optional[SimResult] = None,
) -> Dict[str, Any]:
    info = graph.info
    traced_ns = graph.traced_exec_wall_ns
    dev0 = min(result.n_threads)
    wall = result.wall_ns
    rep: Dict[str, Any] = {
        "query": {
            "label": info.label,
            "index": info.index,
            "uuid": info.uuid,
            "tasks": len(graph.tasks),
            "pipelines": len(graph.pipelines),
            "batches": len(graph.batches),
        },
        "knobs": knobs.to_dict(),
        "knob_warnings": knobs.warnings(),
        "traced_exec_wall_ms": _ms(traced_ns),
        "sim_wall_ms": _ms(wall),
        "sim_vs_traced_pct": (
            100.0 * (wall - traced_ns) / traced_ns if traced_ns else None
        ),
        "binding_constraint": result.binding_constraint(dev0),
        "resources": {
            "devices": {
                str(d): {
                    "threads": result.n_threads[d],
                    "thread_busy_pct": (
                        100.0 * result.thread_busy_ns[d] / (result.n_threads[d] * wall)
                        if wall
                        else 0.0
                    ),
                    "pool_capacity_bytes": result.pool_capacity[d],
                    "pool_peak_bytes": result.peak_pool[d],
                    "pool_peak_pct": (
                        100.0 * result.peak_pool[d] / result.pool_capacity[d]
                        if result.pool_capacity[d]
                        else 0.0
                    ),
                    "blocked_on_threads_ms": _ms(result.block_totals[d]["threads"]),
                    "blocked_on_memory_ms": _ms(result.block_totals[d]["memory"]),
                }
                for d in sorted(result.n_threads)
            },
            "channels": {
                f"{o}->{t}@gpu{d}": {
                    "capacity_gbps": cs.capacity,  # bytes/ns == GB/s
                    "moved_gb": cs.moved_bytes / 1e9,
                    "busy_ms": _ms(cs.busy_ns),
                    "throttled_ms": _ms(cs.throttled_ns),
                    "achieved_gbps": cs.utilization_rate(),
                    "peak_concurrent": cs.peak_active,
                }
                for (o, t, d), cs in sorted(result.channel_stats.items())
            },
        },
        "warn_counters": {
            "forced_admissions": result.forced_admissions,
            "dep_cycle_breaks": result.dep_cycle_breaks,
            "orphan_gpu_batches": result.orphan_gpu_batches,
            **graph.diagnostics,
        },
        "pipelines": per_pipeline_breakdown(graph, result, knobs),
        "operators": per_operator_breakdown(graph, knobs),
        "timelines": {
            "thread_busy": {
                str(d): _downsample(tl) for d, tl in result.thread_timeline.items()
            },
            "pool": {str(d): _downsample(tl) for d, tl in result.pool_timeline.items()},
        },
    }
    if baseline_result is not None:
        b = baseline_result.wall_ns
        rep["sim_baseline_wall_ms"] = _ms(b)
        rep["sim_vs_sim_baseline_pct"] = 100.0 * (wall - b) / b if b else None
    return rep


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = [line(headers), line(["-" * w for w in widths])]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def render_text(rep: Dict[str, Any]) -> str:
    out = io.StringIO()
    q = rep["query"]
    print(
        f"=== {q['label']}  (query #{q['index']}, {q['tasks']} tasks, "
        f"{q['pipelines']} pipelines, {q['batches']} batches) ===",
        file=out,
    )
    kn = (
        ", ".join(
            f"{k}={v:g}" for k, v in rep["knobs"].items() if v is not None and v != 1.0
        )
        or "baseline (all 1.0)"
    )
    print(f"knobs: {kn}", file=out)
    for w in rep["knob_warnings"]:
        print(f"WARNING: {w}", file=out)
    print(f"traced exec wall : {rep['traced_exec_wall_ms']:10.1f} ms", file=out)
    if "sim_baseline_wall_ms" in rep:
        print(f"sim baseline wall: {rep['sim_baseline_wall_ms']:10.1f} ms", file=out)
    delta = rep.get("sim_vs_sim_baseline_pct", rep.get("sim_vs_traced_pct"))
    ref = "sim baseline" if "sim_baseline_wall_ms" in rep else "traced"
    print(
        f"sim wall         : {rep['sim_wall_ms']:10.1f} ms  "
        f"({delta:+.1f}% vs {ref})",
        file=out,
    )
    print(f"binding constraint: {rep['binding_constraint']}", file=out)

    for d, r in rep["resources"]["devices"].items():
        print(
            f"gpu{d}: threads={r['threads']} busy={r['thread_busy_pct']:.0f}%  "
            f"pool peak={_fmt_bytes(r['pool_peak_bytes'])}"
            f" ({r['pool_peak_pct']:.0f}% of {_fmt_bytes(r['pool_capacity_bytes'])})  "
            f"blocked: threads {r['blocked_on_threads_ms']:.1f} ms, "
            f"memory {r['blocked_on_memory_ms']:.1f} ms",
            file=out,
        )
    for name, c in rep["resources"]["channels"].items():
        cap = f"{c['capacity_gbps']:.1f}" if c["capacity_gbps"] else "inf"
        print(
            f"channel {name}: cap {cap} GB/s, moved {c['moved_gb']:.2f} GB, "
            f"busy {c['busy_ms']:.1f} ms, throttled {c['throttled_ms']:.1f} ms, "
            f"achieved {c['achieved_gbps']:.1f} GB/s, "
            f"peak {c['peak_concurrent']} concurrent",
            file=out,
        )
    warn = {k: v for k, v in rep["warn_counters"].items() if v}
    if warn:
        print(f"diagnostics: {warn}", file=out)

    print("\nper-pipeline breakdown:", file=out)
    rows = [
        [
            str(p["pipeline"]),
            str(p["tasks"]),
            f"{p['traced_busy_ms']:.1f}",
            f"{p['sim_busy_ms']:.1f}",
            f"{p['sim_queue_wait_ms']:.1f}",
            f"{p['sim_mem_wait_ms']:.1f}",
            p["chain"][:64],
        ]
        for p in rep["pipelines"]
    ]
    print(
        _table(
            [
                "pipe",
                "tasks",
                "traced ms",
                "sim ms",
                "q-wait ms",
                "mem-wait ms",
                "chain",
            ],
            rows,
        ),
        file=out,
    )

    print("\nper-operator breakdown (Computing spans):", file=out)
    rows = [
        [
            o["operator"],
            str(int(o["calls"])),
            f"{o['traced_ms']:.1f}",
            f"{o['sim_ms']:.1f}",
            _fmt_bytes(o["input_bytes"]),
        ]
        for o in rep["operators"][:15]
    ]
    print(_table(["operator", "calls", "traced ms", "sim ms", "input"], rows), file=out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Machine-readable writers
# ---------------------------------------------------------------------------


def write_json(rep: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=float)


def write_selfcheck_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_sweep_csv(rows: List[Dict[str, Any]], path: str) -> None:
    write_selfcheck_csv(rows, path)

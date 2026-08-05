"""Structural join: PhysicsProfile x QueryGraph -> per-task annotations.

Join key (WS2 section 4.4): (pipeline numeric id, task ordinal within
pipeline, operator position with op_id verification). Never timestamps —
scheduling comes from the unprofiled trace, physics from the paired capture.

Graph-side ordinal: tasks per pipeline sorted by traced execution start
(t_preparing, matching the NVTX task-range open in gpu_pipeline_task::execute).
Profile-side ordinal: task attempts per pipeline sorted by range start.

Every mismatch is counted; unmatched tasks/ops keep v0 conflated scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..model import QueryGraph
from .ingest import _graph_signature, _pipeline_signature, signature_score
from .schema import OpPhysics, PhysicsProfile, PrepPhysics, QueryPhysics


@dataclass
class TaskAnnotation:
    prep: Optional[PrepPhysics] = None
    ops: List[Optional[OpPhysics]] = field(default_factory=list)  # parallels task.ops


@dataclass
class JoinStats:
    matched_query_label: str = ""
    structure_score: float = 0.0
    tasks_total: int = 0
    tasks_matched: int = 0
    ops_total: int = 0
    ops_matched: int = 0
    op_id_mismatches: int = 0
    op_span_ns_total: float = 0.0
    op_span_ns_matched: float = 0.0
    prep_ns_total: float = 0.0
    prep_ns_matched: float = 0.0
    pipelines_missing_in_profile: List[int] = field(default_factory=list)

    @property
    def pct_span_matched(self) -> float:
        tot = self.op_span_ns_total + self.prep_ns_total
        got = self.op_span_ns_matched + self.prep_ns_matched
        return 100.0 * got / tot if tot else 0.0

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["pct_span_matched"] = self.pct_span_matched
        return d

    def warnings(self) -> List[str]:
        w = []
        if self.tasks_total and self.tasks_matched == 0:
            w.append(
                "physics join matched ZERO tasks — the profile does not "
                "structurally match this query (best window was "
                f"{self.matched_query_label or 'none'}, score "
                f"{self.structure_score:.2f}). All spans fall back to v0 "
                "conflated scaling."
            )
        elif self.pct_span_matched < 90.0:
            w.append(
                f"physics join covers only {self.pct_span_matched:.1f}% of "
                f"traced busy time ({self.tasks_matched}/{self.tasks_total} "
                f"tasks, {self.ops_matched}/{self.ops_total} ops); the "
                "unmatched remainder uses v0 conflated scaling."
            )
        if self.op_id_mismatches:
            w.append(
                f"{self.op_id_mismatches} operator-position/op_id mismatches "
                "between trace and profile (plan drift between runs?); those "
                "ops are treated as unmatched."
            )
        if self.pipelines_missing_in_profile:
            w.append(
                "pipelines absent from the nsys profile: "
                f"{sorted(self.pipelines_missing_in_profile)[:12]}"
            )
        return w


def choose_query_physics(
    profile: PhysicsProfile, graph: QueryGraph
) -> Tuple[Optional[QueryPhysics], float]:
    """Pick the capture window that best matches the graph structurally."""
    gsig = _graph_signature(graph)
    best, best_score = None, 0.0
    for qp in profile.queries:
        s = signature_score(_pipeline_signature(qp), gsig)
        if s > best_score:
            best, best_score = qp, s
    return best, best_score


def join_graph(
    profile: PhysicsProfile, graph: QueryGraph
) -> Tuple[Dict[int, TaskAnnotation], JoinStats]:
    stats = JoinStats()
    qp, score = choose_query_physics(profile, graph)
    stats.structure_score = score
    if qp is not None:
        stats.matched_query_label = qp.matched_trace_label

    # graph tasks per pipeline ordinal, sorted by traced execution start
    by_ord: Dict[int, List] = {}
    for t in graph.tasks.values():
        p = graph.pipelines.get(t.pipeline_uuid)
        key = p.ordinal if p is not None else -1
        by_ord.setdefault(key, []).append(t)
    for lst in by_ord.values():
        lst.sort(
            key=lambda t: (
                t.t_preparing if t.t_preparing >= 0 else t.t_created,
                t.tid,
            )
        )

    ann: Dict[int, TaskAnnotation] = {}
    for pid, tasks in sorted(by_ord.items()):
        ptasks = qp.pipelines.get(pid, []) if qp is not None else []
        if qp is not None and not ptasks and pid >= 0:
            stats.pipelines_missing_in_profile.append(pid)
        for i, task in enumerate(tasks):
            stats.tasks_total += 1
            stats.ops_total += len(task.ops)
            stats.op_span_ns_total += task.compute_ns
            stats.prep_ns_total += task.prep_ns
            tp = ptasks[i] if i < len(ptasks) else None
            if tp is None:
                continue
            stats.tasks_matched += 1
            a = TaskAnnotation()
            if tp.prep is not None and task.prep_ns > 0:
                a.prep = tp.prep
                stats.prep_ns_matched += task.prep_ns
            # ops: match by position, verify op_id; fall back to id lookup
            by_id: Dict[int, List[OpPhysics]] = {}
            for op in tp.ops:
                by_id.setdefault(op.op_id, []).append(op)
            used = set()
            for j, (_name, op_id, dur, _b) in enumerate(task.ops):
                cand = tp.ops[j] if j < len(tp.ops) else None
                if cand is not None and cand.op_id != op_id:
                    stats.op_id_mismatches += 1
                    cand = None
                if cand is None:
                    for alt in by_id.get(op_id, []):
                        if id(alt) not in used:
                            cand = alt
                            break
                if cand is not None:
                    used.add(id(cand))
                    stats.ops_matched += 1
                    stats.op_span_ns_matched += dur
                a.ops.append(cand)
            ann[task.tid] = a
    return ann, stats

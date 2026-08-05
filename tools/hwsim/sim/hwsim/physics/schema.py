"""Physics-profile data model + JSON (de)serialization.

The profile is the *output* of `ingest-nsys` and the *input* of
`simulate --physics`. Everything a task span needs to be re-timed honestly is
expressed as **fractions of the nsys-observed span**, which are then applied
to the (differently-timed, unprofiled) Quent spans — the WS2 rule that physics
travels as rates/shares, never as absolute scheduling timestamps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .curves import BandwidthCurve

FORMAT_VERSION = 1

# Cap stored per-copy samples per task (bandwidth-curve inputs are aggregated
# at ingest; the per-task list only drives the size-dependent scale factor).
MAX_COPIES_PER_TASK = 64


@dataclass
class OpPhysics:
    """Decomposition of one operator invocation's wall span (fractions sum to
    ~1; f_unknown covers unclassified kernel time, f_host the CPU remainder)."""

    op_id: int
    op_name: str
    span_ns: float
    f_comp: float = 0.0
    f_membw: float = 0.0
    f_unknown: float = 0.0
    f_h2d: float = 0.0
    f_d2h: float = 0.0
    f_d2d: float = 0.0
    f_host: float = 1.0
    # absolute diagnostics (ns within the nsys op window)
    kernel_ns: float = 0.0
    memcpy_ns: float = 0.0
    launch_ns: float = 0.0
    sync_ns: float = 0.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d: dict) -> "OpPhysics":
        return OpPhysics(**{k: d[k] for k in OpPhysics.__dataclass_fields__ if k in d})


@dataclass
class PrepPhysics:
    """Decomposition of the Preparing window (task start -> first operator)."""

    span_ns: float
    f_xfer: float = 0.0  # explicit-memcpy share (link time)
    f_comp: float = 0.0  # kernel share classified compute (decompress: SM-bound)
    f_membw: float = 0.0
    f_unknown: float = 0.0
    f_host: float = 1.0
    xfer_bytes: int = 0
    dominant_channel: str = ""  # e.g. "Host-to-Device|Pinned|Device"
    copies: List[Tuple[int, float]] = field(default_factory=list)  # (bytes, ns)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["copies"] = [list(c) for c in self.copies[:MAX_COPIES_PER_TASK]]
        return d

    @staticmethod
    def from_dict(d: dict) -> "PrepPhysics":
        kw = {k: d[k] for k in PrepPhysics.__dataclass_fields__ if k in d}
        kw["copies"] = [(int(b), float(t)) for b, t in d.get("copies", [])]
        return PrepPhysics(**kw)


@dataclass
class TaskPhysics:
    pipeline_id: int
    nsys_task_id: int
    attempt: int  # ordinal for re-executed (downgraded) tasks
    start_ns: float  # nsys session time (diagnostics only)
    end_ns: float
    prep: Optional[PrepPhysics] = None
    ops: List[OpPhysics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "nsys_task_id": self.nsys_task_id,
            "attempt": self.attempt,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "prep": self.prep.to_dict() if self.prep else None,
            "ops": [o.to_dict() for o in self.ops],
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskPhysics":
        return TaskPhysics(
            pipeline_id=int(d["pipeline_id"]),
            nsys_task_id=int(d["nsys_task_id"]),
            attempt=int(d.get("attempt", 0)),
            start_ns=float(d.get("start_ns", 0)),
            end_ns=float(d.get("end_ns", 0)),
            prep=PrepPhysics.from_dict(d["prep"]) if d.get("prep") else None,
            ops=[OpPhysics.from_dict(o) for o in d.get("ops", [])],
        )


@dataclass
class QueryPhysics:
    """One `sirius::query` NVTX window in the capture."""

    window: Tuple[float, float]
    # pipeline_id -> tasks ordered by nsys range start (the join ordinal)
    pipelines: Dict[int, List[TaskPhysics]] = field(default_factory=dict)
    matched_trace_label: str = ""  # best structural match in the paired trace

    def n_tasks(self) -> int:
        return sum(len(v) for v in self.pipelines.values())

    def to_dict(self) -> dict:
        return {
            "window": list(self.window),
            "matched_trace_label": self.matched_trace_label,
            "pipelines": {
                str(pid): [t.to_dict() for t in tasks]
                for pid, tasks in self.pipelines.items()
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "QueryPhysics":
        qp = QueryPhysics(
            window=(float(d["window"][0]), float(d["window"][1])),
            matched_trace_label=d.get("matched_trace_label", ""),
        )
        qp.pipelines = {
            int(pid): [TaskPhysics.from_dict(t) for t in tasks]
            for pid, tasks in d.get("pipelines", {}).items()
        }
        return qp


@dataclass
class PhysicsProfile:
    source: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    curves: Dict[str, BandwidthCurve] = field(default_factory=dict)
    queries: List[QueryPhysics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "source": self.source,
            "diagnostics": self.diagnostics,
            "curves": {k: c.to_dict() for k, c in self.curves.items()},
            "queries": [q.to_dict() for q in self.queries],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=float)

    @staticmethod
    def from_dict(d: dict) -> "PhysicsProfile":
        ver = d.get("format_version")
        if ver != FORMAT_VERSION:
            raise ValueError(
                f"physics profile format_version {ver!r} unsupported "
                f"(this build reads version {FORMAT_VERSION}); re-run ingest-nsys"
            )
        p = PhysicsProfile(
            source=d.get("source", {}), diagnostics=d.get("diagnostics", {})
        )
        p.curves = {
            k: BandwidthCurve.from_dict(c) for k, c in d.get("curves", {}).items()
        }
        p.queries = [QueryPhysics.from_dict(q) for q in d.get("queries", [])]
        return p

    @staticmethod
    def load(path: str) -> "PhysicsProfile":
        with open(path) as f:
            return PhysicsProfile.from_dict(json.load(f))

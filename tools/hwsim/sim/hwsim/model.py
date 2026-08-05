"""Data model: parsed trace entities and the per-query simulation graph.

All timestamps stored on per-query objects are integer nanoseconds relative to
that query's ``Executing`` transition (t=0), so they stay well inside float53
range and can be mixed with float arithmetic in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Session-level static entities
# ---------------------------------------------------------------------------


@dataclass
class MemorySpace:
    uuid: str
    name: str  # e.g. "memory_space(tier=GPU, device_id=0, limit=...)"
    tier: str  # "GPU" | "HOST" | "DISK"
    device_id: int
    capacity_bytes: int


@dataclass
class MemoryTier:
    uuid: str
    name: str  # "GPU-0" | "HOST" | "DISK"
    capacity_bytes: int


@dataclass
class ChannelInfo:
    uuid: str
    name: str  # "host-0->gpu-0"


@dataclass
class QueryInfo:
    uuid: str
    index: int  # 0-based ordinal by Init timestamp
    label: str  # synthesized (e.g. tpch_q01_iter2) or trace label
    raw_name: str  # instance_name from the trace ("unnamed_query" here)
    t_init: int  # absolute unix ns
    t_planning: Optional[int]
    t_executing: Optional[int]
    t_exit: Optional[int]

    @property
    def exec_wall_ns(self) -> Optional[int]:
        if self.t_executing is None or self.t_exit is None:
            return None
        return self.t_exit - self.t_executing

    @property
    def total_wall_ns(self) -> Optional[int]:
        if self.t_exit is None:
            return None
        return self.t_exit - self.t_init


@dataclass
class PipelineInfo:
    uuid: str
    query_uuid: str
    ordinal: int  # numeric id parsed from type_name "Pipeline Id N"
    chain: str  # "GPU_SCAN(0) -> ... -> HASH_GROUP_BY(3)"


# ---------------------------------------------------------------------------
# Per-query simulation graph
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """One traced task attempt = one replayed unit of work."""

    tid: int  # numeric ordinal from instance_name "task-N" (session-unique)
    uuid: str
    pipeline_uuid: str
    device: int

    # Traced FSM timestamps, ns relative to query exec start. -1 = absent.
    t_created: int = -1
    t_queued: int = -1
    t_routing: int = -1
    t_reserving: int = -1
    t_preparing: int = -1
    t_first_computing: int = -1
    t_finalizing: int = -1
    t_exit: int = -1
    success: bool = True

    # Memory reservation (granted bytes from the Preparing usage; requested
    # kept for diagnostics).
    reservation_bytes: int = 0
    requested_bytes: int = 0

    # Traced downgrade activity (gap G5). t_downgrading >= 0 means the traced
    # reservation attempt fell short and the engine issued request_downgrade();
    # the Downgrading -> Preparing span (inside grant_ns) is the downgrade wait.
    t_downgrading: int = -1
    dg_shortfall_bytes: int = 0
    dg_partial_bytes: int = 0

    # Preparing phase (input materialization).
    prep_origin: str = ""
    prep_target: str = ""
    prep_bytes: int = 0
    prep_ns: int = 0  # traced Preparing -> first Computing span

    # (op_name, op_id, base_duration_ns, input_bytes) per Computing state.
    ops: List[Tuple[str, int, int, int]] = field(default_factory=list)

    # Replayed fixed overheads (traced spans that are NOT emergent in the sim).
    pre_queue_ns: int = 0  # Created -> Queued
    grant_ns: int = 0  # Reserving -> Preparing (reservation-grant overhead)
    tail_ns: int = 0  # Finalizing -> Exit

    # Explicit dispatch priority (hwsim-sim exports only: parsed from the
    # Routing state's ``qprio=<rank>`` marker). A simulated schedule's enqueue
    # timestamps do NOT encode the queue priority the engine dispatched by
    # (the source trace's queue-entry order), so re-simulating an export by
    # enqueue order can repack the schedule (+67% measured on q9); this field
    # restores the exact order. None on real traces (t_queued order applies).
    queue_prio: Optional[int] = None

    # Dependency structure (filled by build).
    deps: Set[int] = field(default_factory=set)
    creation_lag_ns: int = 0  # Created - max(dep exit), clamped >= 0
    release_offset_ns: Optional[int] = None  # for root tasks: traced Created offset

    input_batches: List[int] = field(default_factory=list)
    output_batches: List[int] = field(default_factory=list)

    # GPU device-compute work of this task's compute phase (gap G4b), in
    # knob-scaled kernel-ns. Set by the physics retime layer when a GPU knob
    # moves; 0 = no device-resource service (fixed-duration replay). The
    # engine serves it through the per-device fluid compute resource so
    # queue-wait under device saturation EMERGES instead of being replayed.
    dev_work_ns: float = 0.0

    @property
    def is_transfer_prep(self) -> bool:
        return bool(self.prep_origin) and self.prep_origin != self.prep_target

    @property
    def compute_ns(self) -> int:
        return sum(d for (_, _, d, _) in self.ops)

    @property
    def service_ns(self) -> int:
        """Thread-occupancy time: prep + compute + finalize tail."""
        return self.prep_ns + self.compute_ns + self.tail_ns


@dataclass
class BatchSpec:
    bid: int  # numeric data_batch_id / batch_id (process-unique)
    nbytes: int = 0
    gpu_resident: bool = False  # tier at registration was a GPU tier
    device: int = 0
    producer_tid: Optional[int] = None
    consumer_tids: Set[int] = field(default_factory=set)
    t_constructed: int = -1  # relative ns; -1 unknown
    ambiguous_producer: bool = False


@dataclass
class EdgeSpec:
    producer_pipeline: str
    consumer_pipeline: str
    full_barrier: bool  # inferred from observed ordering


@dataclass
class QueryGraph:
    info: QueryInfo
    pipelines: Dict[str, PipelineInfo]
    tasks: Dict[int, TaskSpec]  # keyed by tid
    batches: Dict[int, BatchSpec]
    edges: List[EdgeSpec]
    finish_tail_ns: int = 0  # traced query Exit - max task Exit
    diagnostics: Dict[str, int] = field(default_factory=dict)

    @property
    def traced_exec_wall_ns(self) -> int:
        return self.info.exec_wall_ns or 0

    @property
    def has_traced_spill(self) -> bool:
        """True when the traced execution ran under memory pressure: any task
        hit the Downgrading state or was OOM-rescheduled (success=false)."""
        return any(t.t_downgrading >= 0 or not t.success for t in self.tasks.values())


@dataclass
class SessionModel:
    session_dir: str
    session_uuid: str
    memory_spaces: Dict[str, MemorySpace]
    memory_tiers: Dict[str, MemoryTier]
    channels: Dict[str, ChannelInfo]
    n_executor_threads: Dict[int, int]  # device -> thread count
    queries: List[QueryInfo]
    graphs: Dict[str, QueryGraph]  # query uuid -> graph
    # (origin_tier, target_tier, device) -> observed peak aggregate rate
    # over concurrent Preparing transfers, bytes/ns (== GB/s).
    channel_peak_rate: Dict[Tuple[str, str, int], float] = field(default_factory=dict)
    # HOST memory-space capacity (bytes) — bounds how much the spill model can
    # downgrade out of the GPU pool (0 = unknown -> treated as unbounded).
    host_pool_capacity: int = 0

    def graph_by_label(self, label: str) -> QueryGraph:
        for q in self.queries:
            if q.label == label:
                return self.graphs[q.uuid]
        raise KeyError(f"no query labeled {label!r}; use `info` to list labels")

    def graph_by_index(self, index: int) -> QueryGraph:
        for q in self.queries:
            if q.index == index:
                return self.graphs[q.uuid]
        raise KeyError(f"no query with index {index}")

    def gpu_pool_capacity(self, device: int) -> int:
        for ms in self.memory_spaces.values():
            if ms.tier == "GPU" and ms.device_id == device:
                return ms.capacity_bytes
        raise KeyError(f"no GPU memory space for device {device}")

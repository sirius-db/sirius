"""Export a simulated execution as a Quent ndjson session directory (WS17).

This is the INVERSE of ``trace.py``: it takes what the simulator holds after a
run (the ``QueryGraph`` workload + the ``SimResult`` re-timed schedule) and
writes a session directory with the same envelope / entity / file layout a
real Sirius capture has (spec: tools/hwsim/docs/quent-extraction.md section
(d)), so the Quent analyzer UI renders the *predicted* timeline exactly like a
real one, side by side with the source trace.

Contract highlights (see tools/hwsim/docs/quent-export.md):

- engine_name is ``hwsim-sim``; ``engine.Init.custom_attributes`` carries
  ``hwsim.simulated=1`` (I64), ``hwsim.source_session``, ``hwsim.source_query``
  and one ``hwsim.knob.<name>`` (F64) per non-default knob.
- every event is derivable from sim state — re-timed walls are the SIMULATED
  ones, never copied through from the source trace. Event types the sim has
  no data for (io_request, Downgrading, InTransit, tier-change
  self-transitions, OOM-attempt task FSMs) are OMITTED, not fabricated.
  Schema-required scalar fields the sim does not track are emitted with the
  schema's documented unknown markers (0 / nil uuid) — the Rust analyzer's
  serde types have no field defaults, so the current-model WS9 fields
  (input_rows, output_rows/bytes, num_rows/columns, producer_task_uuid) must
  be present on every line or the importer silently truncates the stream
  (WS18 defect 1, tools/hwsim/docs/quent-export-verification.md).
- uuids are deterministic uuid7-style ids given a seed (stable tests); every
  FSM entity gets a contiguous per-entity ``seq`` from 0 with monotone
  timestamps; timestamps are unix-epoch ns with sim t=0 anchored at the source
  query's traced ``Executing`` transition so both sessions align on the UI
  time axis.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple

from .engine import SimResult
from .knobs import Knobs
from .model import QueryGraph, SessionModel

ENGINE_NAME = "hwsim-sim"
NIL_UUID = "00000000-0000-0000-0000-000000000000"
# current_operator_id is u32 in the analyzer model (task.rs); pseudo-operators
# the sim synthesizes (physics PHYS::PREP, op_id -1) export the u32::MAX
# placeholder — real plan operator ids are small.
NO_OPERATOR_ID = (1 << 32) - 1

# Synthetic-timeline pads (ns). Static/session entities are declared inside
# [t0 - _SETUP_PAD, t0); the query Init/Planning pair sits inside
# [t0 - _PLAN_PAD, t0). Producer-less batches (scan-manager staging, orphans)
# are Constructed at t0 - _ORPHAN_PAD: before any task's window, so the
# round-trip parser classifies them as externally available, which is exactly
# how the engine treated them (resident from t=0).
_SETUP_PAD_NS = 2_000_000
_PLAN_PAD_NS = 1_000_000
_ORPHAN_PAD_NS = 10_000_000
_STEP_NS = 1_000  # spacing between consecutive synthetic declaration events


# ---------------------------------------------------------------------------
# Deterministic uuid7-style ids
# ---------------------------------------------------------------------------


class _UuidGen:
    """uuid7-style: 48-bit unix-ms timestamp | ver=7 | 12-bit monotone counter
    | var=10 | 62 bits of seeded-hash randomness. Time-ordered like a real
    UUIDv7 and fully deterministic given (seed, call order)."""

    def __init__(self, seed: str) -> None:
        self._seed = seed.encode()
        self._n = 0

    def next(self, ts_ns: int) -> str:
        ms = (max(0, int(ts_ns)) // 1_000_000) & ((1 << 48) - 1)
        self._n += 1
        digest = hashlib.sha256(self._seed + self._n.to_bytes(8, "big")).digest()
        rand_a = self._n & 0xFFF
        rand_b = int.from_bytes(digest[:8], "big") & ((1 << 62) - 1)
        val = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0x2 << 62) | rand_b
        h = f"{val:032x}"
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# ndjson session writer
# ---------------------------------------------------------------------------


class _SessionWriter:
    def __init__(self, root: str, gen: _UuidGen, anchor_ns: int) -> None:
        self.root = root
        self._gen = gen
        self._anchor = anchor_ns
        self._lines: Dict[str, List[str]] = {}

    def emit(self, entity: str, id_: str, ts_ns: int, data: Any) -> None:
        line = json.dumps(
            {"id": id_, "timestamp": int(ts_ns), "data": data},
            separators=(",", ":"),
        )
        self._lines.setdefault(entity, []).append(line)

    def flush(self) -> None:
        for entity in sorted(self._lines):
            d = os.path.join(self.root, entity)
            os.makedirs(d, exist_ok=True)
            stream = self._gen.next(self._anchor)
            with open(os.path.join(d, stream + ".ndjson"), "w") as f:
                f.write("\n".join(self._lines[entity]) + "\n")


class _Fsm:
    """Per-entity seq counter with monotone-timestamp clamping."""

    def __init__(self, writer: _SessionWriter, entity: str, id_: str) -> None:
        self._w = writer
        self._entity = entity
        self.id = id_
        self._seq = 0
        self._last_ts: Optional[int] = None

    def state(self, ts_ns: int, state: Any) -> None:
        ts = int(ts_ns)
        if self._last_ts is not None and ts < self._last_ts:
            ts = self._last_ts
        self._last_ts = ts
        self._w.emit(self._entity, self.id, ts, {"seq": self._seq, "state": state})
        self._seq += 1

    def exit(self, ts_ns: int) -> None:
        self.state(ts_ns, "Exit")


def _resource_fsm(
    writer: _SessionWriter,
    entity: str,
    id_: str,
    t_init: int,
    t_exit: int,
    init_state: str,
    init_payload: dict,
    operating_state: str,
    operating_payload: Any,
    finalizing_state: str,
) -> None:
    fsm = _Fsm(writer, entity, id_)
    fsm.state(t_init, {init_state: init_payload})
    fsm.state(t_init, {operating_state: operating_payload})
    fsm.state(t_exit, {finalizing_state: None})
    fsm.exit(t_exit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def knob_suffix(knobs: Knobs, physics: bool = False) -> str:
    """Encode the non-default knobs for the exported query label. A
    physics-retimed run carries a trailing ``physics`` marker so a v0 and a
    physics export of the same knob point are distinguishable in the UI."""
    parts = []
    for f in dc_fields(knobs):
        v = getattr(knobs, f.name)
        if v is not None and v != 1.0:
            parts.append(f"{f.name}={v:g}")
    if not parts:
        parts = ["baseline"]
    if physics:
        parts.append("physics")
    return ",".join(parts)


def _usage(resource_id: str, nbytes: Optional[int]) -> dict:
    cap = {"capacity_bytes": int(nbytes)} if nbytes is not None else None
    return {"resource_id": resource_id, "capacity": cap}


def _queue_usage(queue_id: str) -> dict:
    return {"resource_id": queue_id, "capacity": {"capacity_entries": 1}}


def _assign_threads(
    intervals: List[Tuple[float, float, int]], n: int
) -> Dict[int, int]:
    """Greedy interval colouring: assign each task's [admit, finish] to the
    executor-thread slot that frees earliest (deterministic). The engine
    guarantees <= n concurrent tasks, so this reproduces a feasible binding."""
    out: Dict[int, int] = {}
    free = [(0.0, k) for k in range(max(1, n))]
    heapq.heapify(free)
    for start, end, tid in sorted(intervals):
        _avail, k = heapq.heappop(free)
        out[tid] = k
        heapq.heappush(free, (max(end, start), k))
    return out


def _scaled(nbytes: int, tier: str, knobs: Knobs) -> int:
    if tier.startswith("GPU"):
        return int(round(nbytes * knobs.gpu_mem_capacity))
    if tier.startswith("HOST"):
        return int(round(nbytes * knobs.cpu_mem_capacity))
    return int(nbytes)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_session(
    model: SessionModel,
    graph: QueryGraph,
    knobs: Knobs,
    result: SimResult,
    out_dir: str,
    seed: Optional[str] = None,
    physics: Optional[Dict[str, Any]] = None,
) -> str:
    """Write one Quent ndjson session for one simulated query.

    Returns the created session directory path
    (``<out_dir>/<session-uuid>/``). Raises FileExistsError if the
    deterministic session directory already exists.

    ``physics`` (a provenance dict, see ``physics/cli.py``) marks a
    physics-retimed run: ``graph`` must then be the RETIMED graph from
    ``simulate_with_physics(..., return_graph=True)`` — its span durations are
    already knob-scaled (the engine ran with neutralized GPU knobs), so
    per-operator Computing boundaries are laid out with engine-neutral weights
    instead of dividing by the v0 ``op_scale``. The engine Init grows
    ``hwsim.physics=1`` plus profile-provenance attributes and the query label
    carries a ``physics`` marker.
    """
    label = f"{graph.info.label}@{knob_suffix(knobs, physics=physics is not None)}"
    if seed is None:
        seed = f"{model.session_uuid}|{label}"
    gen = _UuidGen(seed)
    # Weight divisor for laying out per-op Computing boundaries: the physics
    # path pre-scales span durations, and the engine keeps only the G1 io
    # knob at the op level (integrate._engine_knobs).
    layout_knobs = Knobs(io_bandwidth=knobs.io_bandwidth) if physics else knobs

    # ---- time anchors (absolute unix ns) ---------------------------------
    t0 = int(graph.info.t_executing or 0)
    t0 = max(t0, _SETUP_PAD_NS + _ORPHAN_PAD_NS)  # keep synthetic ts >= 0
    wall = int(round(result.wall_ns))
    t_end = t0 + wall
    t_setup = t0 - _SETUP_PAD_NS

    def rel(ns: float) -> int:
        return t0 + int(round(ns))

    session_uuid = gen.next(t_setup)
    root = os.path.join(out_dir, session_uuid)
    if os.path.exists(root):
        raise FileExistsError(
            f"exported session already exists: {root} (remove it first)"
        )
    os.makedirs(root)
    w = _SessionWriter(root, gen, t_setup)

    # steadily increasing synthetic timestamps for declarations
    _clock = [t_setup]

    def tick() -> int:
        _clock[0] += _STEP_NS
        return _clock[0]

    devices = sorted(result.n_threads)

    # ---- engine / worker / query_group -----------------------------------
    engine_id = gen.next(t_setup)
    attrs = [
        {"key": "hwsim.simulated", "value": {"I64": 1}},
        {"key": "hwsim.source_session", "value": {"String": model.session_uuid}},
        {"key": "hwsim.source_query", "value": {"String": graph.info.label}},
    ]
    for f in dc_fields(knobs):
        v = getattr(knobs, f.name)
        if v is not None and v != 1.0:
            attrs.append({"key": f"hwsim.knob.{f.name}", "value": {"F64": float(v)}})
    attrs.append({"key": "hwsim.spill_mode", "value": {"String": result.spill_mode}})
    attrs.append({"key": "hwsim.seed", "value": {"String": seed}})
    if physics is not None:
        # Physics-retimed run: mark it and carry the profile provenance so a
        # consumer can tell which capture re-timed this prediction.
        attrs.append({"key": "hwsim.physics", "value": {"I64": 1}})
        for key, akey in (
            ("profile_path", "hwsim.physics_profile"),
            ("nsys_sqlite", "hwsim.physics_nsys_sqlite"),
            ("created_utc", "hwsim.physics_profile_created_utc"),
        ):
            v = physics.get(key)
            if v:
                attrs.append({"key": akey, "value": {"String": str(v)}})
        for key, akey in (
            ("pct_span_matched", "hwsim.physics_pct_span_matched"),
            ("kernel_serial_frac", "hwsim.physics_kernel_serial_frac"),
        ):
            v = physics.get(key)
            if v is not None:
                attrs.append({"key": akey, "value": {"F64": float(v)}})
        if physics.get("device_model_active") is not None:
            attrs.append(
                {
                    "key": "hwsim.physics_device_model",
                    "value": {"I64": int(bool(physics["device_model_active"]))},
                }
            )
    w.emit(
        "engine",
        engine_id,
        tick(),
        {
            "Init": {
                "implementation": {
                    "name": ENGINE_NAME,
                    "version": None,
                    "custom_attributes": attrs,
                },
                "instance_name": ENGINE_NAME,
            }
        },
    )
    worker_id = gen.next(t_setup)
    w.emit(
        "worker",
        worker_id,
        tick(),
        {"Init": {"parent_engine_id": engine_id, "instance_name": "worker-hwsim"}},
    )
    qgroup_id = gen.next(t_setup)
    w.emit(
        "query_group",
        qgroup_id,
        tick(),
        {
            "Declaration": {
                "instance_name": f"{ENGINE_NAME}-session",
                "engine_id": engine_id,
            }
        },
    )

    # ---- gpu devices + thread groups --------------------------------------
    gpu_ids: Dict[int, str] = {}
    tg_exec: Dict[int, str] = {}
    tg_mgr: Dict[int, str] = {}
    tg_shared = gen.next(t_setup)
    w.emit(
        "thread_group",
        tg_shared,
        tick(),
        {"Declaration": {"instance_name": "shared", "parent_group_id": engine_id}},
    )
    for d in devices:
        gpu_ids[d] = gen.next(t_setup)
        w.emit(
            "gpu_device",
            gpu_ids[d],
            tick(),
            {
                "Declaration": {
                    "instance_name": f"gpu-{d}",
                    "parent_group_id": engine_id,
                    "ordinal": d,
                }
            },
        )
        for name, store in (
            ("executor_thread", tg_exec),
            ("task_manager_loop_thread", tg_mgr),
        ):
            store[d] = gen.next(t_setup)
            w.emit(
                "thread_group",
                store[d],
                tick(),
                {
                    "Declaration": {
                        "instance_name": name,
                        "parent_group_id": gpu_ids[d],
                    }
                },
            )

    # ---- memory spaces / tiers (SIMULATED capacities) ----------------------
    # GPU capacities come straight from the engine (already knob-scaled);
    # HOST/DISK come from the source model, scaled by the matching knob.
    spaces: Dict[Tuple[str, int], Tuple[str, int]] = {}  # (tier, dev) -> (uuid, cap)
    for d in devices:
        cap = int(round(result.pool_capacity.get(d, 0)))
        spaces[("GPU", d)] = (gen.next(t_setup), cap)
    for ms in sorted(model.memory_spaces.values(), key=lambda m: m.name):
        key = (ms.tier, ms.device_id)
        if key in spaces:
            continue
        spaces[key] = (gen.next(t_setup), _scaled(ms.capacity_bytes, ms.tier, knobs))
    if ("HOST", 0) not in spaces:
        cap = _scaled(int(getattr(model, "host_pool_capacity", 0) or 0), "HOST", knobs)
        spaces[("HOST", 0)] = (gen.next(t_setup), cap)
    for (tier, dev), (uid, cap) in sorted(spaces.items()):
        _resource_fsm(
            w,
            "memory",
            uid,
            tick(),
            t_end + _STEP_NS,
            "MemoryInitializing",
            {
                "instance_name": (
                    f"memory_space(tier={tier}, device_id={dev}, limit={cap})"
                ),
                "parent_group_id": engine_id,
                "resource_type_name": "memory",
            },
            "MemoryOperating",
            {"capacity_bytes": cap},
            "MemoryFinalizing",
        )

    tiers: Dict[str, str] = {}  # tier name ("GPU-0"/"HOST"/"DISK") -> uuid
    tier_caps: Dict[str, int] = {}
    for d in devices:
        tiers[f"GPU-{d}"] = gen.next(t_setup)
        tier_caps[f"GPU-{d}"] = spaces[("GPU", d)][1]
    for mt in sorted(model.memory_tiers.values(), key=lambda m: m.name):
        if mt.name in tiers:
            continue
        tiers[mt.name] = gen.next(t_setup)
        tier_caps[mt.name] = _scaled(mt.capacity_bytes, mt.name, knobs)
    if "HOST" not in tiers:
        tiers["HOST"] = gen.next(t_setup)
        tier_caps["HOST"] = spaces[("HOST", 0)][1]
    for name in sorted(tiers):
        _resource_fsm(
            w,
            "memory_tier",
            tiers[name],
            tick(),
            t_end + _STEP_NS,
            "MemoryTierInitializing",
            {
                "instance_name": name,
                "parent_group_id": engine_id,
                "resource_type_name": "memory_tier",
            },
            "MemoryTierOperating",
            {"capacity_bytes": tier_caps[name]},
            "MemoryTierFinalizing",
        )

    # ---- channels (full mesh between exported spaces, placeholder caps) ----
    def _space_uuid(endpoint: str) -> Optional[str]:
        # "gpu-0" / "host-0" / "disk-0"
        tier, _, dev = endpoint.partition("-")
        entry = spaces.get((tier.upper(), int(dev or 0)))
        return entry[0] if entry else None

    chan_names = sorted(c.name for c in model.channels.values())
    if not chan_names:
        chan_names = [f"host-0->gpu-{d}" for d in devices] + [
            f"gpu-{d}->host-0" for d in devices
        ]
    for name in chan_names:
        src, _, dst = name.partition("->")
        src_id, dst_id = _space_uuid(src), _space_uuid(dst)
        if not src_id or not dst_id:
            continue
        _resource_fsm(
            w,
            "channel",
            gen.next(t_setup),
            tick(),
            t_end + _STEP_NS,
            "ChannelInitializing",
            {
                "instance_name": name,
                "parent_group_id": engine_id,
                "resource_type_name": "channel",
                "source_id": src_id,
                "target_id": dst_id,
            },
            "ChannelOperating",
            {"capacity_bytes": (1 << 64) - 1},  # placeholder, like real traces
            "ChannelFinalizing",
        )

    # ---- task queues / manager threads / executor threads ------------------
    sched_queue_id = gen.next(t_setup)
    _resource_fsm(
        w,
        "task_queue",
        sched_queue_id,
        tick(),
        t_end + _STEP_NS,
        "TaskQueueInitializing",
        {
            "instance_name": "task-scheduler-gpu-queue",
            "parent_group_id": tg_shared,
            "resource_type_name": "task_queue",
        },
        "TaskQueueOperating",
        {"capacity_entries": (1 << 64) - 1},
        "TaskQueueFinalizing",
    )
    exec_queue_ids: Dict[int, str] = {}
    for d in devices:
        exec_queue_ids[d] = gen.next(t_setup)
        _resource_fsm(
            w,
            "task_queue",
            exec_queue_ids[d],
            tick(),
            t_end + _STEP_NS,
            "TaskQueueInitializing",
            {
                "instance_name": "gpu_pipeline-task-queue",
                "parent_group_id": gpu_ids[d],
                "resource_type_name": "task_queue",
            },
            "TaskQueueOperating",
            {"capacity_entries": (1 << 64) - 1},
            "TaskQueueFinalizing",
        )

    sched_thread_id = gen.next(t_setup)
    _resource_fsm(
        w,
        "task_manager_loop_thread",
        sched_thread_id,
        tick(),
        t_end + _STEP_NS,
        "TaskManagerLoopThreadInitializing",
        {
            "instance_name": "task-scheduler-thread",
            "parent_group_id": tg_shared,
            "resource_type_name": "task_manager_loop_thread",
        },
        "TaskManagerLoopThreadOperating",
        None,
        "TaskManagerLoopThreadFinalizing",
    )
    mgr_ids: Dict[int, str] = {}
    for d in devices:
        mgr_ids[d] = gen.next(t_setup)
        _resource_fsm(
            w,
            "task_manager_loop_thread",
            mgr_ids[d],
            tick(),
            t_end + _STEP_NS,
            "TaskManagerLoopThreadInitializing",
            {
                "instance_name": f"gpu-{d}-exec-manager",
                "parent_group_id": tg_mgr[d],
                "resource_type_name": "task_manager_loop_thread",
            },
            "TaskManagerLoopThreadOperating",
            None,
            "TaskManagerLoopThreadFinalizing",
        )

    exec_thread_ids: Dict[int, List[str]] = {}
    for d in devices:
        exec_thread_ids[d] = []
        for k in range(result.n_threads[d]):
            uid = gen.next(t_setup)
            exec_thread_ids[d].append(uid)
            _resource_fsm(
                w,
                "executor_thread",
                uid,
                tick(),
                t_end + _STEP_NS,
                "ExecutorThreadInitializing",
                {
                    "instance_name": f"gpu_pipeline-gpu{d}-exec-{k}",
                    "parent_group_id": tg_exec[d],
                    "resource_type_name": "executor_thread",
                },
                "ExecutorThreadOperating",
                None,
                "ExecutorThreadFinalizing",
            )

    # ---- query FSM ---------------------------------------------------------
    query_id = gen.next(t0 - _PLAN_PAD_NS)
    qfsm = _Fsm(w, "query", query_id)
    qfsm.state(
        t0 - _PLAN_PAD_NS,
        {"Init": {"instance_name": label, "query_group_id": qgroup_id}},
    )
    qfsm.state(t0 - _PLAN_PAD_NS // 2, {"Planning": {}})
    qfsm.state(t0, {"Executing": {}})

    # ---- plan / operators / ports ------------------------------------------
    # Every pipeline referenced by a task or by graph.pipelines gets an
    # operator declaration plus one receiver + one sender port (the sim only
    # holds pipeline-level dataflow).
    pipe_uuids = dict(graph.pipelines)
    referenced = {t.pipeline_uuid for t in graph.tasks.values()}
    plan_id = gen.next(t0)
    op_ids: Dict[str, str] = {}
    recv_port: Dict[str, str] = {}
    send_port: Dict[str, str] = {}

    def _ordinal(uid: str) -> int:
        p = pipe_uuids.get(uid)
        return p.ordinal if p else -1

    all_pipes = sorted(set(pipe_uuids) | referenced, key=lambda u: (_ordinal(u), u))
    for uid in all_pipes:
        op_ids[uid] = gen.next(t0)
        recv_port[uid] = gen.next(t0)
        send_port[uid] = gen.next(t0)

    edges_out = []
    seen_edges = set()
    for e in graph.edges:
        key = (e.producer_pipeline, e.consumer_pipeline)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        if key[0] in op_ids and key[1] in op_ids:
            edges_out.append({"source": send_port[key[0]], "target": recv_port[key[1]]})
    w.emit(
        "plan",
        plan_id,
        t0,
        {
            "Declaration": {
                "parent": {"query_id": query_id, "plan_id": None},
                "instance_name": "pipeline_plan",
                "edges": edges_out,
                "worker_id": worker_id,
            }
        },
    )
    for uid in all_pipes:
        p = pipe_uuids.get(uid)
        w.emit(
            "operator",
            op_ids[uid],
            t0,
            {
                "Declaration": {
                    "plan_id": plan_id,
                    "parent_operator_ids": [],
                    "instance_name": p.chain if p else "?",
                    "type_name": f"Pipeline Id {p.ordinal if p else -1}",
                    "custom_attributes": [],
                }
            },
        )
        for pid, pname in (
            (recv_port[uid], "default_receiver"),
            (send_port[uid], "default_sender"),
        ):
            w.emit(
                "port",
                pid,
                t0,
                {"Declaration": {"operator_id": op_ids[uid], "instance_name": pname}},
            )

    # ---- task FSMs -----------------------------------------------------------
    # Thread binding: greedy colouring of the [admit, finish] occupancy.
    per_dev_intervals: Dict[int, List[Tuple[float, float, int]]] = {
        d: [] for d in devices
    }
    for tid, task in graph.tasks.items():
        rec = result.task_times.get(tid)
        if rec is None or rec.finish < 0:
            continue
        per_dev_intervals.setdefault(task.device, []).append(
            (rec.admit, rec.finish, tid)
        )
    thread_of: Dict[int, str] = {}
    for d, ivs in per_dev_intervals.items():
        binding = _assign_threads(ivs, result.n_threads.get(d, 1))
        pool = exec_thread_ids.get(d) or exec_thread_ids[devices[0]]
        for tid, k in binding.items():
            thread_of[tid] = pool[k % len(pool)]

    # Dispatch-priority ranks: the engine dispatched released tasks by the
    # source trace's queue-entry order (queue_order="traced"), which the
    # simulated enqueue timestamps do NOT encode — export it explicitly so a
    # re-simulation of this session reproduces the schedule instead of
    # repacking it (measured +67% wall drift on q9 without this).
    def _dispatch_prio(t: int) -> float:
        # mirror engine._enqueue exactly (incl. re-exports of exports)
        task = graph.tasks[t]
        qp = getattr(task, "queue_prio", None)
        if qp is not None:
            return float(qp)
        return float(task.t_queued if task.t_queued >= 0 else task.t_created)

    qprio_rank: Dict[int, int] = {
        tid: rank
        for rank, tid in enumerate(
            sorted(graph.tasks, key=lambda t: (_dispatch_prio(t), t))
        )
    }

    task_uuid_of: Dict[int, str] = {}
    for tid in sorted(graph.tasks):
        task = graph.tasks[tid]
        rec = result.task_times.get(tid)
        if rec is None or rec.finish < 0:
            continue
        d = task.device
        created = rel(rec.release)
        task_uuid_of[tid] = gen.next(created)
        fsm = _Fsm(w, "task", task_uuid_of[tid])
        fsm.state(
            created,
            {
                "Created": {
                    "instance_name": f"task-{tid}",
                    "pipeline_uuid": op_ids.get(task.pipeline_uuid, NIL_UUID),
                }
            },
        )
        enq = rel(rec.enqueue)
        fsm.state(enq, {"Queued": {"queue": _queue_usage(sched_queue_id)}})
        fsm.state(
            enq,
            {
                "Routing": {
                    # dispatch-order marker, parsed back by build.py (the
                    # schema's instance_name is free-form)
                    "instance_name": f"qprio={qprio_rank[tid]}",
                    "preferred_device_id": d,
                    "manager_thread": _usage(sched_thread_id, None),
                }
            },
        )
        fsm.state(
            enq,
            {"Queued": {"queue": _queue_usage(exec_queue_ids.get(d, sched_queue_id))}},
        )
        admit = rel(rec.admit)
        granted = int(
            min(
                float(task.reservation_bytes),
                result.pool_capacity.get(d, float(task.reservation_bytes)),
            )
        )
        requested = task.requested_bytes or task.reservation_bytes
        fsm.state(
            admit,
            {
                "Reserving": {
                    "instance_name": "",
                    "requested_bytes": int(requested),
                    "input_basis": int(task.prep_bytes),
                    "peak_estimate": int(requested),
                    "bytes_to_materialize": int(
                        task.prep_bytes if task.is_transfer_prep else 0
                    ),
                    "manager_thread": _usage(mgr_ids.get(d, sched_thread_id), None),
                }
            },
        )
        thread_usage = _usage(thread_of.get(tid, exec_thread_ids[devices[0]][0]), None)
        reservation_usage = _usage(tiers.get(f"GPU-{d}", tiers["HOST"]), granted)
        prep_start = rel(rec.prep_start)
        fsm.state(
            prep_start,
            {
                "Preparing": {
                    "instance_name": "",
                    "origin_tier": task.prep_origin or "GPU",
                    "target_tier": task.prep_target or "GPU",
                    "input_bytes": int(task.prep_bytes),
                    "executor_thread": thread_usage,
                    "reservation": reservation_usage,
                }
            },
        )
        # Per-operator Computing transitions: the sim schedules the compute
        # phase as one span; op boundaries are laid out proportionally to the
        # knob-scaled base durations (exact at knobs=1, proportional when a
        # fluid device stretched the phase).
        prep_end = rel(rec.prep_end)
        fin_ts = max(prep_end, rel(rec.finish) - int(task.tail_ns))
        weights = [
            dur / layout_knobs.op_scale(name) for (name, _oid, dur, _b) in task.ops
        ]
        total_w = sum(weights)
        cum = 0.0
        for (name, op_id, _dur, in_bytes), wgt in zip(task.ops, weights):
            frac = (cum / total_w) if total_w > 0 else 0.0
            ts = prep_end + int(round(frac * (fin_ts - prep_end)))
            cum += wgt
            fsm.state(
                ts,
                {
                    "Computing": {
                        "instance_name": name,
                        "current_operator_id": (
                            int(op_id) if op_id >= 0 else NO_OPERATOR_ID
                        ),
                        "input_bytes": int(in_bytes),
                        "peak_allocated_bytes": 0,
                        "input_rows": 0,  # WS9 field; 0 = unknown (not tracked)
                        "executor_thread": thread_usage,
                        "reservation": reservation_usage,
                    }
                },
            )
        fsm.state(
            fin_ts,
            {
                "Finalizing": {
                    "instance_name": "",
                    "success": bool(task.success),
                    # WS9 fields; 0 = unknown (the sim tracks no row counts)
                    "output_rows": 0,
                    "output_bytes": 0,
                }
            },
        )
        fsm.exit(rel(rec.finish))

    # ---- data batches + placements -------------------------------------------
    fallback_pipe = all_pipes[0] if all_pipes else None
    for bid in sorted(graph.batches):
        b = graph.batches[bid]
        producer_tid = b.producer_tid if b.producer_tid in task_uuid_of else None
        if producer_tid is not None:
            t_pub = rel(result.task_times[producer_tid].finish)
            producer_pipe = graph.tasks[producer_tid].pipeline_uuid
            producer_task_uuid = task_uuid_of[producer_tid]
        else:
            t_pub = t0 - _ORPHAN_PAD_NS
            producer_pipe = None
            producer_task_uuid = NIL_UUID
        consumers = sorted(t for t in b.consumer_tids if t in task_uuid_of)
        t_done = (
            max(rel(result.task_times[t].finish) for t in consumers)
            if consumers
            else t_end
        )
        tier_name = f"GPU-{b.device}" if b.gpu_resident else "HOST"
        tier_id = tiers.get(tier_name, tiers["HOST"])
        space = spaces.get(("GPU", b.device) if b.gpu_resident else ("HOST", 0))

        dfsm = _Fsm(w, "data_batch", gen.next(t_pub))
        dfsm.state(
            t_pub,
            {
                "Constructed": {
                    "instance_name": "batch",
                    "data_batch_id": int(bid),
                    "producer_pipeline_uuid": op_ids.get(producer_pipe, NIL_UUID),
                    # WS9 fields: producer task known from the sim graph;
                    # rows/columns not tracked (0 = unknown).
                    "producer_task_uuid": producer_task_uuid,
                    "num_rows": 0,
                    "num_columns": 0,
                }
            },
        )
        if space is not None:
            dfsm.state(t_pub, {"Stationary": {"memory": _usage(space[0], b.nbytes)}})
        dfsm.state(t_done, {"Destructed": {}})
        dfsm.exit(t_done)

        tier_usage = _usage(tier_id, b.nbytes)
        if consumers:
            for ctid in consumers:
                crec = result.task_times[ctid]
                cpipe = graph.tasks[ctid].pipeline_uuid
                pfsm = _Fsm(w, "batch_placement", gen.next(t_pub))
                pfsm.state(
                    t_pub,
                    {
                        "BatchRegistered": {
                            "instance_name": f"batch-{bid}",
                            "batch_id": int(bid),
                            "pipeline_uuid": op_ids.get(cpipe, NIL_UUID),
                            "port_uuid": recv_port.get(cpipe, NIL_UUID),
                            "origin": "operator_output",
                            "producer_task_uuid": producer_task_uuid,  # WS9
                            "tier": tier_usage,
                        }
                    },
                )
                pfsm.state(t_pub, {"BatchQueued": {"tier": tier_usage}})
                pfsm.state(
                    rel(crec.admit),
                    {
                        "BatchPackaged": {
                            "instance_name": "",
                            "task_uuid": task_uuid_of[ctid],
                            "tier": tier_usage,
                        }
                    },
                )
                pfsm.state(
                    rel(crec.prep_end),
                    {
                        "BatchProcessing": {
                            "instance_name": "",
                            "task_uuid": task_uuid_of[ctid],
                            "tier": tier_usage,
                        }
                    },
                )
                pfsm.state(
                    rel(crec.finish),
                    {"BatchConsumed": {"instance_name": "", "reason": "processed"}},
                )
                pfsm.exit(rel(crec.finish))
        else:
            anchor_pipe = producer_pipe or fallback_pipe
            pfsm = _Fsm(w, "batch_placement", gen.next(t_pub))
            pfsm.state(
                t_pub,
                {
                    "BatchRegistered": {
                        "instance_name": f"batch-{bid}",
                        "batch_id": int(bid),
                        "pipeline_uuid": op_ids.get(anchor_pipe, NIL_UUID),
                        "port_uuid": recv_port.get(anchor_pipe, NIL_UUID),
                        "origin": "operator_output",
                        "producer_task_uuid": producer_task_uuid,  # WS9
                        "tier": tier_usage,
                    }
                },
            )
            pfsm.state(t_pub, {"BatchQueued": {"tier": tier_usage}})
            pfsm.state(
                t_end,
                {"BatchConsumed": {"instance_name": "", "reason": "query_end"}},
            )
            pfsm.exit(t_end)

    # ---- teardown: query exit, worker/engine exit -----------------------------
    qfsm.exit(t_end)
    w.emit("worker", worker_id, t_end + 2 * _STEP_NS, {"Exit": None})
    w.emit("engine", engine_id, t_end + 3 * _STEP_NS, {"Exit": None})
    w.flush()

    # ---- model.qmi -------------------------------------------------------------
    qmi: Dict[str, Any] = {}
    src_qmi = os.path.join(model.session_dir, "model.qmi") if model.session_dir else ""
    if src_qmi and os.path.isfile(src_qmi):
        try:
            with open(src_qmi) as f:
                qmi = json.load(f)
        except (OSError, ValueError):
            qmi = {}
    qmi.setdefault("quent", {"version": "unknown"})
    qmi.setdefault("model", {"name": "Sirius"})
    qmi["hwsim"] = {
        "generator": "hwsim export-quent v0",
        "engine_name": ENGINE_NAME,
        "source_session": model.session_uuid,
        "source_query": graph.info.label,
        "exported_query": label,
        "knobs": {k: v for k, v in knobs.to_dict().items() if v is not None},
        "spill_mode": result.spill_mode,
        "sim_wall_ns": wall,
        "seed": seed,
    }
    if physics is not None:
        qmi["hwsim"]["physics"] = dict(physics)
    with open(os.path.join(root, "model.qmi"), "w") as f:
        json.dump(qmi, f, indent=2)
    return root

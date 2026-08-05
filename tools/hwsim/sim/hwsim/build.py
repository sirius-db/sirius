"""Build per-query simulation graphs from a parsed RawSession.

This is where the WS1 recipes are applied:
- task FSM -> TaskSpec spans (queue/reservation waits become *emergent* in the
  engine; Created->Queued, Reserving->Preparing and Finalizing->Exit spans are
  replayed as fixed overheads),
- batch->producer-task attribution by (producer pipeline, time window) — the
  documented G2 heuristic, ambiguity flagged,
- task-level dependencies = producers of the packaged input batches, plus
  full-barrier edges inferred from observed ordering,
- per-channel peak aggregate transfer rate (used as the transfer-channel
  capacity C0 that the c2c knob scales).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .model import (
    BatchSpec,
    EdgeSpec,
    PipelineInfo,
    QueryGraph,
    QueryInfo,
    SessionModel,
    TaskSpec,
)
from .trace import RawSession, RawTask, _usage_bytes

# A dependency whose producer finalized more than this after the consumer was
# *created* cannot be real (publish precedes finalize by ~us; the task creator
# adds latency on top). Such edges are attribution mistakes — dropped.
FALSE_DEP_SLACK_NS = 1_000_000  # 1 ms
# Fallback window slack when matching a batch construction timestamp to a
# producer-task execution window.
ATTRIB_SLACK_NS = 5_000_000  # 5 ms

# hwsim-sim exports carry the engine's dispatch order in the Routing state's
# free-form instance_name ("qprio=<rank>"): simulated enqueue timestamps do
# not encode the queue priority the run dispatched by (docs/quent-export.md).
_QPRIO_RE = re.compile(r"\bqprio=(\d+)\b")


def _tier_is_gpu(tier_name: str) -> bool:
    return tier_name.startswith("GPU")


def _build_task_spec(rt: RawTask, t0: int) -> Optional[TaskSpec]:
    spec = TaskSpec(tid=rt.tid, uuid=rt.uuid, pipeline_uuid=rt.pipeline_uuid, device=0)
    computing: List[Tuple[int, str, int, int]] = []  # (ts, name, op_id, input_bytes)
    seen_first_queue = False
    for seq, ts_abs, name, payload in rt.events:
        ts = ts_abs - t0
        if name == "Created":
            spec.t_created = ts
        elif name == "Queued":
            if not seen_first_queue:
                spec.t_queued = ts
                seen_first_queue = True
        elif name == "Routing":
            spec.t_routing = ts
            spec.device = int(payload.get("preferred_device_id", 0)) if payload else 0
            if payload:
                m = _QPRIO_RE.search(payload.get("instance_name", "") or "")
                if m:
                    spec.queue_prio = int(m.group(1))
        elif name == "Reserving":
            spec.t_reserving = ts
            if payload:
                spec.requested_bytes = int(payload.get("requested_bytes", 0) or 0)
        elif name == "Preparing":
            spec.t_preparing = ts
            if payload:
                spec.prep_origin = payload.get("origin_tier", "")
                spec.prep_target = payload.get("target_tier", "")
                spec.prep_bytes = int(payload.get("input_bytes", 0) or 0)
                spec.reservation_bytes = _usage_bytes(payload.get("reservation"))
        elif name == "Computing":
            if spec.t_first_computing < 0:
                spec.t_first_computing = ts
            if payload:
                computing.append(
                    (
                        ts,
                        payload.get("instance_name", "?"),
                        int(payload.get("current_operator_id", -1)),
                        int(payload.get("input_bytes", 0) or 0),
                    )
                )
        elif name == "Downgrading":
            spec.t_downgrading = ts
            if payload:
                spec.dg_shortfall_bytes = int(payload.get("shortfall_bytes", 0) or 0)
                spec.dg_partial_bytes = int(payload.get("partial_bytes", 0) or 0)
        elif name == "Finalizing":
            spec.t_finalizing = ts
            if payload is not None:
                spec.success = bool(payload.get("success", True))
        elif name == "Exit":
            spec.t_exit = ts

    if spec.t_created < 0 or spec.t_exit < 0:
        return None  # incomplete FSM (query aborted mid-trace); skip

    # Derived spans. Missing states (cancelled paths) degrade gracefully.
    if spec.t_queued < 0:
        spec.t_queued = spec.t_created
    spec.pre_queue_ns = max(0, spec.t_queued - spec.t_created)
    end_anchor = spec.t_finalizing if spec.t_finalizing >= 0 else spec.t_exit
    if spec.t_preparing >= 0:
        first_after_prep = (
            spec.t_first_computing if spec.t_first_computing >= 0 else end_anchor
        )
        spec.prep_ns = max(0, first_after_prep - spec.t_preparing)
        if spec.t_reserving >= 0:
            spec.grant_ns = max(0, spec.t_preparing - spec.t_reserving)
    # Per-operator spans: Computing_i -> next transition (last one -> Finalizing).
    for i, (ts, name, op_id, in_bytes) in enumerate(computing):
        nxt = computing[i + 1][0] if i + 1 < len(computing) else end_anchor
        spec.ops.append((name, op_id, max(0, nxt - ts), in_bytes))
    if spec.t_finalizing >= 0:
        spec.tail_ns = max(0, spec.t_exit - spec.t_finalizing)
    return spec


def _label_queries(queries: List[QueryInfo], graphs: Dict[str, QueryGraph]) -> None:
    """Synthesize tpch_qNN_iterK labels when the run is 22 queries x 3 unnamed
    iterations (verified by identical plan shapes within each group of 3)."""
    named = [q for q in queries if q.raw_name and q.raw_name != "unnamed_query"]
    if named:
        counts: Dict[str, int] = defaultdict(int)
        for q in queries:
            base = q.raw_name or f"query{q.index:02d}"
            counts[base] += 1
            q.label = f"{base}_iter{counts[base]}" if q.raw_name else base
        return
    if len(queries) % 3 == 0:

        def shape(q: QueryInfo) -> Tuple[str, ...]:
            g = graphs[q.uuid]
            return tuple(sorted(p.chain for p in g.pipelines.values()))

        ok = all(
            shape(queries[3 * k])
            == shape(queries[3 * k + 1])
            == shape(queries[3 * k + 2])
            for k in range(len(queries) // 3)
        )
        if ok:
            for q in queries:
                q.label = f"tpch_q{q.index // 3 + 1:02d}_iter{q.index % 3 + 1}"
            return
    for q in queries:
        q.label = f"query{q.index:02d}"


def _synthesize_queries(raw: RawSession, verbose: bool = False) -> None:
    """Fallback for truncated sessions (process killed before the buffered
    query/plan ndjson was flushed — e.g. the E2-lo-q9 capacity capture):
    rebuild QueryInfo records from task event extents. Pipelines are grouped
    into queries via plan declarations when present; otherwise they are
    clustered temporally (queries execute sequentially, so a pipeline whose
    first task is created after every task of the current cluster exited
    starts a new query). t_executing/t_exit become first-task-Created /
    last-task-Exit, so the 'traced wall' of a synthesized query excludes the
    (small) planning and result-collection tails."""
    # per-pipeline task extents
    extents: Dict[str, List[int]] = {}
    for rt in raw.tasks.values():
        for _seq, ts, name, _payload in rt.events:
            if name == "Created" or name == "Exit":
                b = extents.get(rt.pipeline_uuid)
                if b is None:
                    extents[rt.pipeline_uuid] = [ts, ts]
                else:
                    if ts < b[0]:
                        b[0] = ts
                    if ts > b[1]:
                        b[1] = ts

    have_plan_map = any(p.query_uuid for p in raw.pipelines.values())
    if not have_plan_map:
        # Group pipelines into queries: temporal clusters (queries execute
        # sequentially) merged by batch dataflow (pipelines of one query
        # exchange batches; distinct queries never do — sequential pipelines
        # of a single query would otherwise over-split).
        ordered = sorted(
            (uid for uid in extents if uid in raw.pipelines),
            key=lambda uid: extents[uid][0],
        )
        cluster_of: Dict[str, int] = {}
        cluster_end = None
        cluster_id = -1
        for uid in ordered:
            t0, t1 = extents[uid]
            if cluster_end is None or t0 > cluster_end:
                cluster_id += 1
                cluster_end = t1
            else:
                cluster_end = max(cluster_end, t1)
            cluster_of[uid] = cluster_id

        parent = list(range(cluster_id + 1))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        batch_producer_cluster: Dict[int, int] = {}
        for bid, (producer_pipe, _ts) in raw.batch_constructed.items():
            c = cluster_of.get(producer_pipe)
            if c is not None:
                batch_producer_cluster[bid] = c
        for pl in raw.placements:
            cc = cluster_of.get(pl.pipeline_uuid)
            cp = batch_producer_cluster.get(pl.batch_id)
            if cc is not None and cp is not None:
                union(cc, cp)

        for uid, c in cluster_of.items():
            raw.pipelines[uid].query_uuid = f"synth-query-{find(c)}"

    bounds: Dict[str, List[int]] = {}
    for uid, (t0, t1) in extents.items():
        p = raw.pipelines.get(uid)
        if p is None or not p.query_uuid:
            continue
        b = bounds.get(p.query_uuid)
        if b is None:
            bounds[p.query_uuid] = [t0, t1]
        else:
            b[0] = min(b[0], t0)
            b[1] = max(b[1], t1)
    recs = sorted((t0, t1, q) for q, (t0, t1) in bounds.items())
    for idx, (t0, t1, quuid) in enumerate(recs):
        raw.queries.append(
            QueryInfo(
                uuid=quuid,
                index=idx,
                label=f"query{idx:02d}",
                raw_name="",
                t_init=t0,
                t_planning=t0,
                t_executing=t0,
                t_exit=t1,
            )
        )
    if verbose and recs:
        print(
            f"[hwsim] query events missing; synthesized {len(recs)} queries "
            "from task extents (walls exclude planning/collection tails)",
            file=sys.stderr,
        )


def build_session_model(raw: RawSession, verbose: bool = False) -> SessionModel:
    tier_name = {uid: t.name for uid, t in raw.memory_tiers.items()}
    if not raw.queries and raw.pipelines:
        _synthesize_queries(raw, verbose=verbose)

    # ---- group tasks / placements / batches by query --------------------
    pipe_to_query = {uid: p.query_uuid for uid, p in raw.pipelines.items()}
    tasks_by_query: Dict[str, List[RawTask]] = defaultdict(list)
    for rt in raw.tasks.values():
        q = pipe_to_query.get(rt.pipeline_uuid)
        if q:
            tasks_by_query[q].append(rt)

    placements_by_query = defaultdict(list)
    for pl in raw.placements:
        q = pipe_to_query.get(pl.pipeline_uuid)
        if q:
            placements_by_query[q].append(pl)

    task_uuid_to_tid = {rt.uuid: rt.tid for rt in raw.tasks.values()}

    # executor thread names look like "gpu_pipeline-gpu<N>-exec-<K>"
    n_threads: Dict[int, int] = defaultdict(int)
    for name in raw.executor_threads.values():
        m = re.search(r"gpu(\d+)-exec-", name)
        n_threads[int(m.group(1)) if m else 0] += 1
    if not n_threads:
        n_threads[0] = 4

    graphs: Dict[str, QueryGraph] = {}
    channel_events: Dict[Tuple[str, str, int], List[Tuple[int, float]]] = defaultdict(
        list
    )

    for q in raw.queries:
        if q.t_executing is None or q.t_exit is None:
            continue
        t0 = q.t_executing
        diag: Dict[str, int] = defaultdict(int)

        pipelines = {
            uid: p for uid, p in raw.pipelines.items() if p.query_uuid == q.uuid
        }

        tasks: Dict[int, TaskSpec] = {}
        for rt in tasks_by_query.get(q.uuid, []):
            spec = _build_task_spec(rt, t0)
            if spec is None:
                diag["incomplete_tasks_skipped"] += 1
                continue
            if not spec.success:
                diag["failed_tasks"] += 1
            if any(name == "Downgrading" for _, _, name, _ in rt.events):
                diag["downgrading_tasks"] += 1
            tasks[spec.tid] = spec

        tasks_by_pipe: Dict[str, List[TaskSpec]] = defaultdict(list)
        for spec in tasks.values():
            tasks_by_pipe[spec.pipeline_uuid].append(spec)
        for lst in tasks_by_pipe.values():
            lst.sort(key=lambda s: s.t_preparing if s.t_preparing >= 0 else s.t_created)

        # ---- batches: bytes/tier/consumers from placements, producer via G2 --
        batches: Dict[int, BatchSpec] = {}
        for pl in placements_by_query.get(q.uuid, []):
            if pl.batch_id < 0:
                diag["placements_without_registered"] += 1
                if not pl.task_uuid:
                    continue
            b = batches.get(pl.batch_id)
            if b is None:
                b = batches[pl.batch_id] = BatchSpec(bid=pl.batch_id)
            b.nbytes = max(b.nbytes, pl.nbytes)
            tname = tier_name.get(pl.tier_uuid, "")
            if _tier_is_gpu(tname):
                b.gpu_resident = True
                b.device = int(tname.split("-")[1]) if "-" in tname else 0
            if pl.task_uuid:
                tid = task_uuid_to_tid.get(pl.task_uuid, -1)
                if tid >= 0 and tid in tasks:
                    b.consumer_tids.add(tid)
                else:
                    diag["packaged_by_unknown_task"] += 1

        for bid, b in batches.items():
            con = raw.batch_constructed.get(bid)
            if con is None:
                diag["batches_without_constructed"] += 1
                continue
            producer_pipe, ts_abs = con
            b.t_constructed = ts_abs - t0
            cands = [
                s
                for s in tasks_by_pipe.get(producer_pipe, [])
                if s.t_preparing >= 0
                and s.t_preparing - ATTRIB_SLACK_NS
                <= b.t_constructed
                <= s.t_exit + ATTRIB_SLACK_NS
            ]
            if not cands:
                # e.g. scan-manager staging batches constructed before any task
                # of the scan pipeline ran -> treated as externally available.
                diag["batches_without_producer_task"] += 1
                continue
            if len(cands) > 1:
                exact = [
                    s for s in cands if s.t_preparing <= b.t_constructed <= s.t_exit
                ]
                if exact:
                    cands = exact
            if len(cands) > 1:
                b.ambiguous_producer = True
                diag["ambiguous_producer_batches"] += 1
            # least-constraining choice: candidate finishing earliest
            b.producer_tid = min(cands, key=lambda s: s.t_exit).tid

        # ---- wire input/output batch lists on tasks -------------------------
        for b in batches.values():
            if b.producer_tid is not None and b.producer_tid in tasks:
                tasks[b.producer_tid].output_batches.append(b.bid)
            for tid in b.consumer_tids:
                tasks[tid].input_batches.append(b.bid)

        # ---- plan edges + barrier inference ---------------------------------
        edges: List[EdgeSpec] = []
        for src_port, dst_port in raw.plan_edges.get(q.uuid, []):
            src_pipe = raw.port_owner.get(src_port, "")
            dst_pipe = raw.port_owner.get(dst_port, "")
            if not src_pipe or not dst_pipe:
                continue
            src_tasks = tasks_by_pipe.get(src_pipe, [])
            dst_tasks = tasks_by_pipe.get(dst_pipe, [])
            full = False
            if src_tasks and dst_tasks:
                full = min(s.t_created for s in dst_tasks) >= max(
                    s.t_finalizing for s in src_tasks
                )
            edges.append(EdgeSpec(src_pipe, dst_pipe, full))

        # ---- dependency sets -------------------------------------------------
        for spec in tasks.values():
            for bid in spec.input_batches:
                p = batches[bid].producer_tid
                if p is not None and p != spec.tid:
                    spec.deps.add(p)
        for e in edges:
            if not e.full_barrier:
                continue
            src_tids = [s.tid for s in tasks_by_pipe.get(e.producer_pipeline, [])]
            for spec in tasks_by_pipe.get(e.consumer_pipeline, []):
                spec.deps.update(t for t in src_tids if t != spec.tid)

        # drop physically-impossible deps (attribution mistakes)
        for spec in tasks.values():
            bad = {
                d
                for d in spec.deps
                if tasks[d].t_finalizing > spec.t_created + FALSE_DEP_SLACK_NS
            }
            if bad:
                diag["dropped_false_deps"] += len(bad)
                spec.deps -= bad

        # ---- release parameters ---------------------------------------------
        for spec in tasks.values():
            if spec.deps:
                dep_done = max(tasks[d].t_exit for d in spec.deps)
                spec.creation_lag_ns = max(0, spec.t_created - dep_done)
                spec.release_offset_ns = None
            else:
                spec.release_offset_ns = max(0, spec.t_created)
                chain = pipelines.get(spec.pipeline_uuid)
                if chain and not chain.chain.startswith("GPU_SCAN"):
                    diag["root_pinned_nonscan_tasks"] += 1

        finish_tail = 0
        if tasks:
            finish_tail = max(
                0, (q.t_exit - t0) - max(s.t_exit for s in tasks.values())
            )

        graphs[q.uuid] = QueryGraph(
            info=q,
            pipelines=pipelines,
            tasks=tasks,
            batches=batches,
            edges=edges,
            finish_tail_ns=finish_tail,
            diagnostics=dict(diag),
        )

        # ---- channel utilization sweep input ---------------------------------
        for spec in tasks.values():
            if spec.is_transfer_prep and spec.prep_ns > 1000 and spec.prep_bytes > 0:
                key = (spec.prep_origin, spec.prep_target, spec.device)
                rate = spec.prep_bytes / spec.prep_ns  # bytes per ns == GB/s
                channel_events[key].append((spec.t_preparing + t0, +rate))
                channel_events[key].append(
                    (spec.t_preparing + spec.prep_ns + t0, -rate)
                )

    # ---- channel capacity: peak concurrent aggregate rate --------------------
    channel_peak: Dict[Tuple[str, str, int], float] = {}
    for key, evs in channel_events.items():
        evs.sort()
        cur = peak = 0.0
        for _, delta in evs:
            cur += delta
            peak = max(peak, cur)
        channel_peak[key] = peak

    host_pool = 0
    for ms in raw.memory_spaces.values():
        if ms.tier == "HOST":
            host_pool = max(host_pool, ms.capacity_bytes)

    model = SessionModel(
        session_dir=raw.session_dir,
        session_uuid=raw.session_uuid,
        memory_spaces=raw.memory_spaces,
        memory_tiers=raw.memory_tiers,
        channels=raw.channels,
        n_executor_threads=dict(n_threads),
        queries=[q for q in raw.queries if q.uuid in graphs],
        graphs=graphs,
        channel_peak_rate=channel_peak,
        host_pool_capacity=host_pool,
    )
    _label_queries(model.queries, model.graphs)
    if verbose:
        total_tasks = sum(len(g.tasks) for g in graphs.values())
        print(
            f"[hwsim] built {len(graphs)} query graphs, {total_tasks} tasks, "
            f"channel peaks: "
            + ", ".join(
                f"{a}->{b}@gpu{d}: {r:.1f} GB/s"
                for (a, b, d), r in sorted(channel_peak.items())
            ),
            file=sys.stderr,
        )
    return model

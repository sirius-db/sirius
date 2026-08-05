"""Streaming parser for a Quent ndjson telemetry session.

Follows the parser spec in tools/hwsim/docs/quent-extraction.md (d):
- one JSON object per line: {"id", "timestamp", "data"}
- FSM entities carry {"seq": N, "state": {...}|"Exit"}; events within one ndjson
  file are NOT globally timestamp-sorted -> per-entity ordering is by seq.
- plain entities carry {"EventName": {...}|null}.

The big files (task 162 MB, data_batch 180 MB, batch_placement 311 MB) are
streamed line by line; for data_batch and batch_placement a cheap substring
pre-filter skips states the v0 simulator does not need before json.loads runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .model import (
    ChannelInfo,
    MemorySpace,
    MemoryTier,
    PipelineInfo,
    QueryInfo,
)

CACHE_FORMAT_VERSION = 9  # bump when RawSession/build outputs change shape


# ---------------------------------------------------------------------------
# Raw (pre-build) containers
# ---------------------------------------------------------------------------


class RawTask:
    __slots__ = ("uuid", "tid", "pipeline_uuid", "events")

    def __init__(self, uuid: str) -> None:
        self.uuid = uuid
        self.tid: int = -1
        self.pipeline_uuid: str = ""
        # (seq, ts_abs, state_name, payload)
        self.events: List[Tuple[int, int, str, Optional[dict]]] = []


class RawPlacement:
    __slots__ = (
        "batch_id",
        "pipeline_uuid",
        "port_uuid",
        "nbytes",
        "tier_uuid",
        "task_uuid",
        "origin",
        "t_registered",
    )

    def __init__(self) -> None:
        self.batch_id: int = -1
        self.pipeline_uuid: str = ""
        self.port_uuid: str = ""
        self.nbytes: int = 0
        self.tier_uuid: str = ""
        self.task_uuid: str = ""
        self.origin: str = ""
        self.t_registered: int = -1


class RawSession:
    """Everything the graph builder needs, still keyed by absolute ns."""

    def __init__(self, session_dir: str) -> None:
        self.session_dir = session_dir
        self.session_uuid = os.path.basename(os.path.normpath(session_dir))
        self.memory_spaces: Dict[str, MemorySpace] = {}
        self.memory_tiers: Dict[str, MemoryTier] = {}
        self.channels: Dict[str, ChannelInfo] = {}
        self.executor_threads: Dict[str, str] = {}  # uuid -> instance_name
        self.thread_group_parent: Dict[str, str] = {}  # group uuid -> parent uuid
        self.gpu_device_ordinal: Dict[str, int] = {}  # gpu_device uuid -> ordinal
        self.queries: List[QueryInfo] = []
        self.pipelines: Dict[str, PipelineInfo] = {}  # operator uuid -> info
        self.port_owner: Dict[str, str] = {}  # port uuid -> operator uuid
        self.plan_edges: Dict[str, List[Tuple[str, str]]] = (
            {}
        )  # query uuid -> [(src_port, dst_port)]
        self.tasks: Dict[str, RawTask] = {}  # task uuid -> RawTask
        # data_batch_id -> (producer_pipeline_uuid, t_constructed_abs)
        self.batch_constructed: Dict[int, Tuple[str, int]] = {}
        self.placements: List[RawPlacement] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ndjson_lines(session_dir: str, entity: str) -> Iterator[str]:
    d = os.path.join(session_dir, entity)
    if not os.path.isdir(d):
        return
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".ndjson"):
            continue
        with open(os.path.join(d, fname), "r", buffering=1 << 20) as f:
            for line in f:
                if line.strip():
                    yield line


def _usage_bytes(usage: Optional[dict]) -> int:
    if not usage:
        return 0
    cap = usage.get("capacity")
    if not cap:
        return 0
    return int(cap.get("capacity_bytes", 0) or 0)


_SPACE_RE = re.compile(r"tier=(\w+), device_id=(\d+), limit=(\d+)")


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_static(raw: RawSession) -> None:
    for line in _ndjson_lines(raw.session_dir, "memory"):
        ev = json.loads(line)
        st = ev["data"].get("state")
        if isinstance(st, dict) and "MemoryInitializing" in st:
            p = st["MemoryInitializing"]
            m = _SPACE_RE.search(p["instance_name"])
            tier, dev, limit = (
                (m.group(1), int(m.group(2)), int(m.group(3))) if m else ("?", 0, 0)
            )
            raw.memory_spaces[ev["id"]] = MemorySpace(
                uuid=ev["id"],
                name=p["instance_name"],
                tier=tier,
                device_id=dev,
                capacity_bytes=limit,
            )
        elif isinstance(st, dict) and "MemoryOperating" in st:
            ms = raw.memory_spaces.get(ev["id"])
            if ms is not None:
                ms.capacity_bytes = int(st["MemoryOperating"]["capacity_bytes"])

    for line in _ndjson_lines(raw.session_dir, "memory_tier"):
        ev = json.loads(line)
        st = ev["data"].get("state")
        if isinstance(st, dict) and "MemoryTierInitializing" in st:
            p = st["MemoryTierInitializing"]
            raw.memory_tiers[ev["id"]] = MemoryTier(
                uuid=ev["id"], name=p["instance_name"], capacity_bytes=0
            )
        elif isinstance(st, dict) and "MemoryTierOperating" in st:
            mt = raw.memory_tiers.get(ev["id"])
            if mt is not None:
                mt.capacity_bytes = int(st["MemoryTierOperating"]["capacity_bytes"])

    for line in _ndjson_lines(raw.session_dir, "channel"):
        ev = json.loads(line)
        st = ev["data"].get("state")
        if isinstance(st, dict) and "ChannelInitializing" in st:
            p = st["ChannelInitializing"]
            raw.channels[ev["id"]] = ChannelInfo(uuid=ev["id"], name=p["instance_name"])

    for line in _ndjson_lines(raw.session_dir, "executor_thread"):
        ev = json.loads(line)
        st = ev["data"].get("state")
        if isinstance(st, dict) and "ExecutorThreadInitializing" in st:
            p = st["ExecutorThreadInitializing"]
            raw.executor_threads[ev["id"]] = p["instance_name"]
            raw.thread_group_parent.setdefault(p["parent_group_id"], "")

    for line in _ndjson_lines(raw.session_dir, "thread_group"):
        ev = json.loads(line)
        p = ev["data"].get("Declaration")
        if p:
            raw.thread_group_parent[ev["id"]] = p["parent_group_id"]

    for line in _ndjson_lines(raw.session_dir, "gpu_device"):
        ev = json.loads(line)
        p = ev["data"].get("Declaration")
        if p:
            raw.gpu_device_ordinal[ev["id"]] = int(p.get("ordinal", 0))


def _parse_queries(raw: RawSession) -> None:
    per_id: Dict[str, Dict[str, Any]] = {}
    for line in _ndjson_lines(raw.session_dir, "query"):
        ev = json.loads(line)
        qid = ev["id"]
        ts = ev["timestamp"]
        st = ev["data"]["state"]
        rec = per_id.setdefault(
            qid,
            {
                "init": None,
                "planning": None,
                "executing": None,
                "exit": None,
                "name": "",
            },
        )
        if st == "Exit":
            rec["exit"] = ts
        elif "Init" in st:
            rec["init"] = ts
            rec["name"] = st["Init"].get("instance_name", "")
        elif "Planning" in st:
            rec["planning"] = ts
        elif "Executing" in st:
            rec["executing"] = ts
    ordered = sorted(per_id.items(), key=lambda kv: kv[1]["init"] or 0)
    for idx, (qid, rec) in enumerate(ordered):
        raw.queries.append(
            QueryInfo(
                uuid=qid,
                index=idx,
                label=f"query{idx:02d}",
                raw_name=rec["name"],
                t_init=rec["init"] or 0,
                t_planning=rec["planning"],
                t_executing=rec["executing"],
                t_exit=rec["exit"],
            )
        )


_PIPE_ID_RE = re.compile(r"Pipeline Id (\d+)")


def _parse_plan_graph(raw: RawSession) -> None:
    plan_to_query: Dict[str, str] = {}
    for line in _ndjson_lines(raw.session_dir, "plan"):
        ev = json.loads(line)
        p = ev["data"].get("Declaration")
        if not p:
            continue
        qid = p["parent"]["query_id"]
        plan_to_query[ev["id"]] = qid
        raw.plan_edges[qid] = [(e["source"], e["target"]) for e in p.get("edges", [])]

    for line in _ndjson_lines(raw.session_dir, "operator"):
        ev = json.loads(line)
        p = ev["data"].get("Declaration")
        if not p:
            continue
        qid = plan_to_query.get(p["plan_id"], "")
        m = _PIPE_ID_RE.search(p.get("type_name", ""))
        raw.pipelines[ev["id"]] = PipelineInfo(
            uuid=ev["id"],
            query_uuid=qid,
            ordinal=int(m.group(1)) if m else -1,
            chain=p.get("instance_name", ""),
        )

    for line in _ndjson_lines(raw.session_dir, "port"):
        ev = json.loads(line)
        p = ev["data"].get("Declaration")
        if p:
            raw.port_owner[ev["id"]] = p["operator_id"]


_TASK_NAME_RE = re.compile(r"task-(\d+)")


def _parse_tasks(raw: RawSession) -> None:
    tasks = raw.tasks
    for line in _ndjson_lines(raw.session_dir, "task"):
        ev = json.loads(line)
        uuid = ev["id"]
        ts = ev["timestamp"]
        d = ev["data"]
        seq = d["seq"]
        st = d["state"]
        t = tasks.get(uuid)
        if t is None:
            t = tasks[uuid] = RawTask(uuid)
        if st == "Exit":
            t.events.append((seq, ts, "Exit", None))
            continue
        name, payload = next(iter(st.items()))
        if name == "Created":
            t.pipeline_uuid = payload["pipeline_uuid"]
            m = _TASK_NAME_RE.search(payload.get("instance_name", ""))
            if m:
                t.tid = int(m.group(1))
        t.events.append((seq, ts, name, payload))
    for t in tasks.values():
        t.events.sort(key=lambda e: e[0])


def _parse_data_batches(raw: RawSession) -> None:
    # Only Constructed carries what v0 needs (producer pipeline + timestamp).
    for line in _ndjson_lines(raw.session_dir, "data_batch"):
        if '"Constructed"' not in line:
            continue
        ev = json.loads(line)
        p = ev["data"]["state"]["Constructed"]
        raw.batch_constructed[int(p["data_batch_id"])] = (
            p["producer_pipeline_uuid"],
            ev["timestamp"],
        )


def _parse_placements(raw: RawSession) -> None:
    # Need BatchRegistered (batch->consumer pipeline/port, bytes, tier) and
    # BatchPackaged (batch->consuming task). Everything else is skipped fast.
    per_id: Dict[str, RawPlacement] = {}
    for line in _ndjson_lines(raw.session_dir, "batch_placement"):
        reg = '"BatchRegistered"' in line
        if not reg and '"BatchPackaged"' not in line:
            continue
        ev = json.loads(line)
        pid = ev["id"]
        pl = per_id.get(pid)
        if pl is None:
            pl = per_id[pid] = RawPlacement()
        st = ev["data"]["state"]
        if reg:
            p = st["BatchRegistered"]
            pl.batch_id = int(p["batch_id"])
            pl.pipeline_uuid = p["pipeline_uuid"]
            pl.port_uuid = p.get("port_uuid", "")
            pl.origin = p.get("origin", "")
            pl.t_registered = ev["timestamp"]
            tier = p.get("tier")
            if tier:
                pl.tier_uuid = tier.get("resource_id", "")
                pl.nbytes = _usage_bytes(tier)
        else:
            p = st["BatchPackaged"]
            pl.task_uuid = p.get("task_uuid", "")
            tier = p.get("tier")
            if tier and not pl.nbytes:
                pl.tier_uuid = pl.tier_uuid or tier.get("resource_id", "")
                pl.nbytes = _usage_bytes(tier)
    raw.placements = list(per_id.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_session(session_dir: str, verbose: bool = False) -> RawSession:
    raw = RawSession(session_dir)
    steps = [
        ("static", _parse_static),
        ("queries", _parse_queries),
        ("plan graph", _parse_plan_graph),
        ("tasks", _parse_tasks),
        ("data batches", _parse_data_batches),
        ("placements", _parse_placements),
    ]
    for name, fn in steps:
        t0 = time.monotonic()
        fn(raw)
        if verbose:
            print(
                f"[hwsim] parsed {name} in {time.monotonic() - t0:.1f}s",
                file=sys.stderr,
            )
    return raw


def _cache_key(session_dir: str) -> str:
    sig = []
    for entity in ("task", "data_batch", "batch_placement", "query"):
        d = os.path.join(session_dir, entity)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                stat = os.stat(p)
                sig.append((f, stat.st_size, int(stat.st_mtime)))
    digest = hashlib.sha1(repr(sig).encode()).hexdigest()[:12]
    return f"v{CACHE_FORMAT_VERSION}-{digest}"


def default_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "hwsim")


def check_ndjson_exports(session_dir: str) -> None:
    """Postcard-trap check (RTX validation defect 4): the bundled
    tpch_telemetry_sirius.yaml ships ``exporter: postcard``, and hwsim parses
    ONLY ndjson — ``_ndjson_lines`` silently skips everything else, yielding
    an empty/near-empty model with no hint why. Warn loudly (error-grade when
    NO ndjson exists at all) before parsing."""
    nd = other = 0
    sample: List[str] = []
    try:
        entries = sorted(os.listdir(session_dir))
    except OSError:
        return
    for d in entries:
        sub = os.path.join(session_dir, d)
        if not os.path.isdir(sub):
            continue
        try:
            files = sorted(os.listdir(sub))
        except OSError:
            continue
        for f in files:
            if not os.path.isfile(os.path.join(sub, f)):
                continue
            if f.endswith(".ndjson"):
                nd += 1
            else:
                other += 1
                if len(sample) < 3:
                    sample.append(f"{d}/{f}")
    if not other:
        return
    hint = (
        "hwsim parses ONLY the ndjson exporter format — capture with "
        "`sirius.telemetry.exporter: ndjson` (the bundled "
        "tpch_telemetry_sirius.yaml ships `exporter: postcard`, which hwsim "
        "cannot read)."
    )
    if nd == 0:
        print(
            f"ERROR: {session_dir}: {other} telemetry file(s) but NONE in "
            f"ndjson format (e.g. {', '.join(sample)}) — this session will "
            f"parse as EMPTY. {hint}",
            file=sys.stderr,
        )
    else:
        print(
            f"WARNING: {session_dir}: {other} non-ndjson telemetry file(s) "
            f"(e.g. {', '.join(sample)}) will be silently ignored. {hint}",
            file=sys.stderr,
        )


def load_session_model(
    session_dir: str, cache_dir: Optional[str] = None, verbose: bool = False
):
    """Parse + build, with a pickle cache keyed by session file signature.

    Returns a SessionModel (import cycle avoided by importing build lazily).
    """
    from .build import build_session_model

    session_dir = os.path.normpath(session_dir)
    check_ndjson_exports(session_dir)  # postcard trap: warn before parsing
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{os.path.basename(session_dir)}-{_cache_key(session_dir)}.pkl"
        )
        if os.path.exists(cache_path):
            if verbose:
                print(f"[hwsim] loading cached model {cache_path}", file=sys.stderr)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    t0 = time.monotonic()
    raw = parse_session(session_dir, verbose=verbose)
    model = build_session_model(raw, verbose=verbose)
    if verbose:
        print(
            f"[hwsim] parse+build total {time.monotonic() - t0:.1f}s", file=sys.stderr
        )
    if cache_path:
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
        if verbose:
            print(f"[hwsim] cached model -> {cache_path}", file=sys.stderr)
    return model

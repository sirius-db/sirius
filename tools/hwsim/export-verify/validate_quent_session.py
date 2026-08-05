#!/usr/bin/env python3
"""Validate a quent telemetry session directory against the analyzer contract.

Contract source: tools/hwsim/export-verify/SCHEMA.md (derived from the Rust
analyzer's deserialization types at rust/crates/telemetry/model/src/ and the
pinned quent crates). Stdlib only.

Checks:
  - file layout (uuid-named session dir, model.qmi, one .ndjson per entity dir)
  - per-line JSON envelope (id / timestamp / data)
  - per-event required fields + types + enum values
  - FSM mechanics: per-id seq contiguity from 0, entry state, legal transitions,
    terminal Exit position, per-id timestamp monotonicity in seq order
  - id-graph integrity (every referenced uuid resolves to the right entity kind)
  - resource-tree integrity (parent_group_id chains terminate at the engine)
  - timestamp sanity (plausible epoch-ns range)
  - --simulated: hwsim exporter conventions (engine name, hwsim.* attributes,
    @knob query-label suffix, v7 uuids)

Exit code: 0 = no errors (warnings allowed), 1 = errors found, 2 = bad usage.

Usage:
  validate_quent_session.py <session_dir> [--simulated] [--allow-legacy]
                            [--max-report N] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid as uuidlib
from collections import defaultdict

NIL_UUID = "00000000-0000-0000-0000-000000000000"

# Plausible epoch-ns range: 2020-01-01 .. 2040-01-01.
TS_MIN = 1_577_836_800_000_000_000
TS_MAX = 2_208_988_800_000_000_000

# ---------------------------------------------------------------------------
# Schema tables (see SCHEMA.md section 4)
# ---------------------------------------------------------------------------

PLAIN_ENTITIES = {
    "engine": {"Init", "Exit"},
    "worker": {"Init", "Exit"},
    "query_group": {"Declaration"},
    "gpu_device": {"Declaration"},
    "thread_group": {"Declaration"},
    "plan": {"Declaration"},
    "operator": {"Declaration", "Statistics"},
    "port": {"Declaration", "Statistics"},
}

# FSM entities: entry state, exit_from states, legal transitions.
FSM_ENTITIES = {
    "query": {
        "entry": "Init",
        "exit_from": {"Executing"},
        "transitions": {("Init", "Planning"), ("Planning", "Executing")},
    },
    "task": {
        "entry": "Created",
        "exit_from": {"Finalizing"},
        "transitions": {
            ("Created", "Queued"),
            ("Queued", "Routing"),
            ("Routing", "Queued"),
            ("Routing", "Reserving"),
            ("Queued", "Reserving"),
            ("Reserving", "Downgrading"),
            ("Reserving", "Preparing"),
            ("Downgrading", "Preparing"),
            ("Preparing", "Computing"),
            ("Computing", "Computing"),
            ("Computing", "Finalizing"),
            ("Created", "Finalizing"),
            ("Queued", "Finalizing"),
            ("Routing", "Finalizing"),
            ("Reserving", "Finalizing"),
            ("Downgrading", "Finalizing"),
            ("Preparing", "Finalizing"),
        },
    },
    "data_batch": {
        "entry": "Constructed",
        "exit_from": {"Destructed"},
        "transitions": {
            ("Constructed", "Stationary"),
            ("Stationary", "InTransit"),
            ("InTransit", "Stationary"),
            ("Stationary", "Stationary"),
            ("Stationary", "Destructed"),
        },
    },
    "batch_placement": {
        "entry": "BatchRegistered",
        "exit_from": {"BatchConsumed"},
        "transitions": {
            ("BatchRegistered", "BatchQueued"),
            ("BatchRegistered", "BatchPackaged"),
            ("BatchQueued", "BatchQueued"),
            ("BatchQueued", "BatchPackaged"),
            ("BatchPackaged", "BatchPackaged"),
            ("BatchPackaged", "BatchProcessing"),
            ("BatchProcessing", "BatchProcessing"),
            ("BatchProcessing", "BatchConsumed"),
            ("BatchPackaged", "BatchConsumed"),
            ("BatchQueued", "BatchConsumed"),
            # Not in the fsm! table but emitted by the real engine under memory
            # pressure (OOM-rescheduled task re-claims a batch mid-processing);
            # observed 44k times in the E2-lo-q9 pressured trace. The analyzer
            # never validates transitions (it time-orders events), so this is
            # reality drift, not a violation. See SCHEMA.md section 7.
            ("BatchProcessing", "BatchPackaged"),
        },
    },
    "io_request": {
        "entry": "Issued",
        "exit_from": {"Completed"},
        "transitions": {("Issued", "Completed")},
    },
}

# Resource FSMs: <Prefix>Initializing -> <Prefix>Operating -> <Prefix>Finalizing -> Exit
RESOURCE_ENTITIES = {
    "task_queue": "TaskQueue",
    "executor_thread": "ExecutorThread",
    "task_manager_loop_thread": "TaskManagerLoopThread",
    "memory": "Memory",
    "memory_tier": "MemoryTier",
    "channel": "Channel",
}
# Operating payload: capacity field name, or None for unit resources (payload null).
RESOURCE_OPERATING_CAPACITY = {
    "task_queue": "capacity_entries",
    "executor_thread": None,
    "task_manager_loop_thread": None,
    "memory": "capacity_bytes",
    "memory_tier": "capacity_bytes",
    "channel": "capacity_bytes",
}

ALL_ENTITIES = set(PLAIN_ENTITIES) | set(FSM_ENTITIES) | set(RESOURCE_ENTITIES)

# Field specs: name -> checker tag.
#   "u"    unsigned int      "i" int          "b" bool       "s" string
#   "U"    uuid string       "Un" uuid or nil "s?" str|null  "U?" uuid|null
# Usage fields are declared separately.
STATE_FIELDS = {
    ("query", "Init"): {"instance_name": "s", "query_group_id": "U"},
    ("query", "Planning"): {},
    ("query", "Executing"): {},
    ("task", "Created"): {"instance_name": "s", "pipeline_uuid": "U"},
    ("task", "Queued"): {},
    ("task", "Routing"): {"instance_name": "s", "preferred_device_id": "i"},
    ("task", "Reserving"): {
        "instance_name": "s", "requested_bytes": "u", "input_basis": "u",
        "peak_estimate": "u", "bytes_to_materialize": "u",
    },
    ("task", "Downgrading"): {
        "instance_name": "s", "shortfall_bytes": "u", "partial_bytes": "u",
    },
    ("task", "Preparing"): {
        "instance_name": "s", "origin_tier": "s", "target_tier": "s",
        "input_bytes": "u",
    },
    ("task", "Computing"): {
        "instance_name": "s", "current_operator_id": "u", "input_bytes": "u",
        "peak_allocated_bytes": "u", "input_rows": "u",
    },
    ("task", "Finalizing"): {
        "instance_name": "s", "success": "b", "output_rows": "u",
        "output_bytes": "u",
    },
    ("data_batch", "Constructed"): {
        "instance_name": "s", "data_batch_id": "u",
        "producer_pipeline_uuid": "U", "producer_task_uuid": "Un",
        "num_rows": "u", "num_columns": "u",
    },
    ("data_batch", "Stationary"): {},
    ("data_batch", "InTransit"): {},
    ("data_batch", "Destructed"): {},
    ("batch_placement", "BatchRegistered"): {
        "instance_name": "s", "batch_id": "u", "pipeline_uuid": "U",
        "port_uuid": "U", "origin": "s", "producer_task_uuid": "Un",
    },
    ("batch_placement", "BatchQueued"): {},
    ("batch_placement", "BatchPackaged"): {"instance_name": "s", "task_uuid": "U"},
    ("batch_placement", "BatchProcessing"): {"instance_name": "s", "task_uuid": "U"},
    ("batch_placement", "BatchConsumed"): {"instance_name": "s", "reason": "s"},
    ("io_request", "Issued"): {
        "instance_name": "s", "task_uuid": "Un", "pipeline_uuid": "U",
        "file_count": "u", "estimated_compressed_bytes": "u",
        "estimated_decoded_bytes": "u",
    },
    ("io_request", "Completed"): {
        "instance_name": "s", "bytes_read": "u", "read_time_ns": "u",
        "read_calls": "u", "rows": "u",
    },
}

# Fields added by WS9 — the only ones --allow-legacy makes optional.
WS9_FIELDS = {
    ("task", "Computing"): {"input_rows"},
    ("task", "Finalizing"): {"output_rows", "output_bytes"},
    ("data_batch", "Constructed"): {"producer_task_uuid", "num_rows", "num_columns"},
    ("batch_placement", "BatchRegistered"): {"producer_task_uuid"},
}

# usage name -> (referenced entity kind, capacity key or None or "any")
STATE_USAGES = {
    ("task", "Queued"): {"queue": ("task_queue", "capacity_entries")},
    ("task", "Routing"): {"manager_thread": ("task_manager_loop_thread", None)},
    ("task", "Reserving"): {"manager_thread": ("task_manager_loop_thread", None)},
    ("task", "Downgrading"): {"manager_thread": ("task_manager_loop_thread", None)},
    ("task", "Preparing"): {
        "executor_thread": ("executor_thread", None),
        "reservation": ("memory_tier", "capacity_bytes"),
    },
    ("task", "Computing"): {
        "executor_thread": ("executor_thread", None),
        "reservation": ("memory_tier", "capacity_bytes"),
    },
    ("data_batch", "Stationary"): {"memory": ("memory", "capacity_bytes")},
    ("data_batch", "InTransit"): {
        "source_memory": ("memory", "capacity_bytes"),
        "dest_memory": ("memory", "capacity_bytes"),
        "channel": ("channel", "capacity_bytes"),
    },
    ("batch_placement", "BatchRegistered"): {"tier": ("memory_tier", "capacity_bytes")},
    ("batch_placement", "BatchQueued"): {"tier": ("memory_tier", "capacity_bytes")},
    ("batch_placement", "BatchPackaged"): {"tier": ("memory_tier", "capacity_bytes")},
    ("batch_placement", "BatchProcessing"): {"tier": ("memory_tier", "capacity_bytes")},
}

ENUMS = {
    ("batch_placement", "BatchRegistered", "origin"): {
        "operator_output", "partition_output", "reschedule_intermediate",
    },
    ("batch_placement", "BatchConsumed", "reason"): {
        "processed", "task_failed", "query_end",
    },
}

TIER_NAME_RE = re.compile(r"^(GPU(-\d+)?|HOST|DISK)$")


def is_uuid(s):
    if not isinstance(s, str) or len(s) != 36:
        return False
    try:
        uuidlib.UUID(s)
        return True
    except ValueError:
        return False


class Reporter:
    def __init__(self, max_report, quiet):
        self.errors = 0
        self.warnings = 0
        self.max_report = max_report
        self.quiet = quiet
        self.counts = defaultdict(int)  # (severity, code) -> n

    def _emit(self, severity, code, where, msg):
        key = (severity, code)
        self.counts[key] += 1
        if severity == "ERROR":
            self.errors += 1
        else:
            self.warnings += 1
        if not self.quiet and self.counts[key] <= self.max_report:
            print(f"{severity} [{code}] {where}: {msg}")
            if self.counts[key] == self.max_report:
                print(f"  ... further [{code}] {severity}s suppressed")

    def error(self, code, where, msg):
        self._emit("ERROR", code, where, msg)

    def warn(self, code, where, msg):
        self._emit("WARN", code, where, msg)


class Validator:
    def __init__(self, session_dir, simulated=False, allow_legacy=False,
                 max_report=10, quiet=False):
        self.dir = os.path.abspath(session_dir)
        self.simulated = simulated
        self.allow_legacy = allow_legacy
        self.rep = Reporter(max_report, quiet)
        # id sets per entity kind (for reference resolution)
        self.ids = defaultdict(set)
        # deferred references: (ref_kind, uuid, where, nil_ok)
        self.refs = []
        # resource-tree: child group/resource id -> parent id
        self.parents = {}
        self.engine_ids = set()
        self.data_batch_numeric_ids = set()
        self.placement_batch_ids = []  # (batch_id, where)
        self.event_totals = defaultdict(int)
        self.engine_init = None  # (implementation dict, where)
        self.query_labels = []  # (label, where)
        self.non_v7_ids = 0
        self.v7_checked = 0

    # -- helpers -----------------------------------------------------------

    def ref(self, kind, value, where, nil_ok=False):
        self.refs.append((kind, value, where, nil_ok))

    def check_scalar(self, tag, val):
        if tag == "u":
            return isinstance(val, int) and not isinstance(val, bool) and val >= 0
        if tag == "i":
            return isinstance(val, int) and not isinstance(val, bool)
        if tag == "b":
            return isinstance(val, bool)
        if tag == "s":
            return isinstance(val, str)
        if tag == "s?":
            return val is None or isinstance(val, str)
        if tag == "U":
            return is_uuid(val)
        if tag == "Un":
            return is_uuid(val)  # nil is a valid uuid string
        if tag == "U?":
            return val is None or is_uuid(val)
        raise AssertionError(tag)

    def check_fields(self, entity, state, payload, where, ref_map=None):
        """Check required attribute fields for (entity, state)."""
        spec = STATE_FIELDS.get((entity, state))
        if spec is None:
            return
        legacy_optional = WS9_FIELDS.get((entity, state), set()) if self.allow_legacy else set()
        for name, tag in spec.items():
            if name not in payload:
                if name in legacy_optional:
                    self.rep.warn("legacy-missing-field", where,
                                  f"{state}.{name} absent (pre-WS9 trace; current "
                                  "Rust analyzer would truncate this stream)")
                    continue
                self.rep.error("missing-field", where, f"{state}.{name} is required")
                continue
            if not self.check_scalar(tag, payload[name]):
                self.rep.error("bad-type", where,
                               f"{state}.{name}={payload[name]!r} does not match {tag}")
        if ref_map:
            for name, (kind, nil_ok) in ref_map.items():
                v = payload.get(name)
                if is_uuid(v):
                    self.ref(kind, v, where, nil_ok=nil_ok)
        # enums
        for (ent, st, field), allowed in ENUMS.items():
            if ent == entity and st == state and field in payload:
                if payload[field] not in allowed:
                    self.rep.error("bad-enum", where,
                                   f"{state}.{field}={payload[field]!r} not in {sorted(allowed)}")

    def check_usages(self, entity, state, payload, where):
        for uname, (rkind, capkey) in STATE_USAGES.get((entity, state), {}).items():
            u = payload.get(uname)
            if u is None:
                self.rep.error("missing-usage", where, f"{state}.{uname} usage is required")
                continue
            if not isinstance(u, dict) or "resource_id" not in u or "capacity" not in u:
                self.rep.error("bad-usage", where,
                               f"{state}.{uname} must be {{resource_id, capacity}}")
                continue
            if not is_uuid(u["resource_id"]):
                self.rep.error("bad-usage", where, f"{state}.{uname}.resource_id not a uuid")
            else:
                self.ref(rkind, u["resource_id"], where)
            cap = u["capacity"]
            if capkey is None:
                if cap is not None:
                    self.rep.warn("unexpected-capacity", where,
                                  f"{state}.{uname}.capacity expected null, got {cap!r}")
            else:
                if not isinstance(cap, dict) or capkey not in cap or not (
                    isinstance(cap[capkey], int) and cap[capkey] >= 0
                ):
                    self.rep.error("bad-usage", where,
                                   f"{state}.{uname}.capacity must be {{{capkey}: u64}}")

    def note_id(self, kind, eid):
        self.ids[kind].add(eid)
        if len(eid) == 36 and eid[14] == "7":
            pass
        else:
            self.non_v7_ids += 1
        self.v7_checked += 1

    # -- layout ------------------------------------------------------------

    def check_layout(self):
        base = os.path.basename(self.dir.rstrip("/"))
        if not is_uuid(base):
            self.rep.error("session-dir-name", self.dir,
                           "session directory name must parse as a UUID "
                           "(sirius-telemetry-server discovery requirement)")
        qmi = os.path.join(self.dir, "model.qmi")
        if not os.path.isfile(qmi):
            self.rep.warn("missing-model-qmi", self.dir,
                          "model.qmi absent (required for `quent open` discovery; "
                          "sirius-telemetry-server does not need it)")
        else:
            try:
                with open(qmi) as f:
                    meta = json.load(f)
                name = meta.get("model", {}).get("name")
                if name != "Sirius":
                    self.rep.warn("model-qmi-name", qmi,
                                  f"model.name={name!r}, expected 'Sirius'")
            except (json.JSONDecodeError, OSError) as e:
                self.rep.error("bad-model-qmi", qmi, f"unreadable/invalid JSON: {e}")

        self.entity_files = {}
        for name in sorted(os.listdir(self.dir)):
            path = os.path.join(self.dir, name)
            if not os.path.isdir(path):
                if name != "model.qmi":
                    self.rep.warn("stray-file", path, "unexpected file in session dir")
                continue
            ndjson = sorted(
                f for f in os.listdir(path) if f.endswith(".ndjson")
            )
            if name not in ALL_ENTITIES:
                self.rep.warn("unknown-entity-dir", path,
                              "not in the current model; the importer ignores it")
                continue
            if len(ndjson) == 0:
                self.rep.error("no-ndjson", path, "entity dir has no .ndjson file "
                               "(importer error: 'no .ndjson file found')")
                continue
            if len(ndjson) > 1:
                self.rep.error("multiple-ndjson", path,
                               f"{len(ndjson)} .ndjson files; the importer reads only ONE "
                               "(first in readdir order) — silent data loss")
            self.entity_files[name] = os.path.join(path, ndjson[0])

        for required in ("engine",):
            if required not in self.entity_files:
                self.rep.error("missing-entity", self.dir,
                               f"'{required}/' stream is required (engine id discovery)")

    # -- per-entity passes ---------------------------------------------------

    def iter_lines(self, entity):
        path = self.entity_files.get(entity)
        if path is None:
            return
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    if i == 1:
                        continue  # tolerate empty file
                    continue
                where = f"{entity}/{os.path.basename(path)}:{i}"
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError as e:
                    self.rep.error("bad-json", where,
                                   f"unparseable line (analyzer would silently truncate "
                                   f"the stream here): {e}")
                    continue
                if not isinstance(ev, dict):
                    self.rep.error("bad-envelope", where, "line is not a JSON object")
                    continue
                eid, ts, data = ev.get("id"), ev.get("timestamp"), ev.get("data")
                if not is_uuid(eid):
                    self.rep.error("bad-envelope", where, f"id={eid!r} is not a uuid")
                    continue
                if not isinstance(ts, int) or isinstance(ts, bool):
                    self.rep.error("bad-envelope", where, f"timestamp={ts!r} is not an int")
                    continue
                if not (TS_MIN <= ts <= TS_MAX):
                    self.rep.error("bad-timestamp", where,
                                   f"timestamp {ts} outside plausible epoch-ns range "
                                   f"[{TS_MIN}, {TS_MAX}]")
                extra = set(ev) - {"id", "timestamp", "data"}
                if extra:
                    self.rep.warn("extra-envelope-keys", where, f"unexpected keys {sorted(extra)}")
                self.event_totals[entity] += 1
                yield where, eid, ts, data

    def scan_plain(self, entity):
        events = PLAIN_ENTITIES[entity]
        for where, eid, ts, data in self.iter_lines(entity):
            if not isinstance(data, dict) or len(data) != 1:
                self.rep.error("bad-event", where,
                               "plain entity data must be a single-key {Event: payload} object")
                continue
            (name, payload), = data.items()
            if name not in events:
                self.rep.error("unknown-event", where,
                               f"{entity} has no event {name!r} (allowed: {sorted(events)})")
                continue
            self.note_id(entity, eid)
            self.dispatch_plain(entity, name, payload, eid, where)

    def dispatch_plain(self, entity, name, payload, eid, where):
        p = payload if isinstance(payload, dict) else {}
        if entity == "engine":
            if name == "Init":
                self.engine_ids.add(eid)
                impl = p.get("implementation")
                if not isinstance(impl, dict):
                    self.rep.error("missing-field", where, "Init.implementation is required")
                else:
                    self.check_custom_attributes(impl.get("custom_attributes"), where)
                    if self.engine_init is None:
                        self.engine_init = (impl, where)
            return
        if entity == "worker":
            if name == "Init":
                self.check_fields_plain(
                    p, {"parent_engine_id": "U", "instance_name": "s"}, name, where)
                self.ref("engine", p.get("parent_engine_id"), where)
            return
        if entity == "query_group":
            self.check_fields_plain(p, {"instance_name": "s", "engine_id": "U"}, name, where)
            self.ref("engine", p.get("engine_id"), where)
            return
        if entity == "gpu_device":
            self.check_fields_plain(
                p, {"instance_name": "s", "parent_group_id": "U", "ordinal": "u"}, name, where)
            if is_uuid(p.get("parent_group_id")):
                self.parents[eid] = p["parent_group_id"]
                self.ref("group", p["parent_group_id"], where)
            return
        if entity == "thread_group":
            self.check_fields_plain(
                p, {"instance_name": "s", "parent_group_id": "U"}, name, where)
            if is_uuid(p.get("parent_group_id")):
                self.parents[eid] = p["parent_group_id"]
                self.ref("group", p["parent_group_id"], where)
            return
        if entity == "plan":
            if name != "Declaration":
                return
            self.check_fields_plain(p, {"instance_name": "s"}, name, where)
            parent = p.get("parent")
            if not isinstance(parent, dict):
                self.rep.error("missing-field", where, "Declaration.parent is required")
            else:
                qid, pid = parent.get("query_id"), parent.get("plan_id")
                if (qid is None) == (pid is None):
                    self.rep.error("bad-plan-parent", where,
                                   "exactly one of parent.query_id/parent.plan_id must be set")
                if qid is not None:
                    self.ref("query", qid, where)
                if pid is not None:
                    self.ref("plan", pid, where)
            edges = p.get("edges")
            if not isinstance(edges, list):
                self.rep.error("missing-field", where, "Declaration.edges must be a list")
            else:
                for e in edges:
                    if not isinstance(e, dict) or not is_uuid(e.get("source")) \
                            or not is_uuid(e.get("target")):
                        self.rep.error("bad-edge", where,
                                       "each edge must be {source: uuid, target: uuid}")
                        continue
                    self.ref("port", e["source"], where)
                    self.ref("port", e["target"], where)
            wid = p.get("worker_id")
            if wid is not None:
                if is_uuid(wid):
                    self.ref("worker", wid, where)
                else:
                    self.rep.error("bad-type", where, f"worker_id={wid!r} not a uuid/null")
            return
        if entity == "operator":
            if name != "Declaration":
                return
            self.check_fields_plain(
                p, {"plan_id": "U", "instance_name": "s", "type_name": "s"}, name, where)
            self.ref("plan", p.get("plan_id"), where)
            for oid in p.get("parent_operator_ids") or []:
                self.ref("operator", oid, where)
            self.check_custom_attributes(p.get("custom_attributes"), where)
            return
        if entity == "port":
            if name != "Declaration":
                return
            self.check_fields_plain(p, {"operator_id": "U", "instance_name": "s"}, name, where)
            self.ref("operator", p.get("operator_id"), where)
            return

    def check_fields_plain(self, payload, spec, event, where):
        for fname, tag in spec.items():
            if fname not in payload:
                self.rep.error("missing-field", where, f"{event}.{fname} is required")
            elif not self.check_scalar(tag, payload[fname]):
                self.rep.error("bad-type", where,
                               f"{event}.{fname}={payload[fname]!r} does not match {tag}")

    def check_custom_attributes(self, attrs, where):
        if attrs is None:
            self.rep.error("missing-field", where, "custom_attributes is required (may be [])")
            return
        if not isinstance(attrs, list):
            self.rep.error("bad-custom-attributes", where,
                           "custom_attributes must be ONE flat array of {key, value} "
                           "(not parallel typed arrays)")
            return
        for a in attrs:
            if not isinstance(a, dict) or not isinstance(a.get("key"), str) \
                    or "value" not in a:
                self.rep.error("bad-custom-attributes", where,
                               f"attribute {a!r} must be {{key: str, value: tagged|null}}")
                continue
            v = a["value"]
            if v is not None and not (isinstance(v, dict) and len(v) == 1):
                self.rep.error("bad-custom-attributes", where,
                               f"value of {a['key']!r} must be a single-key tagged object "
                               f"like {{\"I64\": 1}}, got {v!r}")

    # -- FSM entities --------------------------------------------------------

    def scan_fsm(self, entity, spec):
        # per-id: (next_seq, last_ts, last_state, closed)
        st = {}
        out_of_order = {}
        for where, eid, ts, data in self.iter_lines(entity):
            if not isinstance(data, dict) or "seq" not in data or "state" not in data:
                self.rep.error("bad-event", where,
                               "FSM entity data must be {seq, state}")
                continue
            seq, state = data["seq"], data["state"]
            if not isinstance(seq, int) or seq < 0:
                self.rep.error("bad-seq", where, f"seq={seq!r} must be a non-negative int")
                continue
            if isinstance(state, str):
                if state != "Exit":
                    self.rep.error("unknown-state", where,
                                   f"string state {state!r} (only \"Exit\" is legal)")
                    continue
                sname, payload = "Exit", None
            elif isinstance(state, dict) and len(state) == 1:
                (sname, payload), = state.items()
            else:
                self.rep.error("bad-event", where,
                               "state must be {StateName: payload} or \"Exit\"")
                continue

            rec = st.get(eid)
            if rec is None:
                self.note_id(entity, eid)
                if seq != 0:
                    # events may be buffered out of order across drain batches;
                    # collect and re-check at the end rather than assume file order
                    out_of_order.setdefault(eid, []).append((seq, ts, sname, where))
                    st[eid] = [0, None, None, False]
                    continue
                st[eid] = [1, ts, sname, sname == "Exit"]
                if sname != "Exit" and sname != spec["entry"]:
                    self.rep.error("bad-entry-state", where,
                                   f"first state {sname!r}, expected {spec['entry']!r}")
                if sname == "Exit":
                    self.rep.error("bad-entry-state", where, "FSM opens with Exit")
            else:
                if seq != rec[0]:
                    out_of_order.setdefault(eid, []).append((seq, ts, sname, where))
                    continue
                self.step(entity, spec, eid, rec, seq, ts, sname, where)

            if sname != "Exit":
                if entity in ("query", "task", "data_batch", "batch_placement", "io_request"):
                    self.check_state_payload(entity, sname, payload, eid, where)

        # replay any buffered out-of-order events in seq order
        for eid, pend in out_of_order.items():
            rec = st[eid]
            for seq, ts, sname, where in sorted(pend):
                if seq != rec[0]:
                    self.rep.error("seq-gap", where,
                                   f"id {eid}: seq {seq} but expected {rec[0]} "
                                   "(gap or duplicate)")
                    continue
                if rec[2] is None:
                    rec[0], rec[1], rec[2] = 1, ts, sname
                    rec[3] = sname == "Exit"
                    if sname != "Exit" and sname != spec["entry"]:
                        self.rep.error("bad-entry-state", where,
                                       f"first state {sname!r}, expected {spec['entry']!r}")
                else:
                    self.step(entity, spec, eid, rec, seq, ts, sname, where)

        unclosed = sum(1 for rec in st.values() if not rec[3])
        if unclosed:
            self.rep.warn("unclosed-fsm", entity,
                          f"{unclosed}/{len(st)} {entity} FSM instances lack a terminal "
                          "\"Exit\" (degraded but ingestible)")

    def step(self, entity, spec, eid, rec, seq, ts, sname, where):
        next_seq, last_ts, last_state, closed = rec
        if closed:
            self.rep.error("event-after-exit", where,
                           f"id {eid}: event seq={seq} after terminal Exit")
            return
        if last_ts is not None and ts < last_ts:
            self.rep.error("timestamp-regression", where,
                           f"id {eid}: seq {seq} timestamp {ts} < previous {last_ts}")
        if sname == "Exit":
            if last_state not in spec["exit_from"]:
                self.rep.error("bad-exit", where,
                               f"id {eid}: Exit after {last_state!r}; legal exit_from "
                               f"= {sorted(spec['exit_from'])}")
            rec[3] = True
        else:
            if (last_state, sname) not in spec["transitions"]:
                self.rep.error("illegal-transition", where,
                               f"id {eid}: {last_state} -> {sname} not in the FSM "
                               "transition table")
        rec[0] = seq + 1
        rec[1] = ts
        rec[2] = sname

    def check_state_payload(self, entity, sname, payload, eid, where):
        if (entity, sname) not in STATE_FIELDS:
            self.rep.error("unknown-state", where,
                           f"{entity} has no state {sname!r}")
            return
        p = payload if isinstance(payload, dict) else {}
        if payload is not None and not isinstance(payload, dict):
            self.rep.error("bad-event", where, f"{sname} payload must be an object")
            return
        self.check_fields(entity, sname, p, where)
        self.check_usages(entity, sname, p, where)

        # entity-specific reference & bookkeeping
        if entity == "query" and sname == "Init":
            self.ref("query_group", p.get("query_group_id"), where)
            if isinstance(p.get("instance_name"), str):
                self.query_labels.append((p["instance_name"], where))
        elif entity == "task" and sname == "Created":
            self.ref("operator", p.get("pipeline_uuid"), where)
        # NOTE: task.Preparing.origin_tier/target_tier are plain Strings in the
        # model and reality uses values beyond the memory_tier names (observed:
        # "SOURCE", "UNKNOWN" on scan tasks) — deliberately not restricted here.
        elif entity == "data_batch" and sname == "Constructed":
            self.ref("operator", p.get("producer_pipeline_uuid"), where)
            self.ref("task", p.get("producer_task_uuid"), where, nil_ok=True)
            if isinstance(p.get("data_batch_id"), int):
                self.data_batch_numeric_ids.add(p["data_batch_id"])
        elif entity == "batch_placement" and sname == "BatchRegistered":
            self.ref("operator", p.get("pipeline_uuid"), where)
            self.ref("port", p.get("port_uuid"), where)
            self.ref("task", p.get("producer_task_uuid"), where, nil_ok=True)
            if isinstance(p.get("batch_id"), int):
                self.placement_batch_ids.append((p["batch_id"], where))
        elif entity == "batch_placement" and sname in ("BatchPackaged", "BatchProcessing"):
            self.ref("task", p.get("task_uuid"), where)
        elif entity == "io_request" and sname == "Issued":
            self.ref("task", p.get("task_uuid"), where, nil_ok=True)
            self.ref("operator", p.get("pipeline_uuid"), where)

    # -- resource FSMs --------------------------------------------------------

    def scan_resource(self, entity, prefix):
        init_s = prefix + "Initializing"
        oper_s = prefix + "Operating"
        fin_s = prefix + "Finalizing"
        capkey = RESOURCE_OPERATING_CAPACITY[entity]
        spec = {
            "entry": init_s,
            "exit_from": {fin_s},
            "transitions": {(init_s, oper_s), (oper_s, oper_s), (oper_s, fin_s)},
        }
        st = {}
        for where, eid, ts, data in self.iter_lines(entity):
            if not isinstance(data, dict) or "seq" not in data or "state" not in data:
                self.rep.error("bad-event", where, "resource FSM data must be {seq, state}")
                continue
            seq, state = data["seq"], data["state"]
            if isinstance(state, str):
                sname, payload = state, None
                if sname != "Exit":
                    self.rep.error("unknown-state", where, f"string state {sname!r}")
                    continue
            elif isinstance(state, dict) and len(state) == 1:
                (sname, payload), = state.items()
            else:
                self.rep.error("bad-event", where, "state must be {StateName: payload} or \"Exit\"")
                continue
            if sname not in (init_s, oper_s, fin_s, "Exit"):
                self.rep.error("unknown-state", where,
                               f"{entity} has no state {sname!r}")
                continue

            if sname == init_s:
                p = payload if isinstance(payload, dict) else {}
                self.check_fields_plain(
                    p, {"instance_name": "s", "parent_group_id": "U",
                        "resource_type_name": "s"}, sname, where)
                rtn = p.get("resource_type_name")
                if rtn is not None and rtn != entity:
                    self.rep.error("bad-resource-type-name", where,
                                   f"resource_type_name={rtn!r} must be {entity!r} "
                                   "(hard analyzer ingest error otherwise)")
                if is_uuid(p.get("parent_group_id")):
                    self.parents[eid] = p["parent_group_id"]
                    self.ref("group", p["parent_group_id"], where)
                if entity == "channel":
                    for f in ("source_id", "target_id"):
                        if not is_uuid(p.get(f)):
                            self.rep.error("missing-field", where,
                                           f"{sname}.{f} is required (uuid)")
                        else:
                            self.ref("memory", p[f], where)
                if entity == "memory_tier":
                    iname = p.get("instance_name")
                    if isinstance(iname, str) and not TIER_NAME_RE.match(iname):
                        self.rep.error("bad-tier-name", where,
                                       f"memory_tier instance_name={iname!r} "
                                       "not GPU[-n]/HOST/DISK")
            elif sname == oper_s:
                if capkey is None:
                    if payload is not None:
                        self.rep.warn("unexpected-capacity", where,
                                      f"{sname} expected null payload, got {payload!r}")
                else:
                    if not isinstance(payload, dict) or capkey not in payload or not (
                        isinstance(payload[capkey], int) and payload[capkey] >= 0
                    ):
                        self.rep.error("bad-capacity", where,
                                       f"{sname} payload must be {{{capkey}: u64}}")

            rec = st.get(eid)
            if rec is None:
                self.note_id(entity, eid)
                st[eid] = [1, ts, sname, sname == "Exit"]
                if seq != 0:
                    self.rep.error("seq-gap", where, f"id {eid}: first seen seq={seq}, expected 0")
                if sname != init_s:
                    self.rep.error("bad-entry-state", where,
                                   f"first state {sname!r}, expected {init_s!r}")
            else:
                if seq != rec[0]:
                    self.rep.error("seq-gap", where,
                                   f"id {eid}: seq {seq} but expected {rec[0]}")
                    rec[0] = seq  # resync so one gap doesn't cascade
                self.step(entity, spec, eid, rec, seq, ts, sname, where)
        unclosed = sum(1 for rec in st.values() if not rec[3])
        if unclosed:
            self.rep.warn("unclosed-fsm", entity,
                          f"{unclosed}/{len(st)} {entity} resource FSMs lack a terminal "
                          "\"Exit\"")

    # -- cross-entity checks ---------------------------------------------------

    def resolve_refs(self):
        group_kinds = ("engine", "gpu_device", "thread_group", "worker", "query_group")
        group_ids = set()
        for k in group_kinds:
            group_ids |= self.ids[k]
        missing = defaultdict(int)
        first_where = {}
        for kind, val, where, nil_ok in self.refs:
            if val is None or not isinstance(val, str):
                continue  # field-level checks already reported this
            if val == NIL_UUID:
                if not nil_ok:
                    self.rep.error("nil-uuid", where, f"nil uuid not allowed for ref to {kind}")
                continue
            ok = val in group_ids if kind == "group" else val in self.ids[kind]
            if not ok:
                key = (kind, val)
                missing[key] += 1
                first_where.setdefault(key, where)
        for (kind, val), n in sorted(missing.items(), key=lambda kv: -kv[1]):
            self.rep.error("dangling-ref", first_where[(kind, val)],
                           f"uuid {val} does not resolve to any {kind} "
                           f"({n} reference(s))")

        # resource-tree: every parent chain terminates at an engine id
        for child, parent in self.parents.items():
            seen = set()
            cur = child
            while True:
                if cur in seen:
                    self.rep.error("parent-cycle", cur, "parent_group_id chain has a cycle")
                    break
                seen.add(cur)
                nxt = self.parents.get(cur)
                if nxt is None:
                    if cur not in self.engine_ids and cur != child:
                        # chain ended at a non-engine node with no declared parent
                        if cur not in group_ids:
                            self.rep.error("broken-parent-chain", child,
                                           f"chain ends at undeclared id {cur}")
                        elif cur not in self.engine_ids:
                            self.rep.error("broken-parent-chain", child,
                                           f"chain ends at {cur}, which is not the engine")
                    elif cur == child and child not in self.engine_ids:
                        self.rep.error("broken-parent-chain", child,
                                       "node has no resolvable parent and is not the engine")
                    break
                if nxt in self.engine_ids:
                    break
                cur = nxt

        # placement batch_id -> data_batch numeric id join
        if self.data_batch_numeric_ids or self.placement_batch_ids:
            miss = 0
            first = None
            for bid, where in self.placement_batch_ids:
                if bid not in self.data_batch_numeric_ids:
                    miss += 1
                    first = first or (bid, where)
            if miss:
                self.rep.warn("batch-id-join", first[1],
                              f"{miss} BatchRegistered.batch_id value(s) have no matching "
                              f"data_batch.Constructed.data_batch_id (first: {first[0]}) — "
                              "legal if batch events were partially disabled")

    # -- simulated conventions --------------------------------------------------

    def check_simulated(self):
        where = "engine/Init"
        if self.engine_init is None:
            self.rep.error("sim-no-engine-init", where, "no engine Init event found")
            return
        impl, where = self.engine_init
        name = impl.get("name")
        if name != "hwsim-sim":
            self.rep.error("sim-engine-name", where,
                           f"implementation.name={name!r}, expected 'hwsim-sim'")
        attrs = {}
        for a in impl.get("custom_attributes") or []:
            if isinstance(a, dict) and isinstance(a.get("key"), str):
                attrs[a["key"]] = a.get("value")

        def attr_scalar(key):
            v = attrs.get(key)
            if isinstance(v, dict) and len(v) == 1:
                return next(iter(v.values()))
            return None

        if attr_scalar("hwsim.simulated") != 1:
            self.rep.error("sim-attr", where,
                           "custom_attributes must contain hwsim.simulated = 1 "
                           f"(got {attrs.get('hwsim.simulated')!r})")
        for key in ("hwsim.source_session", "hwsim.source_query"):
            if not isinstance(attr_scalar(key), str):
                self.rep.error("sim-attr", where,
                               f"custom_attributes must contain string {key} "
                               f"(got {attrs.get(key)!r})")
        knobs = sorted(k for k in attrs if k.startswith("hwsim.knob."))
        print(f"  simulated: engine={name!r} knobs={knobs or '(none)'}")

        for label, lwhere in self.query_labels:
            if "@" not in label:
                self.rep.error("sim-query-label", lwhere,
                               f"query label {label!r} missing '@knob=value' suffix")
            else:
                suffix = label.split("@", 1)[1]
                # "@baseline" is WS17's documented all-knobs-default marker; a
                # physics-retimed export appends a ",physics" token (WS20).
                if suffix.endswith(",physics"):
                    suffix = suffix[: -len(",physics")]
                if suffix != "baseline" and "=" not in suffix:
                    self.rep.warn("sim-query-label", lwhere,
                                  f"query label {label!r} has '@' but no 'knob=value' "
                                  "suffix (nor '@baseline')")

        if self.non_v7_ids:
            self.rep.warn("sim-uuid-version", self.dir,
                          f"{self.non_v7_ids}/{self.v7_checked} entity ids are not UUIDv7 "
                          "(convention: fresh time-ordered v7 ids)")

    # -- driver ------------------------------------------------------------------

    def run(self):
        if not os.path.isdir(self.dir):
            print(f"ERROR: {self.dir} is not a directory", file=sys.stderr)
            return 2
        self.check_layout()
        for entity in PLAIN_ENTITIES:
            if entity in self.entity_files:
                self.scan_plain(entity)
        for entity, spec in FSM_ENTITIES.items():
            if entity in self.entity_files:
                self.scan_fsm(entity, spec)
        for entity, prefix in RESOURCE_ENTITIES.items():
            if entity in self.entity_files:
                self.scan_resource(entity, prefix)
        self.resolve_refs()
        if self.simulated:
            self.check_simulated()

        total = sum(self.event_totals.values())
        print(f"\n== {self.dir}")
        print(f"   events: {total} across {len(self.event_totals)} entity streams")
        for entity in sorted(self.event_totals):
            print(f"     {entity:26s} {self.event_totals[entity]}")
        print(f"   entities: " + ", ".join(
            f"{k}={len(v)}" for k, v in sorted(self.ids.items()) if v))
        print(f"   RESULT: {self.rep.errors} error(s), {self.rep.warnings} warning(s)"
              + (" [simulated mode]" if self.simulated else "")
              + (" [legacy allowed]" if self.allow_legacy else ""))
        return 1 if self.rep.errors else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", help="path to <output_dir>/<session-uuid>")
    ap.add_argument("--simulated", action="store_true",
                    help="also enforce hwsim-sim exporter conventions")
    ap.add_argument("--allow-legacy", action="store_true",
                    help="treat WS9-added fields as optional (pre-WS9 traces)")
    ap.add_argument("--max-report", type=int, default=10,
                    help="max diagnostics printed per (severity, code)")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()
    v = Validator(args.session_dir, simulated=args.simulated,
                  allow_legacy=args.allow_legacy, max_report=args.max_report,
                  quiet=args.quiet)
    sys.exit(v.run())


if __name__ == "__main__":
    main()

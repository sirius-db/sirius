#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report how a query was distributed across Sirius StarRocks compute nodes (CNs).

Reads the per-CN telemetry trees and answers one question: did every CN actually do work,
or did the query land on a subset of them?

    <dir>/<prefix><N>/telemetry/<run-uuid>/<record-type>/*.ndjson

Telemetry is buffered in memory and flushed at engine shutdown, and every engine start mints
a NEW run-uuid. A CN directory holding two run-uuids therefore holds two cluster lifetimes,
and a failed cold run looks exactly like a healthy one once their records are pooled. Mixing
them is how a distribution measurement gets contaminated, so this script analyses only the
NEWEST run-uuid per CN by default and shouts if it finds more than one (--all-runs to
override).

A CN that produced NO telemetry is not skipped. It is carried through every table, the
participation list and the balance verdict as an explicit zero, because "this CN never
appeared" is the single most important answer this tool can give -- dropping it silently
turns a 2-of-4 cluster into a "1.00x BALANCED" reading over the two survivors.

ON-DISK SCHEMA
--------------
Every line is one event with a fixed envelope:

    {"id": <entity-uuid>, "timestamp": <unix-ns>, "data": {...}}

`id` is the ENTITY that emitted the event, not an event id -- one entity emits many lines as
it changes state. Two payload shapes exist:

  * declaration entities (operator, plan, port, worker, query_group, gpu_device, thread_group):
        "data": {"Declaration": {...}}        # engine/worker use {"Init": ...} / {"Exit": ...}
  * FSM entities (query, task, data_batch, channel, batch_placement, memory, ...):
        "data": {"seq": <int>, "state": {"<StateName>": {...}}}
    whose terminal transition is the BARE STRING "Exit", not an object:
        "data": {"seq": 3, "state": "Exit"}

CONSEQUENCE: a row count is an EVENT count, not a unit of work. In the reference capture
`task` held 28,044 rows across 3,044 distinct task entities (9.2 rows each), and part of that
multiplier is scheduling churn -- 6,088 Queued events for 3,044 tasks means tasks are requeued.
Distinct entities are therefore the honest denominator for "how was the work placed", and are
what the balance verdict uses by default (--metric rows to switch).

Query attribution follows the reference graph. `query_id` as a literal field name appears on
exactly ONE record type -- `plan` -- so a top-level key scan finds almost nothing; the chain
below was measured at 100% coverage against a real capture:

    plan.Declaration.parent.query_id      -> query id (== the `query` entity's own id)
    operator.Declaration.plan_id          -> plan id
    port.Declaration.operator_id          -> operator id
    task.Created.pipeline_uuid            -> operator id
    data_batch.Constructed.producer_pipeline_uuid   -> operator id
    batch_placement.BatchRegistered.pipeline_uuid   -> operator id
    batch_placement.BatchPackaged.task_uuid         -> task id

Those names are NOT hard-coded. The graph is walked generically: any uuid-valued field is a
candidate edge, and an entity inherits a query when its resolvable references agree. That
generic walk was diffed against the explicit chain above over 88 queries / 11,544 entities and
produced identical counts. Engine-scoped entities (engine, worker, thread_group, memory,
task_queue, executor_thread, channel, ...) legitimately belong to no query and are reported as
unattributed rather than forced into one.

Everything else is schema-defensive: record types are discovered by listing directories, no
field name is required to exist, and a malformed line is counted rather than fatal.

Usage:
    cn-distribution.py [--dir experimental/starrocks] [--prefix .cn] [--json]
                       [--all-runs] [--metric entities|rows] [--headline task]
                       [--tolerance 1.5] [--full-uuid] [--no-attribution]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from collections import Counter, defaultdict

# Key names we accept as a direct query reference.
QUERY_KEY_RE = re.compile(r"(?i)^(query_id|queryid|qid|query)$|_query_id$")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Columns of the headline table, in order. Only those actually present are printed.
DEFAULT_COLUMNS = ["operator", "task", "data_batch", "query", "channel"]
# Cap on refs retained per entity, so a pathological plan cannot blow up memory.
MAX_REFS_PER_ENTITY = 64
MAX_RESOLVE_PASSES = 20


# --------------------------------------------------------------------------- scanning


def uuid_leaves(obj, trail):
    """Yield (trail, key, uuid) for every uuid-valued field.

    Only uuid-valued leaves matter, so the dotted path is assembled by the caller on a hit
    instead of being built for every key of every line.
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str):
                if len(val) == 36 and UUID_RE.match(val):
                    yield trail, key, val
            elif isinstance(val, dict):
                trail.append(key)
                yield from uuid_leaves(val, trail)
                trail.pop()
            elif isinstance(val, list):
                trail.append(key + "[]")
                for item in val:
                    yield from uuid_leaves(item, trail)
                trail.pop()
    elif isinstance(obj, list):
        trail.append("[]")
        for item in obj:
            yield from uuid_leaves(item, trail)
        trail.pop()


def uuid7_millis(name):
    """Decode the mint time of a uuidv7: its first 48 bits are a unix-ms timestamp."""
    if not UUID_RE.match(name) or name[14] != "7":
        return None
    try:
        return int(name[:8] + name[9:13], 16)
    except ValueError:
        return None


def fmt_epoch_ms(ms):
    if ms is None:
        return "-"
    try:
        return _dt.datetime.fromtimestamp(ms / 1000.0, _dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "?"


class RunScan:
    """Counts and reference graph for one <cn>/telemetry/<run-uuid> directory.

    `missing=True` marks a CN that produced no telemetry at all; it still occupies a row in
    every table so it cannot vanish from the verdict.
    """

    def __init__(self, cn, run_uuid, path, missing=False, reason=""):
        self.cn = cn
        self.run_uuid = run_uuid
        self.path = path
        self.missing = missing
        self.reason = reason
        self.rows = Counter()            # record type -> event lines
        self.entities = defaultdict(set)  # record type -> distinct entity ids
        self.parse_errors = Counter()    # record type -> malformed lines
        self.files = Counter()           # record type -> ndjson files
        self.query_key_paths = Counter()  # dotted path -> hits (diagnostic)
        self.refs = defaultdict(set)     # entity id -> referenced uuids
        self.entity_type = {}            # entity id -> record type
        self.direct_query = {}           # entity id -> query id (explicit field)
        self.sidecars = []               # non-directory entries, e.g. model.qmi
        self.ts_min = None
        self.ts_max = None
        self.unattributed = 0
        self.owner = {}

    # -- accessors used by the report; `missing` scans answer 0 for everything -----------
    def n_rows(self, rtype):
        return self.rows.get(rtype, 0)

    def n_entities(self, rtype):
        return len(self.entities.get(rtype, ()))

    def count(self, rtype, metric):
        return self.n_entities(rtype) if metric == "entities" else self.n_rows(rtype)

    def record_types(self):
        return sorted(set(self.rows) | set(self.entities))

    @property
    def started_ms(self):
        return uuid7_millis(self.run_uuid) if self.run_uuid else None

    @property
    def span_s(self):
        if self.ts_min is None or self.ts_max is None:
            return None
        return (self.ts_max - self.ts_min) / 1e9

    def scan(self):
        if self.missing:
            return self
        try:
            entries = sorted(os.listdir(self.path))
        except OSError as exc:
            print(f"cn-distribution: cannot read {self.path}: {exc}", file=sys.stderr)
            return self
        for entry in entries:
            full = os.path.join(self.path, entry)
            if not os.path.isdir(full):
                # model.qmi is a provenance sidecar file, not a record type.
                self.sidecars.append(entry)
                continue
            self._scan_type(entry, full)
        return self

    def _scan_type(self, rtype, dirpath):
        try:
            names = sorted(n for n in os.listdir(dirpath) if n.endswith(".ndjson"))
        except OSError:
            return
        self.files[rtype] = len(names)
        # Ensure the type shows up even with zero rows.
        self.rows.setdefault(rtype, 0)
        self.entities.setdefault(rtype, set())

        for name in names:
            try:
                handle = open(os.path.join(dirpath, name), "r", encoding="utf-8",
                              errors="replace")
            except OSError:
                continue
            with handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    self.rows[rtype] += 1
                    try:
                        obj = json.loads(line)
                    except (ValueError, RecursionError):
                        self.parse_errors[rtype] += 1
                        continue
                    if not isinstance(obj, dict):
                        self.parse_errors[rtype] += 1
                        continue
                    self._ingest(rtype, obj)

    def _ingest(self, rtype, obj):
        ent = obj.get("id")
        if isinstance(ent, str):
            self.entities[rtype].add(ent)
            self.entity_type.setdefault(ent, rtype)

        ts = obj.get("timestamp")
        if isinstance(ts, int):
            if self.ts_min is None or ts < self.ts_min:
                self.ts_min = ts
            if self.ts_max is None or ts > self.ts_max:
                self.ts_max = ts

        refs = self.refs[ent] if isinstance(ent, str) else None
        for trail, key, val in uuid_leaves(obj, []):
            if QUERY_KEY_RE.search(key):
                self.query_key_paths[".".join(trail + [key])] += 1
                if isinstance(ent, str):
                    self.direct_query.setdefault(ent, val)
            if refs is not None and val != ent and not (not trail and key == "id"):
                if len(refs) < MAX_REFS_PER_ENTITY:
                    refs.add(val)

    def resolve_queries(self):
        """Attribute each entity to a query by walking the reference graph to a fixpoint.

        Seeds are (a) every `query` entity -- its own id IS the query id -- and (b) any entity
        carrying an explicit query-id-like field. A query id that is only ever REFERENCED is
        also a seed: in a distributed run the `query` entity may live on the coordinator CN
        while this CN holds only the plan that points at it.
        """
        owner = {}
        for qid in self.entities.get("query", ()):
            owner[qid] = qid
        for ent, qid in self.direct_query.items():
            owner.setdefault(ent, qid)
        for qid in self.direct_query.values():
            owner.setdefault(qid, qid)

        pending = [e for e in self.refs if e not in owner]
        for _ in range(MAX_RESOLVE_PASSES):
            progressed = False
            still = []
            for ent in pending:
                cands = {owner[r] for r in self.refs[ent] if r in owner}
                if len(cands) == 1:
                    owner[ent] = next(iter(cands))
                    progressed = True
                else:
                    still.append(ent)
            pending = still
            if not progressed or not pending:
                break

        table = defaultdict(lambda: defaultdict(int))  # query id -> rtype -> entities
        attributed = 0
        for ent, rtype in self.entity_type.items():
            qid = owner.get(ent)
            if qid is None:
                continue
            table[qid][rtype] += 1
            attributed += 1
        self.owner = owner
        self.unattributed = len(self.entity_type) - attributed
        return table


# --------------------------------------------------------------------------- discovery


def newest_mtime(path):
    best = 0.0
    for root, _dirs, files in os.walk(path):
        try:
            best = max(best, os.path.getmtime(root))
        except OSError:
            pass
        for name in files:
            try:
                best = max(best, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                pass
    return best


def discover_cns(base, prefix):
    """Return [(cn_name, telemetry_dir_or_None)] sorted by trailing CN number."""
    found = []
    try:
        entries = os.listdir(base)
    except OSError as exc:
        print(f"cn-distribution: cannot list {base}: {exc}", file=sys.stderr)
        return found
    for entry in sorted(entries):
        if not entry.startswith(prefix):
            continue
        suffix = entry[len(prefix):]
        if not suffix.isdigit():
            continue
        cn_dir = os.path.join(base, entry)
        if not os.path.isdir(cn_dir):
            continue
        tele = os.path.join(cn_dir, "telemetry")
        found.append((entry, int(suffix), tele if os.path.isdir(tele) else None))
    found.sort(key=lambda t: t[1])
    return [(name, tele) for name, _num, tele in found]


def discover_runs(tele_dir):
    """Return run-uuid dir names, newest last.

    Prefers the uuidv7 mint time embedded in the directory name over mtime: telemetry is
    flushed at shutdown, so mtime records when a lifetime ENDED (and is rewritten by any
    copy or rsync), whereas the uuid records when it STARTED.
    """
    if not tele_dir:
        return []
    try:
        names = [n for n in os.listdir(tele_dir)
                 if os.path.isdir(os.path.join(tele_dir, n))]
    except OSError:
        return []
    if names and all(uuid7_millis(n) is not None for n in names):
        return sorted(names, key=lambda n: (uuid7_millis(n), n))
    return sorted(names, key=lambda n: (newest_mtime(os.path.join(tele_dir, n)), n))


# --------------------------------------------------------------------------- reporting


def ratio_str(values):
    """max/min across CNs, guarding the zero-denominator case."""
    if not values:
        return "n/a", None
    hi, lo = max(values), min(values)
    if lo == 0:
        zeros = sum(1 for v in values if v == 0)
        if hi == 0:
            return "n/a (all zero)", None
        return f"inf ({zeros} CN{'s' if zeros != 1 else ''} at zero)", float("inf")
    return f"{hi / lo:.2f}x", hi / lo


def abbrev(uuid, full):
    """Shorten a uuid but keep BOTH ends.

    uuidv7s minted in the same run share a long time-ordered prefix, so a prefix-only
    abbreviation renders distinct ids identically -- which would silently merge two queries
    into one row.
    """
    if full or not uuid or len(uuid) <= 21:
        return uuid or "-"
    return uuid[:13] + ".." + uuid[-6:]


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.dirname(here)  # experimental/starrocks

    ap = argparse.ArgumentParser(
        description="Per-CN telemetry distribution analyzer for Sirius StarRocks compute nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", default=default_dir,
                    help="directory holding the <prefix>N engine dirs (default: %(default)s)")
    ap.add_argument("--prefix", default=".cn", help="engine dir prefix (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--all-runs", action="store_true",
                    help="analyze every run-uuid per CN (default: newest only)")
    ap.add_argument("--metric", choices=("entities", "rows"), default="entities",
                    help="unit for share%% and the verdict: distinct entities (real units of "
                         "work) or raw event rows (default: %(default)s)")
    ap.add_argument("--headline", default="task",
                    help="record type used for the balance verdict (default: %(default)s)")
    ap.add_argument("--tolerance", type=float, default=1.5,
                    help="max/min below this is called BALANCED (default: %(default)s)")
    ap.add_argument("--full-uuid", action="store_true", help="print run-uuids unabbreviated")
    ap.add_argument("--no-attribution", action="store_true",
                    help="skip the per-query-id breakdown")
    args = ap.parse_args(argv)

    base = os.path.abspath(args.dir)
    cns = discover_cns(base, args.prefix)
    if not cns:
        print(f"cn-distribution: no {args.prefix}<N> directories under {base}")
        return 1

    scans = []
    multi_run = []
    for cn, tele in cns:
        runs = discover_runs(tele)
        if not runs:
            # Keep the CN in the report as an explicit zero -- see module docstring.
            reason = ("no telemetry/ directory" if tele is None
                      else "telemetry/ exists but holds no run directory")
            scans.append(RunScan(cn, None, None, missing=True, reason=reason))
            continue
        if len(runs) > 1:
            multi_run.append((cn, runs))
        chosen = runs if args.all_runs else runs[-1:]
        for run in chosen:
            scans.append(RunScan(cn, run, os.path.join(tele, run)).scan())

    live = [s for s in scans if not s.missing]
    dead = [s for s in scans if s.missing]

    per_query = {}
    if not args.no_attribution:
        for scan in live:
            per_query[(scan.cn, scan.run_uuid)] = scan.resolve_queries()

    if args.json:
        return emit_json(base, args, scans, multi_run, per_query)

    metric = args.metric
    unit = "distinct entities" if metric == "entities" else "event rows"

    print(f"cn-distribution: {base}   prefix={args.prefix}   "
          f"mode={'ALL runs' if args.all_runs else 'newest run per CN'}   metric={metric}")
    print()

    if not live:
        print(f"  found {len(cns)} CN dir(s) but NO telemetry run directories at all.")
        for s in dead:
            print(f"    {s.cn}: {s.reason}")
        print("  Nothing to analyze -- telemetry is flushed only at engine shutdown, so a")
        print("  cluster that is still running (or was killed mid-query) writes nothing.")
        return 1

    # ---- contamination warning ------------------------------------------------------
    if multi_run:
        bar = "!" * 88
        print(bar)
        print("!! CONTAMINATION WARNING -- more than one run-uuid found")
        for cn, runs in multi_run:
            print(f"!!   {cn} holds {len(runs)} run-uuids (that many cluster lifetimes):")
            prev = None
            for r in runs:
                ms = uuid7_millis(r)
                gap = ""
                if ms is not None and prev is not None:
                    gap = f"   (+{(ms - prev) / 1000.0:,.1f}s after previous)"
                prev = ms if ms is not None else prev
                print(f"!!     {r}  started {fmt_epoch_ms(ms)}{gap}")
        if not args.all_runs:
            print("!! Analyzing the NEWEST uuid per CN only. Re-run with --all-runs to see them")
            print("!! all, or clean first (with the cluster down):  rm -rf .cn*/telemetry/*")
        else:
            print("!! --all-runs given: rows below MIX cluster lifetimes and the verdict is NOT")
            print("!! a single-run distribution measurement.")
        print(bar)
        print()

    # ---- runs analyzed ---------------------------------------------------------------
    uw = max([len(abbrev(s.run_uuid, args.full_uuid)) for s in scans] + [8])
    cw = max([len(s.cn) for s in scans] + [4])

    print("RUNS ANALYZED")
    rhdr = (f"  {'CN':<{cw}}  {'run-uuid':<{uw}}  {'engine start (UTC)':<19}  "
            f"{'events span':>11}  {'ndjson':>7}")
    print(rhdr)
    print("  " + "-" * (len(rhdr) - 2))
    for s in scans:
        if s.missing:
            print(f"  {s.cn:<{cw}}  {'-':<{uw}}  {'-':<19}  {'-':>11}  {'-':>7}"
                  f"   <-- {s.reason}")
            continue
        span = f"{s.span_s:,.1f}s" if s.span_s is not None else "-"
        print(f"  {s.cn:<{cw}}  {abbrev(s.run_uuid, args.full_uuid):<{uw}}  "
              f"{fmt_epoch_ms(s.started_ms):<19}  {span:>11}  {sum(s.files.values()):>7,}")
    print()

    # ---- headline table --------------------------------------------------------------
    all_types = sorted({t for s in scans for t in s.record_types()})
    cols = [c for c in DEFAULT_COLUMNS if c in all_types]
    if args.headline in all_types and args.headline not in cols:
        cols.append(args.headline)
    headline = args.headline

    def emit_counts(title, getter):
        totals = [getter(s, headline) for s in scans]
        grand = sum(totals)
        hdr = (f"{'CN':<{cw}}  {'run-uuid':<{uw}}  "
               + "  ".join(f"{c:>11}" for c in cols)
               + f"  {'share%':>8}")
        print(title)
        print(hdr)
        print("-" * len(hdr))
        for s in scans:
            share = (getter(s, headline) / grand * 100) if grand else 0.0
            mark = "  <-- NO TELEMETRY" if s.missing else ""
            row = f"{s.cn:<{cw}}  {abbrev(s.run_uuid, args.full_uuid):<{uw}}  "
            row += "  ".join(f"{getter(s, c):>11,}" for c in cols)
            row += f"  {share:>7.1f}%"
            print(row + mark)
        print(f"{'TOTAL':<{cw}}  {'':<{uw}}  "
              + "  ".join(f"{sum(getter(s, c) for s in scans):>11,}" for c in cols))
        print()
        return hdr

    emit_counts(f"WORK DISTRIBUTION -- distinct entities (the real unit of work; "
                f"share% on '{headline}')",
                lambda s, c: s.n_entities(c))
    emit_counts("EVENT ROWS -- state transitions (one entity emits many; inflated vs above)",
                lambda s, c: s.n_rows(c))

    # ---- other record types ----------------------------------------------------------
    others = [t for t in all_types if t not in cols]
    if others:
        print(f"OTHER RECORD TYPES ({unit})")
        ow = max(len(t) for t in others)
        labels = [f"{s.cn}" if not args.all_runs else
                  f"{s.cn}/{(s.run_uuid or '-')[:8]}" for s in scans]
        lw = max(max(len(x) for x in labels), 10)
        line = f"{'record type':<{ow}}  " + "  ".join(f"{x:>{lw}}" for x in labels)
        print(line)
        print("-" * len(line))
        for t in others:
            print(f"{t:<{ow}}  "
                  + "  ".join(f"{s.count(t, metric):>{lw},}" for s in scans))
        print()

    # ---- parse errors ----------------------------------------------------------------
    errs = [(s, t, n) for s in scans for t, n in s.parse_errors.items() if n]
    if errs:
        print("MALFORMED LINES (counted, not fatal)")
        for scan, t, n in errs:
            print(f"  {scan.cn} {abbrev(scan.run_uuid, args.full_uuid)} {t}: {n}")
        print()

    # ---- the sharp signal ------------------------------------------------------------
    print("PARTICIPATION (zero query + zero channel records == this CN never joined the")
    print("query/exchange machinery at all)")
    for s in scans:
        q = s.count("query", metric)
        ch = s.count("channel", metric)
        if s.missing:
            verdict = f"NEVER STARTED / NEVER FLUSHED  <-- {s.reason}"
        elif q == 0 and ch == 0:
            verdict = "NEVER JOINED  <-- zero query AND zero channel records"
        elif q == 0:
            verdict = "no query records (channel present)"
        elif ch == 0:
            verdict = "no channel records (query present)"
        else:
            verdict = "participated"
        print(f"  {s.cn:<{cw}} {abbrev(s.run_uuid, args.full_uuid):<{uw}}  "
              f"query={q:<7,} channel={ch:<7,}  {verdict}")
    print()

    # ---- per-query breakdown ---------------------------------------------------------
    if not args.no_attribution:
        qids = sorted({q for tbl in per_query.values() for q in tbl})
        direct = sorted({p for s in scans for p in s.query_key_paths})
        if not qids:
            print("PER-QUERY BREAKDOWN: unavailable -- no query-id-like field and no `query`")
            print("  entities were found, so events cannot be attributed to a query id.")
            print()
        else:
            print(f"PER-QUERY BREAKDOWN (distinct entities attributed via the reference "
                  f"graph; {len(qids)} query id(s))")
            if direct:
                print(f"  explicit query-id field found at: {', '.join(direct)}")
            qcols = [c for c in cols if c != "query"]
            qw = max(len(abbrev(q, args.full_uuid)) for q in qids)
            line = (f"{'query id':<{qw}}  {'CN':<{cw}}  "
                    + "  ".join(f"{c:>11}" for c in qcols))
            print(line)
            print("-" * len(line))
            for qid in qids:
                for scan in live:
                    counts = per_query.get((scan.cn, scan.run_uuid), {}).get(qid)
                    if not counts:
                        continue
                    print(f"{abbrev(qid, args.full_uuid):<{qw}}  {scan.cn:<{cw}}  "
                          + "  ".join(f"{counts.get(c, 0):>11,}" for c in qcols))
            unattr = sum(s.unattributed for s in live)
            if unattr:
                print(f"  ({unattr} entities tied to no query -- engine/worker/thread level "
                      "entities exist outside any query)")
            print()

    # ---- verdict ---------------------------------------------------------------------
    print(f"VERDICT (max/min across {len(scans)} CNs)")
    for c in cols + others:
        ev = [s.n_entities(c) for s in scans]
        rv = [s.n_rows(c) for s in scans]
        etxt, _ = ratio_str(ev)
        rtxt, _ = ratio_str(rv)
        print(f"  {c:<24} entities {min(ev):>7,} .. {max(ev):<7,} -> {etxt:<26}"
              f" rows {min(rv):>8,} .. {max(rv):<8,} -> {rtxt}")

    hv = [s.count(headline, metric) for s in scans]
    txt, num = ratio_str(hv)
    zero = [s.cn for s in scans if s.count(headline, metric) == 0]
    print()
    if max(hv, default=0) == 0:
        call = f"NO DATA -- every CN reported zero '{headline}' {unit}"
    elif num == float("inf"):
        call = (f"IMBALANCED (inf) -- {len(zero)} of {len(scans)} CNs did ZERO "
                f"'{headline}' work: {', '.join(zero)}")
    elif num < args.tolerance:
        call = (f"BALANCED ({txt} spread on '{headline}' {unit} across {len(scans)} CNs)")
    else:
        call = f"IMBALANCED ({txt} on '{headline}' {unit} across {len(scans)} CNs)"
    print(f"  ==> {call}")
    if args.all_runs and multi_run:
        print("  NOTE: --all-runs mixes cluster lifetimes; this is not a single-run reading.")
    return 0


def emit_json(base, args, scans, multi_run, per_query):
    out = {
        "dir": base,
        "prefix": args.prefix,
        "mode": "all-runs" if args.all_runs else "newest-per-cn",
        "metric": args.metric,
        "headline": args.headline,
        "multi_run_warning": [
            {"cn": cn,
             "runs": [{"run_uuid": r, "started_utc": fmt_epoch_ms(uuid7_millis(r))}
                      for r in runs]}
            for cn, runs in multi_run],
        "cns": [],
    }
    for s in scans:
        tbl = per_query.get((s.cn, s.run_uuid), {})
        out["cns"].append({
            "cn": s.cn,
            "run_uuid": s.run_uuid,
            "missing": s.missing,
            "reason": s.reason or None,
            "started_utc": fmt_epoch_ms(s.started_ms),
            "events_span_s": s.span_s,
            "rows": dict(s.rows),
            "entities": {t: len(v) for t, v in s.entities.items()},
            "files": dict(s.files),
            "parse_errors": {k: v for k, v in s.parse_errors.items() if v},
            "sidecars": s.sidecars,
            "query_key_paths": dict(s.query_key_paths),
            "zero_query": s.count("query", args.metric) == 0,
            "zero_channel": s.count("channel", args.metric) == 0,
            "per_query": {q: dict(c) for q, c in tbl.items()},
            "unattributed_entities": s.unattributed,
        })
    types = sorted({t for s in scans for t in s.record_types()})
    verdict = {}
    for t in types:
        ev = [s.n_entities(t) for s in scans]
        rv = [s.n_rows(t) for s in scans]
        etxt, enum = ratio_str(ev)
        rtxt, rnum = ratio_str(rv)
        verdict[t] = {
            "entities": {"min": min(ev), "max": max(ev), "ratio": etxt,
                         "ratio_value": None if enum in (None, float("inf")) else enum},
            "rows": {"min": min(rv), "max": max(rv), "ratio": rtxt,
                     "ratio_value": None if rnum in (None, float("inf")) else rnum},
        }
    hv = [s.count(args.headline, args.metric) for s in scans]
    txt, num = ratio_str(hv)
    if max(hv, default=0) == 0:
        call = "NO_DATA"
    elif num == float("inf") or (num is not None and num >= args.tolerance):
        call = "IMBALANCED"
    else:
        call = "BALANCED"
    out["verdict"] = verdict
    out["balance"] = {
        "call": call,
        "ratio": txt,
        "metric": args.metric,
        "zero_cns": [s.cn for s in scans if s.count(args.headline, args.metric) == 0],
        "missing_cns": [s.cn for s in scans if s.missing],
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

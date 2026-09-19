#!/usr/bin/env python3
"""Per-query engine time and pipeline shape from a Sirius (Quent) telemetry directory.

The engine's own query window (`query/*.ndjson`, Executing -> Exit) is what the GPU actually
spent; `run-tpch.sh`'s engine_ms is the whole execute_substrait call (Substrait lowering +
planning + execution + Arrow conversion) and the transparent path's wall time adds DuckDB's
own parse/bind/optimize. A large gap between the two is time outside the engine — that is how
the SF100 lowering cost (plan-doc handoff, 2026-09-19) was found. The pipeline declarations
(`operator/*.ndjson`) give the physical plan shape, so two paths can be compared per query
without a log sink.

    scripts/telemetry-plans.py DIR [--json FILE] [--skip-values]

    DIR            one engine-instance directory: <telemetry root>/<instance uuid>/ (bench.sh
                   writes one root per run; the transparent path creates one instance per
                   shell process, the backend one per start)
    --json FILE    also write [[window_ms, [pipeline, ...]], ...] in query order
    --skip-values  drop the GPU_VALUES-only queries (the timestamp SELECTs of
                   run-tpch-duckdb.sh), so the transparent path lines up with the 22 queries

Plans are attributed to queries by time: a plan's first declaration lands between the query's
Executing and Exit events. Queries are printed in execution order.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--json")
    ap.add_argument("--skip-values", action="store_true")
    args = ap.parse_args()
    d = args.dir.rstrip("/")
    if not os.path.isdir(os.path.join(d, "query")):
        print(f"error: {d} has no query/ directory (pass the instance directory, not the telemetry root)", file=sys.stderr)
        return 1

    queries = {}
    for f in glob.glob(os.path.join(d, "query", "*.ndjson")):
        for line in open(f):
            o = json.loads(line)
            st = o["data"]["state"]
            q = queries.setdefault(o["id"], {})
            if isinstance(st, dict) and "Executing" in st:
                q["start"] = o["timestamp"]
            if st == "Exit":
                q["end"] = o["timestamp"]

    decls = defaultdict(list)
    for f in glob.glob(os.path.join(d, "operator", "*.ndjson")):
        for line in open(f):
            o = json.loads(line)
            dd = o["data"]
            if isinstance(dd, dict) and "Declaration" in dd:
                decl = dd["Declaration"]
                decls[decl["plan_id"]].append((o["timestamp"], decl["instance_name"]))

    ordered = sorted((q["start"], q.get("end"), qid) for qid, q in queries.items() if "start" in q)
    plans = sorted((min(t for t, _ in v), pid) for pid, v in decls.items())
    out = []
    for start, end, _ in ordered:
        mine = [pid for t, pid in plans if start <= t <= (end if end else start + 10**13)]
        pipes = [name for pid in mine for _, name in sorted(decls[pid])]
        if args.skip_values and len(pipes) == 1 and pipes[0].startswith("GPU_VALUES"):
            continue
        out.append(((end - start) / 1e6 if end else None, pipes))

    for i, (ms, pipes) in enumerate(out, 1):
        window = f"{ms:8.0f} ms" if ms is not None else "  (no Exit)"
        print(f"#{i:<3d} {window}  {len(pipes)} pipeline(s)")
        for p in pipes:
            print(f"        {p}")
    if args.json:
        json.dump(out, open(args.json, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

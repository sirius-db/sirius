#!/usr/bin/env python3
"""Join the FE's audit log onto a run-tpch.sh timings.csv.

    scripts/fe-audit.py --timings log/tpch/timings.csv [--audit .doris-fe/fe/log/fe.audit.log]
                        [--wait 30]

The FE writes one `|QueryId=…|Timestamp=…|…|Time(ms)=…|PlanTimesMs={…}|ScheduleTimesMs={…}|`
line per statement to fe.audit.log, a few seconds after the statement (the audit event queue
is flushed on a timer), so this runs after a round and waits for the lines to land. Each
query of timings.csv is matched to the SELECT audit line whose Timestamp (the statement's
start) falls inside the client's [start_ms, end_ms] window — the harness runs one statement
at a time, so the match is unique. The FE-side numbers are appended as columns:

    fe_ms          Time(ms): the FE's own end-to-end time (wall_ms minus the client process
                   and connection setup)
    plan_ms        PlanTimesMs.plan: Nereids planning
    schedule_ms    ScheduleTimesMs.schedule_time_ms: fragment assignment + the exec RPCs; on
                   the Sirius backend the phase-1 RPC (rpc1_ms) is where the engine runs
    rpc1_ms        ScheduleTimesMs.fragment_rpc_phase_1_time_ms
    cpu_ms         CpuTimeMS as reported by the BE (0 on the Sirius backend)
    peak_mem_bytes PeakMemoryBytes as reported by the BE (0 on the Sirius backend)
    scan_bytes     ScanBytes as reported by the BE (0 on the Sirius backend)
    scan_rows      ScanRows as reported by the BE (0 on the Sirius backend)

and query_id is filled in from the audit line when the backend log did not provide it
(the native BE). Unmatched queries keep `-` in every new column.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / ".doris-fe/fe/log/fe.audit.log"
NEW_COLUMNS = ["fe_ms", "plan_ms", "schedule_ms", "rpc1_ms", "cpu_ms", "peak_mem_bytes", "scan_bytes", "scan_rows"]
FIELD_RE = re.compile(r"\|(?P<key>[A-Za-z()]+)=(?P<value>[^|]*)")


def parse_audit(path: Path) -> list[dict[str, str]]:
    """Every SELECT query line of the audit log as a dict of its |key=value| fields plus
    `ts_ms` (Timestamp as local epoch milliseconds)."""
    events = []
    if not path.exists():
        return events
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if "|IsQuery=true|" not in line or "|StmtType=SELECT|" not in line:
                continue
            fields = {m.group("key"): m.group("value") for m in FIELD_RE.finditer(line)}
            try:
                ts = datetime.strptime(fields["Timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            except (KeyError, ValueError):
                continue
            fields["ts_ms"] = str(int(ts.timestamp() * 1000))
            events.append(fields)
    return events


def json_field(fields: dict[str, str], key: str, member: str) -> str:
    try:
        value = json.loads(fields.get(key, "") or "{}").get(member)
    except json.JSONDecodeError:
        return "-"
    return "-" if value is None else str(value)


def match(events: list[dict[str, str]], start_ms: int, end_ms: int, query_id: str) -> dict[str, str] | None:
    # The mysql client's own `select @@version_comment` / `select $$` on connect are SELECTs
    # in the same window, without a database; the harness's statement runs after `USE tpch`.
    hits = [e for e in events if start_ms <= int(e["ts_ms"]) <= end_ms and e.get("Db")]
    if not hits:
        return None
    by_id = [e for e in hits if e.get("QueryId") == query_id]
    if by_id:
        return by_id[0]
    return max(hits, key=lambda e: int(e["ts_ms"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timings", type=Path, required=True, help="run-tpch.sh --out/timings.csv (rewritten in place)")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help=f"the FE's audit log (default {DEFAULT_AUDIT})")
    parser.add_argument("--wait", type=float, default=30, help="seconds to wait for the audit lines to land (default 30)")
    args = parser.parse_args()

    with args.timings.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("fe-audit: nothing to join (empty timings)", file=sys.stderr)
        return 0
    if "start_ms" not in rows[0]:
        print("fe-audit: timings.csv has no start_ms/end_ms columns (older run-tpch.sh)", file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.wait
    while True:
        events = parse_audit(args.audit)
        matched = {
            row["query"]: match(events, int(row["start_ms"]), int(row["end_ms"]), row.get("query_id", "-"))
            for row in rows
        }
        missing = [q for q, e in matched.items() if e is None]
        if not missing or time.monotonic() >= deadline:
            break
        time.sleep(1)

    columns = list(rows[0].keys()) + [c for c in NEW_COLUMNS if c not in rows[0]]
    for row in rows:
        event = matched[row["query"]]
        if event is None:
            for column in NEW_COLUMNS:
                row.setdefault(column, "-")
            continue
        if row.get("query_id", "-") in ("", "-"):
            row["query_id"] = event.get("QueryId", "-")
        row["fe_ms"] = event.get("Time(ms)", "-")
        row["plan_ms"] = json_field(event, "PlanTimesMs", "plan")
        row["schedule_ms"] = json_field(event, "ScheduleTimesMs", "schedule_time_ms")
        row["rpc1_ms"] = json_field(event, "ScheduleTimesMs", "fragment_rpc_phase_1_time_ms")
        row["cpu_ms"] = event.get("CpuTimeMS", "-")
        row["peak_mem_bytes"] = event.get("PeakMemoryBytes", "-")
        row["scan_bytes"] = event.get("ScanBytes", "-")
        row["scan_rows"] = event.get("ScanRows", "-")
    with args.timings.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if missing:
        print(f"fe-audit: no audit line for {', '.join(missing)} (audit log {args.audit})", file=sys.stderr)
        return 1
    print(f"fe-audit: joined {len(rows)} querie(s) from {args.audit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

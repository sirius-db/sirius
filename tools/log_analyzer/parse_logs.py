#!/usr/bin/env python3
"""CLI entry point for the Sirius log analyzer.

Usage:
    python -m tools.log_analyzer.parse_logs <log_file> [--out <dir>]

    # or directly:
    python tools/log_analyzer/parse_logs.py <log_file> [--out <dir>]

Outputs to <out>/ (default ./log_analysis/<log_basename>/):
    _index.csv                 # one row per kept analytical query
    _summary.json              # totals, parser version, format-drift warnings
    _pipeline_aggregates.csv   # one row per (query, pipeline_id)
    <ts>/                      # one folder per kept query, named by QueryBegin ts
        query.sql
        query_meta.json
        pipeline_plan.json
        pipeline_plan.txt
        memory_reservations.csv
        task_inputs.csv
        task_outputs.csv
        memory_history.csv
        downgrades.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Support both `python -m tools.log_analyzer.parse_logs` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from tools.log_analyzer import aggregator, patterns, plan_parser, segmenter
    from tools.log_analyzer.metrics import (
        downgrade,
        memory_history,
        memory_reservation,
        task_input,
        task_output,
    )
    from tools.log_analyzer.validators import FormatWarnings
else:
    from . import aggregator, patterns, plan_parser, segmenter
    from .metrics import (
        downgrade,
        memory_history,
        memory_reservation,
        task_input,
        task_output,
    )
    from .validators import FormatWarnings


def _check_trace_level(lines):
    """Fail fast unless the log was captured with at least trace/debug enabled."""
    has_trace = any(patterns.TRACE_TAG in ln for ln in lines)
    has_debug = any(patterns.DEBUG_TAG in ln for ln in lines)
    if not (has_trace and has_debug):
        print(
            "ERROR: the logs were not capture with the trace log level",
            file=sys.stderr,
        )
        sys.exit(2)


def _ts_to_folder_name(ts: str) -> str:
    # "2026-05-20 14:25:02.368" -> "2026-05-20_14-25-02.368"
    return ts.replace(" ", "_").replace(":", "-")


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _extract_overview_text(lines):
    """Return the raw === Pipeline Overview === block text for archival."""
    out = []
    inside = False
    for line in lines:
        if patterns.PIPELINE_OVERVIEW_HEADER in line:
            inside = True
            out.append(line.rstrip("\n"))
            continue
        if inside:
            s = line.rstrip("\n")
            if not s.strip() or any(
                marker in s for marker in patterns.PIPELINE_OVERVIEW_END_MARKERS
            ):
                break
            out.append(s)
    return "\n".join(out) if out else ""


def _extract_pool_stats(lines) -> dict:
    """Extract host (pinned) and GPU device memory pool stats from boundary lines.

    Returns a dict with keys "host" and "gpu":
      - host: {"begin": {allocated_bytes, peak_bytes, free_blocks}, "end": {...}}
      - gpu: [{"device_id": N, "begin": {allocated_bytes, peak_bytes, reserved_bytes}, "end": {...}}, ...]
    """
    host: dict = {}
    gpu: dict = {}  # device_id -> {"begin": ..., "end": ...}

    for line in lines:
        if patterns.QUERY_BEGIN_ANCHOR in line or patterns.QUERY_END_ANCHOR in line:
            m = patterns.QUERY_POOL_RE.match(line)
            if m:
                key = "begin" if m.group("tag") == "QueryBegin" else "end"
                host[key] = {
                    "allocated_bytes": int(m.group("allocated_bytes")),
                    "peak_bytes": int(m.group("peak_bytes")),
                    "free_blocks": int(m.group("free_blocks")),
                }
        if patterns.GPU_POOL_ANCHOR in line:
            m = patterns.GPU_POOL_RE.match(line)
            if m:
                dev_id = int(m.group("gpu_id"))
                key = "begin" if m.group("tag") == "QueryBegin" else "end"
                if dev_id not in gpu:
                    gpu[dev_id] = {}
                gpu[dev_id][key] = {
                    "allocated_bytes": int(m.group("allocated_bytes")),
                    "peak_bytes": int(m.group("peak_bytes")),
                    "reserved_bytes": int(m.group("reserved_bytes")),
                }

    return {
        "host": host,
        "gpu": [
            {"device_id": dev_id, **dev_stats}
            for dev_id, dev_stats in sorted(gpu.items())
        ],
    }


def process_query(seg, out_dir: Path, warnings: FormatWarnings) -> dict:
    """Process a single query segment. Returns the index-row dict."""
    folder_name = _ts_to_folder_name(seg.begin_ts)
    qdir = out_dir / folder_name
    qdir.mkdir(parents=True, exist_ok=True)

    # query.sql
    (qdir / "query.sql").write_text(seg.sql + "\n")

    # metrics
    mr_rows = memory_reservation.parse(seg.lines, warnings)
    ti_rows = task_input.parse(seg.lines, warnings)
    to_rows = task_output.parse(seg.lines, warnings)
    mh_rows = memory_history.parse(seg.lines, warnings)
    dg_rows = downgrade.parse(seg.lines, warnings)

    _write_csv(qdir / "memory_reservations.csv", memory_reservation.COLUMNS, mr_rows)
    _write_csv(qdir / "task_inputs.csv", task_input.COLUMNS, ti_rows)
    _write_csv(qdir / "task_outputs.csv", task_output.COLUMNS, to_rows)
    _write_csv(qdir / "memory_history.csv", memory_history.COLUMNS, mh_rows)
    _write_csv(qdir / "downgrades.csv", downgrade.COLUMNS, dg_rows)

    # plan
    plan = plan_parser.parse_plan(seg.lines, warnings)
    if plan is not None:
        (qdir / "pipeline_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    overview_text = _extract_overview_text(seg.lines)
    if overview_text:
        (qdir / "pipeline_plan.txt").write_text(overview_text + "\n")

    # meta
    duration_ms = None
    if seg.begin_ts and seg.end_ts:
        try:
            from datetime import datetime

            fmt = "%Y-%m-%d %H:%M:%S.%f"
            duration_ms = (
                datetime.strptime(seg.end_ts, fmt)
                - datetime.strptime(seg.begin_ts, fmt)
            ).total_seconds() * 1000.0
        except ValueError:
            duration_ms = None

    sql_preview = seg.sql[:120]
    pool_stats = _extract_pool_stats(seg.lines)
    meta = {
        "begin_ts": seg.begin_ts,
        "end_ts": seg.end_ts,
        "status": seg.status,
        "duration_ms": duration_ms,
        "sql_preview": sql_preview,
        "pool_stats": pool_stats,
        "counts": {
            "memory_reservations": len(mr_rows),
            "task_inputs": len(ti_rows),
            "task_outputs": len(to_rows),
            "memory_history": len(mh_rows),
            "downgrades": len(dg_rows),
            "pipelines_in_plan": plan["counts"]["pipelines"] if plan else 0,
        },
    }
    (qdir / "query_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Aggregate rows for this query (to be written cross-query later).
    agg_rows = aggregator.build_rows(
        query_begin_ts=seg.begin_ts,
        sql_preview=sql_preview,
        task_inputs=ti_rows,
        task_outputs=to_rows,
        memory_reservations=mr_rows,
        memory_history=mh_rows,
    )

    return {
        "index_row": {
            "query_begin_ts": seg.begin_ts,
            "query_end_ts": seg.end_ts,
            "status": seg.status,
            "duration_ms": duration_ms,
            "folder": folder_name,
            "sql_preview": sql_preview,
        },
        "agg_rows": agg_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Parse Sirius logs into per-query folders + aggregates."
    )
    p.add_argument("log_file", help="Path to the Sirius log file to parse.")
    p.add_argument(
        "--out",
        help="Output directory (default ./log_analysis/<log_basename>/).",
    )
    args = p.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_file():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else Path("log_analysis") / log_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("r", errors="replace") as f:
        lines = f.readlines()

    _check_trace_level(lines)

    segments = segmenter.segment(lines)
    warnings = FormatWarnings()

    index_rows = []
    agg_rows_all = []
    for seg in segments:
        result = process_query(seg, out_dir, warnings)
        index_rows.append(result["index_row"])
        agg_rows_all.extend(result["agg_rows"])

    # _index.csv
    _write_csv(
        out_dir / "_index.csv",
        [
            "query_begin_ts",
            "query_end_ts",
            "status",
            "duration_ms",
            "folder",
            "sql_preview",
        ],
        index_rows,
    )

    # _pipeline_aggregates.csv
    _write_csv(
        out_dir / "_pipeline_aggregates.csv",
        aggregator.AGG_COLUMNS,
        agg_rows_all,
    )

    # _summary.json
    log_ts_first = next(
        (patterns.TS_RE.match(ln).group(1) for ln in lines if patterns.TS_RE.match(ln)),
        None,
    )
    log_ts_last = None
    for ln in reversed(lines):
        m = patterns.TS_RE.match(ln)
        if m:
            log_ts_last = m.group(1)
            break

    def _query_list_entry(r):
        return {
            "query_begin_ts": r["query_begin_ts"],
            "folder": r["folder"],
            "sql_preview": r["sql_preview"],
        }

    complete_list = [
        _query_list_entry(r) for r in index_rows if r["status"] == "complete"
    ]
    incomplete_list = [
        _query_list_entry(r) for r in index_rows if r["status"] == "incomplete"
    ]

    summary = {
        "parser_shape_version": patterns.SHAPE_VERSION,
        "log_file": str(log_path.resolve()),
        "log_time_range": {"first_ts": log_ts_first, "last_ts": log_ts_last},
        "queries": {
            "kept_analytical": len(index_rows),
            "complete": len(complete_list),
            "incomplete": len(incomplete_list),
            "complete_queries": complete_list,
            "incomplete_queries": incomplete_list,
        },
        "format_warnings": warnings.to_dict(),
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"Parsed {len(index_rows)} analytical queries into {out_dir}. "
        f"Format warnings: {sum(warnings.drift_counts.values())}."
    )
    if warnings.has_drift:
        print(
            "WARNING: log format drift detected; "
            "the data parsing scripts or the skill may need to be updated. "
            "See _summary.json -> format_warnings for details.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

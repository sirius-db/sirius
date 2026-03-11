#!/usr/bin/env python3
"""
Parse Sirius GPU pipeline task logs and aggregate operator execution statistics.

Usage:
    python parse_pipeline_log.py <log_file> [--pretty]

Arguments:
    log_file   Path to the log file to parse.
    --pretty   Pretty-print the output as a terminal table instead of CSV.
"""

import re
import sys
from collections import OrderedDict

# Matches: Pipeline 1: operator TABLE_SCAN (id=0) prepare execution time: 20.25 ms
PREPARE_RE = re.compile(
    r"Pipeline (\d+): operator (\w+) \(id=(\d+)\) prepare execution time: ([\d.]+) ms"
)

# Matches: Pipeline 1: operator TABLE_SCAN (id=0) executing on 1 batches with num row: 79776599
# Also handles multi-batch: executing on 120 batches with num row: 0  0  1  0  ...
EXECUTING_RE = re.compile(
    r"Pipeline (\d+): operator (\w+) \(id=(\d+)\) executing on (\d+) batches with num row: ([\d\s]+)"
)

# Matches: Pipeline 1: operator TABLE_SCAN (id=0) produced 1 batches with num rows: 41980874  , execution time: 25.29 ms
# Also handles multi-batch: produced 120 batches with num rows: 0  0  1  0  ...  , execution time: 7.99 ms
PRODUCED_RE = re.compile(
    r"Pipeline (\d+): operator (\w+) \(id=(\d+)\) produced (\d+) batches with num rows: ([\d\s]+?)\s*, execution time: ([\d.]+) ms"
)

# Matches: Pipeline 1: operator TABLE_SCAN (id=0) sink execution time: 0.00 ms
SINK_RE = re.compile(
    r"Pipeline (\d+): operator (\w+) \(id=(\d+)\) sink execution time: ([\d.]+) ms"
)


def _make_op_entry(pipeline_id, op_id, op_type, phase):
    entry = {
        "pipeline_id": pipeline_id,
        "operator_id": op_id,
        "operator_type": op_type,
        "phase": phase,
        "num_tasks": 0,
        "total_exec_time_ms": 0.0,
    }
    # Row/batch counts are only meaningful for the execute phase
    if phase == "execute":
        entry["input_batches"] = 0
        entry["input_rows"] = 0
        entry["output_batches"] = 0
        entry["output_rows"] = 0
    return entry


def _process_line(line, operators):
    """Process one log line, updating operators in-place.

    Returns True if the line was a RESULT_COLLECTOR produced event (used as a
    run-boundary signal), False otherwise.
    """
    m = PREPARE_RE.search(line)
    if m:
        pipeline_id, op_type, op_id = int(m.group(1)), m.group(2), int(m.group(3))
        key = (pipeline_id, op_id, "prepare")
        if key not in operators:
            operators[key] = _make_op_entry(pipeline_id, op_id, op_type, "prepare")
        operators[key]["num_tasks"] += 1
        operators[key]["total_exec_time_ms"] += float(m.group(4))
        return False

    m = EXECUTING_RE.search(line)
    if m:
        pipeline_id, op_type, op_id = int(m.group(1)), m.group(2), int(m.group(3))
        key = (pipeline_id, op_id, "execute")
        if key not in operators:
            operators[key] = _make_op_entry(pipeline_id, op_id, op_type, "execute")
        operators[key]["input_batches"] += int(m.group(4))
        operators[key]["input_rows"] += sum(int(x) for x in m.group(5).split())
        return False

    m = PRODUCED_RE.search(line)
    if m:
        pipeline_id, op_type, op_id = int(m.group(1)), m.group(2), int(m.group(3))
        key = (pipeline_id, op_id, "execute")
        if key not in operators:
            operators[key] = _make_op_entry(pipeline_id, op_id, op_type, "execute")
        operators[key]["num_tasks"] += 1
        operators[key]["output_batches"] += int(m.group(4))
        operators[key]["output_rows"] += sum(int(x) for x in m.group(5).split())
        operators[key]["total_exec_time_ms"] += float(m.group(6))
        return op_type == "RESULT_COLLECTOR"

    m = SINK_RE.search(line)
    if m:
        pipeline_id, op_type, op_id = int(m.group(1)), m.group(2), int(m.group(3))
        key = (pipeline_id, op_id, "sink")
        if key not in operators:
            operators[key] = _make_op_entry(pipeline_id, op_id, op_type, "sink")
        operators[key]["num_tasks"] += 1
        operators[key]["total_exec_time_ms"] += float(m.group(4))
        return False

    return False


def parse_log(path):
    """Parse a single-run log. Returns a flat list of operator dicts."""
    return parse_log_runs(path, n_runs=1)[0]


def parse_log_runs(path, n_runs):
    """Parse a log that may contain multiple sequential query runs.

    Runs are delimited by RESULT_COLLECTOR 'produced' events: the log is
    expected to contain exactly n_runs * K such events, where K is the number
    of RESULT_COLLECTORs per query.  A new run begins after every K events.

    Returns a list of n_runs operator lists (each list is ordered by first
    appearance, same as parse_log).  Raises ValueError if the RESULT_COLLECTOR
    event count is not divisible by n_runs.
    """
    # First pass: count total RESULT_COLLECTOR 'produced' events
    rc_total = 0
    with open(path, "r") as f:
        for line in f:
            m = PRODUCED_RE.search(line)
            if m and m.group(2) == "RESULT_COLLECTOR":
                rc_total += 1

    if rc_total == 0:
        # No RESULT_COLLECTOR events — return one run with whatever we find
        return [_parse_into_list(path)]

    if rc_total % n_runs != 0:
        raise ValueError(
            f"{path}: found {rc_total} RESULT_COLLECTOR 'produced' events "
            f"but expected a multiple of n_runs={n_runs}"
        )

    rc_per_run = rc_total // n_runs

    # Second pass: accumulate operators, splitting at every rc_per_run RC events
    runs = []
    operators = OrderedDict()
    rc_seen = 0

    with open(path, "r") as f:
        for line in f:
            is_rc = _process_line(line, operators)
            if is_rc:
                rc_seen += 1
                if rc_seen % rc_per_run == 0:
                    runs.append(list(operators.values()))
                    operators = OrderedDict()

    # Catch any trailing events (shouldn't happen in well-formed logs)
    if operators:
        runs.append(list(operators.values()))

    return runs


def _parse_into_list(path):
    """Single-pass parse with no run splitting. Returns a list of operator dicts."""
    operators = OrderedDict()
    with open(path, "r") as f:
        for line in f:
            _process_line(line, operators)
    return list(operators.values())


COLUMNS = [
    ("pipeline_id", "Pipeline"),
    ("operator_id", "Op ID"),
    ("operator_type", "Operator Type"),
    ("phase", "Phase"),
    ("num_tasks", "Num Tasks"),
    ("input_batches", "Input Batches"),
    ("input_rows", "Input Rows"),
    ("output_batches", "Output Batches"),
    ("output_rows", "Output Rows"),
    ("total_exec_time_ms", "Exec Time (ms)"),
]


def format_value(key, value):
    if value is None:
        return ""
    if key == "total_exec_time_ms":
        return f"{value:.2f}"
    return str(value)


def print_csv(rows):
    headers = [col_name for _, col_name in COLUMNS]
    print(",".join(headers))
    for row in rows:
        values = [format_value(key, row.get(key)) for key, _ in COLUMNS]
        print(",".join(values))


def print_pretty(rows):
    headers = [col_name for _, col_name in COLUMNS]
    keys = [key for key, _ in COLUMNS]

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, key in enumerate(keys):
            col_widths[i] = max(col_widths[i], len(format_value(key, row.get(key))))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )

    print(sep)
    print(header_line)
    print(sep)

    prev_pipeline = None
    for row in rows:
        if prev_pipeline is not None and row["pipeline_id"] != prev_pipeline:
            print(sep)
        prev_pipeline = row["pipeline_id"]

        values = [
            format_value(key, row.get(key)).rjust(col_widths[i])
            for i, key in enumerate(keys)
        ]
        print("| " + " | ".join(values) + " |")

    print(sep)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    log_file = args[0]
    pretty = "--pretty" in args

    rows = parse_log(log_file)

    if not rows:
        print("No operator execution data found in the log file.", file=sys.stderr)
        sys.exit(1)

    if pretty:
        print_pretty(rows)
    else:
        print_csv(rows)


if __name__ == "__main__":
    main()

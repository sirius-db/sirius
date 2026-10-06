#!/usr/bin/env python3
"""Run AC-13 and verify concurrent query-window log segmentation.

From the repository root:
    python -m tools.log_analyzer.verify_query_lifecycle_segments
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Support module and direct-script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from tools.log_analyzer import patterns, segmenter
else:
    from . import patterns, segmenter


ROOT = Path(__file__).resolve().parent.parent.parent
CHILD_CASE = "query lifecycle slot watchdog child runner"
CHILD_VARIANT = "ac13_concurrent_logging"

MAIN_SQL = {
    "A1": ("SELECT sum(i) + 101 FROM ac13_a;", "A"),
    "A2": ("SELECT sum(i) + 102 FROM ac13_a;", "A"),
    "B1": ("SELECT sum(i) + 201 FROM ac13_b;", "B"),
    "B2": ("SELECT sum(i) + 202 FROM ac13_b;", "B"),
}
PENDING_SQL = "SELECT sum(i) + 301 FROM ac13_a;"

SqlKey = tuple[str, int, int]
WindowKey = tuple[str, int, int]
FullKey = tuple[str, int, int, int]


class VerifyError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def normalized(sql: str) -> str:
    return " ".join(sql.split())


def strict(regex, line: str, kind: str):
    match = regex.fullmatch(line)
    check(match is not None, f"malformed {kind} line: {line}")
    return match


@dataclass(frozen=True)
class WindowEvent:
    line: int
    event: str
    instance: str
    connection: int
    window: int
    query: int
    outcome: str

    @property
    def key(self) -> WindowKey:
        return (self.instance, self.connection, self.window)

    @property
    def full_key(self) -> FullKey:
        return (self.instance, self.connection, self.window, self.query)


def read_result(path: Path) -> None:
    check(path.is_file(), f"child result is missing: {path}")
    fields: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        name, separator, value = line.partition("\t")
        check(bool(separator), f"malformed child result line: {line!r}")
        check(name not in fields, f"duplicate child result field: {name}")
        fields[name] = value
    check(
        set(fields) == {"ERROR", "STARTED", "COMPLETED"},
        f"unexpected child result fields: {sorted(fields)}",
    )
    check(not fields["ERROR"], f"child scenario failed: {fields['ERROR']}")
    check(fields["STARTED"] == "1", "child workload did not start")
    check(fields["COMPLETED"] == "1", "child workload did not complete")


def read_logs(log_dir: Path) -> tuple[list[str], list[Path]]:
    check(log_dir.is_dir(), f"log directory is missing: {log_dir}")
    files = sorted(path for path in log_dir.glob("sirius*.log") if path.is_file())
    if not files:
        files = sorted(path for path in log_dir.iterdir() if path.is_file())
    check(bool(files), f"no log file found under {log_dir}")

    # Preserve chronology if spdlog rotates at midnight during the run.
    parts = []
    for path in files:
        lines = path.read_text(errors="replace").splitlines()
        first_timestamp = next(
            (match.group(1) for line in lines if (match := patterns.TS_RE.match(line))),
            "",
        )
        parts.append((first_timestamp, path.name, path, lines))
    parts.sort(key=lambda part: (part[0] == "", part[0], part[1]))
    lines = [line for _, _, _, part_lines in parts for line in part_lines]
    check(
        any(patterns.is_window_line(line) for line in lines),
        "logs contain no keyed window events",
    )
    return lines, [part[2] for part in parts]


def parse_raw(
    lines: list[str],
) -> tuple[
    dict[SqlKey, str],
    dict[SqlKey, int],
    dict[WindowKey, WindowEvent],
    dict[WindowKey, WindowEvent],
]:
    sql_by_key: dict[SqlKey, str] = {}
    sql_line_by_key: dict[SqlKey, int] = {}
    events: dict[WindowKey, list[WindowEvent]] = defaultdict(list)

    for line_number, line in enumerate(lines):
        if patterns.is_query_sql_line(line):
            match = strict(patterns.QUERY_SQL_RE, line, "query-SQL")
            if match.group("instance") is not None:
                key = (
                    match.group("instance"),
                    int(match.group("connection_id")),
                    int(match.group("query_id")),
                )
                sql = normalized(match.group("sql"))
                check(
                    key not in sql_by_key or sql_by_key[key] == sql,
                    f"SQL key {key} maps to multiple statements",
                )
                sql_by_key[key] = sql
                sql_line_by_key.setdefault(key, line_number)

        if patterns.is_window_line(line):
            match = strict(patterns.WINDOW_RE, line, "window")
            event = WindowEvent(
                line_number,
                match.group("event"),
                match.group("instance"),
                int(match.group("connection_id")),
                int(match.group("window_id")),
                int(match.group("query_id")),
                match.group("outcome"),
            )
            events[event.key].append(event)

        if patterns.is_query_begin_line(line) or patterns.is_query_end_line(line):
            strict(patterns.HOST_POOL_RE, line, "host-pool")
        if patterns.GPU_POOL_ANCHOR in line and (
            "QueryBegin" in line or "QueryEnd" in line
        ):
            strict(patterns.GPU_POOL_RE, line, "GPU-pool")

    check(bool(sql_by_key), "no keyed SQL records found")
    check(bool(events), "no strict window records found")

    begins: dict[WindowKey, WindowEvent] = {}
    ends: dict[WindowKey, WindowEvent] = {}
    for key, key_events in events.items():
        key_begins = [event for event in key_events if event.event == "begin"]
        key_ends = [event for event in key_events if event.event == "end"]
        check(len(key_begins) == 1, f"window {key} has {len(key_begins)} begins")
        check(len(key_ends) <= 1, f"window {key} has {len(key_ends)} ends")

        begin = key_begins[0]
        check(begin.outcome == "-", f"window {key} begin outcome is {begin.outcome}")
        begins[key] = begin
        if key_ends:
            end = key_ends[0]
            check(end.line > begin.line, f"window {key} ends before it begins")
            # segmenter pairs on instance/connection/window. Check query
            # independently so a malformed end cannot create a false green.
            check(
                end.query == begin.query,
                f"window {key} begin query={begin.query}, end query={end.query}",
            )
            ends[key] = end
    return sql_by_key, sql_line_by_key, begins, ends


def pool_key(match) -> FullKey:
    names = ("instance", "connection_id", "window_id", "query_id")
    check(
        all(match.group(name) is not None for name in names),
        f"pool boundary lacks a window key: {match.group(0)}",
    )
    return (
        match.group("instance"),
        int(match.group("connection_id")),
        int(match.group("window_id")),
        int(match.group("query_id")),
    )


def verify_pool_keys(query_segment, expected: FullKey) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    for line in query_segment.lines:
        if patterns.is_query_begin_line(line) or patterns.is_query_end_line(line):
            match = strict(patterns.HOST_POOL_RE, line, "host-pool")
            actual = pool_key(match)
            check(actual == expected, f"host-pool key {actual} != window {expected}")
            counts[("host", match.group("tag"))] += 1
        elif patterns.GPU_POOL_ANCHOR in line and (
            "QueryBegin" in line or "QueryEnd" in line
        ):
            match = strict(patterns.GPU_POOL_RE, line, "GPU-pool")
            actual = pool_key(match)
            check(actual == expected, f"GPU-pool key {actual} != window {expected}")
            counts[("gpu", match.group("tag"))] += 1

    for pool in ("host", "gpu"):
        check(
            counts[(pool, "QueryBegin")] > 0,
            f"{query_segment.sql!r} lacks a keyed {pool} begin",
        )
        if query_segment.status == "complete":
            check(
                counts[(pool, "QueryEnd")] > 0,
                f"{query_segment.sql!r} lacks a keyed {pool} end",
            )


def verify(lines: list[str]) -> tuple[str, int]:
    sql_by_key, sql_line_by_key, raw_begins, raw_ends = parse_raw(lines)
    parsed = segmenter.segment(lines)
    check(bool(parsed), "segmenter returned no analytical windows")

    allowed_sql = {normalized(sql) for sql, _ in MAIN_SQL.values()}
    allowed_sql.add(normalized(PENDING_SQL))
    by_sql = defaultdict(list)
    by_window = {}

    for query_segment in parsed:
        check(
            all(
                value is not None
                for value in (
                    query_segment.instance,
                    query_segment.connection_id,
                    query_segment.window_id,
                    query_segment.query_id,
                )
            ),
            f"segment lacks a window key: {query_segment}",
        )
        sql = normalized(query_segment.sql)
        check(sql in allowed_sql, f"unexpected window SQL: {sql!r}")
        window_key = (
            query_segment.instance,
            query_segment.connection_id,
            query_segment.window_id,
        )
        check(window_key not in by_window, f"duplicate segment {window_key}")
        by_window[window_key] = query_segment
        by_sql[sql].append(query_segment)

        begin = raw_begins.get(window_key)
        check(begin is not None, f"segment {window_key} has no raw begin")
        check(
            query_segment.begin_line_idx == begin.line,
            f"segment {window_key} selected the wrong begin",
        )
        check(
            bool(query_segment.lines) and query_segment.lines[0] == lines[begin.line],
            f"segment {window_key} slice does not start at its begin",
        )
        check(
            begin.query == query_segment.query_id,
            f"segment {window_key} query differs from its begin",
        )
        sql_key = (
            query_segment.instance,
            query_segment.connection_id,
            query_segment.query_id,
        )
        check(
            sql_by_key.get(sql_key) == sql,
            f"segment {window_key} joined SQL from the wrong query key",
        )

        end = raw_ends.get(window_key)
        if query_segment.status == "complete":
            check(end is not None, f"complete segment {window_key} has no raw end")
            check(
                query_segment.end_line_idx == end.line,
                f"segment {window_key} selected the wrong end",
            )
            check(
                query_segment.lines[-1] == lines[end.line],
                f"segment {window_key} slice does not end at its matching end",
            )
            check(
                query_segment.outcome == end.outcome,
                f"segment {window_key} outcome differs from its end",
            )
        else:
            check(end is None, f"incomplete segment {window_key} has a raw end")
            check(
                query_segment.end_line_idx is None and query_segment.outcome is None,
                f"incomplete segment {window_key} exposes end metadata",
            )
        verify_pool_keys(query_segment, begin.full_key)

    check(
        set(raw_begins) == set(by_window),
        "raw and parsed windows differ: "
        f"raw-only={sorted(set(raw_begins) - set(by_window))}, "
        f"parsed-only={sorted(set(by_window) - set(raw_begins))}",
    )
    complete_keys = {
        key
        for key, query_segment in by_window.items()
        if query_segment.status == "complete"
    }
    check(set(raw_ends) == complete_keys, "raw ends and complete segments differ")

    main = {}
    for name, (expected_sql, _) in MAIN_SQL.items():
        matches = by_sql[normalized(expected_sql)]
        check(len(matches) == 1, f"{name} has {len(matches)} segments")
        query_segment = matches[0]
        check(
            query_segment.status == "complete" and query_segment.outcome == "ok",
            f"{name} ended as {query_segment.status}/{query_segment.outcome}",
        )
        main[name] = query_segment

    instances = {query_segment.instance for query_segment in main.values()}
    check(len(instances) == 1, f"main windows span instances {sorted(instances)}")
    instance = next(iter(instances))
    connection_a = main["A1"].connection_id
    connection_b = main["B1"].connection_id
    check(main["A2"].connection_id == connection_a, "A1/A2 owner mismatch")
    check(main["B2"].connection_id == connection_b, "B1/B2 owner mismatch")
    check(connection_a != connection_b, "A and B use the same connection id")

    # The one-shot gate makes A1 the held window and B1 its waiter. Preserve the
    # concurrency shape that used to defeat positional segmentation: B1's keyed
    # SQL must be emitted while A1's execution window is still open.
    a1 = main["A1"]
    b1 = main["B1"]
    a1_window_key = (a1.instance, a1.connection_id, a1.window_id)
    b1_sql_key = (b1.instance, b1.connection_id, b1.query_id)
    check(b1_sql_key in sql_line_by_key, "B1 has no keyed SQL line index")
    check(a1_window_key in raw_ends, "A1 has no matching raw end")
    check(
        raw_begins[a1_window_key].line
        < sql_line_by_key[b1_sql_key]
        < raw_ends[a1_window_key].line,
        "B1 SQL was not emitted inside the held A1 window",
    )

    owners = {"A": connection_a, "B": connection_b}
    expected_records = (*MAIN_SQL.values(), (PENDING_SQL, "A"))
    for expected_sql, owner in expected_records:
        keys = [
            key
            for key, keyed_sql in sql_by_key.items()
            if keyed_sql == normalized(expected_sql)
        ]
        check(bool(keys), f"{expected_sql!r} has no keyed SQL record")
        for sql_instance, connection, _ in keys:
            check(sql_instance == instance, f"{expected_sql!r} instance mismatch")
            check(connection == owners[owner], f"{expected_sql!r} owner mismatch")

    pending = by_sql[normalized(PENDING_SQL)]
    check(len(pending) <= 1, f"pending query produced {len(pending)} windows")
    if pending:
        query_segment = pending[0]
        check(
            query_segment.instance == instance
            and query_segment.connection_id == connection_a,
            "pending window owner mismatch",
        )
        check(
            query_segment.status == "incomplete"
            or query_segment.outcome in {"ok", "unwind"},
            f"pending window ended as "
            f"{query_segment.status}/{query_segment.outcome}",
        )

    check(
        len(parsed) == len(MAIN_SQL) + len(pending),
        f"unexpected segment count: {len(parsed)}",
    )
    return instance, len(pending)


def tail(text: str | bytes | None) -> str:
    if text is None:
        return "<empty>"
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return "\n".join(text.splitlines()[-30:]) or "<empty>"


def run_child(
    unittest: Path, config: Path, work_dir: Path, timeout: float
) -> tuple[list[str], list[Path]]:
    result = work_dir / "result.tsv"
    log_dir = result.parent / f"{result.stem}_logs"
    log_dir.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "SIRIUS_SLOT_WATCHDOG_VARIANT": CHILD_VARIANT,
            "SIRIUS_SLOT_WATCHDOG_OUTPUT": str(result),
            "SIRIUS_SLOT_WATCHDOG_CONFIG": str(config),
            "SIRIUS_CONFIG_FILE": str(config),
            "SIRIUS_LOG_BACKEND": "spdlog",
            "SIRIUS_LOG_LEVEL": "debug",
            "SIRIUS_LOG_DIR": str(log_dir),
        }
    )
    env.pop("SIRIUS_DISABLE", None)
    try:
        process = subprocess.run(
            [str(unittest), CHILD_CASE],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise VerifyError(
            f"child exceeded {timeout:g}s\nstdout:\n{tail(error.stdout)}\n"
            f"stderr:\n{tail(error.stderr)}"
        ) from error
    check(
        process.returncode == 0,
        f"child exited {process.returncode}\nstdout:\n{tail(process.stdout)}\n"
        f"stderr:\n{tail(process.stderr)}",
    )
    read_result(result)
    return read_logs(log_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AC-13 and verify keyed query-window segmentation."
    )
    parser.add_argument(
        "--unittest",
        type=Path,
        default=ROOT / "build/release/extension/sirius/test/cpp/sirius_unittest",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "test/cpp/integration/integration.yaml",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    unittest = args.unittest if args.unittest.is_absolute() else ROOT / args.unittest
    config = args.config if args.config.is_absolute() else ROOT / args.config
    if not unittest.is_file() or not os.access(unittest, os.X_OK):
        print(
            f"ERROR: unittest binary is missing or not executable: {unittest}",
            file=sys.stderr,
        )
        return 2
    if not config.is_file():
        print(f"ERROR: config is missing: {config}", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("ERROR: --timeout must be positive", file=sys.stderr)
        return 2

    work_dir = Path(tempfile.mkdtemp(prefix="sirius_ac13_"))
    passed = False
    try:
        lines, _ = run_child(unittest, config, work_dir, args.timeout)
        instance, pending_count = verify(lines)
        passed = True
        print(
            "PASS: AC-13 verified 4 main windows, "
            f"{pending_count} pending window(s), instance={instance}"
        )
        if args.keep_artifacts:
            print(f"Artifacts: {work_dir}")
        return 0
    except (OSError, VerifyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(f"Artifacts preserved at: {work_dir}", file=sys.stderr)
        return 1
    finally:
        if passed and not args.keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

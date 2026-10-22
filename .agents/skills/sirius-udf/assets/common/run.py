#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Compare CPU/reference/GPU outputs and benchmark only after GPU proof passes."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parent
STATS_SQL = (
    "SELECT executions, plan_fallbacks, runtime_fallbacks FROM sirius_execution_stats()"
)


def require_gpu(before, after, min_executions=1):
    delta = {name: after[name] - before[name] for name in before}
    if delta["executions"] < min_executions:
        raise AssertionError(
            "candidate did not execute on the GPU for every measured query"
        )
    if delta["plan_fallbacks"] or delta["runtime_fallbacks"]:
        raise AssertionError(f"candidate fell back: {delta}")
    return delta


def compare(reference, actual, label):
    if reference["types"] != actual["types"]:
        raise AssertionError(f"{label}: logical result types differ")
    if reference["rows"] != actual["rows"]:
        raise AssertionError(f"{label}: result rows/values/NULL masks differ")


def encode_cell(value, logical_type):
    if value is None:
        return None
    if isinstance(value, float):
        width = "f" if logical_type == "FLOAT" else "d"
        return {"ieee_bits": struct.pack("!" + width, value).hex()}
    return str(value)


def validation_log_path(database):
    path = database.parent.resolve() / "log"
    path.mkdir(mode=0o700, exist_ok=True)
    return path


@contextmanager
def private_validation_logs(database):
    previous = os.environ.get("SIRIUS_LOG_DIR")
    os.environ["SIRIUS_LOG_DIR"] = str(validation_log_path(database))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SIRIUS_LOG_DIR", None)
        else:
            os.environ["SIRIUS_LOG_DIR"] = previous


class PythonDuckDB:
    def __init__(self, database, extension):
        try:
            import duckdb
        except ImportError as error:
            raise RuntimeError(
                "install the Sirius-matching DuckDB Python package"
            ) from error
        if not callable(getattr(duckdb, "connect", None)):
            raise RuntimeError(
                "the imported duckdb is not a runnable DuckDB Python package"
            )

        self.connection = duckdb.connect(
            str(database), config={"allow_unsigned_extensions": "true"}
        )
        path = str(extension.resolve()).replace("'", "''")
        self.connection.execute(f"LOAD '{path}'")
        self.connection.execute("SET gpu_execution=false")

    def execute(self, query):
        self.connection.execute(query)

    def query(self, query):
        cursor = self.connection.execute(query)
        # Preserve NULLs and IEEE bits, including signed zero and NaN payloads.
        types = [str(description[1]) for description in cursor.description]
        rows = [
            [
                encode_cell(value, logical_type)
                for value, logical_type in zip(row, types)
            ]
            for row in cursor.fetchall()
        ]
        return {"types": types, "rows": rows}

    def snapshot(self):
        self.connection.execute("SET gpu_execution=false")
        try:
            values = self.connection.execute(STATS_SQL).fetchone()
            return dict(
                zip(("executions", "plan_fallbacks", "runtime_fallbacks"), values)
            )
        finally:
            self.connection.execute("SET gpu_execution=true")

    def close(self):
        self.connection.close()


def benchmark(adapter, query, warmups, iterations, gpu):
    adapter.execute(f"SET gpu_execution={'true' if gpu else 'false'}")
    samples = []
    before = adapter.snapshot() if gpu else None
    for iteration in range(warmups + iterations):
        started = time.perf_counter()
        adapter.query(query)
        elapsed = time.perf_counter() - started
        if iteration >= warmups:
            samples.append(elapsed)
    evidence = (
        require_gpu(before, adapter.snapshot(), warmups + iterations) if gpu else None
    )
    return {
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "gpu_counters": evidence,
    }


def run_python(args, database, setup, registration, reference, candidate):
    with private_validation_logs(database):
        return _run_python_queries(
            args, database, setup, registration, reference, candidate
        )


def _run_python_queries(args, database, setup, registration, reference, candidate):
    adapter = PythonDuckDB(database, args.extension)
    try:
        adapter.execute(setup)
        adapter.execute(registration)
        adapter.execute("CHECKPOINT")
        expected = adapter.query(reference)
        compare(expected, adapter.query(candidate), "CPU candidate")
        adapter.execute("SET enable_duckdb_fallback=false")
        adapter.execute("SET gpu_execution=true")
        before = adapter.snapshot()
        compare(expected, adapter.query(candidate), "GPU candidate")
        evidence = require_gpu(before, adapter.snapshot())
        return {
            "validation": "passed",
            "rows": len(expected["rows"]),
            "gpu_counters": evidence,
            "timing": "query wall time including result fetch",
            "cpu_baseline": "reference.sql",
            "cpu": benchmark(adapter, reference, args.warmups, args.iterations, False),
            "gpu": benchmark(adapter, candidate, args.warmups, args.iterations, True),
        }
    finally:
        adapter.close()


def validate_native_report(result, benchmark_queries):
    if not isinstance(result, dict) or result.get("validation") != "passed":
        raise RuntimeError("native host did not report passing validation")
    benchmark = result.get("gpu")
    if not isinstance(benchmark, dict):
        raise RuntimeError("native host omitted benchmark GPU execution evidence")
    for report, minimum in ((result, 1), (benchmark, benchmark_queries)):
        counters = report.get("gpu_counters")
        names = ("executions", "plan_fallbacks", "runtime_fallbacks")
        if not isinstance(counters, dict) or any(
            type(counters.get(name)) is not int for name in names
        ):
            raise RuntimeError("native host omitted valid GPU execution counters")
        before = {name: 0 for name in names}
        require_gpu(before, counters, minimum)
    return result


def run_native(args, database, setup, registration, reference, candidate):
    sql_paths = []
    for name, query in (
        ("setup", setup),
        ("registration", registration),
        ("reference", reference),
        ("candidate", candidate),
    ):
        path = database.parent / f"{name}.sql"
        path.write_text(query)
        sql_paths.append(str(path))
    # Use explicit argv; source SQL never enters a shell command.
    environment = os.environ.copy()
    environment["SIRIUS_LOG_DIR"] = str(validation_log_path(database))
    completed = subprocess.run(
        [
            str(args.host.resolve()),
            str(database),
            str(args.plugin.resolve()) if args.plugin else "-",
            *sql_paths,
            str(args.warmups),
            str(args.iterations),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"native validation failed ({completed.returncode}):\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("SIRIUS_UDF_RESULT="):
            result = json.loads(line.removeprefix("SIRIUS_UDF_RESULT="))
            return validate_native_report(result, args.warmups + args.iterations)
    raise RuntimeError("native host did not emit a validation result")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension", type=Path, help="SQL target: matching Sirius extension"
    )
    parser.add_argument(
        "--host",
        type=Path,
        help="matching generic udf_host; required for native targets",
    )
    parser.add_argument("--plugin", type=Path, help="native target: generated plugin")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.rows < 1 or args.warmups < 0 or args.iterations < 1:
        parser.error("rows/iterations must be positive and warmups nonnegative")
    target = json.loads((ROOT / "project.json").read_text())["target"]
    if target == "sql" and not args.extension and not args.host:
        parser.error("SQL target requires --extension or --host")
    if target != "sql" and (not args.host or not args.plugin):
        parser.error("native target requires --host and --plugin")
    for path in (args.extension, args.host, args.plugin):
        if path is not None and not path.is_file():
            parser.error(f"missing file: {path}")
    setup = (ROOT / "setup.sql").read_text().replace("@ROWS@", str(args.rows))
    registration = (ROOT / "registration.sql").read_text()
    reference = (ROOT / "reference.sql").read_text()
    candidate = (ROOT / "candidate.sql").read_text()
    with tempfile.TemporaryDirectory(prefix="sirius-udf-validation-") as directory:
        database = Path(directory) / "validation.db"
        result = (
            run_python(args, database, setup, registration, reference, candidate)
            if target == "sql" and args.host is None
            else run_native(args, database, setup, registration, reference, candidate)
        )
    result["target"] = target
    result["reference_basis"] = json.loads((ROOT / "project.json").read_text())[
        "reference"
    ]
    serialized = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()

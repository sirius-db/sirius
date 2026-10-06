# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""Read-only shared-runtime benchmark implementation for performance_test.py.

The supervisor never opens a database. Its child owns every connection and worker,
so killing a stuck child also disposes the complete shared runtime. No result rows
are retained between requests unless validation is enabled.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

from tpch_stream_permutations import stream_order


FIELDS = (
    "engine",
    "stream_id",
    "query",
    "iteration",
    "sequence",
    "start_s",
    "end_s",
    "runtime_s",
    "status",
    "row_count",
    "error",
)


def check_arguments(args):
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    if args.concurrency == 1:
        if args.stream_order is not None:
            raise SystemExit("--stream-order requires --concurrency > 1")
        return
    if args.mode is not None:
        raise SystemExit(
            "--mode is not supported with concurrency > 1; use --stream-order"
        )
    if args.profile == "cold":
        raise SystemExit(
            "--profile cold resets shared caches per query; incompatible with concurrency > 1"
        )
    if args.precmd != "none" or args.nsys_profile:
        raise SystemExit(
            "per-query --precmd/--nsys-profile cannot measure concurrent clients in one runtime"
        )
    if args.pin_after_iteration != 0:
        raise SystemExit(
            "--pin-after-iteration is not supported with concurrency > 1; only an initial union pin is allowed"
        )
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    if (
        not math.isfinite(args.run_timeout)
        or args.run_timeout < 0
        or args.query_timeout < 0
    ):
        raise SystemExit("timeouts must be finite and nonnegative (0 disables)")
    if args.validation_memory_limit_mb <= 0:
        raise SystemExit("--validation-memory-limit-mb must be positive")
    from performance_test import parse_query_spec

    queries = parse_query_spec(args.queries)
    if (
        not queries
        or len(set(queries)) != len(queries)
        or any(q < 1 or q > 22 for q in queries)
    ):
        raise SystemExit(
            "--queries must select unique TPC-H query numbers between 1 and 22"
        )
    if args.pin_compression and args.pin not in ("gpu", "host"):
        raise SystemExit("--pin-compression requires --pin gpu or host")
    if args.pin == "parquet" and args.data_source != "parquet":
        raise SystemExit("--pin parquet requires --data-source parquet")


def orders_for(queries, concurrency, order):
    return [
        (
            list(queries)
            if order == "same"
            else [q for q in stream_order(i + 1) if q in queries]
        )
        for i in range(concurrency)
    ]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def input_identity(args):
    """Cheap provenance, not a full data hash. Remote objects cannot be fingerprinted here."""
    source = args.input
    if source.startswith("s3://"):
        return {"source": source, "verified": False}
    p = Path(source).resolve()
    files = [p] if p.is_file() else sorted(p.rglob("*.parquet"))
    manifest = [
        (
            str(f.relative_to(p) if f != p else f.name),
            f.stat().st_size,
            f.stat().st_mtime_ns,
        )
        for f in files
    ]
    return {
        "source": str(p),
        "file_count": len(files),
        "verified": True,
        "size_mtime_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
    }


def configured_limit(config_path):
    # Resolve the same documented startup search locations when no --config was given.
    import yaml

    candidates = (
        [Path(config_path)]
        if config_path
        else [Path("sirius.yaml"), Path.home() / ".sirius/sirius.yaml"]
    )
    for path in candidates:
        if path.is_file():
            doc = yaml.safe_load(path.read_text()) or {}
            sirius = doc.get("sirius", {})
            legacy = (
                sirius.get("executor", {})
                .get("scan_manager", {})
                .get("max_concurrent_queries", 1)
            )
            return sirius.get("max_concurrent_queries", legacy), str(path.resolve())
    return 1, None


def split_keyed_logs(directory):
    """Only attribute lines carrying a known identity; never infer temporal spans."""
    files = sorted((directory / "log_dir").glob("sirius*.log"))
    key_pattern = re.compile(r"instance=(\S+) connection=(\d+).*?query=(\d+)")
    pending, queries, windows = {}, {}, {}
    for path in files:
        with path.open(errors="replace") as source:
            for line in source:
                key = key_pattern.search(line)
                if not key:
                    continue
                identity = key.groups()
                connection = identity[:2]
                label = re.search(
                    r"SQL: CALL sirius_set_query_label\('(readonly_s\d+_q\d+_i\d+)'\)",
                    line,
                )
                if label:
                    pending[connection] = label[1]
                elif (
                    "QueryBegin:" in line and " SQL: " in line and connection in pending
                ):
                    queries[identity] = pending.pop(connection)
                window = re.search(r"window=(\d+)", line)
                if window and identity in queries:
                    windows[(identity[0], window[1])] = queries[identity]
    destinations = {}
    try:
        for path in files:
            with path.open(errors="replace") as source:
                for line in source:
                    key = key_pattern.search(line)
                    label = queries.get(key.groups()) if key else None
                    if label is None:
                        window = re.search(r"instance=(\S+).*?window=(\d+)", line)
                        if window:
                            label = windows.get(window.groups())
                    if label:
                        if label not in destinations:
                            folder = directory / "sirius" / "execution_logs"
                            folder.mkdir(parents=True, exist_ok=True)
                            destinations[label] = (folder / f"{label}.log").open("w")
                        destinations[label].write(line)
    finally:
        for output in destinations.values():
            output.close()
    write_json(
        directory / "log_index.json",
        {
            "queries": [
                {
                    "instance": k[0],
                    "connection": k[1],
                    "query_ordinal": k[2],
                    "label": v,
                }
                for k, v in queries.items()
            ],
            "windows": [
                {"instance": k[0], "window": k[1], "label": v}
                for k, v in windows.items()
            ],
            "note": "Execution logs contain only attributable lines. Shared originals retain all messages.",
        },
    )


def supervise(args, directory, config_path, validate, references, compression_plan):
    import performance_test as perf

    directory = Path(directory).resolve()
    queries = perf.parse_query_spec(args.queries)
    query_texts = perf.queries_for_scale_factor(args.scale_factor)
    limit, resolved_config = configured_limit(config_path)
    if resolved_config:
        # Freeze configuration for the child, including implicit startup discovery.
        import shutil

        destination = directory / "concurrent_config.yml"
        shutil.copyfile(resolved_config, destination)
        os.environ["SIRIUS_CONFIG_FILE"] = str(destination)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "schema_version": 2,
            "concurrency": args.concurrency,
            "sirius_max_concurrent_queries": limit,
            "stream_order": args.stream_order or "permuted",
            "stream_queries": orders_for(
                queries, args.concurrency, args.stream_order or "permuted"
            ),
            "executions_per_engine": args.concurrency * len(queries) * args.iterations,
            "pin_policy": "union-before-run" if args.pin != "none" else "none",
            "validation": validate,
            "validation_memory_limit_mb": args.validation_memory_limit_mb,
            "query_timeout_s": args.query_timeout,
            "run_timeout_s": args.run_timeout,
            "input_identity": input_identity(args),
            "query_sha256": {
                str(q): hashlib.sha256(query_texts[f"q{q}"].encode()).hexdigest()
                for q in queries
            },
            "runtime_file": "csv/concurrent_runtimes.csv",
            "status": "running",
            "effective_config": "concurrent_config.yml" if resolved_config else None,
            "cache_policy": "shared; no resets during execution; profile does not determine query order",
            "result_capture": (
                "in-memory-until-validation" if validate else "discard-after-fetch"
            ),
            "gpu_fallback": "disabled",
        }
    )
    write_json(metadata_path, metadata)
    spec = {
        "args": vars(args),
        "directory": str(directory),
        "validate": validate,
        "references": references,
        "compression_plan": compression_plan,
    }
    spec_path = directory / "concurrent_request.json"
    write_json(spec_path, spec)
    print(
        f"Concurrent clients: {args.concurrency}; configured Sirius admission: {limit}",
        flush=True,
    )
    termination = "completed"
    try:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(spec_path)],
            timeout=args.run_timeout or None,
            check=False,
        )
        status = child.returncode
        if status:
            termination = "nonzero_exit"
    except subprocess.TimeoutExpired:
        status = 124
        termination = "watchdog_timeout"
        print(
            f"Concurrent benchmark exceeded --run-timeout={args.run_timeout}s; shared-runtime child killed",
            flush=True,
        )
    except KeyboardInterrupt:
        status = 130
        termination = "interrupted"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "success" if status == 0 else "failed"
    metadata["exit_code"] = status
    metadata["termination"] = termination
    write_json(metadata_path, metadata)
    summary_path = directory / "concurrent_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update(
        {"status": metadata["status"], "exit_code": status, "termination": termination}
    )
    write_json(summary_path, summary)
    return status if status >= 0 else 1


class CaptureBudget:
    """Conservative Python-object accounting; admission of captures is atomic."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.lock = threading.Lock()

    def add(self, rows):
        # Allow for over-allocation of the accumulating result list, too.
        size = 2 * sys.getsizeof(rows)
        for row in rows:
            size += sys.getsizeof(row) + sum(sys.getsizeof(value) for value in row)
        with self.lock:
            if self.used + size > self.limit:
                raise MemoryError(
                    "validation capture budget exceeded; increase --validation-memory-limit-mb"
                )
            self.used += size
        return size

    def release(self, size):
        with self.lock:
            self.used -= size


def execute_request(connection, sql, timeout, capture, budget):
    expired = threading.Event()

    def interrupt():
        expired.set()
        connection.interrupt()

    timer = threading.Timer(timeout, interrupt) if timeout else None
    rows = [] if capture else None
    count = charged = 0
    start = time.perf_counter()
    if timer:
        timer.start()
    try:
        connection.execute(sql)
        while True:
            chunk = connection.fetchmany(2048)
            if not chunk:
                break
            count += len(chunk)
            if capture:
                charged += budget.add(chunk)
                rows.extend(chunk)
        end = time.perf_counter()
        if expired.is_set():
            raise TimeoutError("query timeout, including admission wait")
        return start, end, count, rows, "success", ""
    except Exception as exc:
        end = time.perf_counter()
        if charged:
            budget.release(charged)
        return (
            start,
            end,
            count,
            None,
            "timeout" if expired.is_set() else "error",
            str(exc),
        )
    finally:
        if timer:
            timer.cancel()
            # No late interrupt may reach the next query on this connection.
            timer.join()


def run_phase(
    root,
    args,
    engine,
    directory,
    queries,
    capture,
    budget,
    writer,
    output,
    s3_input=None,
):
    """Worker connections all derive from root; SQL never executes under the record lock."""
    import performance_test as perf

    query_texts = perf.queries_for_scale_factor(args.scale_factor)
    use_gpu = engine == "sirius"
    orders = orders_for(queries, args.concurrency, args.stream_order or "permuted")
    records, captures, setup_errors = [], {}, []
    lock = threading.Lock()
    stop = threading.Event()
    epoch = [None]
    barrier = threading.Barrier(
        args.concurrency + 1, action=lambda: epoch.__setitem__(0, time.perf_counter())
    )

    def worker(stream, order):
        con = None
        try:
            con = root.cursor()
            if use_gpu:
                con.execute("SET gpu_execution=true")
                con.execute("SET enable_duckdb_fallback=false")
                con.execute(
                    f"CALL sirius_set_session_label('readonly_stream_{stream}')"
                )
            elif args.duckdb_profiling:
                con.execute("PRAGMA enable_profiling='json'")
                con.execute("PRAGMA profiling_mode='detailed'")
            barrier.wait()
            sequence = 0
            for iteration in range(args.iterations):
                for query in order:
                    if stop.is_set():
                        return
                    key = (engine, stream, query, iteration)
                    label = f"readonly_s{stream}_q{query}_i{iteration}"
                    try:
                        if s3_input is not None:
                            s3_input.refresh_secret(con, use_gpu)
                        if use_gpu:
                            con.execute(f"CALL sirius_set_query_label('{label}')")
                        elif args.duckdb_profiling:
                            path = directory / engine / f"stream{stream}" / f"q{query}"
                            path.mkdir(parents=True, exist_ok=True)
                            sql_path = str(
                                path / f"profile_iter{iteration}.json"
                            ).replace("'", "''")
                            con.execute(f"PRAGMA profiling_output='{sql_path}'")
                        start, end, count, rows, status, error = execute_request(
                            con,
                            query_texts[f"q{query}"],
                            args.query_timeout,
                            capture,
                            budget,
                        )
                    except Exception as exc:
                        start = end = time.perf_counter()
                        count, rows, status, error = 0, None, "error", str(exc)
                    record = dict(
                        zip(
                            FIELDS,
                            (
                                engine,
                                stream,
                                f"q{query}",
                                iteration,
                                sequence,
                                start - epoch[0],
                                end - epoch[0],
                                end - start,
                                status,
                                count,
                                error,
                            ),
                        )
                    )
                    with lock:
                        records.append(record)
                        if rows is not None:
                            captures[key] = rows
                        writer.writerow(record)
                        output.flush()
                    sequence += 1
                    if (
                        "validation capture budget exceeded" in error
                        or "runtime is unavailable" in error.lower()
                    ):
                        stop.set()
        except threading.BrokenBarrierError:
            pass
        except BaseException as exc:
            with lock:
                setup_errors.append(f"stream {stream}: {exc}")
            stop.set()
            barrier.abort()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception as exc:
                    with lock:
                        setup_errors.append(f"stream {stream} close: {exc}")

    threads = [
        threading.Thread(target=worker, args=(i + 1, order))
        for i, order in enumerate(orders)
    ]
    try:
        for thread in threads:
            thread.start()
        # Barrier action records the epoch after all connection setup, before any request.
        barrier.wait()
    except (threading.BrokenBarrierError, RuntimeError) as exc:
        if not setup_errors:
            setup_errors.append(str(exc))
        stop.set()
        barrier.abort()
    finally:
        for thread in threads:
            if thread.ident is not None:
                thread.join()
    return records, captures, setup_errors


def latency_summary(values):
    if not values:
        return {"samples": 0}
    values = sorted(values)
    return {
        "samples": len(values),
        "median_s": (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2,
        "p95_s": values[math.ceil(len(values) * 0.95) - 1],
    }


def summarize(records, expected, setup_errors):
    successes = [r for r in records if r["status"] == "success"]
    elapsed = max((r["end_s"] for r in records), default=0)
    failed = len(records) - len(successes)
    complete = len(records) == expected and not failed and not setup_errors
    by_query = defaultdict(list)
    streams = {}
    for r in records:
        streams[r["stream_id"]] = max(streams.get(r["stream_id"], 0), r["end_s"])
        if r["status"] == "success":
            by_query[r["query"]].append(r["runtime_s"])
    return {
        "expected": expected,
        "submitted": len(records),
        "completed": len(successes),
        "failed": failed,
        "not_submitted": expected - len(records),
        "timeouts": sum(r["status"] == "timeout" for r in records),
        "complete": complete,
        "setup_errors": setup_errors,
        "elapsed_s": elapsed,
        "successful_queries_per_s": len(successes) / elapsed if elapsed else None,
        "latency": latency_summary([r["runtime_s"] for r in successes]),
        "per_query_latency": {q: latency_summary(v) for q, v in by_query.items()},
        "stream_completion_s": streams,
    }


def matching_rows(expected, actual):
    """Tolerance-aware multiset equality, preserving duplicate multiplicity.

    Remove exact matches first. Approximate rows are grouped by their non-float
    values and matched one-to-one, avoiding string-sort pairing of nearby floats.
    Like the existing runner this validates row contents, not ORDER BY semantics.
    """
    import performance_test as perf

    if len(expected) != len(actual):
        return False
    left, right = Counter(expected), Counter(actual)
    common = left & right
    left.subtract(common)
    right.subtract(common)

    def grouped(counter):
        groups = defaultdict(list)
        for row, count in counter.items():
            key = tuple(
                (i, value)
                for i, value in enumerate(row)
                if not isinstance(value, float)
            )
            groups[(len(row), key)].extend([row] * count)
        return {k: v for k, v in groups.items() if v}

    a, b = grouped(left), grouped(right)
    if a.keys() != b.keys():
        return False
    for key, rows in a.items():
        candidates = b[key]
        if len(rows) != len(candidates):
            return False
        # Augmenting paths give a maximum matching even when tolerance is non-transitive.
        matched = {}
        for i in range(len(rows)):
            pending, parent, seen = [i], {}, {i}
            found = None
            for x in pending:
                for j, row in enumerate(candidates):
                    if j in parent or not perf._rows_match(
                        rows[x], row, perf.VALIDATION_ABS_TOL
                    ):
                        continue
                    parent[j] = x
                    if j not in matched:
                        found = j
                        break
                    if matched[j] not in seen:
                        seen.add(matched[j])
                        pending.append(matched[j])
                if found is not None:
                    break
            if found is None:
                return False
            while found is not None:
                x = parent[found]
                previous = next((j for j, owner in matched.items() if owner == x), None)
                matched[found] = x
                found = previous
    return True


def validate_captures(captures, references, queries, external=None):
    """One verdict per captured execution; no query is re-executed for validation."""
    import performance_test as perf

    if external:
        references = {
            q: perf._load_result_file(str(Path(external) / f"q{q}" / "result.txt"))
            for q in queries
        }
    verdicts = []
    for (engine, stream, q, iteration), rows in sorted(captures.items()):
        ok = q in references and matching_rows(references[q], rows)
        verdicts.append(
            {
                "engine": engine,
                "stream_id": stream,
                "query": f"q{q}",
                "iteration": iteration,
                "status": "success" if ok else "mismatch",
            }
        )
    return verdicts


def check_references(path, metadata, queries):
    if not path:
        return
    directory = Path(path)
    for q in queries:
        if not (directory / f"q{q}" / "result.txt").is_file():
            raise ValueError(f"missing reference q{q}/result.txt")
    manifest = directory.parent / "metadata.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    if "input_identity" not in old or "query_sha256" not in old:
        print(
            "Reference provenance unavailable (legacy reference files); validation will compare rows only",
            flush=True,
        )
        return
    if old["input_identity"] != metadata["input_identity"]:
        raise ValueError("reference input identity differs from this benchmark")
    for q in queries:
        if old["query_sha256"].get(str(q)) != metadata["query_sha256"][str(q)]:
            raise ValueError(f"reference SQL differs for q{q}")


def run(spec):
    import performance_test as perf

    args = argparse.Namespace(**spec["args"])
    directory = Path(spec["directory"])
    perf.PIN_COMPRESSION_PLAN_DIR = spec["compression_plan"]
    queries = perf.parse_query_spec(args.queries)
    capture = spec["validate"]
    budget = CaptureBudget(args.validation_memory_limit_mb * 1024 * 1024)
    metadata = json.loads((directory / "metadata.json").read_text())
    check_references(spec["references"], metadata, queries)
    summaries, captures, references = {}, {}, {}
    # Only an initial drop is allowed; no worker ever resets shared cache or pins.
    perf.drop_os_cache(args.input, args.data_source)
    s3_input = perf.S3Input(args.input) if perf.is_s3_input(args.input) else None
    path = directory / "csv" / "concurrent_runtimes.csv"
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        output.flush()
        for engine, gpu in perf.resolve_engine_modes(args.engine):
            root = perf.open_connection(
                args.input,
                gpu_execution=gpu,
                data_source=args.data_source,
                s3_input=s3_input,
            )
            pinned = False
            try:
                if gpu and args.pin != "none":
                    if spec["compression_plan"]:
                        root.execute("SET pin_table_compression=true")
                        plan = spec["compression_plan"].replace("'", "''")
                        root.execute(
                            f"SET pin_table_input_compression_plan_dir='{plan}'"
                        )
                    perf._execute_multi(
                        root,
                        perf.emit_pin_all(
                            args.input,
                            args.data_source,
                            s3_input.pin_globs() if s3_input else None,
                            queries=queries,
                        ),
                    )
                    pinned = True
                records, phase_captures, errors = run_phase(
                    root,
                    args,
                    engine,
                    directory,
                    queries,
                    capture,
                    budget,
                    writer,
                    output,
                    s3_input=s3_input,
                )
                captures.update(phase_captures)
                if engine == "duckdb" and capture:
                    for (_, _, q, _), rows in sorted(phase_captures.items()):
                        references.setdefault(q, rows)
                summaries[engine] = summarize(
                    records, args.concurrency * args.iterations * len(queries), errors
                )
                write_json(
                    directory / "concurrent_summary.json",
                    {
                        "engines": summaries,
                        "validation": "pending" if capture else "not_requested",
                    },
                )
                print(f"[{engine}] {json.dumps(summaries[engine])}", flush=True)
            finally:
                try:
                    if pinned:
                        perf._execute_multi(root, perf.emit_unpin_all(queries))
                finally:
                    root.close()
    if input_identity(args) != metadata["input_identity"]:
        raise RuntimeError(
            "input files changed during the benchmark; results are invalid"
        )
    split_keyed_logs(directory)
    if capture:
        # Write actual measured rows after timing; unique paths retain every execution.
        for (engine, stream, q, iteration), rows in captures.items():
            result_path = directory / engine / f"stream{stream}" / f"q{q}"
            result_path.mkdir(parents=True, exist_ok=True)
            with (result_path / f"result_iter{iteration}.txt").open("w") as out:
                for row in rows:
                    out.write(repr(row) + "\n")
        # A canonical CPU result is convenient for later --duckdb-results reuse.
        for q, rows in references.items():
            perf._write_result(str(directory), "duckdb", q, rows)
    verdicts = (
        validate_captures(captures, references, queries, spec["references"])
        if capture
        else []
    )
    if capture:
        with (directory / "validation.csv").open("w", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=("engine", "stream_id", "query", "iteration", "status"),
            )
            writer.writeheader()
            writer.writerows(verdicts)
        print(
            f"Validation: {len(verdicts)} executions checked, "
            f"{sum(v['status'] != 'success' for v in verdicts)} mismatches",
            flush=True,
        )
    ok = all(v["complete"] for v in summaries.values()) and all(
        v["status"] == "success" for v in verdicts
    )
    write_json(
        directory / "concurrent_summary.json",
        {
            "engines": summaries,
            "validation": (
                {
                    "checked": len(verdicts),
                    "mismatches": sum(v["status"] != "success" for v in verdicts),
                }
                if capture
                else "not_requested"
            ),
            "captured_python_bytes": budget.used,
            "status": "success" if ok else "failed",
            "notes": "Client latency includes admission and result fetch; throughput includes client overhead. Shared logs remain intact; no serial log splitting.",
        },
    )
    return 0 if ok else 1


if __name__ == "__main__":
    specification = json.loads(Path(sys.argv[1]).read_text())
    try:
        raise SystemExit(run(specification))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

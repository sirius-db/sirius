#!/usr/bin/env python3
"""Run correctness-gated, end-to-end Sirius UDF query benchmarks.

Run inside the repository's locked Pixi environment. The external CMake project
in this directory builds the matching native host and UDF modules first.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
WORKLOADS = ("sql", "cudf", "cuda", "udfbench_q1")
AGGREGATE_WORKLOADS = ("sum_squares", "weighted_mean")
AGGREGATE_KINDS = ("global", "grouped_uniform", "grouped_skewed")
DEFAULT_SIZES = [100_000, 1_000_000, 10_000_000]
LABEL_EXPR = """CASE WHEN i%20=0 THEN NULL WHEN i%100=1 THEN ''
 WHEN i%100=2 THEN chr(9)||'  '||chr(9)
 WHEN i%100=3 THEN repeat(md5(i::VARCHAR),4)
 ELSE CASE WHEN i%3=0 THEN ' '||chr(9) WHEN i%3=1 THEN chr(9) ELSE '' END
 || substr(repeat(md5(i::VARCHAR)||upper(md5((i+19)::VARCHAR)),2),1,(hash(i)%113)::INTEGER)
 ||' a-'||chr(9)||'B_9 '
 ||CASE WHEN i%4=0 THEN chr(9)||' ' WHEN i%4=1 THEN ' ' WHEN i%4=2 THEN chr(9) ELSE '' END END"""
MIX_VECTOR = """WITH words AS (SELECT id,
 (xor(value,value>>17)::UHUGEINT*11400714785074694791::UHUGEINT)
 &18446744073709551615::UHUGEINT AS w FROM bench_input)
 SELECT id,xor((((w<<23)|(w>>41))&18446744073709551615::UHUGEINT)::UBIGINT,
 15111065706836454659::UBIGINT) AS result FROM words"""


def parse_args(argv=None, *, allow_aggregate=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS + (AGGREGATE_WORKLOADS if allow_aggregate else ()),
        default=list(WORKLOADS[:3]),
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--pin", action="store_true", help="Also time GPU-resident input"
    )
    parser.add_argument(
        "--projection", action="store_true", help="Also materialize every output row"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--q1-data", type=Path, help="Prepared UDFBench CSV from q1/prepare.py"
    )
    args = parser.parse_args(argv)
    if any(n <= 0 for n in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("--sizes must contain distinct positive row counts")
    if len(set(args.workloads)) != len(args.workloads):
        parser.error("--workloads must not contain duplicates")
    if args.warmups < 1 or args.repetitions < 5 or args.threads < 1:
        parser.error("require --warmups >= 1, --repetitions >= 5, and --threads >= 1")
    args.build_dir = args.build_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == REPO or REPO in args.output_dir.parents:
        parser.error("--output-dir must be outside the checkout")
    if args.output_dir.exists():
        parser.error("--output-dir must not exist; each run needs a fresh directory")
    host = args.build_dir / "bench_host"
    if not host.is_file() or not os.access(host, os.X_OK):
        parser.error(f"missing executable {host}; build the benchmark CMake project")
    for workload in args.workloads:
        if workload not in ("sql", *AGGREGATE_WORKLOADS):
            suffix = "q1" if workload == "udfbench_q1" else workload
            module = args.build_dir / f"benchmark_{suffix}.so"
            if not module.is_file():
                parser.error(f"missing UDF module {module}")
    if "udfbench_q1" in args.workloads:
        if args.q1_data is None or not args.q1_data.expanduser().is_file():
            parser.error("udfbench_q1 requires --q1-data from q1/prepare.py")
        args.q1_data = args.q1_data.expanduser().resolve()
    for path in (args.build_dir, args.output_dir):
        if "\n" in str(path) or "\r" in str(path):
            parser.error("paths must not contain line breaks")
    return args


def definitions(workload, rows, args):
    """Return deterministic input SQL and the original benchmark query variants."""
    if workload == "sql":
        value = (
            "CASE WHEN i%20=0 THEN NULL "
            "WHEN i%101=0 THEN '-9223372036854775808'::BIGINT "
            "WHEN i%103=0 THEN '9223372036854775807'::BIGINT "
            "ELSE (hash(i)%161)::BIGINT-64 END"
        )
        modes = [("native", "Vectorized CPU SQL")]
        bodies = {"native": "SELECT id,bounded_band(value) AS result FROM bench_input"}
        projection, aggregate = (
            "result",
            "count(result),min(result),max(result),sum(result)",
        )
    elif workload == "cudf":
        value = LABEL_EXPR
        modes = [
            ("native", "Generated CPU callback"),
            ("fast", "Typed vectorized CPU UDF"),
            ("vector", "Vectorized CPU SQL"),
        ]
        bodies = {
            "native": "SELECT id,normalize_label(value) AS result FROM bench_input",
            "fast": "SELECT id,bench_normalize_cpu(value) AS result FROM bench_input",
            "vector": "SELECT id,upper(replace(trim(value,chr(32)||chr(9)),'-','_')) AS result FROM bench_input",
        }
        projection, aggregate = "result", "count(result),sum(length(result))"
    elif workload == "cuda":
        value = (
            "CASE WHEN i%20=0 THEN NULL "
            "WHEN i%101=0 THEN 18446744073709551615::UBIGINT "
            "WHEN i%103=0 THEN 0::UBIGINT ELSE hash(i) END"
        )
        modes = [
            ("native", "Generated CPU callback"),
            ("fast", "Typed vectorized CPU UDF"),
            ("vector", "Vectorized CPU SQL (128-bit wrap)"),
        ]
        bodies = {
            "native": "SELECT id,mix_word_cuda(value) AS result FROM bench_input",
            "fast": "SELECT id,bench_mix_cpu(value) AS result FROM bench_input",
            "vector": MIX_VECTOR,
        }
        projection, aggregate = "result", "count(result),min(result),max(result)"
    elif workload == "udfbench_q1":
        path = str(args.q1_data).replace("'", "''")
        scan = (
            f"read_csv('{path}',header=true,columns={{'seq':'BIGINT','id':'VARCHAR',"
            "'value':'VARCHAR','expected_year':'BIGINT','expected_month':'BIGINT',"
            "'expected_day':'BIGINT'},nullstr='__NULL__')"
        )
        setup = (
            f"CREATE TABLE q1_oracle AS SELECT * FROM {scan};\n"
            "CREATE TABLE bench_input AS SELECT i::BIGINT AS id,"
            f"source.id AS artifact_id,source.value FROM range({rows}) input(i) "
            "JOIN q1_oracle source ON i%50000=source.seq;"
        )
        modes = [
            ("native", "Q1 native CPU callback"),
            ("vector", "Vectorized CPU SQL"),
        ]
        bodies = {
            "native": "SELECT id,artifact_id,extractyear(value) AS year,extractmonth(value) AS month,extractday(value) AS day FROM bench_input",
            "vector": "SELECT id,artifact_id,try_cast(nullif(substr(value,1,4),'') AS BIGINT) AS year,try_cast(nullif(substr(value,6,2),'') AS BIGINT) AS month,try_cast(nullif(substr(value,9,2),'') AS BIGINT) AS day FROM bench_input",
        }
        return (
            setup,
            modes,
            bodies,
            "artifact_id,year,month,day",
            "count(year),sum(year),sum(month),sum(day)",
        )
    else:
        raise ValueError(f"unknown workload: {workload}")
    setup = (
        f"CREATE TABLE bench_input AS SELECT i::BIGINT AS id,{value} AS value "
        f"FROM range({rows}) input(i);"
    )
    return setup, modes, bodies, projection, aggregate


def valid_gpu_counters(value, executions):
    expected = {"executions": executions, "plan_fallbacks": 0, "runtime_fallbacks": 0}
    return (
        isinstance(value, dict)
        and set(value) == set(expected)
        and all(
            type(value[name]) is int and value[name] == count
            for name, count in expected.items()
        )
    )


def query_kinds(workload):
    return (
        AGGREGATE_KINDS
        if workload in AGGREGATE_WORKLOADS
        else ("aggregate", "projection")
    )


def make_aggregate_job(workload, rows, args):
    """Generate SQL-only aggregate rewrites and equivalent CPU SQL baselines."""
    job = args.output_dir / "jobs" / workload / str(rows)
    job.mkdir(parents=True)
    value = "(CASE WHEN i%29=0 THEN NULL ELSE (i%127)::INTEGER-63 END)::DOUBLE"
    if workload == "weighted_mean":
        value = f"({value})::DOUBLE/8.0"
    setup = (
        f"CREATE TABLE bench_input AS SELECT i::BIGINT AS id, {value} AS value,"
        "CASE WHEN i%31=0 THEN NULL ELSE (i%7+1)::DOUBLE END AS weight,"
        "(i%1024)::INTEGER AS g_uniform,"
        "(CASE WHEN i%8<>0 THEN 0 ELSE (i//8)%1024 END)::INTEGER AS g_skewed "
        f"FROM range({rows}) input(i);\n"
    )
    expression = (
        "sum(value*value)"
        if workload == "sum_squares"
        else "sum(CASE WHEN value IS NOT NULL AND weight IS NOT NULL THEN value*weight END)"
        "/nullif(sum(CASE WHEN value IS NOT NULL AND weight IS NOT NULL THEN weight END),0.0)"
    )
    signature = "value" if workload == "sum_squares" else "value,weight"
    registration = f"CREATE MACRO bench_{workload}({signature}) AS {expression};\n"
    (job / "setup.sql").write_text(setup)
    (job / "registration.sql").write_text(registration)
    (job / "source_input.sql").write_text(
        "SELECT id,value,weight,g_uniform,g_skewed FROM bench_input ORDER BY id;\n"
    )
    (job / "cpu_modes.tsv").write_text("native\tEquivalent DuckDB CPU SQL\n")
    (job / "query_kinds.tsv").write_text(
        "".join(f"{kind}\n" for kind in AGGREGATE_KINDS)
    )
    for kind, column in (
        ("global", None),
        ("grouped_uniform", "g_uniform"),
        ("grouped_skewed", "g_skewed"),
    ):
        prefix = f"{column} AS group_key," if column else ""
        suffix = f" GROUP BY {column} ORDER BY {column}" if column else ""
        for mode, expr in (
            ("native", expression),
            ("gpu", f"bench_{workload}({signature})"),
        ):
            (job / f"{kind}_{mode}.sql").write_text(
                f"SELECT {prefix}{expr} AS result FROM bench_input{suffix};\n"
            )
    config = {
        "workload": workload,
        "rows": rows,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "threads": args.threads,
        "plugin": "-",
        "pin": 0,
        "projection": 0,
        "validate_only": 0,
        "aggregate_rewrite": 1,
    }
    (job / "config.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in config.items())
    )
    return job


def make_job(workload, rows, args):
    if workload in AGGREGATE_WORKLOADS:
        return make_aggregate_job(workload, rows, args)
    job = args.output_dir / "jobs" / workload / str(rows)
    job.mkdir(parents=True)
    setup, modes, bodies, projection, aggregate = definitions(workload, rows, args)
    (job / "setup.sql").write_text(setup + "\n")
    registration = (
        (ROOT / "udfs/sql/registration.sql").read_text()
        if workload == "sql"
        else "-- UDF module registered by the native host.\n"
    )
    (job / "registration.sql").write_text(registration)
    module = "q1" if workload == "udfbench_q1" else workload
    config = {
        "workload": workload,
        "rows": rows,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "threads": args.threads,
        "plugin": (
            str(args.build_dir / f"benchmark_{module}.so") if workload != "sql" else "-"
        ),
        "pin": int(args.pin),
        "projection": int(args.projection),
        "validate_only": int(args.validate_only),
    }
    (job / "config.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in config.items())
    )
    (job / "cpu_modes.tsv").write_text(
        "".join(f"{ident}\t{label}\n" for ident, label in modes)
    )
    for ident, body in {**bodies, "gpu": bodies["native"]}.items():
        for kind, expressions in (("projection", projection), ("aggregate", aggregate)):
            (job / f"{kind}_{ident}.sql").write_text(
                f"SELECT {expressions} FROM ({body}) outputs;\n"
            )
        (job / f"sample_{ident}.sql").write_text(
            f"SELECT id,{projection} FROM ({body}) outputs WHERE id<4096 ORDER BY id;\n"
        )
    return job


def validate_measurements(measurements, args, workload, rows):
    """Fail closed if a successful process supplies incomplete GPU evidence."""
    if not isinstance(measurements, dict):
        raise ValueError("host did not supply a measurement object")
    if measurements.get("correctness_verified") is not True:
        raise ValueError("host did not confirm correctness comparisons")
    gpu_checks = measurements.get("gpu_validation_queries", 0)
    if not isinstance(gpu_checks, int) or gpu_checks < (2 if args.pin else 1):
        raise ValueError("host did not confirm GPU validation in every cache mode")
    if measurements.get("input_rows") != rows:
        raise ValueError("host supplied the wrong input row count")
    if measurements.get("correctness_sample_rows", 0) < 1:
        raise ValueError("host did not compare sampled output rows")
    if measurements.get("dispersed_sample_rows", 0) < 1:
        raise ValueError("host did not compare dispersed output rows")
    caches = ["warm_storage", "pinned_gpu"] if args.pin else ["warm_storage"]
    kinds = ["aggregate", "projection"] if args.projection else ["aggregate"]
    expected_validation = {
        (cache, sample) for cache in caches for sample in ("prefix", "dispersed")
    }
    if workload == "udfbench_q1":
        expected_validation.add(("warm_storage", "q1_original_python_oracle"))
    if args.validate_only:
        expected_validation.update((cache, kind) for cache in caches for kind in kinds)
    validation = measurements.get("validation_records")
    if not isinstance(validation, list) or any(
        not isinstance(record, dict) for record in validation
    ):
        raise ValueError("host did not supply per-cache correctness evidence")
    validation_keys = {
        (record.get("cache_mode"), record.get("sample")) for record in validation
    }
    if (
        len(validation) != len(expected_validation)
        or validation_keys != expected_validation
    ):
        raise ValueError(
            "host supplied missing, duplicated or unexpected correctness gates"
        )
    for record in validation:
        if (
            record.get("gpu_executions") != 1
            or record.get("plan_fallbacks") != 0
            or record.get("runtime_fallbacks") != 0
        ):
            raise ValueError(
                "correctness gate did not confirm one GPU query without fallback"
            )
        sample = record["sample"]
        compared = record.get("rows_compared", 0)
        if sample == "projection":
            if (
                record.get("rows_checked") != rows
                or compared != 0
                or record.get("verification") != "row_count_and_logical_types"
            ):
                raise ValueError(
                    "full-projection validation did not check row count and types"
                )
        elif not isinstance(compared, int) or compared < 1:
            raise ValueError("correctness gate did not compare output values")
        elif sample == "q1_original_python_oracle" and compared != 50_000:
            raise ValueError(
                "Q1 correctness gate did not compare all public oracle rows"
            )
        elif sample == "aggregate" and (
            record.get("rows_checked") != 1
            or compared != 1
            or record.get("verification") != "exact_values"
        ):
            raise ValueError("aggregate validation did not compare its complete result")
    if sum(record["gpu_executions"] for record in validation) != gpu_checks:
        raise ValueError(
            "total GPU validation queries disagree with per-query evidence"
        )
    records = measurements.get("records")
    if not isinstance(records, list) or (not args.validate_only and not records):
        raise ValueError("host did not supply benchmark records")
    if args.validate_only and records:
        raise ValueError("validate-only run unexpectedly supplied timing records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("host supplied an invalid timing record")
    if not args.validate_only:
        modes = definitions(workload, rows, args)[1]
        expected = {
            (label, cache, f"{kind} / {cache}")
            for _, label in modes
            for cache in caches
            for kind in kinds
        }
        actual = {
            (row.get("cpu_baseline"), row.get("cache_mode"), row.get("query_kind"))
            for row in records
        }
        if len(records) != len(expected) or actual != expected:
            raise ValueError(
                "host supplied incomplete or duplicated benchmark variants"
            )
    for record in records:
        if record.get("workload") != workload or record.get("rows") != rows:
            raise ValueError("host supplied a timing record for the wrong job")
        if record.get("verification") != "passed":
            raise ValueError("host supplied an unverified timing record")
        counters = record.get("gpu_counters", {})
        if counters != {
            "executions": args.warmups + args.repetitions,
            "plan_fallbacks": 0,
            "runtime_fallbacks": 0,
        }:
            raise ValueError(f"unexpected GPU execution/fallback counters: {counters}")
        for field in ("cpu_samples_ms", "gpu_samples_ms"):
            samples = record.get(field, [])
            if (
                not isinstance(samples, list)
                or len(samples) != args.repetitions
                or any(
                    not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0
                    for ms in samples
                )
            ):
                raise ValueError(f"incomplete or invalid {field}")
        record["cpu_median_ms"] = statistics.median(record["cpu_samples_ms"])
        record["gpu_median_ms"] = statistics.median(record["gpu_samples_ms"])
        record["speedup"] = record["cpu_median_ms"] / record["gpu_median_ms"]


def command_output(command, cwd=None):
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, timeout=10, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def cpu_model():
    try:
        with Path("/proc/cpuinfo").open() as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unavailable"


def metadata(args):
    try:
        build = json.loads((args.build_dir / "benchmark-build.json").read_text())
        if not isinstance(build, dict):
            build = {"status": "unavailable"}
    except (OSError, ValueError):
        build = {"status": "unavailable"}
    tree_state = command_output(["git", "status", "--porcelain"], REPO)
    return {
        "title": "Sirius UDF end-to-end query benchmark",
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "revision": command_output(["git", "rev-parse", "HEAD"], REPO),
        "source_dirty": None if tree_state == "unavailable" else bool(tree_state),
        "environment_lock_sha256": hashlib.sha256(
            (REPO / "pixi.lock").read_bytes()
        ).hexdigest(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "build": build,
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "threads": args.threads,
        "validate_only": args.validate_only,
        "timing_scope": "Serialized warm query wall time, including binding/planning, execution, transfers and result materialization; excludes build, startup, data generation/checkpoint, pinning, correctness checks and counter reads.",
        "ordering": "Rotate CPU alternatives and GPU execution order each round.",
        "correctness": "Exact typed and NULL-aware CPU/GPU prefix and dispersed samples before timing in each cache mode; every full aggregate checked after its timer. Full-projection row counts are checked, with sampled cell comparisons, not exhaustive output equivalence.",
        "cache_modes": "warm_storage uses a checkpointed DuckDB table and warm CPU/OS caches; optional pinned_gpu additionally keeps input on GPU, excluding pin setup from query timing.",
        "cpu_baselines": "Generated CPU callbacks, optimized typed vectorized CPU UDFs, and SQL alternatives where applicable. Compare GPU against the fastest measured CPU alternative in each cache mode/query shape.",
        "limitations": [
            "Synthetic workloads are independently authored; no confidential UDFs or data.",
            "SQL macros already use supported Sirius operators; this route is not acceleration newly enabled by native UDF registration.",
            "CUDA aggregation uses COUNT/MIN/MAX because UBIGINT SUM requires an unsupported unsigned-to-HUGEINT GPU cast.",
            "Optional Q1 is derived from a pinned UDFBench tiny dataset; the adapter covers nullable, empty and ASCII YYYY-MM-DD-shaped strings without calendar validation, not arbitrary original-UDF inputs or the full official suite.",
            "No device-only UDF/kernel timing, cold-cache measurement, Python/JVM runtime comparison, or arbitrary-UDF speedup guarantee.",
            "GPU clocks and other machine activity are uncontrolled; timings vary by hardware and load.",
        ],
    }


def save_results(output, directory):
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "results.json")
    fields = [
        "workload",
        "rows",
        "query_kind",
        "cache_mode",
        "cpu_baseline",
        "cpu_median_ms",
        "gpu_median_ms",
        "speedup",
        "pin_setup_seconds",
        "verification",
    ]
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output["results"])


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = {"metadata": metadata(args), "results": [], "jobs": [], "failures": []}
    save_results(output, args.output_dir)
    environment = os.environ.copy()
    environment.pop("SIRIUS_DISABLE", None)
    environment["SIRIUS_CONFIG_FILE"] = str(
        REPO / "test/cpp/integration/integration.yaml"
    )
    for workload in args.workloads:
        sizes = args.sizes
        if workload == "udfbench_q1" and sizes == DEFAULT_SIZES:
            sizes = [50_000, 1_000_000, 10_000_000]
        for rows in sizes:
            job = make_job(workload, rows, args)
            environment["SIRIUS_LOG_DIR"] = str(job / "engine-log")
            started = time.perf_counter()
            try:
                with (job / "run.log").open("w") as log:
                    process = subprocess.run(
                        [str(args.build_dir / "bench_host"), str(job)],
                        cwd=job,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode:
                    raise RuntimeError(f"benchmark host exited {process.returncode}")
                measurements = json.loads((job / "measurements.json").read_text())
                validate_measurements(measurements, args, workload, rows)
                output["results"].extend(measurements.pop("records"))
                output["jobs"].append(
                    {
                        "workload": workload,
                        "rows": rows,
                        "path": str(job),
                        "wall_seconds": time.perf_counter() - started,
                        **measurements,
                    }
                )
            except (OSError, ValueError, RuntimeError) as error:
                failure = {
                    "workload": workload,
                    "rows": rows,
                    "log": str(job / "run.log"),
                    "error": str(error),
                }
                output["failures"].append(failure)
                save_results(output, args.output_dir)
                print(f"FAILED {workload} {rows:,}: {error}", file=sys.stderr)
                print(f"Retained logs and queries: {job}", file=sys.stderr)
                return 1
            save_results(output, args.output_dir)
            print(
                f"{workload} {rows:,}: passed in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    print(f"Results: {args.output_dir / 'results.json'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

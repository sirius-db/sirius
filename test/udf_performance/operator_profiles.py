"""Collect measured operator work from CPU profiles and keyed Sirius TRACE logs.

Times are exclusive per operator invocation, accumulated across tasks/threads.
They are not a partition of query wall time: workers can overlap. GPU times
include stream synchronization in the existing engine instrumentation and are
rounded to 0.01 ms. Projection includes every expression, not only the UDF.
"""

import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.log_analyzer import patterns, segmenter  # noqa: E402
from tools.log_analyzer.metrics import downgrade, task_output  # noqa: E402
from tools.log_analyzer.validators import FormatWarnings  # noqa: E402

STAGES = (
    "input_prepare_scan",
    "projection_expressions",
    "aggregation",
    "output_transfer_materialization",
    "other",
)
TAG = re.compile(r"/\*sirius_stage:([A-Za-z0-9_-]+)\*/")
AUXILIARY = re.compile(
    r"Pipeline (?P<pipeline_id>\d+): operator (?P<op_type>\w+) "
    r"\(id=(?P<op_id>\d+)\) (?P<kind>prepare|sink) execution time: "
    r"(?P<execution_time_ms>[\d.]+) ms"
)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _category(operator_type):
    name = operator_type.upper()
    if "SCAN" in name:
        return "input_prepare_scan"
    if "PROJECTION" in name:
        return "projection_expressions"
    if any(part in name for part in ("AGGREGAT", "GROUP_BY", "GROUPBY")):
        return "aggregation"
    if "COLLECTOR" in name:
        return "output_transfer_materialization"
    return "other"


def _contained_path(job, filename):
    path = (job / filename).resolve()
    if not path.is_relative_to(job.resolve()):
        raise ValueError("CPU profile path escapes its benchmark job")
    if not path.is_file():
        raise ValueError(f"Missing CPU profile: {filename}")
    return path


def _cpu_operators(document):
    """Read exclusive operator timing, never cpu_time/cumulative/root latency."""
    rows = []

    def visit(node, path):
        if isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            raise ValueError(f"Invalid CPU profile operator at {path}")
        # The current DuckDB profile has root.operator; legacy profiles have
        # root.children. Some serializers wrap those under metrics.operator.
        metrics = node.get("metrics", {})
        if metrics and not isinstance(metrics, dict):
            raise ValueError("Invalid CPU profile metrics object")
        values = node
        if "operator_type" not in values and "type" not in values:
            if "operator_type" in metrics or "type" in metrics:
                values = metrics
        operator_type = values.get("operator_type", values.get("type"))
        if operator_type is not None:
            if not isinstance(operator_type, str) or not operator_type:
                raise ValueError("Invalid CPU operator type")
            timing_key = "operator_timing" if "operator_timing" in values else "timing"
            if timing_key not in values:
                raise ValueError(f"Missing exclusive timing for CPU {operator_type}")
            rows.append(
                {
                    "operator_type": operator_type,
                    "execution_time_ms": _number(
                        values[timing_key], "CPU operator timing"
                    )
                    * 1000,
                    "stage": _category(operator_type),
                    "kind": "execute",
                    "profile_path": path,
                    "rows": values.get(
                        "operator_cardinality", values.get("intermediate_rows")
                    ),
                    "size_bytes": values.get(
                        "result_set_size", values.get("intermediate_size_bytes")
                    ),
                }
            )
        for container in (node, metrics):
            for key in ("operator", "children"):
                if key in container:
                    children = container[key]
                    if not isinstance(children, list):
                        raise ValueError(f"CPU profile {key} must be an array")
                    visit(children, f"{path}.{key}")

    visit(document, "root")
    if not rows or not any(row["stage"] == "projection_expressions" for row in rows):
        raise ValueError("CPU profile has no timed projection operator")
    return rows


def _gpu_proof(profile):
    counters = profile.get("gpu_counters")
    expected = {"executions": 1, "plan_fallbacks": 0, "runtime_fallbacks": 0}
    if not isinstance(counters, dict) or any(
        type(counters.get(key)) is not int or counters[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("GPU profile must prove one execution and zero fallbacks")


def _gpu_windows(job, profiles):
    if not profiles:
        return {}
    files = sorted((job / "engine-log").rglob("*.log"))
    if not files:
        raise ValueError("Missing Sirius TRACE engine log")
    # The fresh per-job log can rotate. File-order is intentionally not used to
    # pair windows across files: a split window must fail rather than silently
    # attribute another query's work to the sample.
    windows = {}
    expected_ids = {profile["sample_id"] for profile in profiles}
    for filename in files:
        lines = filename.read_text().splitlines()
        for window in segmenter.segment(lines):
            tags = TAG.findall(window.sql)
            relevant = [tag for tag in tags if tag in expected_ids]
            if not relevant:
                continue
            if len(tags) != 1 or len(relevant) != 1:
                raise ValueError("Ambiguous Sirius stage sample SQL tag")
            sample_id = relevant[0]
            if sample_id in windows:
                raise ValueError(f"Duplicate GPU execution window for {sample_id}")
            if window.status != "complete" or window.outcome != "ok":
                raise ValueError(f"Incomplete or unsuccessful GPU window: {sample_id}")
            if window.instance is None:
                raise ValueError(
                    "GPU stage profiles require keyed execution-window logs"
                )
            # Metric lines are not keyed. Refuse overlapping GPU windows rather
            # than make an invalid attribution; CPU observer SQL does not open
            # a GPU window and does not interrupt this window.
            begins = sum(
                bool(match and match.group("event") == "begin")
                for line in window.lines
                if (match := patterns.WINDOW_RE.match(line))
            )
            if begins != 1:
                raise ValueError("Overlapping GPU windows cannot be profiled")
            windows[sample_id] = window
    missing = expected_ids - windows.keys()
    if missing:
        raise ValueError(f"Missing GPU execution windows: {', '.join(sorted(missing))}")
    return windows


def _gpu_operators(window):
    warnings = FormatWarnings()
    rows = task_output.parse(window.lines, warnings)
    for row in rows:
        row["stage"] = _category(row["operator_type"])
        row["kind"] = "execute"
    for line in window.lines:
        if (
            " prepare execution time:" not in line
            and " sink execution time:" not in line
        ):
            continue
        match = AUXILIARY.search(line)
        if not match:
            raise ValueError("Unrecognized Sirius prepare/sink timing line")
        kind = match.group("kind")
        operator_type = match.group("op_type")
        rows.append(
            {
                "pipeline_id": int(match.group("pipeline_id")),
                "operator_id": int(match.group("op_id")),
                "operator_type": operator_type,
                "kind": kind,
                "execution_time_ms": _number(
                    float(match.group("execution_time_ms")), "GPU timing"
                ),
                "stage": (
                    "input_prepare_scan"
                    if kind == "prepare"
                    else _category(operator_type)
                ),
            }
        )
    downgrades = downgrade.parse(window.lines, warnings)
    if warnings.has_drift:
        raise ValueError(f"Sirius TRACE format drift: {warnings.to_dict()}")
    if not any(
        row["kind"] == "execute" and row["stage"] == "projection_expressions"
        for row in rows
    ):
        raise ValueError("GPU window has no measured projection operator")
    if not any(
        row["kind"] == "sink" and row["stage"] == "output_transfer_materialization"
        for row in rows
    ):
        raise ValueError("GPU window has no measured result collector sink")
    pools = []
    for line in window.lines:
        if "[gpu_pool]" in line:
            match = patterns.GPU_POOL_RE.search(line)
        elif "[host_pool]" in line:
            match = patterns.HOST_POOL_RE.search(line)
        else:
            continue
        if not match:
            raise ValueError("Unrecognized Sirius memory pool observation")
        pool = {
            key: value for key, value in match.groupdict().items() if value is not None
        }
        for key in ("allocated_bytes", "peak_bytes", "reserved_bytes", "free_blocks"):
            if key in pool:
                pool[key] = int(pool[key])
        pools.append(pool)
    observations = {
        "pool_boundaries": pools,
        "pool_peak_scope": "process_pool_high_water_mark; not per-query peak or delta",
        "task_allocator_peaks_bytes": [
            row["peak_allocated_bytes"] for row in rows if "peak_allocated_bytes" in row
        ],
        "task_peak_scope": "allocator-reported stream peak; not a query-global high-water mark",
        "downgrades": downgrades,
        "downgrade_count": len(downgrades),
        "downgrade_bytes": sum(row["total_bytes"] for row in downgrades),
        "oom_event_count": sum(
            any(
                marker in line
                for marker in (
                    "OOM at operator",
                    "OOM preparing batches for processing",
                    "OOM restoring a deferral at the sink",
                )
            )
            for line in window.lines
        ),
        "reschedule_count": sum(
            "GPU Pipeline Executor: reschedule (retry " in line for line in window.lines
        ),
        "fallback_event_count": sum(
            "Transparent execution fallback" in line for line in window.lines
        ),
    }
    observations["pressure_observed"] = any(
        observations[key]
        for key in ("downgrade_count", "oom_event_count", "reschedule_count")
    )
    if observations["fallback_event_count"]:
        raise ValueError("GPU window contains fallback events")
    return rows, observations


def collect(job: Path, manifest: dict, threads: int) -> list[dict]:
    """Return one stage-work record per manifest profile; missing proof fails."""
    job = Path(job)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Stage manifest has no profiles")
    ids = []
    for profile in profiles:
        sample_id = profile.get("sample_id")
        if not isinstance(sample_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", sample_id
        ):
            raise ValueError("Invalid stage sample_id")
        if TAG.findall(profile.get("sql", "")) != [sample_id]:
            raise ValueError("Stage SQL must contain its unique sample_id tag")
        ids.append(sample_id)
        _number(profile.get("query_wall_ms"), "query_wall_ms")
        if profile.get("engine") not in ("cpu", "gpu"):
            raise ValueError("Unknown stage profile engine")
        if profile["engine"] == "gpu":
            _gpu_proof(profile)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate stage sample_id")
    windows = _gpu_windows(
        job, [profile for profile in profiles if profile["engine"] == "gpu"]
    )
    records = []
    for profile in profiles:
        if profile["engine"] == "cpu":
            document = json.loads(
                _contained_path(job, profile.get("profile_file", "")).read_text()
            )
            query = document.get("query", {}) if isinstance(document, dict) else {}
            query_sql = query.get("sql") if isinstance(query, dict) else None
            if query_sql is None and isinstance(document, dict):
                query_sql = document.get("query_name")
            if query_sql is not None and (
                not isinstance(query_sql, str)
                or TAG.findall(query_sql) != [profile["sample_id"]]
            ):
                raise ValueError("CPU profile SQL does not match its stage sample")
            operators = _cpu_operators(document)
            memory = {
                "available": False,
                "reason": "CPU operator profiler does not measure GPU memory",
            }
        else:
            operators, memory = _gpu_operators(windows[profile["sample_id"]])
        stages = {stage: 0.0 for stage in STAGES}
        available = dict.fromkeys(STAGES, False)
        for operator in operators:
            stage = operator["stage"]
            stages[stage] += _number(
                operator["execution_time_ms"], "operator execution_time_ms"
            )
            available[stage] = True
        records.append(
            {
                **profile,
                "threads": threads,
                "stages": stages,
                "stage_available": available,
                "raw_operators": operators,
                "memory_observations": memory,
                "total_operator_work_ms": sum(stages.values()),
                "timing_model": "cumulative_worker_wall",
                "timings_not_additive_to_query_wall": True,
                "precision_resolution_ms": 0.01 if profile["engine"] == "gpu" else None,
                "projection_scope": "all projection expressions, not isolated UDF time",
            }
        )
    return records

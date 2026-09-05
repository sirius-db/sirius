#!/usr/bin/env python3
"""Summarize immutable two-CN benchmark blocks without rerunning queries.

Example:
  python scripts/local-gb10/analyze-multi-cn.py \
    --baseline results/multi-cn-throughput/baseline \
    --optimized results/multi-cn-throughput/optimized \
    --output results/multi-cn-throughput/analysis

Only derived files below --output are written. Missing, skipped, failed, and
uncommitted attempts remain visible; none become zero-second successful samples.
"""

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
_execution_spec = importlib.util.spec_from_file_location(
    "multi_cn_execution_validation", Path(__file__).with_name("execution_validation.py")
)
_execution_validation = importlib.util.module_from_spec(_execution_spec)
_execution_spec.loader.exec_module(_execution_validation)
QUERIES = [f"q{i:02}" for i in range(1, 23)]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r'\b(\w+)=("(?:\\.|[^"\\])*"|[^\s]+)')
UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
FRAGMENT = re.compile(r"\bfragment run (started|finished|failed)\b")
EVENT_NAMES = {
    "ingress": "[exchange_ingress]",
    "reload": "[exchange_reload]",
    "pack": "[exchange_pack]",
    "pack_wait": "optimized exchange frame packed",
    "transfer": "transmitted batches via nixl",
    "retirement": "owned ingress query retirement accounting",
    "credit_grant": "owned receive credit granted",
    "credit_return": "owned receive credit returned after ingress completion",
}
NOT_EXECUTED = {
    "MISSING",
    "NOT_RUN_YET",
    "NOT_PLANNED",
    "UNCOMMITTED_RESULT",
    "SKIPPED_AFTER_FAILURE",
    "MISSING_MANIFEST",
}
# These two frozen CNs were built with async_sender_dispatch_from_env() returning
# false when unset (95bec853 baseline and this experiment's optimized build).
# Restrict the legacy inference to their recorded identities: future binaries may
# change defaults. New harnesses record the switch explicitly.
LEGACY_ASYNC_DEFAULT_OFF_CNS = {
    "bd76ce167b6c2844a1b44925e4960a283437564fcdf3d333ce43e9293ef0eee4",
    "b6a274a93d116177726b77afdd5e0623ee46f4e5204d25c3d2f5defc20856cd3",
}
# This inspected stack.py records every SIRIUS_CN_* variable in launch.json's
# transport_environment. Absence in an unrecognized/partial snapshot proves nothing.
COMPLETE_CN_ENV_RECORDERS = {
    "684e6001a17c451891da2591b76cdf2c5f2f711dbe37e521e20f376274fd087c"
}


def read_json(path, warnings=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        if warnings is not None:
            warnings.append(f"{path}: {error}")
        return None


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def line_timestamp(line):
    if line.startswith("["):
        return timestamp(line[1 : line.find("]")])
    return timestamp(line.split(" ", 1)[0])


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(str(value).removeprefix("Some(").removesuffix(")"))
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def fields(line):
    return {key: value.strip('"') for key, value in FIELD.findall(line)}


def digest(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def locate(value, base, fallback):
    candidates = [Path(value)] if value else []
    if value and not Path(value).is_absolute():
        candidates.insert(0, base / value)
    candidates.append(fallback)
    return next((p.resolve() for p in candidates if p.exists()), fallback.resolve())


def failure_class(status, detail):
    if status == "PASS":
        return None
    if status in NOT_EXECUTED:
        return status
    text = f"{status} {detail}".lower()
    for needles, category in [
        (("mismatch", "wrong result", "bad_cells"), "WRONG_RESULT"),
        (("ingress_capacity", "evacuation reservation"), "INGRESS_CAPACITY"),
        (("export_capacity",), "EXPORT_CAPACITY"),
        (("arena exhausted", "staging arena", "lease of"), "STAGING_CAPACITY_OR_LEASE"),
        (("out_of_memory", "bad_alloc", "oom retry"), "MEMORY_ALLOCATION_FAILURE"),
        (("watchdog",), "ENGINE_WATCHDOG"),
        (("nixl write", "transfer timeout"), "TRANSFER_FAILURE"),
        (("epoch mismatch", "lease token", "immutable", "protocol"), "LEASE_PROTOCOL"),
        (("timeout", "deadline"), "TIMEOUT"),
        (("runner_error", "startup", "topology"), "SETUP_OR_RUNNER"),
    ]:
        if any(word in text for word in needles):
            return category
    return "ENGINE_OR_TRANSPORT_ERROR"


def assess(row, *, require_runner=True):
    """Keep reported PASS separate from a success supported by correctness/topology evidence."""
    issues = []
    comparison = row.get("comparison", {})
    execution = row.get("starrocks", {})
    seconds = number(execution.get("elapsed_seconds"))
    if row.get("status") != "PASS":
        issues.append("status is not PASS")
    if comparison.get("match") is not True:
        issues.append("no affirmative oracle comparison")
    if comparison.get("bad_cells", 0) != 0:
        issues.append("comparison reports bad cells")
    if any(
        type(comparison.get(key)) is not int or comparison[key] < 0
        for key in ("actual_rows", "expected_rows")
    ):
        issues.append("missing or invalid comparison row counts")
    elif comparison.get("actual_rows") != comparison.get("expected_rows"):
        issues.append("result row counts differ")
    if execution.get("status") != "OK" or execution.get("returncode") != 0:
        issues.append("SQL client did not report successful completion")
    if row.get("runner_returncode") != 0 and (
        require_runner or "runner_returncode" in row
    ):
        issues.append("missing or nonzero benchmark runner return code")
    if seconds is None or seconds < 0:
        issues.append("missing or invalid query time")
    for when in ("before", "after"):
        if row.get(f"topology_{when}") != {"compute-nodes": 2, "backends": 0}:
            issues.append(f"missing or unexpected topology_{when}")
    return not issues, seconds, issues


def artifact_info(attempt, query, reference):
    directory = attempt / query
    paths = {
        name: str(directory / name)
        for name in (
            "source.sql",
            "duckdb.sql",
            "starrocks.sql",
            "explain.tsv",
            "explain.stderr",
            "starrocks.tsv",
            "starrocks.stderr",
            "comparison.json",
        )
        if (directory / name).exists()
    }
    paths.update(
        {
            name: str(attempt / name)
            for name in ("runner.log", "manifest.json", "results.json")
            if (attempt / name).exists()
        }
    )
    fingerprint = directory / "oracle-fingerprint.txt"
    previous = reference / query / "oracle-fingerprint.txt"
    try:
        matches = fingerprint.read_text().strip() == previous.read_text().strip()
    except OSError:
        matches = None
    runner = read_json(attempt / "manifest.json") or {}
    return {
        "paths": paths,
        "source_sql_sha256": digest(directory / "source.sql"),
        "duckdb_sql_sha256": digest(directory / "duckdb.sql"),
        "reference_oracle_fingerprint_match": matches,
        "runner_protocol": {
            key: runner.get(key)
            for key in (
                "relative_tolerance",
                "absolute_tolerance",
                "timeout_seconds",
                "session_sql",
            )
        },
    }


def load_arm(root, arm, repetitions, reference):
    warnings, blocks, samples = [], [], []
    execution_indexes = {}
    paths = sorted(root.rglob("manifest.json")) if root.exists() else []
    for path in paths:
        manifest = read_json(path, warnings)
        if (
            not isinstance(manifest, dict)
            or manifest.get("arm") != arm
            or not isinstance(manifest.get("runs"), list)
        ):
            continue
        status = read_json(path.parent / "status.json") or {}
        block = {key: value for key, value in manifest.items() if key != "runs"}
        block.update(path=str(path.resolve()), status=status)
        blocks.append(block)
        observed = defaultdict(list)
        for row in manifest["runs"]:
            observed[(row.get("query"), row.get("repetition"))].append(row)
        planned_queries = manifest.get("queries", [])
        planned_repetitions = manifest.get("repetitions", repetitions)
        if not isinstance(planned_repetitions, int) or planned_repetitions < 1:
            warnings.append(f"{path}: invalid repetition count {planned_repetitions!r}")
            planned_repetitions = repetitions
        if planned_repetitions != repetitions:
            warnings.append(
                f"{path}: declares {planned_repetitions} repetitions; requested protocol is {repetitions}"
            )
        slots = {
            (q, r)
            for q in planned_queries
            if q in QUERIES
            for r in range(max(repetitions, planned_repetitions))
        }
        slots.update(
            key for key in observed if key[0] in QUERIES and isinstance(key[1], int)
        )
        for query, repetition in sorted(slots):
            rows = observed.get((query, repetition), [])
            if len(rows) > 1:
                warnings.append(
                    f"{path}: duplicate slot {query} r{repetition}; retained all records"
                )
            if not rows:
                result_path = path.parent / query / f"r{repetition:02}" / "results.json"
                missing_status = (
                    "NOT_PLANNED"
                    if repetition >= planned_repetitions
                    else (
                        "MISSING"
                        if status.get("phase") == "complete"
                        else "NOT_RUN_YET"
                    )
                )
                row = {
                    "query": query,
                    "repetition": repetition,
                    "status": missing_status,
                }
                if result_path.exists():
                    row.update(
                        status="UNCOMMITTED_RESULT",
                        detail=f"Result exists outside the parent manifest: {result_path}",
                    )
                rows = [row]
            for index, row in enumerate(rows):
                sample = dict(row)
                phase = "cold" if repetition == 0 else "warm"
                if row.get("phase", phase) != phase:
                    warnings.append(
                        f"{path}: phase/repetition disagreement for {query} r{repetition}"
                    )
                valid, seconds, issues = assess(row)
                if len(rows) > 1:
                    valid = False
                    issues.append("duplicate manifest slot")
                attempt = locate(
                    row.get("attempt"),
                    path.parent,
                    path.parent / query / f"r{repetition:02}",
                )
                cluster = locate(
                    row.get("cluster"), path.parent, path.parent / "unknown-cluster"
                )
                sample.update(
                    arm=arm,
                    query=query,
                    repetition=repetition,
                    phase=phase,
                    sample_id=f"{path.parent.resolve()}::{query}::r{repetition}::{index}",
                    harness_manifest=str(path.resolve()),
                    attempt=str(attempt),
                    cluster=str(cluster),
                    valid_success=valid,
                    query_seconds=seconds,
                    validation_issues=issues,
                    failure_class=failure_class(
                        row.get("status", "MISSING"), row.get("detail", "")
                    ),
                    artifacts=artifact_info(attempt, query, reference),
                )
                if sample["artifacts"]["reference_oracle_fingerprint_match"] is False:
                    sample["valid_success"] = False
                    sample["validation_issues"].append(
                        "oracle fingerprint differs from SF500 reference"
                    )
                if not sample["valid_success"] and sample["status"] == "PASS":
                    sample["failure_class"] = "INCOMPLETE_VALIDATION"
                execution = row.get("starrocks")
                if isinstance(execution, dict):
                    if cluster not in execution_indexes:
                        execution_indexes[cluster] = (
                            _execution_validation.load_execution_evidence(cluster)
                        )
                    if row.get("execution_validation") is not None:
                        sample["recorded_execution_validation"] = row[
                            "execution_validation"
                        ]
                    audit = _execution_validation.validate_execution(
                        cluster,
                        execution.get("started_utc"),
                        execution.get("finished_utc"),
                        evidence=execution_indexes[cluster],
                    )
                    sample["execution_validation"] = audit
                    sample["execution_validation_status"] = audit["status"]
                    excluded = audit["status"] == "INELIGIBLE" or (
                        audit["status"] == "UNKNOWN" and audit["detected_retry"]
                    )
                    if excluded:
                        sample["valid_success"] = False
                        sample["validation_issues"].append(audit["issue"])
                        if sample["status"] == "PASS":
                            sample["failure_class"] = audit["failure_class"]
                    if audit["status"] != "VALID":
                        warnings.append(
                            f"{arm} {query} r{repetition:02}: execution validation {audit['status']}: {audit['issue']}"
                        )
                samples.append(sample)
    duplicated_slots = defaultdict(list)
    for sample in samples:
        duplicated_slots[(sample["query"], sample["repetition"])].append(sample)
    for slot, records in duplicated_slots.items():
        if len({record["harness_manifest"] for record in records}) > 1:
            warnings.append(
                f"{arm}: {slot} occurs in multiple harness blocks; retained but excluded from timing aggregates"
            )
            for record in records:
                record["valid_success"] = False
                record["validation_issues"].append(
                    "duplicate query/repetition across harness blocks"
                )
                if record["status"] == "PASS":
                    record["failure_class"] = "INCOMPLETE_VALIDATION"
    covered = {sample["query"] for sample in samples}
    for query in QUERIES:
        if query not in covered:
            for repetition in range(repetitions):
                samples.append(
                    {
                        "arm": arm,
                        "query": query,
                        "repetition": repetition,
                        "phase": "cold" if repetition == 0 else "warm",
                        "status": "MISSING_MANIFEST",
                        "valid_success": False,
                        "query_seconds": None,
                        "validation_issues": [
                            "no harness manifest declares this query"
                        ],
                        "failure_class": "MISSING_MANIFEST",
                        "artifacts": {"paths": {}},
                        "sample_id": f"{arm}::{query}::r{repetition}::missing",
                    }
                )
    if not blocks:
        warnings.append(f"No {arm} harness manifests found below {root}")
    return {
        "root": str(root.resolve()),
        "blocks": blocks,
        "samples": samples,
        "warnings": warnings,
    }


def phase_summary(samples, phase):
    selected = [sample for sample in samples if sample["phase"] == phase]
    times = [sample["query_seconds"] for sample in selected if sample["valid_success"]]
    return {
        "planned_samples": len(selected),
        "successful_samples": len(times),
        "complete": bool(selected) and len(times) == len(selected),
        "seconds": times,
        "median_seconds": statistics.median(times) if times else None,
        "min_seconds": min(times) if times else None,
        "max_seconds": max(times) if times else None,
    }


def summarize_arm(arm):
    grouped = defaultdict(list)
    for sample in arm["samples"]:
        grouped[sample["query"]].append(sample)
    arm["queries"] = {
        query: {
            phase: phase_summary(grouped[query], phase) for phase in ("cold", "warm")
        }
        for query in QUERIES
    }
    counts = Counter(sample["status"] for sample in arm["samples"])
    arm["counts"] = {
        "planned_samples": len(arm["samples"]),
        "recorded_status_counts": dict(counts),
        "oracle_matching_pass_samples": sum(
            s["status"] == "PASS" and s.get("comparison", {}).get("match") is True
            for s in arm["samples"]
        ),
        "validated_successful_samples": sum(s["valid_success"] for s in arm["samples"]),
        "executed_or_failed_samples": sum(
            s["status"] not in NOT_EXECUTED for s in arm["samples"]
        ),
        "queries_with_any_success": sum(
            any(s["valid_success"] for s in grouped[q]) for q in QUERIES
        ),
        "queries_with_every_sample_successful": sum(
            all(s["valid_success"] for s in grouped[q]) for q in QUERIES
        ),
    }
    arm["successful_suite_seconds"] = {
        phase: (
            sum(arm["queries"][q][phase]["median_seconds"] for q in QUERIES)
            if all(arm["queries"][q][phase]["complete"] for q in QUERIES)
            else None
        )
        for phase in ("cold", "warm")
    }


def new_metrics():
    return {
        "fragment_sets": defaultdict(lambda: defaultdict(set)),
        "events": {},
        "retirements": {},
        "pool": {
            "samples": 0,
            "max_sampled_allocated_bytes": None,
            "max_reported_lifetime_peak_bytes": None,
        },
        "evidence": defaultdict(list),
        "lines_scanned": 0,
    }


def record_metrics(metrics, kind, values, evidence):
    metrics["lines_scanned"] += 1
    if len(metrics["evidence"][kind]) < 8:
        metrics["evidence"][kind].append(evidence)
    if kind.startswith("fragment_"):
        query, fragment = values.get("query_id", ""), values.get(
            "fragment_instance_id", ""
        )
        if UUID.fullmatch(query) and UUID.fullmatch(fragment):
            metrics["fragment_sets"][query][kind.removeprefix("fragment_")].add(
                fragment
            )
    elif kind == "pool":
        metrics["pool"]["samples"] += 1
        for source, target in (
            ("allocated", "max_sampled_allocated_bytes"),
            ("peak", "max_reported_lifetime_peak_bytes"),
        ):
            value = number(values.get(source))
            if value is not None:
                previous = metrics["pool"][target]
                metrics["pool"][target] = (
                    max(value, previous) if previous is not None else value
                )
    elif kind == "retirement":
        query = values.get("query_id")
        if query:
            # Counters are cumulative for one FE query. Repeated cancellation is not new traffic.
            metrics["retirements"][query] = {"fields": values, "evidence": evidence}
    else:
        event = metrics["events"].setdefault(kind, {"records": 0, "numeric_fields": {}})
        event["records"] += 1
        for name in (
            "bytes",
            "rows",
            "batches",
            "elapsed_us",
            "pack_wait_us",
            "live_bytes",
            "peak_bytes",
            "quarantined_bytes",
        ):
            value = number(values.get(name))
            if value is None:
                continue
            field = event["numeric_fields"].setdefault(name, {"sum": 0, "max": value})
            field["sum"] += value
            field["max"] = max(field["max"], value)


def finish_metrics(metrics):
    distribution = {
        query: {
            state: len(ids.get(state, set()))
            for state in ("started", "finished", "failed")
        }
        for query, ids in metrics.pop("fragment_sets").items()
    }
    metrics["fragments_by_fe_query_id"] = distribution
    metrics["fragment_counts"] = {
        state: sum(row[state] for row in distribution.values())
        for state in ("started", "finished", "failed")
    }
    metrics["evidence"] = dict(metrics["evidence"])
    return metrics


def telemetry_distribution(cluster, warnings):
    """Read the saved official analyzer output; never infer engine/FE UUID identity."""
    path = cluster / "telemetry-distribution.json"
    result = {"path": str(path), "available": False, "data": None, "notes": []}
    if not path.exists():
        result["notes"].append(
            "Saved telemetry-distribution.json is unavailable; telemetry may not yet have flushed at CN shutdown."
        )
        return result
    data = read_json(path, warnings)
    if not isinstance(data, dict) or not isinstance(data.get("cns"), list):
        result["notes"].append(
            "Saved telemetry distribution is unreadable or has no CN records."
        )
        return result
    result.update(available=True, data=data)
    counts = Counter(node.get("cn") for node in data["cns"] if isinstance(node, dict))
    absent = [cn for cn in ("cn0", "cn1") if cn not in counts]
    if absent:
        result["notes"].append(f"Telemetry omits expected CN records: {absent}.")
    if data.get("multi_run_warning") or any(count > 1 for count in counts.values()):
        result["notes"].append(
            "Telemetry contains multiple engine process lifetimes; the saved aggregate balance is not a single-cluster-lifetime comparison."
        )
    for node in data["cns"]:
        if not isinstance(node, dict):
            result["notes"].append(
                "An invalid CN telemetry record was preserved in the raw distribution."
            )
            continue
        if node.get("missing"):
            result["notes"].append(
                f"{node.get('cn')} telemetry is missing: {node.get('reason')}."
            )
        if node.get("parse_errors"):
            result["notes"].append(
                f"{node.get('cn')} telemetry parse errors: {node['parse_errors']}."
            )
    result["notes"].extend(
        [
            "Saved telemetry is preferred for work distribution; fragment log counts remain separate execution evidence.",
            "Task/entity counts describe logged work units, not equal scan rows, transferred bytes, GPU utilization, or time.",
            "Engine query UUIDs are not FE query UUIDs and are not joined by UUID or inferred event order.",
            "Telemetry covers the cluster lifetime, including warm repetitions and administrative queries; it has no per-attempt attribution here.",
        ]
    )
    return result


def scan_cluster(cluster, samples, warnings):
    windows = {}
    for sample in samples:
        execution = sample.get("starrocks", {})
        start = timestamp(execution.get("started_utc") or sample.get("started_utc"))
        end = timestamp(execution.get("finished_utc") or sample.get("finished_utc"))
        if start is not None and end is not None and end >= start:
            windows[sample["sample_id"]] = (start, end)
    report = {
        "directory": str(cluster),
        "queries": sorted({s["query"] for s in samples}),
        "cns": {},
        "sample_windows": windows,
        "notes": [
            "Cluster aggregates include all cold/warm repetitions and administrative SQL on this cluster, not a single attempt.",
            "Cluster ERROR/failed-fragment events can come from administrative queries; TPC-H failure status comes only from the recorded benchmark attempt.",
            "Per-attempt metrics use recorded UTC windows; timestamp-less evidence stays cluster-scoped.",
            "gpu_pool peak is a reported process-lifetime high-water mark, not a reset per-query peak.",
            "Fragment counts prove logged execution, not equal rows, bytes, or GPU utilization.",
            "Event byte sums are separate physical stages and must not be added as unique data volume.",
            "Credit live/peak values are gauges: their maxima matter, not the sum of snapshots.",
        ],
    }
    for cn in ("cn0", "cn1"):
        cn_log = cluster / f"{cn}.log"
        engine_logs = sorted((cluster / f"{cn}-engine-log").glob("*.log"))
        if not engine_logs:
            engine_logs = sorted((cluster / "engine" / cn / "log").glob("*.log"))
        totals, per_sample = new_metrics(), {key: new_metrics() for key in windows}
        sources = [(cn_log, False)] + [(path, True) for path in engine_logs]
        seen = set()
        existing = []
        for path, engine in sources:
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                existing.append(str(path))
                with path.open(errors="replace") as source:
                    for line_number, raw in enumerate(source, 1):
                        line = ANSI.sub("", raw).rstrip()
                        values = fields(line)
                        match = FRAGMENT.search(line) if not engine else None
                        if match:
                            kind = "fragment_" + match[1]
                        elif "[gpu_pool]" in line and engine:
                            kind = "pool"
                        else:
                            kind = next(
                                (
                                    key
                                    for key, marker in EVENT_NAMES.items()
                                    if marker in line
                                ),
                                None,
                            )
                            if kind is None and re.search(
                                r"reschedul|OOM retry|not convertible", line, re.I
                            ):
                                kind = "reschedule_or_capacity"
                            elif (
                                kind is None and "fused" in line and "fragment" in line
                            ):
                                kind = "fusion"
                            elif kind is None and re.search(
                                r"\bERROR\b|\[error\]", line
                            ):
                                kind = "error"
                        if kind is None:
                            continue
                        # C++ metrics may also be mirrored to CN stderr; count their engine log once.
                        if (
                            not engine
                            and engine_logs
                            and kind in {"pool", "ingress", "reload", "pack"}
                        ):
                            continue
                        evidence = {
                            "path": str(path),
                            "line": line_number,
                            "text": line[:1800],
                        }
                        record_metrics(totals, kind, values, evidence)
                        instant = line_timestamp(line)
                        if instant is not None:
                            for key, (start, end) in windows.items():
                                if start <= instant <= end:
                                    record_metrics(
                                        per_sample[key], kind, values, evidence
                                    )
            except OSError as error:
                warnings.append(f"{path}: {error}")
        report["cns"][cn] = {
            "cn_log_missing": not cn_log.exists(),
            "engine_logs_missing": not engine_logs,
            "sources": existing,
            "cluster_metrics": finish_metrics(totals),
            "per_sample": {
                key: finish_metrics(value) for key, value in per_sample.items()
            },
        }
    query_ids = sorted(
        {
            query
            for node in report["cns"].values()
            for query in node["cluster_metrics"]["fragments_by_fe_query_id"]
        }
    )
    report["finished_fragment_distribution"] = {
        query: {
            cn: (
                None
                if node["cn_log_missing"]
                else node["cluster_metrics"]["fragments_by_fe_query_id"]
                .get(query, {})
                .get("finished", 0)
            )
            for cn, node in report["cns"].items()
        }
        for query in query_ids
    }
    report["profile_artifacts"] = (
        sorted(
            str(path)
            for pattern in (
                "*.ndjson",
                "*.jsonl",
                "*distribution*.json",
                "*.substrait",
                "fragment-*.txt",
            )
            for path in cluster.rglob(pattern)
        )
        if cluster.exists()
        else []
    )
    report["activity_artifacts"] = [
        str(cluster / name)
        for name in ("activity.json", "cn-activity.json")
        if (cluster / name).exists()
    ]
    report["telemetry_distribution"] = telemetry_distribution(cluster, warnings)
    return report


def common_comparison(first, second, phase):
    common = [
        q
        for q in QUERIES
        if first["queries"][q][phase]["complete"]
        and second["queries"][q][phase]["complete"]
    ]
    a = sum(first["queries"][q][phase]["median_seconds"] for q in common)
    b = sum(second["queries"][q][phase]["median_seconds"] for q in common)
    return {
        "phase": phase,
        "queries": common,
        "query_count": len(common),
        "baseline_seconds": a if common else None,
        "optimized_seconds": b if common else None,
        "baseline_over_optimized": a / b if common and b > 0 else None,
        "definition": "ratio of sums of per-query phase medians on the same fully successful query subset; not whole-suite speedup",
    }


def historic_reference(path, warnings):
    data = read_json(path / "results.json", warnings) or {}
    manifest = data.get("manifest", {})
    queries = {}
    for row in data.get("results", []):
        valid, seconds, issues = assess(row, require_runner=False)
        queries[row["query"]] = {
            "status": row.get("status"),
            "valid_success": valid,
            "query_seconds": seconds,
            "detail": row.get("detail", ""),
            "validation_issues": issues,
        }
    return {
        "path": str(path),
        "validation_scope": "Historical rows predate parent-runner exit metadata: validate their SQL client exit, correctness, time and availability topology. Missing parent-runner return codes remain unavailable, not inferred as zero.",
        "queries": queries,
        "passed": sum(row["valid_success"] for row in queries.values()),
        "attempted": len(queries),
        "successful_suite_seconds": (
            sum(row["query_seconds"] for row in queries.values())
            if set(queries) == set(QUERIES)
            and all(row["valid_success"] for row in queries.values())
            else None
        ),
        "successful_subset_seconds": sum(
            row["query_seconds"] for row in queries.values() if row["valid_success"]
        ),
        "source_sha": manifest.get("engine_source", {}).get("git_head"),
        "execution_note": manifest.get("execution_note"),
        "session_sql": manifest.get("session_sql"),
        "timeout_seconds": manifest.get("timeout_seconds"),
        "relative_tolerance": manifest.get("relative_tolerance"),
        "absolute_tolerance": manifest.get("absolute_tolerance"),
    }


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def safe(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def normalized_setting(field, value):
    if field in {"warmup", "async_sender_dispatch"}:
        if isinstance(value, bool):
            return value
        if value in ("1", "true", "TRUE", "on", "ON"):
            return True
        if value in ("0", "false", "FALSE", "off", "OFF"):
            return False
        return None
    if field == "transfer_window":
        if type(value) is int and 1 <= value <= 8:
            return value
        if isinstance(value, str) and value.isdigit() and 1 <= int(value) <= 8:
            return int(value)
        return None
    if field == "config_sha256":
        return (
            value.lower()
            if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value)
            else None
        )
    return value if value is not None else None


def block_runtime_settings(block, samples):
    """Read only small launch metadata; never inspect telemetry or execute a CN."""
    directory = Path(block["path"]).parent
    paths = set(directory.glob("cluster-*/launch.json"))
    paths.update(
        Path(sample["cluster"]) / "launch.json"
        for sample in samples
        if sample.get("harness_manifest") == block["path"]
        and sample.get("cluster")
        and not sample["cluster"].endswith("unknown-cluster")
    )
    launches = []
    for path in sorted(paths):
        data = read_json(path)
        env = data.get("transport_environment") if isinstance(data, dict) else None
        launches.append(
            {"path": str(path), "environment": env if isinstance(env, dict) else None}
        )
    return launches


def resolve_block_setting(block, field, launches):
    declared = normalized_setting(field, block.get(field))
    evidence = [
        {
            "source": block["path"],
            "field": field,
            "raw": block.get(field),
            "value": declared,
            "status": "recorded" if declared is not None else "missing_or_invalid",
        }
    ]
    env_key = {
        "warmup": "SIRIUS_CN_NIXL_WARMUP",
        "async_sender_dispatch": "SIRIUS_CN_ASYNC_SENDER_DISPATCH",
        "log_filter": "RUST_LOG",
        "transfer_window": "SIRIUS_CN_NIXL_TRANSFER_WINDOW",
    }.get(field)
    unknown_runtime = 0
    for launch in launches if env_key else []:
        env = launch["environment"]
        raw = env.get(env_key) if env is not None else None
        value = normalized_setting(field, raw)
        entry = {
            "source": launch["path"],
            "field": env_key,
            "raw": raw,
            "value": value,
            "status": "recorded" if value is not None else "missing_or_invalid",
        }
        if field == "async_sender_dispatch" and env is not None and env_key not in env:
            source_hashes = block.get("untracked_source_sha256") or {}
            binaries = block.get("binary_sha256") or {}
            recorder = (
                source_hashes.get("scripts/local-gb10/stack.py")
                if isinstance(source_hashes, dict)
                else None
            )
            cn_hashes = {
                value
                for path, value in (
                    binaries.items() if isinstance(binaries, dict) else []
                )
                if Path(path).name == "sirius-starrocks-cn"
            }
            if (
                recorder in COMPLETE_CN_ENV_RECORDERS
                and len(cn_hashes) == 1
                and cn_hashes <= LEGACY_ASYNC_DEFAULT_OFF_CNS
            ):
                entry.update(
                    value=False,
                    status="inferred_from_verified_unset_default",
                    reason="Known launcher records all SIRIUS_CN_* keys; async key is absent. This known frozen CN defaults unset async dispatch to off.",
                    recorder_sha256=recorder,
                    cn_sha256=next(iter(cn_hashes)),
                )
        # The baseline transport is serial and ignores this optimized-only knob.
        # Keep its raw environment value, but compare the manifest's effective window.
        if field == "transfer_window" and block.get("arm") == "baseline":
            entry["comparison_role"] = "ignored_by_legacy_transport"
        elif entry["value"] is None:
            unknown_runtime += 1
        evidence.append(entry)
    values = {
        json.dumps(entry["value"], sort_keys=True)
        for entry in evidence
        if entry["value"] is not None
        and entry.get("comparison_role") != "ignored_by_legacy_transport"
    }
    if len(values) > 1:
        status, value = "conflicting", None
    elif len(values) == 1 and (declared is not None or unknown_runtime == 0):
        status, value = "known", json.loads(next(iter(values)))
    else:
        status, value = "unknown", None
    return {"status": status, "value": value, "evidence": evidence}


def compare_run_settings(arms, warnings):
    across = (
        "topology",
        "budgets_per_cn",
        "settings",
        "config_sha256",
        "warmup",
        "async_sender_dispatch",
        "log_filter",
    )
    checked = (*across, "transfer_window")
    result = {
        "across_arm_fields": list(across),
        "within_arm_fields": list(checked),
        "arms": {},
    }
    for name, arm in arms.items():
        for block in arm["blocks"]:
            launches = block_runtime_settings(block, arm["samples"])
            block["resolved_run_settings"] = {
                field: resolve_block_setting(block, field, launches)
                for field in checked
            }
        fields = {}
        for field in checked:
            settings = [
                block["resolved_run_settings"][field] for block in arm["blocks"]
            ]
            values = sorted(
                {
                    json.dumps(setting["value"], sort_keys=True)
                    for setting in settings
                    if setting["status"] == "known"
                }
            )
            unknown = sum(setting["status"] == "unknown" for setting in settings)
            conflicting = sum(
                setting["status"] == "conflicting" for setting in settings
            )
            relevant = not (field == "transfer_window" and name == "baseline")
            fields[field] = {
                "known_values": [json.loads(value) for value in values],
                "blocks": len(settings),
                "unknown_blocks": unknown,
                "conflicting_blocks": conflicting,
                "within_arm_stable": bool(settings)
                and len(values) == 1
                and not unknown
                and not conflicting,
                "relevant": relevant,
            }
            if relevant and (unknown or not settings):
                warnings.append(
                    f"{name} {field}: unknown in {unknown}/{len(settings)} blocks; missing metadata is not a default."
                )
            if relevant and (len(values) > 1 or conflicting):
                warnings.append(
                    f"{name} mixes or contradicts {field}: known values={values}, conflicting blocks={conflicting}."
                )
        identities = [block.get("binary_sha256") for block in arm["blocks"]]
        if any(not identity for identity in identities) or not identities:
            warnings.append(
                f"{name} has unknown frozen binary identities in one or more blocks."
            )
        if (
            len(
                {
                    json.dumps(identity, sort_keys=True)
                    for identity in identities
                    if identity
                }
            )
            > 1
        ):
            warnings.append(
                f"{name} mixes frozen binary identities; do not interpret its aggregate as one build"
            )
        result["arms"][name] = fields
    for field in across:
        first, second = (
            result["arms"][name][field] for name in ("baseline", "optimized")
        )
        if (
            not first["within_arm_stable"]
            or not second["within_arm_stable"]
            or first["known_values"] != second["known_values"]
        ):
            warnings.append(
                f"Matching {field} is not established across every baseline/optimized block; see settings_comparability and per-block resolved_run_settings."
            )
    return result


def render(report, output):
    arms = report["arms"]
    lines = [
        "# Two-CN SF500 benchmark results",
        "",
        *report["protocol_notes"],
        "",
        "| Arm | Oracle-matching PASS samples | Eligible samples / planned | Queries with any eligible result | All-sample eligible queries | Cold all-22 sum (s) | Warm all-22 sum of medians (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        counts = arm["counts"]
        lines.append(
            f"| {name} | {counts['oracle_matching_pass_samples']} | {counts['validated_successful_samples']}/{counts['planned_samples']} | {counts['queries_with_any_success']}/22 | {counts['queries_with_every_sample_successful']}/22 | {fmt(arm['successful_suite_seconds']['cold'])} | {fmt(arm['successful_suite_seconds']['warm'])} |"
        )
    lines += [
        "",
        "| Phase | Common fully successful queries | Baseline sum (s) | Optimized sum (s) | Baseline / optimized |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison in report["comparisons"]:
        lines.append(
            f"| {comparison['phase']} | {comparison['query_count']}/22 | {fmt(comparison['baseline_seconds'])} | {fmt(comparison['optimized_seconds'])} | {fmt(comparison['baseline_over_optimized'])} |"
        )
    lines += [
        "",
        "Ratios use the same eligible query subset; they exclude failed/skipped/missing samples and degraded or unresolved FE retry placement. They are not full-suite speedups. Warm values are sums of per-query medians, not wall-clock suite duration. Raw correctness PASS and SQL time remain recorded even when execution placement makes a sample ineligible.",
        "",
        "| Query | Historical single attempt (s) | Baseline cold / warm samples (s) | Optimized cold / warm samples (s) | Profile |",
        "|---|---:|---|---|---|",
    ]
    for query in QUERIES:
        old = report["historical"]["queries"].get(query, {})
        previous = (
            fmt(old.get("query_seconds"))
            if old.get("valid_success")
            else old.get("status", "MISSING")
        )
        cells = []
        for arm in arms.values():
            values = []
            for phase in ("cold", "warm"):
                selected = [
                    s
                    for s in arm["samples"]
                    if s["query"] == query and s["phase"] == phase
                ]
                values.append(
                    ", ".join(
                        (
                            fmt(s["query_seconds"])
                            if s["valid_success"]
                            else (
                                s.get("failure_class", "INCOMPLETE_VALIDATION")
                                if s["status"] == "PASS"
                                else s["status"]
                            )
                        )
                        for s in selected
                    )
                )
            cells.append(" / ".join(values))
        lines.append(
            f"| {query.upper()} | {previous} | {cells[0]} | {cells[1]} | [evidence](profiles/{query}.md) |"
        )
    lines += ["", "## Failures and incomplete evidence", ""]
    for name, arm in arms.items():
        for sample in arm["samples"]:
            if sample["valid_success"] or sample["status"] in {
                "MISSING_MANIFEST",
                "NOT_RUN_YET",
                "NOT_PLANNED",
            }:
                continue
            detail = sample.get("detail") or "; ".join(sample["validation_issues"])
            lines.append(
                f"- {name} {sample['query']} r{sample['repetition']:02}: {sample['status']} ({sample.get('failure_class')}); {safe(detail)[:1800]}"
            )
    lines += ["", "## Comparability and warnings", ""]
    lines.extend(f"- {safe(warning)}" for warning in report["warnings"])
    lines += [
        "",
        "Raw outputs were read without modification. See `analysis.json` for every sample, source manifest, comparison gate, and scoped profile metric; `samples.csv` contains the flat sample ledger.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))
    profiles = output / "profiles"
    profiles.mkdir(exist_ok=True)
    for query in QUERIES:
        text = [
            f"# {query.upper()} profiling evidence",
            "",
            "CN/engine logs provide execution and memory evidence. They do not alone prove GPU utilization, equal scan work, or the cause of a throughput change. Missing events are reported as unavailable.",
            "",
        ]
        for name, arm in arms.items():
            text += [f"## {name}", ""]
            for sample in arm["samples"]:
                if sample["query"] != query:
                    continue
                text.append(
                    f"- r{sample['repetition']:02} {sample['phase']}: {sample['status']}; validated={sample['valid_success']}; query seconds={fmt(sample['query_seconds'])}."
                )
                comparison = sample.get("comparison", {})
                if comparison:
                    text.append(
                        f"  Oracle match={comparison.get('match')}; maximum absolute numeric error={comparison.get('max_absolute_numeric_error', 'unavailable')}; maximum relative numeric error={comparison.get('max_relative_numeric_error', 'unavailable')}. A match within tolerance need not be exact."
                    )
                audit = sample.get("execution_validation")
                if audit:
                    text.append(
                        f"  Execution placement={audit['status']}; detected FE retry={audit['detected_retry']}; initial CNs={audit['initial_nodes']}; final CNs={audit['final_nodes']}; final FE query IDs={audit['final_query_ids']}; issue={audit['issue'] or 'none'}."
                    )
                for label, path in sample["artifacts"]["paths"].items():
                    if label in {"explain.tsv", "starrocks.stderr", "comparison.json"}:
                        text.append(f"  [{label}]({path})")
            clusters = sorted(
                {
                    s.get("cluster")
                    for s in arm["samples"]
                    if s["query"] == query and s.get("cluster")
                }
            )
            for cluster in clusters:
                evidence = report["clusters"].get(cluster)
                if not evidence:
                    continue
                text += [
                    "",
                    f"Cluster: `{cluster}`; aggregates include all repetitions and administrative SQL on it. Administrative failures do not classify the TPC-H attempt as failed.",
                    "",
                    "| CN | Finished fragments | Max sampled GPU allocation | Reported lifetime GPU peak | Transfer bytes | Ingress bytes | Reload bytes | Pack bytes |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
                for cn, node in evidence["cns"].items():
                    metrics = node["cluster_metrics"]
                    stages = [
                        metrics["events"]
                        .get(stage, {})
                        .get("numeric_fields", {})
                        .get("bytes", {})
                        .get("sum")
                        for stage in ("transfer", "ingress", "reload", "pack")
                    ]
                    finished = (
                        None
                        if node["cn_log_missing"]
                        else metrics["fragment_counts"]["finished"]
                    )
                    values = [
                        finished,
                        metrics["pool"]["max_sampled_allocated_bytes"],
                        metrics["pool"]["max_reported_lifetime_peak_bytes"],
                        *stages,
                    ]
                    text.append(
                        f"| {cn} | "
                        + " | ".join(
                            "unavailable" if value is None else f"{value:,.0f}"
                            for value in values
                        )
                        + " |"
                    )
                text += [
                    "",
                    "Per-FE-query finished-fragment distribution: `"
                    + json.dumps(
                        evidence["finished_fragment_distribution"], sort_keys=True
                    )
                    + "`.",
                    "",
                ]
                telemetry = evidence["telemetry_distribution"]
                if telemetry["available"]:
                    data = telemetry["data"]
                    headline, metric = data.get("headline", "task"), data.get(
                        "metric", "entities"
                    )
                    text += [
                        f"Saved [CN telemetry distribution]({telemetry['path']}), preferred for work-unit distribution:",
                        "",
                        f"| CN | Engine process UUID | {safe(headline)} {safe(metric)} | Missing | Parse errors |",
                        "|---|---|---:|---|---|",
                    ]
                    for node in data["cns"]:
                        if not isinstance(node, dict):
                            continue
                        value = node.get(metric, {}).get(headline)
                        text.append(
                            f"| {safe(node.get('cn'))} | {safe(node.get('run_uuid'))} | {value if value is not None else 'unavailable'} | {node.get('missing')} | {safe(node.get('parse_errors', {}))} |"
                        )
                    text += [
                        "",
                        "Saved balance verdict (work-unit counts): `"
                        + json.dumps(data.get("balance"), sort_keys=True)
                        + "`.",
                        "",
                    ]
                text.extend(f"- {note}" for note in telemetry["notes"])
                text += [
                    "",
                    "Timed SQL windows (administrative activity outside these windows is excluded):",
                    "",
                    "| Attempt | CN | Started / finished / failed fragments | Max sampled GPU allocation | Ingress bytes | Pack wait max (µs) |",
                    "|---|---|---|---:|---:|---:|",
                ]
                for sample in arm["samples"]:
                    if sample["query"] != query or sample.get("cluster") != cluster:
                        continue
                    for cn, node in evidence["cns"].items():
                        metrics = node["per_sample"].get(sample["sample_id"])
                        if metrics is None:
                            text.append(
                                f"| r{sample['repetition']:02} | {cn} | unavailable UTC window | — | — | — |"
                            )
                            continue
                        fragments = (
                            "unavailable"
                            if node["cn_log_missing"]
                            else " / ".join(
                                str(metrics["fragment_counts"][state])
                                for state in ("started", "finished", "failed")
                            )
                        )
                        ingress = (
                            metrics["events"]
                            .get("ingress", {})
                            .get("numeric_fields", {})
                            .get("bytes", {})
                            .get("sum")
                        )
                        wait = (
                            metrics["events"]
                            .get("pack_wait", {})
                            .get("numeric_fields", {})
                            .get("pack_wait_us", {})
                            .get("max")
                        )
                        text.append(
                            f"| r{sample['repetition']:02} | {cn} | {fragments} | {fmt(metrics['pool']['max_sampled_allocated_bytes'])} | {fmt(ingress)} | {fmt(wait)} |"
                        )
                text += [
                    "",
                    "Credit retirement records are cumulative per FE query; their last snapshots are preserved in `analysis.json`. Missing credit events are unavailable, not proof of zero pressure.",
                    "",
                ]
                for cn, node in evidence["cns"].items():
                    metrics = node["cluster_metrics"]
                    for kind in (
                        "pool",
                        "retirement",
                        "ingress",
                        "reload",
                        "pack",
                        "reschedule_or_capacity",
                        "error",
                    ):
                        records = metrics["evidence"].get(kind, [])
                        if records:
                            record = records[-1]
                            text.append(
                                f"- {cn} {kind}: [{Path(record['path']).name}:{record['line']}]({record['path']}#L{record['line']}) — `{safe(record['text'])}`"
                            )
                text += [
                    "",
                    "Per-attempt window metrics, cumulative retirement counters, and additional source lines are in `analysis.json`. No timestamp window means no per-attempt attribution; cluster-level evidence remains available.",
                    "",
                ]
        (profiles / f"{query}.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "build/tpch-starrocks-sf500-2cn"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Expected slots per query, including cold (default 3)",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    output = args.output.resolve()
    if output in {
        args.baseline.resolve(),
        args.optimized.resolve(),
        args.reference.resolve(),
    }:
        parser.error(
            "--output must be a separate directory from the raw arm/reference roots"
        )
    output.mkdir(parents=True, exist_ok=True)
    warnings = []
    arms = {
        name: load_arm(path.resolve(), name, args.repetitions, args.reference.resolve())
        for name, path in (("baseline", args.baseline), ("optimized", args.optimized))
    }
    for arm in arms.values():
        summarize_arm(arm)
        warnings.extend(arm["warnings"])
    clusters = defaultdict(list)
    for arm in arms.values():
        for sample in arm["samples"]:
            if sample.get("cluster") and not sample["cluster"].endswith(
                "unknown-cluster"
            ):
                clusters[sample["cluster"]].append(sample)
    cluster_reports = {
        path: scan_cluster(Path(path), samples, warnings)
        for path, samples in clusters.items()
    }
    settings_comparability = compare_run_settings(arms, warnings)
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_notes": [
            "Historical reference: one measured attempt per query, with clusters restarted after failure; no controlled cold/warm repetition distribution.",
            f"Current protocol expects all 22 queries in each arm with {args.repetitions} slots per query: one fresh-application-cluster cold sample followed by warm samples. OS page cache is uncontrolled and transport warmup precedes timing.",
            "Each per-query A/B block is retained separately. Raw start timestamps/manifests reveal actual ordering; this analyzer does not assume alternation or combine repeated blocks by selecting their fastest sample.",
            "Run-setting warnings do not change recorded correctness or timing samples. Missing settings remain unknown; any legacy async-dispatch default inference includes the inspected launcher and frozen CN identities in resolved_run_settings. Transfer-window stability is checked within the optimized arm because the legacy baseline ignores that knob.",
            "Query time is the SQL client interval. Cluster startup, EXPLAIN COSTS, oracle checks, and restart time are excluded. Failed client durations remain visible but never count as successful query throughput.",
            "Historical and current watchdog/client deadlines differ; compare those manifests before interpreting failure timing. The current harness records forced CTE reuse and two CNs sharing one physical GPU.",
            "PASS requires an affirmative oracle comparison, successful client/runner exit, and exactly two CNs/zero BEs before and after the query. The documented comparison uses 1e-6 relative and 1e-8 absolute tolerance; per-attempt runner settings and maximum numeric errors are retained. A match within tolerance need not be exact. A missing comparison, row count, time, or topology is incomplete evidence.",
            "Execution eligibility additionally excludes a distributed FE execution whose automatic retry finishes on fewer CNs, or an observed retry whose placement cannot be resolved. Legitimate single-CN plans without a retry retain their existing eligibility. Missing execution logs without an observed retry remain explicitly UNKNOWN; registered-node availability alone does not establish actual placement.",
        ],
        "historical": historic_reference(args.reference.resolve(), warnings),
        "arms": arms,
        "comparisons": [
            common_comparison(arms["baseline"], arms["optimized"], phase)
            for phase in ("cold", "warm")
        ],
        "clusters": cluster_reports,
        "settings_comparability": settings_comparability,
        "warnings": warnings,
    }
    with (output / "samples.csv").open("w", newline="") as target:
        columns = [
            "arm",
            "query",
            "repetition",
            "phase",
            "status",
            "valid_success",
            "query_seconds",
            "failure_class",
            "execution_validation_status",
            "harness_manifest",
            "attempt",
            "cluster",
            "detail",
        ]
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for arm in arms.values():
            writer.writerows(arm["samples"])
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    render(report, output)
    print(output / "RESULTS.md")


if __name__ == "__main__":
    main()

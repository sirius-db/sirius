# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Finding dedup, repro artifacts and the run summary."""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
import shutil
import time
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Any

from .classify import SEVERITY, Verdict, normalize_reason
from .artifacts import seal, write_json, sql_literal
from .config import FuzzConfig, FUZZ_DIR


@dataclass
class QueryRecord:
    worker: int
    dataset: str  # dataset file stem, e.g. "w0-d3"
    seed: int
    sql: str
    verdict: str
    reason: str = ""  # short classification text (error reason / setting name)
    detail: str = ""  # comparator detail or full error
    labels: list[str] = field(default_factory=list)
    reduced_sql: str | None = None
    reduced_labels: list[str] = field(default_factory=list)
    reduced_reason: str = ""  # error reason reported by the reduced query
    elapsed_cpu: float = 0.0
    elapsed_gpu: float = 0.0
    variant: dict[str, Any] | None = None
    diffs: list[str] = field(default_factory=list)
    reduction_steps: int = 0
    comparison: str = "multiset"
    evidence: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnownIssue:
    pattern: str  # regex over the finding's reason and detail text
    issue: str
    note: str = ""
    verdicts: list[str] = field(default_factory=list)
    sql_pattern: str = ""  # optional regex over the (reduced) query; both must match

    def matches(self, rec: QueryRecord) -> bool:
        if self.verdicts and rec.verdict not in self.verdicts:
            return False
        flags = re.IGNORECASE | re.MULTILINE
        if re.search(self.pattern, rec.reason + "\n" + rec.detail, flags) is None:
            return False
        if self.sql_pattern:
            return (
                re.search(self.sql_pattern, rec.reduced_sql or rec.sql, flags)
                is not None
            )
        return True


def load_known_issues(path: pathlib.Path | None) -> list[KnownIssue]:
    if path is None or not path.exists():
        return []
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    out = []
    for item in data.get("issue", []):
        if "pattern" not in item or "issue" not in item:
            raise ValueError(f"known issue entry needs pattern and issue: {item}")
        out.append(
            KnownIssue(
                item["pattern"],
                item["issue"],
                item.get("note", ""),
                list(item.get("verdicts", [])),
                item.get("sql_pattern", ""),
            )
        )
    return out


EXTRA_REPRODUCERS = 5  # additional query-<n>.sql files kept per deduplicated finding


def signature(rec: QueryRecord) -> str:
    """Group errors by reason; retain distinct mismatch and timeout observations."""
    v = rec.verdict
    if v in (Verdict.MISMATCH, Verdict.VARIANT_MISMATCH, Verdict.TIMEOUT):
        # Findings are persisted before reduction, which may itself crash. Operator
        # labels are too coarse to discard later examples: retain each input and
        # observed difference, including the value of a variant setting.
        evidence = {
            "dataset": rec.dataset,
            "sql": rec.sql,
            "comparison": rec.comparison,
            "variant": rec.variant,
            "reason": rec.reason,
            "detail": rec.detail,
            "diffs": rec.diffs,
            "phase": rec.context.get("phase"),
            "stage": rec.context.get("stage"),
            "results": [
                (op.get("phase"), op.get(f"fingerprint_{rec.comparison}"))
                for op in rec.evidence.get("operations", [])
            ],
        }
        digest = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, default=str).encode()
        )
        key = f"{v}|{digest.hexdigest()}"
    elif v in (
        Verdict.PLAN_FALLBACK,
        Verdict.RUNTIME_FALLBACK,
        Verdict.GPU_ERROR,
        Verdict.GPU_INTERNAL_ERROR,
        Verdict.GPU_OOM,
    ):
        key = f"{v}|{normalize_reason(rec.reason)}"
    elif v == Verdict.CRASH and ("SIG" in rec.reason or "terminate" in rec.reason):
        key = f"{v}|{normalize_reason(rec.reason)}"
    else:
        key = f"{v}|{','.join(rec.reduced_labels or rec.labels)}"
    return key


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


class Report:
    def __init__(
        self,
        run_dir: pathlib.Path,
        config: FuzzConfig,
        known_issues: list[KnownIssue],
        seed: int,
        mode: str = "correctness",
    ):
        self.run_dir = run_dir
        self.cfg = config
        self.known = known_issues
        self.seed = seed
        self.mode = mode
        self.counts: collections.Counter[str] = collections.Counter()
        self.cpu_error_reasons: collections.Counter[str] = collections.Counter()
        self.findings: dict[str, dict[str, Any]] = {}
        self.feature_stats: collections.Counter[str] = collections.Counter()
        self.started = time.time()
        self.queries = 0
        self.status = "complete"
        self.stop_reason = "budget completed"
        self.worker_context: dict[int, dict[str, Any]] = {}
        self.record_paths: dict[tuple[str, str], pathlib.Path] = {}
        self.record_signatures: dict[tuple[str, str], str] = {}
        self.datasets = 0
        (run_dir / "findings").mkdir(parents=True, exist_ok=True)
        (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "queries.jsonl"
        self._log = open(self.log_path, "a", encoding="utf-8")

    # -- ingestion -------------------------------------------------------------

    def add(self, rec: QueryRecord) -> str | None:
        """Record a query; returns the finding directory name when a new finding was written."""
        self.queries += 1
        verdict = Verdict(rec.verdict)
        for ki in self.known:
            if verdict.is_finding() and ki.matches(rec):
                rec.reason = f"{ki.issue}: {rec.reason}"
                rec.verdict = Verdict.KNOWN_ISSUE.value
                verdict = Verdict.KNOWN_ISSUE
                break
        self.counts[rec.verdict] += 1
        if verdict == Verdict.CPU_ERROR:
            self.cpu_error_reasons[normalize_reason(rec.reason)] += 1
        self._log.write(json.dumps(asdict(rec), default=str) + "\n")
        self._log.flush()
        if not verdict.is_finding() and verdict != Verdict.KNOWN_ISSUE:
            return None
        sig = signature(rec)
        self.record_signatures[(rec.dataset, rec.sql)] = sig
        if sig in self.findings:
            entry = self.findings[sig]
            entry["count"] += 1
            entry["gpu_seconds"] += rec.elapsed_gpu
            if entry["count"] <= EXTRA_REPRODUCERS + 1:
                name = f"{entry['name']}/additional/{entry['count']:03d}"
                self._write_finding(name, rec, sig)
            return None
        name = f"{len(self.findings):03d}-{rec.verdict}-{short_hash(sig)}"
        self.findings[sig] = {
            "name": name,
            "count": 1,
            "verdict": rec.verdict,
            "reason": rec.reason,
            "worker": rec.worker,
            "sql": rec.sql,
            "labels": rec.labels,
            "reduced_sql": None,
            "reduced_labels": [],
            "reduced_reason": "",
            "gpu_seconds": rec.elapsed_gpu,  # over every query with this signature
        }
        self._write_finding(name, rec, sig)
        return name

    def add_feature_stats(self, stats: dict[str, int]) -> None:
        self.feature_stats.update(stats)

    def dataset_path(self, stem: str) -> pathlib.Path:
        return self.run_dir / "datasets" / f"{stem}.sql"

    # -- artifacts -------------------------------------------------------------

    def _write_finding(self, name: str, rec: QueryRecord, sig: str) -> None:
        d = self.run_dir / "findings" / name
        d.mkdir(parents=True, exist_ok=True)
        self.record_paths[(rec.dataset, rec.sql)] = d
        (d / "query.sql").write_text(rec.sql.rstrip() + ";\n")
        repro_sql = rec.reduced_sql or rec.sql
        if rec.reduced_sql:
            (d / "reduced.sql").write_text(rec.reduced_sql.rstrip() + ";\n")
        ds_src = self.dataset_path(rec.dataset)
        if ds_src.exists():
            shutil.copy(ds_src, d / "dataset.sql")
        (d / "config.toml").write_text(self.cfg.to_toml())
        meta = asdict(rec)
        context = self.worker_context.get(rec.worker, {})
        for source, dest in (("environment.json", "environment.json"),):
            path = self.run_dir / source
            if path.exists():
                shutil.copy(path, d / dest)
        for source, dest in (
            (context.get("sirius_config"), "sirius.yaml"),
            (context.get("runtime"), "runtime.json"),
            (context.get("stderr"), "worker.stderr"),
        ):
            if source and pathlib.Path(source).exists():
                shutil.copy(source, d / dest)
        meta.update(
            {
                "signature": sig,
                "run_seed": self.seed,
                "config_hash": self.cfg.config_hash(),
                "bundle_version": 1,
                "execution": {
                    "cpu_only": context.get("cpu_only"),
                    "sirius_config_required": bool(context.get("sirius_config")),
                    "allow_metadata_mismatch": context.get(
                        "allow_metadata_mismatch", False
                    ),
                },
            }
        )
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        detail = [
            f"verdict: {rec.verdict}",
            f"reason: {rec.reason}",
            f"detail: {rec.detail}",
        ]
        if rec.variant:
            detail.append(f"variant: {rec.variant}")
        detail += rec.diffs
        detail.append("")
        detail.append(
            "Reproduce from a Sirius checkout (use this bundle's current absolute path):"
        )
        detail.append(
            "  pixi run -e duckdb-python fuzz replay /absolute/path/to/bundle"
        )
        detail.append("See REPLAY.md and repro.sql for standalone shell instructions.")
        (d / "detail.txt").write_text("\n".join(detail) + "\n")
        (d / "repro_catch2.cpp").write_text(
            catch2_snippet(name, rec, ds_src if ds_src.exists() else None, repro_sql)
        )
        # Relative files make the SQL usable after copying the entire bundle. Invoke
        # from the bundle directory with the matching Sirius-linked DuckDB shell.
        query = rec.sql.rstrip(";\n") + ";\n"
        variant_sql = "".join(
            f"SET {key} = {sql_literal(value) if isinstance(value, str) else value};\n"
            for key, value in (rec.variant or {}).items()
        )
        (d / "repro.sql").write_text(
            "-- Run in this directory with a matching Sirius-loaded DuckDB shell.\n"
            "-- Use a fresh database path for each attempt.\n"
            "SET gpu_execution = false;\nATTACH 'repro.duckdb' AS repro;\nUSE repro;\n"
            ".read dataset.sql\nCHECKPOINT;\n"
            "-- Reference result\n"
            + query
            + "SET enable_duckdb_fallback = false;\nSET gpu_execution = true;\n"
            "-- GPU baseline\n"
            + query
            + ("-- Recorded variant\n" + variant_sql + query if variant_sql else "")
        )
        (d / "REPLAY.md").write_text(
            "Run from a Sirius checkout (replace the path):\n\n"
            "```sh\npixi run -e duckdb-python fuzz replay /absolute/path/to/this/bundle\n```\n\n"
            "Replay restores config.toml, sirius.yaml and the recorded variant. It writes a new output directory; this bundle is not modified.\n"
            "Use --extension to select the matching binary, or deliberately test another build (the new fingerprint is recorded). "
            "Check environment.json and runtime.json for the original environment. YAML may reference machine-specific spill paths: "
            "use --sirius-config for a local copy when necessary; the override is recorded.\n\n"
            "For a standalone shell reproduction, start the matching DuckDB shell in this directory, set SIRIUS_CONFIG_FILE to sirius.yaml, "
            "LOAD the matching Sirius extension, then `.read repro.sql`. Use a fresh directory/database for each attempt. "
            "repro.sql prints CPU, strict GPU and recorded variant results as applicable; it does not compare them.\n"
        )
        seal(d)

    def add_reduction(self, rec: QueryRecord) -> None:
        d = self.record_paths.get((rec.dataset, rec.sql))
        if d is None or not rec.reduced_sql:
            return
        (d / "reduced.sql").write_text(rec.reduced_sql.rstrip() + ";\n")
        write_json(
            d / "reduction.json",
            {
                "sql": rec.reduced_sql,
                "reason": rec.reduced_reason,
                "steps": rec.reduction_steps,
                "comparison": "multiset",
                "variant": rec.variant,
                "note": "Original evidence retained in meta.json; reduced SQL requires a fresh replay.",
            },
        )
        # The summary shows the smallest reproducer seen for the signature.
        entry = self.findings.get(
            self.record_signatures.get((rec.dataset, rec.sql), "")
        )
        if entry is not None and (
            entry["reduced_sql"] is None
            or len(rec.reduced_sql) < len(entry["reduced_sql"])
        ):
            entry.update(
                reduced_sql=rec.reduced_sql,
                reduced_labels=rec.reduced_labels,
                reduced_reason=rec.reduced_reason,
            )

    @staticmethod
    def gaps(findings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Gap findings grouped by the reason their smallest reproducer reports.

        ``runtime`` fallbacks come first, ordered by the GPU time they threw away,
        since those are the checks worth moving to plan time; ``plan`` rejections
        follow, ordered by count. Reduction narrows an "Unsupported <expression>"
        rejection to the function Sirius cannot translate, so findings that
        started from different expressions around the same function share a
        group. A group shows the smallest reduced query it has, or an original
        query when nothing in it was reduced.
        """
        groups: dict[str, dict[str, dict[str, Any]]] = {"runtime": {}, "plan": {}}
        for f in findings:
            verdict = Verdict(f["verdict"])
            if not verdict.is_gap():
                continue
            kind = "runtime" if verdict == Verdict.RUNTIME_FALLBACK else "plan"
            reduced = f["reduced_sql"] is not None
            reason = f["reduced_reason"] or f["reason"]
            sql = f["reduced_sql"] or f["sql"]
            labels = f["reduced_labels"] or f["labels"]
            key = normalize_reason(reason)
            group = groups[kind].setdefault(
                key,
                {
                    "verdict": f["verdict"],
                    "key": key,
                    "reason": reason,
                    "count": 0,
                    "gpu_seconds": 0.0,
                    "sql": sql,
                    "labels": labels,
                    "reduced": reduced,
                    "findings": [],
                },
            )
            group["count"] += f["count"]
            group["gpu_seconds"] += f.get("gpu_seconds", 0.0)
            group["findings"].append(f["name"])
            if (reduced, -len(sql)) > (group["reduced"], -len(group["sql"])):
                group.update(reason=reason, sql=sql, labels=labels, reduced=reduced)
        return {
            "runtime": sorted(
                groups["runtime"].values(),
                key=lambda g: (-g["gpu_seconds"], -g["count"], g["key"]),
            ),
            "plan": sorted(
                groups["plan"].values(), key=lambda g: (-g["count"], g["key"])
            ),
        }

    def finish(self) -> dict[str, Any]:
        self._log.close()
        elapsed = time.time() - self.started
        ordered = sorted(
            self.findings.values(),
            key=lambda f: (
                (
                    SEVERITY.index(Verdict(f["verdict"]))
                    if Verdict(f["verdict"]) in SEVERITY
                    else 99
                ),
                -f["count"],
            ),
        )
        summary = {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "mode": self.mode,
            "seed": self.seed,
            "config_hash": self.cfg.config_hash(),
            "config_path": self.cfg.source_path,
            "elapsed_seconds": round(elapsed, 1),
            "queries": self.queries,
            "datasets": self.datasets,
            "counts": dict(self.counts),
            "findings": ordered,
            "gaps": self.gaps(ordered),
            "cpu_error_reasons": self.cpu_error_reasons.most_common(20),
            "feature_stats": dict(sorted(self.feature_stats.items())),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        text = self.render_summary(summary)
        (self.run_dir / "summary.txt").write_text(text)
        return summary

    def render_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            f"siriusfuzz run: {self.run_dir}",
            f"status={summary.get('status', 'unknown')}: {summary.get('stop_reason', '')}",
            f"mode={summary.get('mode', 'correctness')} seed={summary['seed']} config={summary['config_hash']} "
            f"queries={summary['queries']} datasets={summary['datasets']} elapsed={summary['elapsed_seconds']}s",
            "",
            "verdict counts:",
        ]
        for k, v in sorted(summary["counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:20s} {v}")
        gaps = summary.get("gaps") or {}
        runtime, plan = gaps.get("runtime", []), gaps.get("plan", [])

        def names(g: dict[str, Any]) -> str:
            more = len(g["findings"]) - 3
            return (
                "findings: "
                + ", ".join(g["findings"][:3])
                + (f" (+{more})" if more > 0 else "")
            )

        def reasons(groups: list[dict[str, Any]]) -> str:
            return f"{len(groups)} reason" + ("" if len(groups) == 1 else "s")

        if runtime:
            seconds = sum(g["gpu_seconds"] for g in runtime)
            lines.append("")
            lines.append(
                f"runtime fallbacks ({reasons(runtime)}, "
                f"{sum(g['count'] for g in runtime)} queries, {seconds:.1f}s of GPU work "
                "thrown away); smallest query that passes the planner and still fails, "
                "and its features:"
            )
            for g in runtime:
                lines.append(
                    f"  x{g['count']:<4d} {g['gpu_seconds']:6.1f}s  {g['reason'][:110]}"
                )
                lines.append(f"      {' '.join(g['sql'].split())[:220]}")
                lines.append(f"      features: {', '.join(g['labels'])}   {names(g)}")
        if plan:
            lines.append("")
            lines.append(
                f"plan-time fallbacks ({reasons(plan)}, "
                f"{sum(g['count'] for g in plan)} queries):"
            )
            for g in plan:
                shown = g["reason"] if g["reduced"] else g["key"]
                lines.append(f"  x{g['count']:<4d} {shown[:110]}   {names(g)}")
                if g["reduced"]:
                    lines.append(f"      {' '.join(g['sql'].split())[:220]}")
                    lines.append(f"      features: {', '.join(g['labels'])}")
        findings = [
            f for f in summary["findings"] if not Verdict(f["verdict"]).is_gap()
        ]
        lines.append("")
        lines.append(f"findings ({len(findings)} unique):")
        for f in findings:
            lines.append(
                f"  [{f['verdict']}] x{f['count']:<4d} {f['name']}  {f['reason'][:100]}"
            )
        if summary["cpu_error_reasons"]:
            lines.append("")
            lines.append("skipped (CPU error) reasons, top 20:")
            for reason, n in summary["cpu_error_reasons"]:
                lines.append(f"  {n:5d}  {reason}")
        never = [
            f for f in _CLAIMED_FEATURES(self.cfg) if self.feature_stats.get(f, 0) == 0
        ]
        lines.append("")
        lines.append(f"features emitted: {len(summary['feature_stats'])} distinct")
        if never and summary["queries"] >= 100:
            lines.append("WARNING enabled features never emitted: " + ", ".join(never))
        return "\n".join(lines) + "\n"


def _CLAIMED_FEATURES(cfg: FuzzConfig) -> list[str]:
    """Feature counters that must be non-zero for a run of any length, given the config."""
    f = cfg.features
    claimed = ["where"]
    if f.aggregates.functions:
        claimed += ["group_by", "ungrouped_aggregate"]
    claimed += [f"join:{jt}" for jt in f.joins.types]
    if f.full_join:
        claimed.append("join:full")
    if f.cross_join:
        claimed.append("join:cross")
    if f.grouping_sets:
        claimed.append("grouping_sets")
    if f.order_by.enabled:
        claimed.append("order_by")
    if f.limit.enabled:
        claimed.append("limit")
    if f.cte.materialized:
        claimed.append("cte")
    for op, on in (
        ("UNION ALL", f.set_ops.union_all),
        ("UNION", f.set_ops.union),
        ("EXCEPT", f.set_ops.except_),
        ("INTERSECT", f.set_ops.intersect),
    ):
        if on:
            claimed.append(f"setop:{op}")
    if f.subqueries.exists:
        claimed.append("subquery:exists")
    if f.subqueries.in_:
        claimed.append("subquery:in")
    if f.subqueries.scalar:
        claimed.append("subquery:scalar")
    if f.window_functions:
        claimed.append("window")
    if f.distinct:
        claimed.append("distinct")
    return claimed


def catch2_snippet(
    name: str, rec: QueryRecord, dataset_path: pathlib.Path | None, query_sql: str
) -> str:
    schema_hint = (
        f"// Data: see dataset.sql next to this file ({dataset_path.name}); paste its CREATE/INSERT\n"
        "// statements into run_ok() calls, or load it with the shell before running the query.\n"
        if dataset_path
        else "// Data: dataset SQL was not captured.\n"
    )
    comparator = (
        "compare_gpu_vs_cpu_ordered"
        if rec.comparison == "ordered"
        else "compare_gpu_vs_cpu"
    )
    comparison = f"  {comparator}(query);\n"
    if rec.variant:
        comparison += (
            "  auto baseline = con->Query(query);\n"
            "  REQUIRE(baseline);\n"
            "  REQUIRE_FALSE(baseline->HasError());\n"
        )
        restores = ""
        for i, (setting, value) in enumerate(rec.variant.items()):
            current_sql = json.dumps(f"SELECT current_setting({sql_literal(setting)});")
            literal = (
                sql_literal(value) if isinstance(value, str) else str(value).lower()
            )
            set_sql = json.dumps(f"SET {setting} = {literal};")
            restore_prefix = json.dumps(f"SET {setting} = ")
            comparison += (
                f"  auto original_{i} = con->Query({current_sql});\n"
                f"  REQUIRE(original_{i});\n"
                f"  REQUIRE_FALSE(original_{i}->HasError());\n"
                f"  run_ok({set_sql});\n"
            )
            restores = (
                f'  run_ok({restore_prefix} + original_{i}->GetValue(0, 0).ToSQLString() + ";");\n'
                + restores
            )
        sort = "false" if rec.comparison == "ordered" else "true"
        comparison += (
            "  auto before = sirius::test::get_transparent_execution_stats(*con);\n"
            "  auto variant = con->Query(query);\n"
            "  auto after = sirius::test::get_transparent_execution_stats(*con);\n"
            "  // Restore settings before asserting on the variant result.\n"
            + restores
            + "  REQUIRE(variant);\n"
            "  REQUIRE_FALSE(variant->HasError());\n"
            "  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);\n"
            "  REQUIRE(baseline->ColumnCount() == variant->ColumnCount());\n"
            f"  auto baseline_rows = collect_rows(baseline->Cast<duckdb::MaterializedQueryResult>(), {sort});\n"
            f"  auto variant_rows = collect_rows(variant->Cast<duckdb::MaterializedQueryResult>(), {sort});\n"
            "  REQUIRE(baseline_rows == variant_rows);\n"
        )
    return (
        "// Generated by siriusfuzz; drop into test/cpp/integration/test_gpu_execution_fuzz.cpp\n"
        f"{schema_hint}"
        "TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,\n"
        f'                 "fuzz repro {name}",\n'
        '                 "[integration][gpu_execution][fuzz]")\n'
        "{\n"
        '  run_ok(R"SQL(\n'
        "    -- schema + data from dataset.sql\n"
        '  )SQL");\n'
        '  run_ok("CHECKPOINT;");\n'
        f"  // verdict: {rec.verdict}; reason: {rec.reason[:100]}\n"
        '  const std::string query = R"SQL(\n'
        f"{query_sql}\n"
        '  )SQL";\n' + comparison + "}\n"
    )


def default_known_issues_path(cfg: FuzzConfig) -> pathlib.Path | None:
    if not cfg.oracle.known_issues:
        return None
    p = pathlib.Path(cfg.oracle.known_issues)
    if p.is_absolute():
        return p
    if cfg.source_path:
        candidate = pathlib.Path(cfg.source_path).resolve().parent / p
        if candidate.exists():
            return candidate
    return FUZZ_DIR / p

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
from .artifacts import write_json
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


EXTRA_REPRODUCERS = 5  # further query/dataset pairs kept under more/ per finding


def signature(rec: QueryRecord) -> str:
    """Group errors by reason; keep each distinct mismatch or timeout input apart."""
    v = rec.verdict
    if v in (Verdict.MISMATCH, Verdict.VARIANT_MISMATCH, Verdict.TIMEOUT):
        variant = json.dumps(rec.variant, sort_keys=True) if rec.variant else ""
        key = f"{v}|{rec.dataset}|{rec.comparison}|{variant}|{rec.sql}"
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
        self._environment: dict[str, Any] | None = None
        self._runtime: dict[str, dict[str, Any]] = {}
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
        observed = verdict = Verdict(rec.verdict)
        reason = rec.reason
        issue = None
        for ki in self.known:
            if verdict.is_finding() and ki.matches(rec):
                issue = ki.issue
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
                self._write_more(entry["name"], entry["count"], rec)
            return None
        name = f"{len(self.findings):03d}-{rec.verdict}-{short_hash(sig)}"
        self.findings[sig] = {
            "name": name,
            "count": 1,
            "verdict": rec.verdict,
            "observed": observed.value,  # the verdict before a known issue claimed it
            "issue": issue,
            "reason": reason,
            "detail": rec.detail,
            "diffs": list(rec.diffs[:3]),
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
        if rec.reduced_sql:
            (d / "reduced.sql").write_text(rec.reduced_sql.rstrip() + ";\n")
        ds_src = self.dataset_path(rec.dataset)
        if ds_src.exists():
            shutil.copy(ds_src, d / "dataset.sql")
        (d / "config.toml").write_text(self.cfg.to_toml())
        context = self.worker_context.get(rec.worker, {})
        yaml = context.get("sirius_config")
        if yaml and pathlib.Path(yaml).exists():
            shutil.copy(yaml, d / "sirius.yaml")
        stderr = context.get("stderr")
        if (
            rec.verdict in (Verdict.CRASH.value, Verdict.TIMEOUT.value)
            and stderr
            and pathlib.Path(stderr).exists()
        ):
            shutil.copy(stderr, d / "worker.stderr")
        meta = asdict(rec)
        meta.update(
            {
                "signature": sig,
                "run_seed": self.seed,
                "config_hash": self.cfg.config_hash(),
                "bundle_version": 2,
                "execution": {
                    "cpu_only": context.get("cpu_only"),
                    "sirius_config_required": bool(yaml),
                },
                "environment": self._environment_json(),
                "runtime": self._runtime_json(context.get("runtime")),
            }
        )
        write_json(d / "meta.json", meta)
        self._write_finding_md(d, name, rec)

    def _write_more(self, finding: str, count: int, rec: QueryRecord) -> None:
        """A further query with the same signature: just its SQL and data."""
        d = self.run_dir / "findings" / finding / "more" / f"{count:03d}"
        d.mkdir(parents=True, exist_ok=True)
        self.record_paths[(rec.dataset, rec.sql)] = d
        (d / "query.sql").write_text(rec.sql.rstrip() + ";\n")
        ds_src = self.dataset_path(rec.dataset)
        if ds_src.exists():
            shutil.copy(ds_src, d / "dataset.sql")

    def _environment_json(self) -> dict[str, Any]:
        if self._environment is None:
            path = self.run_dir / "environment.json"
            self._environment = json.loads(path.read_text()) if path.exists() else {}
        return self._environment

    def _runtime_json(self, path: str | None) -> dict[str, Any]:
        if not path:
            return {}
        if path not in self._runtime:
            p = pathlib.Path(path)
            self._runtime[path] = json.loads(p.read_text()) if p.exists() else {}
        return self._runtime[path]

    @staticmethod
    def _write_finding_md(d: pathlib.Path, name: str, rec: QueryRecord) -> None:
        """The one file to read: what happened, the query, how to replay."""
        lines = [
            f"# {name}",
            "",
            f"- verdict: `{rec.verdict}`",
            f"- reason: {rec.reason}",
        ]
        if rec.variant:
            lines.append(f"- setting variant: `{rec.variant}`")
        lines.append(f"- comparison: {rec.comparison}; dataset: {rec.dataset}")
        if rec.reduced_sql:
            lines += [
                "",
                f"## Query (reduced in {rec.reduction_steps} steps; original in query.sql)",
                "",
                "```sql",
                rec.reduced_sql.rstrip().rstrip(";") + ";",
                "```",
            ]
            if rec.reduced_reason and rec.reduced_reason != rec.reason:
                lines.append(f"\nReduced query's reason: {rec.reduced_reason}")
        else:
            lines += [
                "",
                "## Query",
                "",
                "```sql",
                rec.sql.rstrip().rstrip(";") + ";",
                "```",
            ]
        if rec.detail or rec.diffs:
            lines += ["", "## What differed", ""]
            if rec.detail:
                lines.append(rec.detail)
            lines += [f"- {line}" for line in rec.diffs]
        lines += [
            "",
            "## Replay",
            "",
            "```sh",
            f"pixi run fuzz replay {d.resolve()}",
            "```",
            "",
            "Replay restores config.toml, sirius.yaml and the recorded setting variant, writes a new",
            "directory and leaves this one unchanged. `--original` replays query.sql instead of",
            "reduced.sql; `--extension` tests another build. If this directory was copied, use its",
            "new path. `fuzz recheck <run-dir>` replays every finding of the run at once.",
            "",
            "## Files",
            "",
            "query.sql and dataset.sql are the original inputs; reduced.sql and reduction.json exist",
            "when reduction made progress; meta.json holds the full record, provenance and session",
            "settings; worker.stderr is kept for crashes and hangs; more/<n>/ holds further",
            "query/dataset pairs with the same signature.",
            "",
        ]
        (d / "FINDING.md").write_text("\n".join(lines))

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
        if (d / "FINDING.md").exists():
            self._write_finding_md(d, d.name, rec)
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
        query when nothing in it was reduced. A gap that matched known_issues.toml
        stays in its table, tagged with the issue, so the list is complete after
        the issue is filed.
        """
        groups: dict[str, dict[str, dict[str, Any]]] = {"runtime": {}, "plan": {}}
        for f in findings:
            observed = Verdict(f.get("observed", f["verdict"]))
            if not observed.is_gap():
                continue
            kind = "runtime" if observed == Verdict.RUNTIME_FALLBACK else "plan"
            reduced = f["reduced_sql"] is not None
            reason = f["reduced_reason"] or f["reason"]
            sql = f["reduced_sql"] or f["sql"]
            labels = f["reduced_labels"] or f["labels"]
            key = normalize_reason(reason)
            group = groups[kind].setdefault(
                key,
                {
                    "verdict": observed.value,
                    "key": key,
                    "reason": reason,
                    "count": 0,
                    "gpu_seconds": 0.0,
                    "sql": sql,
                    "labels": labels,
                    "reduced": reduced,
                    "findings": [],
                    "issues": [],
                },
            )
            group["count"] += f["count"]
            group["gpu_seconds"] += f.get("gpu_seconds", 0.0)
            group["findings"].append(f["name"])
            if f.get("issue") and f["issue"] not in group["issues"]:
                group["issues"].append(f["issue"])
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

        def known(g: dict[str, Any]) -> str:
            return f"   known: {', '.join(g['issues'])}" if g.get("issues") else ""

        def one_line(sql: str) -> str:
            return " ".join(sql.split())[:220]

        # Findings first: a crash or a wrong answer outranks any gap, in either mode.
        # Gaps, known or not, are in the tables below.
        findings = [
            f
            for f in summary["findings"]
            if not Verdict(f.get("observed", f["verdict"])).is_gap()
        ]
        lines.append("")
        lines.append(
            f"findings ({len(findings)} unique); smallest query that reproduces:"
        )
        for f in findings:
            reason = f"{f['issue']}: {f['reason']}" if f.get("issue") else f["reason"]
            lines.append(
                f"  [{f['verdict']}] x{f['count']:<4d} {f['name']}  {reason[:100]}"
            )
            lines.append(f"      {one_line(f.get('reduced_sql') or f.get('sql', ''))}")
            labels = f.get("reduced_labels") or f.get("labels") or []
            if labels:
                lines.append(f"      features: {', '.join(labels)}")
            for diff in f.get("diffs", [])[:2]:
                lines.append(f"      {diff[:160]}")
        if runtime:
            seconds = sum(g["gpu_seconds"] for g in runtime)
            lines.append("")
            lines.append(
                f"runtime fallbacks ({reasons(runtime)}, "
                f"{sum(g['count'] for g in runtime)} queries, {seconds:.1f}s of GPU work "
                "thrown away); smallest query that passes the planner and still fails:"
            )
            for g in runtime:
                lines.append(
                    f"  x{g['count']:<4d} {g['gpu_seconds']:6.1f}s  {g['reason'][:110]}"
                    + known(g)
                )
                lines.append(f"      {one_line(g['sql'])}")
                lines.append(f"      features: {', '.join(g['labels'])}   {names(g)}")
        if plan:
            lines.append("")
            lines.append(
                f"plan-time fallbacks ({reasons(plan)}, "
                f"{sum(g['count'] for g in plan)} queries):"
            )
            for g in plan:
                shown = g["reason"] if g["reduced"] else g["key"]
                lines.append(
                    f"  x{g['count']:<4d} {shown[:110]}{known(g)}   {names(g)}"
                )
                if g["reduced"]:
                    lines.append(f"      {one_line(g['sql'])}")
                    lines.append(f"      features: {', '.join(g['labels'])}")
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

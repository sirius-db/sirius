# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Command line entry point: ``python -m siriusfuzz <command>``."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
import shutil
import tempfile
import math

from .artifacts import provenance, write_json
from .isolation import supervise
from . import __version__
from .classify import Verdict
from .config import (
    FUZZ_DIR,
    MODES,
    REPO_ROOT,
    FuzzConfig,
    load_config,
    resolve_repo_path,
)
from .report import Report, default_known_issues_path, load_known_issues
from .runner import Orchestrator, OrchestratorOptions
from .schema_gen import DataGenerator

SELFTEST_DEFAULT_OVERRIDES = (
    "data.rows=[8,16]",
    "features.joins.max_tables=2",
    "features.subqueries.max_depth=1",
)
DEFAULT_DURATION_SECONDS = 600.0  # a run given neither --duration nor --queries
DOCTOR_TIMEOUT_SECONDS = 120.0
BUILD_MODULE_HINT = (
    "build it from the submodule with: pixi run -e duckdb-python build-duckdb-python"
)


def cli_path(text: str, what: str) -> pathlib.Path:
    """Resolve a path from the command line: from the current directory first, then
    from the repository root, so repo-relative paths work from any directory."""
    given = pathlib.Path(text).expanduser()
    bases = [pathlib.Path.cwd()]
    if REPO_ROOT != bases[0]:
        bases.append(REPO_ROOT)
    for base in bases:
        candidate = base / given  # keep the path as typed; no symlink resolution
        if candidate.exists():
            return candidate
    raise ValueError(
        f"{what} not found: {given} (looked in {' and '.join(map(str, bases))})"
    )


def check_duckdb_module() -> None:
    """Fail early, with the fix, when the DuckDB Python module is missing.

    From the repository root a bare ``import duckdb`` can resolve to the
    ``duckdb/`` source checkout as a namespace package instead of failing.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ValueError(
            f"the duckdb Python module is not importable ({exc}); {BUILD_MODULE_HINT}"
        ) from exc
    if not hasattr(duckdb, "connect"):
        where = getattr(duckdb, "__path__", None) or getattr(duckdb, "__file__", "?")
        raise ValueError(
            f"'import duckdb' found {where}, not the Python module; {BUILD_MODULE_HINT}"
        )


def parse_duration(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip().lower()
    if not t:
        raise ValueError("duration must not be empty")
    mult = 1.0
    if t.endswith("ms"):
        return float(t[:-2]) / 1000
    if t[-1] in "smh":
        mult = {"s": 1, "m": 60, "h": 3600}[t[-1]]
        t = t[:-1]
    return float(t) * mult


def _common_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default=None,
        help="TOML configuration (default: config/default.toml)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. features.window_functions=true",
    )


def _mode_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mode",
        choices=MODES,
        default="correctness",
        help="correctness (default): fuzz the configuration's enabled features, where any "
        "fallback is a finding. gaps: find the queries Sirius accepts at plan time and "
        "hands to the CPU at runtime; every generator feature is on regardless of the "
        "configuration, setting variants are skipped, each runtime fallback is reduced to "
        "the smallest query that passes the planner and still fails, and plan-time "
        "rejections are listed by reason without reduction",
    )


def _common_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--extension", help="path to sirius.duckdb_extension (default: from config)"
    )
    p.add_argument(
        "--sirius-config",
        action="append",
        default=None,
        help="Sirius YAML config; repeat to round-robin across workers",
    )
    p.add_argument(
        "--allow-metadata-mismatch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="explicitly bypass version metadata only for independently verified matching builds",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="do not load Sirius; compare CPU against CPU (harness self-check)",
    )


def _load(args: argparse.Namespace) -> FuzzConfig:
    path = cli_path(args.config, "configuration") if args.config else None
    return load_config(path, args.set, getattr(args, "mode", "correctness"))


def _engine(args: argparse.Namespace, cfg: FuzzConfig) -> tuple[str | None, list[str]]:
    check_duckdb_module()
    if args.cpu_only:
        return None, []
    if args.extension:
        ext = cli_path(args.extension, "extension")
    else:
        ext = resolve_repo_path(cfg.sirius.extension)
        if not ext.exists():
            raise ValueError(
                f"extension not found: {ext} (build Sirius first with pixi run make, "
                "or pass --extension / --cpu-only)"
            )
    if args.sirius_config:
        resolved = [str(cli_path(c, "Sirius config")) for c in args.sirius_config]
    else:
        resolved = []
        for c in cfg.sirius.configs:
            p = resolve_repo_path(c)
            if not p.exists():
                raise ValueError(f"Sirius config not found: {p}")
            resolved.append(str(p))
    return str(ext), resolved


def _make_run_dir(out: str | None, seed: int) -> pathlib.Path:
    base = pathlib.Path(out) if out else FUZZ_DIR / "out"
    base.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(
        tempfile.mkdtemp(
            prefix=f"run-{time.strftime('%Y%m%d-%H%M%S')}-seed{seed}-", dir=base
        )
    ).resolve()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    validate_limits(args)
    extension, sirius_configs = _engine(args, cfg)
    seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().randrange(1, 2**31)
    )
    run_dir = _make_run_dir(args.out, seed)
    (run_dir / "config.toml").write_text(cfg.to_toml())
    sirius_configs = snapshot_configs(run_dir, sirius_configs)
    write_json(run_dir / "environment.json", provenance(extension))
    write_json(
        run_dir / "invocation.json",
        {k: v for k, v in vars(args).items() if k != "func"},
    )
    if not getattr(args, "no_doctor", False):
        print("Readiness check (skip with --no-doctor):", flush=True)
        result = run_doctor(
            extension,
            sirius_configs,
            run_dir / "doctor",
            DOCTOR_TIMEOUT_SECONDS,
            getattr(args, "allow_metadata_mismatch", False),
        )
        if result["status"] != "ok":
            return 130 if result["status"] == "cancelled" else 2
    known = load_known_issues(default_known_issues_path(cfg))
    report = Report(run_dir, cfg, known, seed, mode=args.mode)
    opts = OrchestratorOptions(
        workers=args.workers,
        duration=parse_duration(args.duration),
        max_queries=args.queries,
        extension=extension,
        sirius_configs=sirius_configs,
        reduce=not args.no_reduce,
        quiet=args.quiet,
        max_respawns=args.max_respawns,
        allow_metadata_mismatch=getattr(args, "allow_metadata_mismatch", False),
        mode=args.mode,
    )
    if opts.duration is None and opts.max_queries is None:
        opts.duration = DEFAULT_DURATION_SECONDS
        budget = (
            f"{DEFAULT_DURATION_SECONDS / 60:g}m (default; set --duration or --queries)"
        )
    elif opts.duration is not None and opts.max_queries is not None:
        budget = f"{args.duration} or {opts.max_queries} queries, whichever comes first"
    elif opts.duration is not None:
        budget = args.duration
    else:
        budget = f"{opts.max_queries} queries"
    print(
        f"siriusfuzz {__version__}: mode={args.mode} config={cfg.config_hash()} seed={seed} "
        f"workers={opts.workers} budget={budget} "
        f"{'cpu-only' if extension is None else extension} -> {run_dir}",
        file=sys.stderr,
    )
    summary = Orchestrator(cfg, report, seed, opts).run()
    print(report.render_summary(summary))
    if summary["findings"]:
        first = run_dir / "findings" / summary["findings"][0]["name"]
        print(
            f"Findings: {run_dir / 'findings'}\n"
            f"Replay one with: pixi run -e duckdb-python fuzz replay {first}"
        )
    # Gaps are the expected output of a gaps run, not a failure of it.
    findings = [
        f
        for f in summary["findings"]
        if f["verdict"] != Verdict.KNOWN_ISSUE.value
        and not (args.mode == "gaps" and Verdict(f["verdict"]).is_gap())
    ]
    if summary["status"] == "cancelled":
        return 130
    if summary["status"] != "complete":
        return 2
    if args.fail_on_findings and findings:
        return 1
    return 0


def snapshot_configs(work: pathlib.Path, paths: list[str]) -> list[str]:
    saved = []
    for index, source in enumerate(paths):
        dest = work / f"sirius-{index}.yaml"
        shutil.copy(source, dest)
        saved.append(str(dest))
    return saved


def validate_limits(args: argparse.Namespace) -> None:
    for name in ("workers", "queries", "timeout"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name} must be positive and finite")
    duration = getattr(args, "duration", None)
    if duration is not None and (
        not math.isfinite(parse_duration(duration)) or parse_duration(duration) <= 0
    ):
        raise ValueError("--duration must be positive and finite")
    if getattr(args, "max_respawns", 0) < 0:
        raise ValueError("--max-respawns must be nonnegative")


def run_doctor(
    extension: str | None,
    configs: list[str],
    work: pathlib.Path,
    timeout: float,
    allow_metadata_mismatch: bool | None,
) -> dict:
    """Readiness checks in a disposable subprocess; prints one line per check."""
    work.mkdir(parents=True, exist_ok=True)
    configs = snapshot_configs(work, configs)
    write_json(work / "environment.json", provenance(extension))
    result = supervise(
        {
            "operation": "doctor",
            "extension": extension,
            "sirius_config": configs[0] if configs else None,
            "allow_metadata_mismatch": allow_metadata_mismatch,
        },
        work,
        timeout,
    )
    if result["status"] == "ok":
        for check in result["checks"]:
            print(f"PASS  {check}")
        print(
            "PASS  GPU interception and execution verified"
            if result["gpu_verified"]
            else "PASS  CPU-only checks; GPU readiness was not tested"
        )
    else:
        print(
            f"FAIL  {result['status']}: {result.get('error', result.get('exitcode', ''))}"
        )
        print(
            f"Check stderr.log and outcome.json in {work}. Build matching sources with "
            "pixi run make and pixi run -e duckdb-python build-duckdb-python. Check the "
            "GPU and selected Sirius YAML if initialization failed."
        )
    return result


def cmd_doctor(args: argparse.Namespace) -> int:
    validate_limits(args)
    cfg = _load(args)
    extension, configs = _engine(args, cfg)
    work = _make_run_dir(args.out, 0)
    (work / "config.toml").write_text(cfg.to_toml())
    print(f"PASS  TOML configuration and output directory: {work}", flush=True)
    result = run_doctor(
        extension, configs, work, args.timeout, args.allow_metadata_mismatch
    )
    if result["status"] == "ok":
        print(
            "Ready to fuzz."
            if result["gpu_verified"]
            else "Ready for CPU-only selftest."
        )
        return 0
    return 130 if result["status"] == "cancelled" else 2


def replay(args: argparse.Namespace, target: pathlib.Path, work: pathlib.Path) -> dict:
    """Replay one finding directory or SQL file into ``work``; returns the outcome.

    A finding directory restores its own configuration, Sirius YAML, comparison
    mode and recorded setting variant. ``args`` may be updated with values the
    bundle recorded (cpu-only, YAML, metadata override), so pass a copy when
    replaying several bundles.
    """
    bundle = target if target.is_dir() else None
    meta = {}
    if bundle:
        for required in ("query.sql", "dataset.sql", "config.toml", "meta.json"):
            if not (bundle / required).is_file():
                raise ValueError(f"incomplete reproducer: missing {required}")
        meta = json.loads((bundle / "meta.json").read_text())
        config_path = (
            cli_path(args.config, "configuration")
            if args.config
            else bundle / "config.toml"
        )
        query = bundle / (
            "query.sql"
            if args.original or not (bundle / "reduced.sql").exists()
            else "reduced.sql"
        )
        if args.sirius_config is None:
            if (bundle / "sirius.yaml").exists():
                args.sirius_config = [str(bundle / "sirius.yaml")]
            elif (
                not (args.cpu_only or meta.get("execution", {}).get("cpu_only"))
                and meta.get("execution", {}).get("sirius_config_required") is not False
            ):
                raise ValueError(
                    "incomplete reproducer: missing saved Sirius YAML; supply --sirius-config explicitly"
                )
        args.cpu_only = args.cpu_only or meta.get("execution", {}).get(
            "cpu_only", False
        )
        if args.allow_metadata_mismatch is None:
            args.allow_metadata_mismatch = meta.get("execution", {}).get(
                "allow_metadata_mismatch", False
            )
            if args.allow_metadata_mismatch:
                print(
                    "NOTE restoring the recorded metadata-mismatch override.",
                    file=sys.stderr,
                )
    else:
        config_path = cli_path(args.config, "configuration") if args.config else None
        query = target
    if getattr(args, "query_override", None):
        query = pathlib.Path(args.query_override)
    cfg = load_config(config_path, args.set)
    extension, configs = _engine(args, cfg)
    dataset = (
        pathlib.Path(args.dataset).resolve()
        if args.dataset
        else (bundle / "dataset.sql" if bundle else None)
    )
    if dataset is None and args.dataset_seed is None:
        raise ValueError(
            "SQL-file replay needs --dataset or an explicit --dataset-seed"
        )
    if not query.is_file() or (dataset is not None and not dataset.is_file()):
        raise ValueError("incomplete reproducer: query or dataset file missing")
    work.mkdir(parents=True, exist_ok=True)
    configs = snapshot_configs(work, configs)
    (work / "config.toml").write_text(cfg.to_toml())
    shutil.copy(query, work / "query.sql")
    if dataset is not None:
        shutil.copy(dataset, work / "dataset.sql")
    else:
        ds = DataGenerator(cfg, random.Random(args.dataset_seed)).generate(
            args.dataset_seed
        )
        (work / "dataset.sql").write_text(ds.schema_sql() + "\n" + ds.data_sql())
    environment = provenance(extension)
    write_json(work / "environment.json", environment)
    original_environment = meta.get("environment") or (
        json.loads((bundle / "environment.json").read_text())
        if bundle and (bundle / "environment.json").exists()  # older bundles
        else {}
    )
    if (
        original_environment.get("extension")
        and environment.get("extension")
        and original_environment["extension"]["sha256"]
        != environment["extension"]["sha256"]
    ):
        print(
            "NOTE extension differs from the original finding; this attempt records the new binary.",
            file=sys.stderr,
        )
    comparison = meta.get("comparison", "multiset")
    if bundle and query.name == "reduced.sql" and (bundle / "reduction.json").exists():
        reduction = json.loads((bundle / "reduction.json").read_text())
        comparison = reduction.get("comparison", "multiset")
    if args.ordered:
        comparison = "ordered"
    write_json(
        work / "replay.json",
        {
            "source": str(target),
            "source_metadata": meta,
            "selected_query": str(query),
            "original_environment": original_environment,
            "overrides": {k: v for k, v in vars(args).items() if k != "func"},
        },
    )
    baseline_settings = meta.get("runtime", {}).get("session_settings") or (
        json.loads((bundle / "runtime.json").read_text()).get("session_settings", {})
        if bundle and (bundle / "runtime.json").exists()  # older bundles
        else {}
    )
    result = supervise(
        {
            "operation": "replay",
            "extension": extension,
            "sirius_config": configs[0] if configs else None,
            "allow_metadata_mismatch": args.allow_metadata_mismatch,
            "query": str(work / "query.sql"),
            "dataset": str(work / "dataset.sql"),
            "config": str(work / "config.toml"),
            "variant": meta.get("variant"),
            "comparison": comparison,
            "session_settings": baseline_settings,
        },
        work,
        args.timeout,
    )
    replay_meta = result.get(
        "record",
        {"verdict": result["status"], "context": result.get("last_operation", {})},
    )
    replay_meta.update(
        {
            "comparison": comparison,
            "variant": meta.get("variant"),
            "execution": {
                "cpu_only": args.cpu_only,
                "allow_metadata_mismatch": args.allow_metadata_mismatch,
                "sirius_config_required": bool(configs),
            },
        }
    )
    write_json(work / "meta.json", replay_meta)
    if configs:
        shutil.copy(configs[0], work / "sirius.yaml")
    return result


def replay_verdict(result: dict) -> str:
    """The verdict a replay produced, or the supervisor status when it did not run."""
    return result.get("record", {}).get("verdict", result["status"])


def cmd_replay(args: argparse.Namespace) -> int:
    validate_limits(args)
    target = pathlib.Path(args.target).resolve()
    work = _make_run_dir(args.out, 0)
    print(f"Replay evidence: {work}", flush=True)
    result = replay(args, target, work)
    print(json.dumps(result, indent=2, default=str))
    if result["status"] == "cancelled":
        return 130
    if result["status"] == "setup_error":
        return 2
    if result["status"] != "ok":
        return 1
    return 0 if replay_verdict(result) == Verdict.OK.value else 1


def cmd_recheck(args: argparse.Namespace) -> int:
    """Replay every finding of a run against the current build; print old and new verdicts."""
    validate_limits(args)
    run_dir = pathlib.Path(args.run_dir).resolve()
    findings_dir = run_dir / "findings"
    bundles = (
        sorted(p for p in findings_dir.iterdir() if p.is_dir())
        if findings_dir.is_dir()
        else []
    )
    if not bundles:
        raise ValueError(f"no findings under {run_dir}")
    work = _make_run_dir(args.out, 0)
    print(f"Recheck of {len(bundles)} findings from {run_dir}; evidence in {work}\n")
    print(f"  {'finding':<40} {'recorded':>18}    {'now':<18} reason")
    rows = []
    cancelled = False
    for bundle in bundles:
        meta_path = bundle / "meta.json"
        before = (
            json.loads(meta_path.read_text()).get("verdict", "?")
            if meta_path.exists()
            else "?"
        )
        attempt = argparse.Namespace(**vars(args))
        result = replay(attempt, bundle, work / bundle.name)
        after = replay_verdict(result)
        reason = result.get("record", {}).get("reason") or result.get("error") or ""
        rows.append(
            {
                "finding": bundle.name,
                "recorded": before,
                "now": after,
                "reason": reason,
                "evidence": str(work / bundle.name),
            }
        )
        print(
            f"  {bundle.name:<40} {before:>18} -> {after:<18} {reason[:60]}", flush=True
        )
        if result["status"] == "cancelled":
            cancelled = True
            break
    cleared = sum(r["now"] == Verdict.OK.value for r in rows)
    write_json(
        work / "recheck.json",
        {
            "run": str(run_dir),
            "status": "cancelled" if cancelled else "complete",
            "cleared": cleared,
            "findings": rows,
        },
    )
    print(
        f"\n{cleared} of {len(rows)} findings no longer reproduce; details in {work / 'recheck.json'}"
    )
    if cancelled:
        return 130
    return 0 if cleared == len(rows) else 1


def cmd_show_config(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(cfg.to_toml())
    print(f"# mode: {args.mode}")
    print(f"# config hash: {cfg.config_hash()}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """CPU-vs-CPU run: proves the generator, comparator, reducer and report work without a GPU."""
    # A developer smoke should exercise the harness without materializing millions of rows.
    # Explicit --set values come last so callers can still choose a larger workload.
    args.set = [*SELFTEST_DEFAULT_OVERRIDES, *args.set]
    args.cpu_only = True
    args.extension = None
    args.sirius_config = None
    args.no_reduce = False
    args.quiet = True
    args.fail_on_findings = True
    args.duration = None
    args.max_respawns = 0
    args.allow_metadata_mismatch = False
    if args.queries is None:
        args.queries = 150
    if args.seed is None:
        args.seed = 20260922
    rc = cmd_run(args)
    print("selftest " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="siriusfuzz", description="Generative differential fuzzer for Sirius"
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate, run and compare queries")
    _common_config_args(run)
    _mode_arg(run)
    _common_engine_args(run)
    run.add_argument("--seed", type=int)
    run.add_argument(
        "--duration",
        help="time budget, e.g. 30m, 2h, 90s (default 10m when --queries is not given either)",
    )
    run.add_argument(
        "--queries",
        type=int,
        help="total query budget; with --duration too, whichever is reached first stops the run",
    )
    run.add_argument(
        "--no-doctor",
        action="store_true",
        help="skip the readiness check (the doctor command) that run performs first",
    )
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--out", help=f"output root (default {FUZZ_DIR / 'out'})")
    run.add_argument("--no-reduce", action="store_true")
    run.add_argument(
        "--max-respawns",
        type=int,
        default=200,
        help="worker restarts after crashes/hangs before giving up",
    )
    run.add_argument("--quiet", action="store_true")
    run.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit 1 when any non-known finding was recorded",
    )
    run.set_defaults(func=cmd_run)

    doctor = sub.add_parser(
        "doctor", help="check configuration, module, extension and GPU readiness"
    )
    _common_config_args(doctor)
    _common_engine_args(doctor)
    doctor.add_argument("--out")
    doctor.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="hard deadline for the setup probe (seconds)",
    )
    doctor.set_defaults(func=cmd_doctor)

    rp = sub.add_parser("replay", help="re-run one finding directory or .sql file")
    _common_config_args(rp)
    _common_engine_args(rp)
    rp.add_argument("target")
    rp.add_argument(
        "--dataset", help="dataset.sql to load (default: the finding's dataset.sql)"
    )
    rp.add_argument("--dataset-seed", type=int)
    rp.add_argument(
        "--original",
        action="store_true",
        help="replay query.sql instead of reduced.sql",
    )
    rp.add_argument("--out")
    rp.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="hard deadline for the entire replay subprocess, including setup and cleanup (seconds)",
    )
    rp.add_argument(
        "--ordered",
        action="store_true",
        help="compare ordered rows for a custom SQL file with deterministic ordering",
    )
    rp.set_defaults(func=cmd_replay)

    rc = sub.add_parser(
        "recheck",
        help="replay every finding of a run against the current build and compare verdicts",
    )
    _common_config_args(rc)
    _common_engine_args(rc)
    rc.add_argument("run_dir", help="a run directory with a findings/ subdirectory")
    rc.add_argument(
        "--original",
        action="store_true",
        help="replay each finding's query.sql instead of its reduced.sql",
    )
    rc.add_argument("--out")
    rc.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="hard deadline per finding, including setup (seconds)",
    )
    rc.set_defaults(
        func=cmd_recheck,
        dataset=None,
        dataset_seed=None,
        ordered=False,
        query_override=None,
    )

    sc = sub.add_parser("show-config", help="print the effective configuration")
    _common_config_args(sc)
    _mode_arg(sc)
    sc.set_defaults(func=cmd_show_config)

    stp = sub.add_parser("selftest", help="CPU-only harness check (no GPU needed)")
    _common_config_args(stp)
    _mode_arg(stp)
    stp.add_argument("--seed", type=int)
    stp.add_argument("--queries", type=int)
    stp.add_argument("--workers", type=int, default=1)
    stp.add_argument("--out")
    stp.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

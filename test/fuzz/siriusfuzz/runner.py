# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Worker processes and the orchestrator.

A worker owns one Sirius session and a stream of generated datasets; for each
query it runs CPU, then GPU, classifies, filters ambiguity, runs setting
variants and reduces findings. The orchestrator spawns workers, consumes their
messages, detects crashes (worker exit) and hangs (no progress past the
timeout), respawns, and feeds the report.
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import pathlib
import queue as queue_mod
import random
import re
import shutil
import signal
import json
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .classify import Verdict, classify_gpu_error, normalize_reason, uses_plan_fallback
from .artifacts import runtime_info, write_json
from .compare import ResultSet, Tolerances, compare_mode, compare_results
from .config import FuzzConfig
from .query_gen import QueryGenerator
from .reduce import reduce_query
from .report import QueryRecord, Report
from .schema_gen import DataGenerator, Dataset
from .session import RunResult, Session
from .sqlast import Query, labels

HANG_GRACE_SECONDS = 30.0


class Mailbox:
    """Atomic messages survive native worker death and deliberate termination.

    A multiprocessing.Queue can be corrupted when its writer is killed midway
    through a frame. Small on-disk messages avoid blocking the supervisor on a
    damaged pipe and retain completed observations at cancellation.
    """

    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def put(self, message: dict[str, Any]) -> None:
        write_json(
            self.directory / f"{time.time_ns():020d}-{uuid.uuid4().hex}.json", message
        )

    def get_nowait(self) -> dict[str, Any]:
        paths = sorted(self.directory.glob("*.json"))
        if not paths:
            raise queue_mod.Empty
        path = paths[0]
        message = json.loads(path.read_text())
        path.unlink()
        return message

    def get(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.get_nowait()
            except queue_mod.Empty:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)


@dataclass
class WorkerArgs:
    worker_id: int
    seed: int
    run_dir: str
    extension: str | None
    sirius_config: str | None
    max_queries: int | None  # per worker
    deadline: float | None  # time.time() at which to stop
    reduce: bool = True
    stderr_path: str | None = None  # per-worker capture of native backtraces
    allow_metadata_mismatch: bool = False
    spawn_id: int = 0


# --------------------------------------------------------------------------
# evaluation of one query
# --------------------------------------------------------------------------


class Evaluator:
    def __init__(self, cfg: FuzzConfig, session: Session, log: Callable[[str], None]):
        self.cfg = cfg
        self.s = session
        self.log = log
        self.tol = Tolerances(
            cfg.oracle.float32_rel_tol, cfg.oracle.float64_rel_tol, cfg.oracle.abs_tol
        )
        self.timeout = cfg.oracle.query_timeout_seconds
        self.variants: dict[str, list[Any]] = {}
        if session.gpu_available:
            for name, values in cfg.variants.settings.items():
                if session.setting_supported(name):
                    self.variants[name] = list(values)
                else:
                    log(f"variant setting {name!r} not available; skipping")
        self.dataset: Dataset | None = None
        self.alias = ""
        self.perm_alias: str | None = None
        self.rng = random.Random()
        self.sqlsmith = cfg.oracle.reduce_sqlsmith != "off"
        self.forced_variant: dict[str, Any] | None = None
        self.replay_mode: str | None = None

    def set_dataset(self, ds: Dataset, alias: str) -> None:
        self.dataset = ds
        self.alias = alias
        self.perm_alias = None

    # -- main entry --------------------------------------------------------

    def evaluate(
        self, query: Query | None, sql: str, worker: int, dataset_stem: str, seed: int
    ) -> QueryRecord:
        mode = self.replay_mode or compare_mode(query)
        rec = QueryRecord(
            worker=worker,
            dataset=dataset_stem,
            seed=seed,
            sql=sql,
            verdict=Verdict.OK.value,
            comparison=mode,
        )
        rec.labels = labels(query) if query is not None else []
        cpu = self.s.run(sql, gpu=False, timeout=self.timeout)
        rec.elapsed_cpu = cpu.elapsed
        if cpu.status == "timeout":
            rec.verdict = Verdict.CPU_TIMEOUT.value
            return rec
        if cpu.status == "error":
            rec.verdict = Verdict.CPU_ERROR.value
            rec.reason = cpu.error.split("\n", 1)[0][:200]
            return rec
        assert cpu.result is not None
        cols = self.s.describe(sql)
        if cols is not None and len(cols) == len(cpu.result.columns):
            cpu.result.columns = cols
        if not self.s.gpu_available:
            # cpu-only harness check: a second CPU run must agree with the first.
            gpu = self.s.run(sql, gpu=False, timeout=self.timeout)
        else:
            gpu = self.s.run(sql, gpu=True, timeout=self.timeout)
        rec.elapsed_gpu = gpu.elapsed
        if gpu.status == "timeout":
            rec.verdict = Verdict.TIMEOUT.value
            rec.reason = f"GPU run exceeded {self.timeout}s"
            return rec
        if gpu.status == "error":
            verdict, reason = classify_gpu_error(gpu.error)
            rec.verdict = verdict.value
            rec.reason = reason
            rec.detail = gpu.error.split("\n", 1)[0][:500]
            if verdict == Verdict.PLAN_FALLBACK:
                self._handle_plan_fallback(rec, cpu.result, mode)
            return rec
        assert gpu.result is not None
        gpu.result.columns = cpu.result.columns
        outcome = compare_results(cpu.result, gpu.result, mode == "ordered", self.tol)
        if not outcome.equal:
            if self.cfg.oracle.ambiguity_filter and self._is_ambiguous(
                sql, cpu.result, mode
            ):
                rec.verdict = Verdict.AMBIGUOUS.value
                rec.detail = outcome.detail
                return rec
            rec.verdict = Verdict.MISMATCH.value
            rec.reason = outcome.detail
            rec.detail = f"compare={mode}: {outcome.detail}"
            rec.diffs = outcome.diffs
            return rec
        self._run_variants(rec, sql, gpu.result, mode)
        return rec

    # -- helpers -------------------------------------------------------------

    def _handle_plan_fallback(
        self, rec: QueryRecord, cpu_result: ResultSet, mode: str
    ) -> None:
        policy = self.cfg.oracle.on_plan_fallback
        if policy != "count" or not self.s.gpu_available:
            return
        rec.context["plan_fallback_reason"] = rec.reason
        # Frontier: the fallback path itself must stay clean. Re-run with fallback on.
        self.s.begin_fallback_retry(rec.reason)
        try:
            self.s.con.execute("SET enable_duckdb_fallback = true")
            fb = self.s.run(rec.sql, gpu=True, timeout=self.timeout)
        finally:
            try:
                self.s.con.execute("SET enable_duckdb_fallback = false")
            finally:
                self.s.end_fallback_retry()
        if fb.status == "ok" and fb.result is not None:
            fb.result.columns = cpu_result.columns
            outcome = compare_results(
                cpu_result, fb.result, mode == "ordered", self.tol
            )
            if not outcome.equal:
                rec.verdict = Verdict.FALLBACK_MISMATCH.value
                rec.reason = outcome.detail
                rec.detail = f"fallback path: {outcome.detail}"
                rec.diffs = outcome.diffs
        elif fb.status == "timeout":
            rec.verdict = Verdict.TIMEOUT.value
            rec.reason = f"fallback run exceeded {self.timeout}s"
            rec.detail = f"fallback path timed out: {fb.error[:300]}"
        elif fb.status == "error":
            verdict, rec.reason = classify_gpu_error(fb.error, fallback_enabled=True)
            rec.verdict = verdict.value
            rec.detail = f"fallback path failed: {fb.error[:300]}"

    def _is_ambiguous(self, sql: str, cpu_result: ResultSet, mode: str) -> bool:
        """DQP filter: re-run on CPU over the same rows inserted in a different order."""
        if self.dataset is None:
            return False
        try:
            if self.perm_alias is None:
                self.perm_alias = f"{self.alias}p"
                self.s.load_dataset(
                    self.dataset,
                    self.perm_alias,
                    permutation_seed=self.dataset.seed + 1,
                )
            self.s.use(self.perm_alias)
            perm = self.s.run(sql, gpu=False, timeout=self.timeout)
        finally:
            self.s.use(self.alias)
        if perm.status != "ok" or perm.result is None:
            return False
        perm.result.columns = cpu_result.columns
        return not compare_results(
            cpu_result, perm.result, mode == "ordered", self.tol
        ).equal

    def _run_variants(
        self, rec: QueryRecord, sql: str, baseline: ResultSet, mode: str
    ) -> None:
        if self.forced_variant:
            choices = list(self.forced_variant.items())
        elif not self.variants or self.cfg.variants.per_query <= 0:
            return
        else:
            names = self.rng.sample(
                sorted(self.variants),
                min(self.cfg.variants.per_query, len(self.variants)),
            )
            choices = [(name, self.rng.choice(self.variants[name])) for name in names]
        for name, value in choices:
            try:
                self.s.set(name, value)
                res = self.s.run(sql, gpu=True, timeout=self.timeout)
            finally:
                self.s.restore(name)
            if res.status == "timeout":
                rec.verdict = Verdict.TIMEOUT.value
                rec.reason = f"variant {name}={value} exceeded {self.timeout}s"
                rec.variant = {name: value}
                return
            if res.status == "error":
                verdict, reason = classify_gpu_error(res.error)
                rec.verdict = verdict.value
                rec.reason = f"variant {name}={value}: {reason}"
                rec.detail = res.error.split("\n", 1)[0][:500]
                rec.variant = {name: value}
                return
            assert res.result is not None
            res.result.columns = baseline.columns
            outcome = compare_results(baseline, res.result, mode == "ordered", self.tol)
            if not outcome.equal:
                rec.verdict = Verdict.VARIANT_MISMATCH.value
                rec.reason = f"{name}={value}: {outcome.detail}"
                rec.detail = f"compare={mode}: {outcome.detail}"
                rec.diffs = outcome.diffs
                rec.variant = {name: value}
                return

    # -- reduction -------------------------------------------------------------

    def reduce(self, rec: QueryRecord, query: Query | None) -> None:
        if query is None or rec.verdict in (Verdict.TIMEOUT.value, Verdict.CRASH.value):
            return
        check = self._make_still_fails(rec)
        if check is None:
            return
        self.s.mark_auxiliary("reduction", rec.sql)
        smith = None
        if self.sqlsmith and self.s.load_sqlsmith():
            smith = self._sqlsmith_candidates
        elif self.cfg.oracle.reduce_sqlsmith == "on":
            self.log("sqlsmith extension unavailable; string-level reduction skipped")
        result = reduce_query(
            query, rec.sql, check, self.cfg.oracle.reduce_max_steps, smith
        )
        if result.sql != rec.sql:
            rec.reduced_sql = result.sql
            rec.reduction_steps = result.steps + result.sqlsmith_steps
            if result.query is not None:
                rec.reduced_labels = labels(result.query)

    def _sqlsmith_candidates(self, sql: str) -> list[str]:
        self.s.mark_auxiliary("sqlsmith_reduction", sql)
        try:
            rows = self.s.con.execute(
                "SELECT sql FROM reduce_sql_statement(?)", [sql]
            ).fetchall()
        except Exception:
            return []
        return [r[0] for r in rows if r and r[0]]

    def _make_still_fails(self, rec: QueryRecord) -> Callable[[str], bool] | None:
        verdict = Verdict(rec.verdict)
        variant = rec.variant or {}
        gpu = self.s.gpu_available
        fallback = uses_plan_fallback(rec.verdict, rec.context)

        def cpu_ok(sql: str) -> RunResult | None:
            res = self.s.run(sql, gpu=False, timeout=self.timeout)
            return res if res.status == "ok" and res.result is not None else None

        def plan_rejected(sql: str) -> str | None:
            strict = self.s.run(sql, gpu=gpu, timeout=self.timeout)
            if strict.status == "error":
                classified, reason = classify_gpu_error(strict.error)
                if classified == Verdict.PLAN_FALLBACK:
                    return reason
            return None

        def run_fallback(sql: str, reason: str) -> RunResult:
            self.s.begin_fallback_retry(reason)
            try:
                self.s.con.execute("SET enable_duckdb_fallback = true")
                return self.s.run(sql, gpu=gpu, timeout=self.timeout)
            finally:
                try:
                    self.s.con.execute("SET enable_duckdb_fallback = false")
                finally:
                    self.s.end_fallback_retry()

        if verdict in (Verdict.MISMATCH, Verdict.FALLBACK_MISMATCH):

            def check(sql: str) -> bool:
                cpu = cpu_ok(sql)
                if cpu is None:
                    return False
                rejection = plan_rejected(sql) if fallback else None
                if fallback and rejection is None:
                    return False
                g = (
                    run_fallback(sql, rejection)
                    if rejection is not None
                    else self.s.run(sql, gpu=gpu, timeout=self.timeout)
                )
                if g.status != "ok" or g.result is None:
                    return False
                g.result.columns = cpu.result.columns  # type: ignore[union-attr]
                return not compare_results(cpu.result, g.result, False, self.tol).equal  # type: ignore[arg-type]

            return check
        if verdict == Verdict.VARIANT_MISMATCH and variant:
            name, value = next(iter(variant.items()))

            def check_variant(sql: str) -> bool:
                cpu = cpu_ok(sql)
                if cpu is None:
                    return False
                base = self.s.run(sql, gpu=gpu, timeout=self.timeout)
                if base.status != "ok" or base.result is None:
                    return False
                # A variant finding starts from a correct baseline. Reject edits
                # that introduce a CPU/GPU mismatch before changing the setting.
                base.result.columns = cpu.result.columns  # type: ignore[union-attr]
                if not compare_results(cpu.result, base.result, False, self.tol).equal:  # type: ignore[arg-type]
                    return False
                try:
                    self.s.set(name, value)
                    var = self.s.run(sql, gpu=gpu, timeout=self.timeout)
                finally:
                    self.s.restore(name)
                if var.status != "ok" or var.result is None:
                    return False
                var.result.columns = base.result.columns
                return not compare_results(
                    base.result, var.result, False, self.tol
                ).equal

            return check_variant
        if verdict in (
            Verdict.PLAN_FALLBACK,
            Verdict.GPU_ERROR,
            Verdict.GPU_INTERNAL_ERROR,
            Verdict.GPU_OOM,
        ):
            original_reason = rec.reason.split(": ", 1)[-1] if variant else rec.reason
            want = normalize_reason(original_reason)

            def check_error(sql: str) -> bool:
                if cpu_ok(sql) is None:
                    return False
                rejection = plan_rejected(sql) if fallback else None
                if fallback and rejection is None:
                    return False
                if variant:
                    name, value = next(iter(variant.items()))
                    self.s.set(name, value)
                try:
                    g = (
                        run_fallback(sql, rejection)
                        if rejection is not None
                        else self.s.run(sql, gpu=gpu, timeout=self.timeout)
                    )
                finally:
                    if variant:
                        self.s.restore(next(iter(variant)))
                if g.status != "error":
                    return False
                v2, reason2 = classify_gpu_error(g.error, fallback_enabled=fallback)
                if v2 != verdict:
                    return False
                return normalize_reason(reason2) == want

            return check_error
        return None


# --------------------------------------------------------------------------
# worker process
# --------------------------------------------------------------------------


def worker_main(args: WorkerArgs, cfg: FuzzConfig, out: Any, stop: Any) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    w = args.worker_id
    run_dir = pathlib.Path(args.run_dir)
    if args.stderr_path:
        # Sirius prints its signal-handler backtrace and std::terminate messages to fd 2; the
        # orchestrator reads this file when the process dies.
        pathlib.Path(args.stderr_path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.stderr_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 2)
        os.close(fd)

    def send(msg: dict[str, Any]) -> None:
        msg["worker"] = w
        msg["spawn"] = args.spawn_id
        out.put(msg)

    def log(text: str) -> None:
        send({"type": "log", "msg": text})

    evidence_path = (
        pathlib.Path(args.stderr_path).with_suffix(".active.json")
        if args.stderr_path
        else None
    )
    session = Session(
        args.extension,
        args.sirius_config,
        run_dir / "db",
        w,
        allow_metadata_mismatch=args.allow_metadata_mismatch,
        evidence_path=evidence_path,
    )
    try:
        session.open()
        if args.stderr_path:
            write_json(
                pathlib.Path(args.stderr_path).with_suffix(".runtime.json"),
                runtime_info(session),
            )
    except Exception as e:  # noqa: BLE001
        send({"type": "done", "reason": f"session open failed: {e}"})
        return
    rng = random.Random(args.seed)
    data_gen = DataGenerator(cfg, rng)
    evaluator = Evaluator(cfg, session, log)
    evaluator.rng = random.Random(args.seed ^ 0x5EED)
    done_queries = 0
    dataset_idx = 0
    try:
        while not stop.is_set():
            if args.deadline and time.time() >= args.deadline:
                break
            if args.max_queries is not None and done_queries >= args.max_queries:
                break
            ds_seed = args.seed * 1000 + dataset_idx
            ds = data_gen.generate(ds_seed)
            stem = f"w{w}-s{args.spawn_id}-d{dataset_idx}"
            alias = f"fz{w}_{dataset_idx}"
            (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
            (run_dir / "datasets" / f"{stem}.sql").write_text(
                f"-- siriusfuzz dataset {stem} (seed {ds_seed})\n{ds.schema_sql()}\n{ds.data_sql()}\nCHECKPOINT;\n"
            )
            try:
                session.load_dataset(ds, alias)
            except Exception as e:  # noqa: BLE001
                log(f"dataset load failed: {e}")
                send({"type": "done", "reason": f"dataset load failed: {e}"})
                return
            evaluator.set_dataset(ds, alias)
            send(
                {
                    "type": "dataset",
                    "stem": stem,
                    "rows": sum(len(t.rows) for t in ds.tables),
                }
            )
            if dataset_idx == 0 and session.gpu_available:
                ok, why = session.check_interception()
                send({"type": "canary", "ok": ok, "why": why})
                if not ok:
                    send(
                        {
                            "type": "done",
                            "reason": f"Sirius does not intercept queries: {why}",
                        }
                    )
                    return
            qgen = QueryGenerator(cfg, ds, random.Random(ds_seed * 31 + 7), data_gen)
            for i in range(cfg.sirius.dataset_queries):
                if stop.is_set() or (args.deadline and time.time() >= args.deadline):
                    break
                if args.max_queries is not None and done_queries >= args.max_queries:
                    break
                query = qgen.generate()
                sql = query.sql()
                send(
                    {
                        "type": "begin",
                        "sql": sql,
                        "dataset": stem,
                        "labels": labels(query),
                    }
                )
                try:
                    session.evidence = {}
                    session.stage = "evaluation"
                    rec = evaluator.evaluate(query, sql, w, stem, ds_seed)
                    rec.evidence = dict(session.evidence)
                    send({"type": "result", "record": asdict(rec)})
                    if (
                        args.reduce
                        and cfg.oracle.reduce
                        and Verdict(rec.verdict).is_finding(cfg.oracle.on_plan_fallback)
                    ):
                        session.stage = "reduction"
                        evaluator.reduce(rec, query)
                        send({"type": "reduction", "record": asdict(rec)})
                except (
                    Exception
                ) as e:  # noqa: BLE001 - harness bug, report and continue
                    log("harness exception: " + traceback.format_exc()[-800:])
                    send({"type": "done", "reason": f"harness error: {e}"})
                    return
                send({"type": "idle"})
                done_queries += 1
            send({"type": "stats", "stats": dict(qgen.stats)})
            session.drop_dataset(alias)
            if evaluator.perm_alias:
                session.drop_dataset(evaluator.perm_alias)
            dataset_idx += 1
    finally:
        session.close()
        send({"type": "done", "reason": "finished"})


# --------------------------------------------------------------------------
# crash reasons
# --------------------------------------------------------------------------

_SIGNAL_RE = re.compile(r"\*\*\* (SIG[A-Z]+)")
_WHAT_RE = re.compile(r"what\(\):\s*(.+)")
_TERMINATE_RE = re.compile(r"terminate called after throwing an instance of '([^']+)'")
_EXT_FRAME_RE = re.compile(r"#\d+\s+\S*?sirius\.duckdb_extension\((\+0x[0-9a-f]+)\)")
_HANDLER_NAMES = ("segfault_handler", "signal_handler", "sigaction", "backtrace")


def symbolize(extension: str | None, offsets: list[str]) -> list[str]:
    """Function names for extension offsets via addr2line; offsets when unavailable."""
    if not extension or not offsets or shutil.which("addr2line") is None:
        return offsets
    try:
        out = subprocess.run(
            ["addr2line", "-f", "-C", "-e", extension, *offsets],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return offsets
    names = out[0::2]  # -f prints function, then file:line
    return [n if n and n != "??" else off for n, off in zip(names, offsets)] or offsets


def extract_crash_reason(
    stderr_tail: str, exitcode: int | None, extension: str | None = None
) -> str:
    """Short, dedup-friendly description of a worker death from its captured stderr.

    Prefers the std::terminate what() text, then the signal plus the first frames inside
    the extension below Sirius's own signal handler (symbolized when addr2line exists;
    offsets are stable within one build), then the bare exit code.
    """
    blocks = stderr_tail.split("*** end backtrace ***")
    text = blocks[-2] if len(blocks) >= 2 else stderr_tail  # last crash block only
    what = _WHAT_RE.search(text)
    term = _TERMINATE_RE.search(text)
    sig = _SIGNAL_RE.search(text)
    if what:
        prefix = sig.group(1) if sig else "terminate"
        exc = f" ({term.group(1)})" if term else ""
        return f"{prefix}{exc}: {what.group(1).strip()}"
    if sig:
        offsets = _EXT_FRAME_RE.findall(text)[:5]
        names = symbolize(extension, offsets)
        frames = [n for n in names if not any(h in n for h in _HANDLER_NAMES)]
        if frames and frames == names and len(frames) > 1:
            frames = frames[
                1:
            ]  # unsymbolized: the first extension frame is the handler
        where = " in " + " < ".join(f[:80] for f in frames[:3]) if frames else ""
        return f"{sig.group(1)}{where}"
    return f"worker exited with code {exitcode}"


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------


@dataclass
class OrchestratorOptions:
    workers: int = 1
    duration: float | None = None  # seconds
    max_queries: int | None = None  # total
    extension: str | None = None
    sirius_configs: list[str] = dataclasses.field(default_factory=list)
    reduce: bool = True
    progress_every: float = 10.0
    quiet: bool = False
    max_respawns: int = 200  # crashes are findings, not a reason to stop the run
    allow_metadata_mismatch: bool = False
    startup_timeout: float = 120.0


class Orchestrator:
    def __init__(
        self, cfg: FuzzConfig, report: Report, seed: int, opts: OrchestratorOptions
    ):
        self.cfg = cfg
        self.report = report
        self.seed = seed
        self.opts = opts
        self.ctx = mp.get_context("spawn")
        self.queue = Mailbox(report.run_dir / "messages")
        self.stop = self.ctx.Event()
        self.procs: dict[int, Any] = {}
        self.inflight: dict[int, tuple[str, str, float, list[str]]] = {}
        self.stderr_paths: dict[int, str] = {}
        self.last_seen: dict[int, float] = {}
        self.respawns = 0
        self.spawn_count = 0
        self.total_results = 0
        self.active_paths: dict[int, pathlib.Path] = {}
        self.deadline: float | None = None
        self.worker_spawns: dict[int, int] = {}

    def _spawn(self, worker_id: int) -> None:
        self.spawn_count += 1
        self.worker_spawns[worker_id] = self.spawn_count
        seed = self.seed * 100 + worker_id + 1000 * self.spawn_count
        cfg_path = None
        if self.opts.sirius_configs:
            cfg_path = self.opts.sirius_configs[
                worker_id % len(self.opts.sirius_configs)
            ]
        per_worker = None
        if self.opts.max_queries is not None:
            per_worker = max(1, -(-self.opts.max_queries // self.opts.workers))
        deadline = self.deadline
        stderr_path = str(
            self.report.run_dir / "logs" / f"w{worker_id}-s{self.spawn_count}.stderr"
        )
        args = WorkerArgs(
            worker_id,
            seed,
            str(self.report.run_dir),
            self.opts.extension,
            cfg_path,
            per_worker,
            deadline,
            self.opts.reduce,
            stderr_path,
            self.opts.allow_metadata_mismatch,
            self.spawn_count,
        )
        self.stderr_paths[worker_id] = stderr_path
        self.active_paths[worker_id] = pathlib.Path(stderr_path).with_suffix(
            ".active.json"
        )
        self.report.worker_context[worker_id] = {
            "sirius_config": cfg_path,
            "stderr": stderr_path,
            "runtime": str(pathlib.Path(stderr_path).with_suffix(".runtime.json")),
            "cpu_only": self.opts.extension is None,
            "allow_metadata_mismatch": self.opts.allow_metadata_mismatch,
        }
        p = self.ctx.Process(
            target=worker_main,
            args=(args, self.cfg, self.queue, self.stop),
            daemon=True,
        )
        p.start()
        self.procs[worker_id] = p
        self.last_seen[worker_id] = time.time()
        self.inflight.pop(worker_id, None)

    def _say(self, text: str) -> None:
        if not self.opts.quiet:
            print(text, file=sys.stderr, flush=True)

    def run(self) -> dict[str, Any]:
        try:
            self._run()
        except KeyboardInterrupt:
            self.report.status = "cancelled"
            self.report.stop_reason = "interrupted by user; completed findings retained"
        except Exception as exc:
            self.report.status = "incomplete"
            self.report.stop_reason = f"orchestrator error: {exc}"
        finally:
            self.stop.set()
            for p in self.procs.values():
                if p.is_alive():
                    p.kill()
                p.join(timeout=5)
            # Workers already completed these records before cancellation/deadline.
            # Do not spawn or attribute deliberately stopped work as a hang.
            while True:
                try:
                    msg = self.queue.get_nowait()
                except queue_mod.Empty:
                    break
                if msg["type"] in ("result", "reduction", "stats"):
                    self._handle(msg, set())
        return self.report.finish()

    def _run(self) -> None:
        start = time.time()
        self.deadline = start + self.opts.duration if self.opts.duration else None
        for w in range(self.opts.workers):
            self._spawn(w)
        last_progress = start
        finished: set[int] = set()
        timeout = self.cfg.oracle.query_timeout_seconds + HANG_GRACE_SECONDS
        while True:
            try:
                msg = self.queue.get(timeout=0.5)
            except queue_mod.Empty:
                msg = None
            if msg is not None:
                self._handle(msg, finished)
            drained = 0
            for _ in range(256):
                try:
                    self._handle(self.queue.get_nowait(), finished)
                    drained += 1
                except queue_mod.Empty:
                    break
            now = time.time()
            if self.opts.duration and now - start >= self.opts.duration:
                self.stop.set()
                self.report.stop_reason = (
                    "duration budget reached; in-flight work stopped"
                )
                break
            if (
                self.opts.max_queries is not None
                and self.total_results >= self.opts.max_queries
            ):
                self.stop.set()
                break
            for w, p in list(self.procs.items()):
                if w in finished:
                    continue
                if not p.is_alive():
                    if drained < 256:
                        self._worker_died(w, p, finished)
                    continue
                sql_info = self.inflight.get(w)
                active_path = self.active_paths.get(w)
                since = sql_info[2] if sql_info else self.last_seen[w]
                if sql_info and active_path and active_path.exists():
                    active = json.loads(active_path.read_text())
                    since = max(since, active.get("started_at", since))
                if sql_info is not None and now - since > timeout:
                    self._worker_hung(w, p, sql_info, finished)
                elif (
                    sql_info is None
                    and now - self.last_seen[w] > self.opts.startup_timeout
                ):
                    self.report.status = "incomplete"
                    self.report.stop_reason = (
                        "worker stalled outside a query (startup/dataset/cleanup)"
                    )
                    p.kill()
                    finished.add(w)
            if now - last_progress >= self.opts.progress_every:
                last_progress = now
                self._progress(start)
            if len(finished) >= len(self.procs) and all(
                w in finished for w in self.procs
            ):
                break
        for p in self.procs.values():
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def _handle(self, msg: dict[str, Any], finished: set[int]) -> None:
        w = msg.get("worker", -1)
        if msg.get("spawn", self.worker_spawns.get(w)) != self.worker_spawns.get(w):
            return
        self.last_seen[w] = time.time()
        t = msg["type"]
        if t == "begin":
            self.inflight[w] = (
                msg["sql"],
                msg["dataset"],
                time.time(),
                msg.get("labels", []),
            )
        elif t == "result":
            rec = QueryRecord(**msg["record"])
            name = self.report.add(rec)
            self.total_results += 1
            if name:
                self._say(f"[w{w}] NEW FINDING {name}: {rec.reason[:100]}")
        elif t == "reduction":
            self.report.add_reduction(QueryRecord(**msg["record"]))
        elif t == "idle":
            self.inflight.pop(w, None)
        elif t == "dataset":
            self.report.datasets += 1
        elif t == "stats":
            self.report.add_feature_stats(msg["stats"])
        elif t == "canary":
            self._say(
                f"[w{w}] interception canary: {'ok' if msg['ok'] else 'FAILED'} ({msg['why']})"
            )
        elif t == "log":
            self._say(f"[w{w}] {msg['msg']}")
        elif t == "done":
            finished.add(w)
            self.inflight.pop(w, None)
            if msg.get("reason") != "finished":
                self.report.status = "incomplete"
                self.report.stop_reason = msg.get("reason", "worker stopped")
                self._say(f"[w{w}] stopped: {msg.get('reason')}")

    def _stderr_tail(self, w: int) -> str:
        path = self.stderr_paths.get(w)
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 16000))
            return fh.read().decode("utf-8", "replace")

    def _worker_died(self, w: int, p: Any, finished: set[int]) -> None:
        info = self.inflight.pop(w, None)
        finished.add(w)
        tail = self._stderr_tail(w)
        reason = extract_crash_reason(tail, p.exitcode, self.opts.extension)
        if info is not None:
            sql, stem, _, query_labels = info
            rec = QueryRecord(
                w,
                stem,
                self.seed,
                sql,
                Verdict.CRASH.value,
                reason=reason,
                detail=tail[-4000:],
                labels=query_labels,
            )
            self._attach_active(w, rec)
            if self._reducer_outside_query(rec):
                self.report.status = "incomplete"
                operation = rec.context.get("operation", "between reduction queries")
                self.report.stop_reason = f"reducer failed during {operation}: {reason}"
                self._say(f"[w{w}] {self.report.stop_reason}")
                return
            if rec.context.get("phase") == "cpu":
                rec.reason = "CPU phase: " + rec.reason
            self.report.add(rec)
            self.total_results += 1
            self._say(f"[w{w}] CRASH (exit {p.exitcode}) during query: {reason[:120]}")
        elif p.exitcode not in (0, None):
            self.report.status = "incomplete"
            self.report.stop_reason = f"worker startup/cleanup failed: {reason}"
            self._say(
                f"[w{w}] worker exited with code {p.exitcode} outside a query: {reason[:120]}"
            )
            return  # startup/cleanup failures are not recoverable query findings
        else:
            self.report.status = "incomplete"
            self.report.stop_reason = "worker exited without reporting completion"
            return
        self._maybe_respawn(w, finished)

    def _worker_hung(
        self,
        w: int,
        p: Any,
        info: tuple[str, str, float, list[str]],
        finished: set[int],
    ) -> None:
        sql, stem, since, query_labels = info
        self.inflight.pop(w, None)
        p.kill()
        p.join(timeout=5)
        finished.add(w)
        tail = self._stderr_tail(w)
        rec = QueryRecord(
            w,
            stem,
            self.seed,
            sql,
            Verdict.TIMEOUT.value,
            reason=f"no progress for {time.time() - since:.0f}s; worker killed",
            detail=tail[-4000:],
            labels=query_labels,
        )
        self._attach_active(w, rec)
        if self._reducer_outside_query(rec):
            self.report.status = "incomplete"
            operation = rec.context.get("operation", "between reduction queries")
            self.report.stop_reason = f"reducer hung during {operation}; worker killed"
            self._say(f"[w{w}] {self.report.stop_reason}")
            return
        if rec.context.get("phase") == "cpu":
            rec.verdict = Verdict.CPU_TIMEOUT.value
        self.report.add(rec)
        self.total_results += 1
        self._say(f"[w{w}] HANG: killed after {time.time() - since:.0f}s")
        self._maybe_respawn(w, finished)

    def _maybe_respawn(self, w: int, finished: set[int]) -> None:
        if (
            self.opts.max_queries is not None
            and self.total_results >= self.opts.max_queries
        ):
            self.stop.set()
        if self.stop.is_set() or self.respawns >= self.opts.max_respawns:
            if not self.stop.is_set():
                self.report.status = "incomplete"
                self.report.stop_reason = "worker restart budget exhausted"
                self._say(
                    f"[w{w}] respawn cap {self.opts.max_respawns} reached; worker not restarted"
                )
            return
        self.respawns += 1
        finished.discard(w)
        self._spawn(w)
        self._say(f"[w{w}] respawned ({self.respawns}/{self.opts.max_respawns})")

    def _attach_active(self, w: int, rec: QueryRecord) -> None:
        path = self.active_paths.get(w)
        if path and path.exists():
            active = json.loads(path.read_text())
            rec.context = active
            rec.sql = active.get("sql", rec.sql)
            rec.variant = active.get("settings") or None
            observed = path.with_suffix(".observed.json")
            if observed.exists():
                rec.evidence = json.loads(observed.read_text())

    @staticmethod
    def _reducer_outside_query(rec: QueryRecord) -> bool:
        active = rec.context
        return active.get("stage") == "reduction" and (
            "operation" in active or active.get("status") != "running"
        )

    def _progress(self, start: float) -> None:
        elapsed = time.time() - start
        counts = self.report.counts
        rate = self.total_results / elapsed if elapsed > 0 else 0.0
        self._say(
            f"[{elapsed:6.0f}s] queries={self.total_results} ({rate:.1f}/s) findings={len(self.report.findings)} "
            f"ok={counts.get('ok', 0)} skip={counts.get('cpu_error', 0)} fallback={counts.get('plan_fallback', 0)} "
            f"mismatch={counts.get('mismatch', 0)} gpu_err={counts.get('gpu_error', 0) + counts.get('gpu_internal_error', 0)}"
        )

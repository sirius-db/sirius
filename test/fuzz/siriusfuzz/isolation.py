# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Bounded subprocess probes: native crashes and uninterruptible GPU work stay contained."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pathlib
import signal
import time
import traceback
from dataclasses import asdict
from typing import Any

from .artifacts import runtime_info, sql_literal, write_json
from .config import load_config
from .session import Session, SessionError, single_select


def _child(payload: dict[str, Any], work: str) -> None:
    directory = pathlib.Path(work)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for number, name in ((1, "stdout.log"), (2, "stderr.log")):
        fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fd, number)
        os.close(fd)
    session = Session(
        payload["shell"],
        payload.get("extension"),
        payload.get("sirius_config"),
        directory / "db",
        0,
        cpu_only=payload.get("cpu_only", False),
        evidence_path=directory / "active.json",
        log_path=directory / "shell.stderr",
    )
    result: dict[str, Any] = {"status": "setup_error"}
    try:
        session.open()
        for name, value in payload.get("session_settings", {}).items():
            if not session.setting_supported(name):
                raise SessionError(f"recorded session setting unavailable: {name}")
            session.set(name, value)
        session.active_settings.clear()  # baseline settings are saved in runtime.json
        write_json(directory / "runtime.json", runtime_info(session))
        session.db_dir.mkdir(parents=True, exist_ok=True)
        replay_db = session.db_dir / "replay.duckdb"
        session.execute(f"ATTACH {sql_literal(str(replay_db))} AS replay")
        session.execute("USE replay")
        session.current_alias = "replay"
        session.attached["replay"] = replay_db
        if payload["operation"] == "doctor":
            session.execute(
                "CREATE TABLE fuzz_probe(k INTEGER); INSERT INTO fuzz_probe VALUES (1), (2), (NULL); CHECKPOINT"
            )
        else:
            # The shell parses the entire script; quoted semicolons and comments survive.
            session.execute(pathlib.Path(payload["dataset"]).read_text())
            session.execute("CHECKPOINT")
        if session.gpu_available:
            ok, why = session.check_interception()
            write_json(directory / "canary.json", {"ok": ok, "detail": why})
            if not ok:
                raise SessionError(f"GPU interception check failed: {why}")
        if payload["operation"] == "doctor":
            probe = session.run(
                "SELECT CAST(sum(k) AS BIGINT), count(*) FROM fuzz_probe",
                session.gpu_available,
                120,
            )
            if (
                probe.status != "ok"
                or probe.result is None
                or probe.result.rows != [(3, 3)]
            ):
                raise SessionError(
                    f"GPU execution probe failed: {probe.error or probe.status}"
                )
            result = {
                "status": "ok",
                "gpu_verified": session.gpu_available,
                "checks": [
                    "DuckDB shell started",
                    "Sirius session opened",
                    "file-backed dataset loaded",
                    "query result verified",
                ],
            }
        else:
            from .runner import Evaluator

            cfg = load_config(payload["config"])
            ev = Evaluator(cfg, session, lambda message: print(message, flush=True))
            ev.forced_variant = payload.get("variant")
            ev.replay_mode = payload.get("comparison", "multiset")
            ev.variants = (
                {}
            )  # replay only the recorded variant, never a random replacement
            if ev.forced_variant:
                for name in ev.forced_variant:
                    if not session.gpu_available or not session.setting_supported(name):
                        raise SessionError(f"saved variant setting unavailable: {name}")
            sql = pathlib.Path(payload["query"]).read_text().strip().rstrip(";")
            if not single_select(sql):
                raise ValueError("replay requires exactly one SELECT or WITH query")
            session.begin_query(sql)
            record = ev.evaluate(None, sql, 0, "replay", 0)
            record.evidence = session.evidence
            record.context.update(
                {
                    "ambiguity_filter": "not rerun for SQL-only replay; manual confirmation required"
                }
            )
            result = {"status": "ok", "record": asdict(record)}
    except Exception as exc:
        result = {
            "status": "setup_error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        # Write evidence before closing: cleanup can itself hang or crash.
        write_json(directory / "result.json", result)
        session.close()


def supervise(
    payload: dict[str, Any],
    directory: pathlib.Path,
    timeout: float,
    target: Any = _child,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "request.json", payload)
    result_path = directory / "result.json"
    result_path.unlink(missing_ok=True)
    process = mp.get_context("spawn").Process(
        target=target, args=(payload, str(directory))
    )
    start = time.monotonic()
    status = None
    try:
        # SIGINT can arrive during spawn, after request.json is written but
        # before the child is ready. Preserve a cancelled outcome in that case.
        process.start()
        while process.is_alive():
            if time.monotonic() - start >= timeout:
                status = "timeout"
                break
            process.join(timeout=0.1)
    except KeyboardInterrupt:
        status = "cancelled"
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
    active = directory / "active.json"
    if result_path.exists():
        completed = json.loads(result_path.read_text())
        if status or process.exitcode != 0:
            cleanup_status = status or "crash"
            result = {
                "status": "cleanup_error",
                "error": f"probe cleanup {cleanup_status}; subprocess stopped",
                "query_result": completed,
                "cleanup": {"status": cleanup_status},
            }
            if cleanup_status == "timeout":
                result["cleanup"]["deadline_seconds"] = timeout
            else:
                result["cleanup"]["exitcode"] = process.exitcode
        else:
            result = completed
    elif status:
        result = {
            "status": status,
            "error": f"probe {status}; subprocess stopped",
            "deadline_seconds": timeout,
        }
    elif process.exitcode != 0:
        result = {"status": "crash", "exitcode": process.exitcode}
    else:
        result = {
            "status": "setup_error",
            "error": "subprocess exited without a result",
        }
    if active.exists():
        result["last_operation"] = json.loads(active.read_text())
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    write_json(directory / "outcome.json", result)
    return result

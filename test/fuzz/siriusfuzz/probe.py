# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Doctor and replay: one session, one dataset, one query, in this process.

The shell is the isolation boundary: a native fault or a hung GPU call ends the
shell process, which the session reports within the statement's own timeout, so
these probes need no supervising subprocess of their own.
"""

from __future__ import annotations

import pathlib
import traceback
from dataclasses import asdict
from typing import Any

from .artifacts import runtime_info, sql_literal, write_json
from .config import load_config
from .session import Session, SessionError, single_select

DOCTOR_CHECKS = [
    "DuckDB shell started",
    "Sirius session opened",
    "file-backed dataset loaded",
    "query result verified",
]


def execute_probe(payload: dict[str, Any], work: pathlib.Path) -> dict[str, Any]:
    """Run the doctor check or a replay described by ``payload``; return the outcome.

    ``payload`` carries ``operation`` (``doctor`` or ``replay``), ``shell``,
    ``extension``, ``cpu_only``, ``sirius_config`` and ``timeout`` (seconds per
    query); a replay adds ``query``, ``dataset``, ``config``, ``variant``,
    ``comparison`` and ``session_settings``. Evidence lands in ``work``.
    """
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "request.json", payload)
    timeout = float(payload.get("timeout") or 180)
    session = Session(
        payload["shell"],
        payload.get("extension"),
        payload.get("sirius_config"),
        work / "db",
        0,
        cpu_only=payload.get("cpu_only", False),
        evidence_path=work / "active.json",
        log_path=work / "shell.stderr",
    )
    result: dict[str, Any] = {"status": "setup_error"}
    try:
        session.open()
        for name, value in payload.get("session_settings", {}).items():
            if not session.setting_supported(name):
                raise SessionError(f"recorded session setting unavailable: {name}")
            session.set(name, value)
        session.active_settings.clear()  # baseline settings are saved in runtime.json
        write_json(work / "runtime.json", runtime_info(session))
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
            write_json(work / "canary.json", {"ok": ok, "detail": why})
            if not ok:
                raise SessionError(f"GPU interception check failed: {why}")
        if payload["operation"] == "doctor":
            probe = session.run(
                "SELECT CAST(sum(k) AS BIGINT), count(*) FROM fuzz_probe",
                session.gpu_available,
                timeout,
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
                "checks": list(DOCTOR_CHECKS),
            }
        else:
            from .runner import Evaluator

            cfg = load_config(payload["config"])
            ev = Evaluator(cfg, session, lambda message: print(message, flush=True))
            ev.timeout = timeout
            ev.forced_variant = payload.get("variant")
            ev.replay_mode = payload.get("comparison", "multiset")
            ev.variants = {}  # replay only the recorded variant, never a random one
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
    except Exception as exc:  # noqa: BLE001 - every failure is reported as the outcome
        result = {
            "status": "setup_error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
    write_json(work / "outcome.json", result)
    return result

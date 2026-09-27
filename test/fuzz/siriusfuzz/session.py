# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""One DuckDB connection with Sirius loaded, in strict mode.

Generated tables live in an ATTACHed file-backed database (in-memory tables
never reach the GPU native scan) and are CHECKPOINTed so the scan sees them on
disk. Queries run with ``enable_duckdb_fallback = false`` so plan-time and
runtime fallbacks surface as errors instead of silent CPU runs.
"""

from __future__ import annotations

import os
import hashlib
import pathlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from . import sqltypes as st
from .artifacts import sql_literal, write_json
from .compare import ColumnInfo, ResultSet, canonical
from .schema_gen import Dataset

CANARY_SETTING = "sirius_test_inject_transparent_gpu_error"


class SessionError(RuntimeError):
    pass


@dataclass
class RunResult:
    status: str  # ok | error | timeout
    result: ResultSet | None = None
    error: str = ""
    elapsed: float = 0.0


class Session:
    def __init__(
        self,
        extension: str | None,
        sirius_config: str | None,
        db_dir: pathlib.Path,
        worker_id: int,
        strict: bool = True,
        allow_metadata_mismatch: bool = False,
        evidence_path: pathlib.Path | None = None,
    ):
        self.extension = extension
        self.sirius_config = sirius_config
        self.sirius_config_mode: str | None = None
        self.db_dir = db_dir
        self.worker_id = worker_id
        self.strict = strict
        self.allow_metadata_mismatch = bool(allow_metadata_mismatch)
        self.evidence_path = evidence_path
        self.active_settings: dict[str, Any] = {}
        self.evidence: dict[str, Any] = {}
        self.plan_fallback_reason: str | None = None
        self.stage = "evaluation"
        self.con: Any = None
        self.gpu_available = extension is not None
        self.current_alias: str | None = None
        self._setting_defaults: dict[str, Any] = {}
        self.sqlsmith_loaded = False
        self.version_mismatch_bypassed = False

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        import duckdb

        self._configure_sirius()
        # Registers the TEST ONLY fault-injection option used as the interception canary.
        os.environ.setdefault("SIRIUS_ENABLE_TEST_OPTIONS", "1")
        self.con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
        self.con.execute("SET enable_progress_bar = false")
        if self.extension:
            self._load_extension()
            if self.strict:
                self.con.execute("SET enable_duckdb_fallback = false")
        self.set_gpu(False)

    def _configure_sirius(self) -> None:
        if not self.extension:
            self.sirius_config_mode = "cpu_only"
            return
        if self.sirius_config:
            selected = pathlib.Path(self.sirius_config).resolve()
            if not selected.is_file():
                raise SessionError(f"Sirius configuration is not a file: {selected}")
            self.sirius_config = str(selected)
            os.environ["SIRIUS_CONFIG_FILE"] = self.sirius_config
            self.sirius_config_mode = "explicit_yaml"
            return
        # Mirror Sirius's discovery order. An absent snapshot must never silently
        # select settings that the finding cannot carry to another host.
        ambient = None
        if "SIRIUS_CONFIG_FILE" in os.environ:
            ambient = "SIRIUS_CONFIG_FILE"
        else:
            candidates = [pathlib.Path.cwd() / "sirius.yaml"]
            if "HOME" in os.environ:
                candidates.append(
                    pathlib.Path(os.environ["HOME"]) / ".sirius/sirius.yaml"
                )
            ambient = next((str(path) for path in candidates if path.exists()), None)
        if ambient is not None:
            raise SessionError(
                f"Unrecorded ambient Sirius configuration: {ambient}. "
                "Select the intended file with --sirius-config so it is saved, "
                "or remove the ambient configuration to use built-in defaults."
            )
        self.sirius_config_mode = "builtin_defaults"

    def _load_extension(self) -> None:
        try:
            self.con.execute(f"LOAD {sql_literal(self.extension)}")
        except Exception as e:  # noqa: BLE001
            if "built specifically for DuckDB version" not in str(e):
                raise
            if not self.allow_metadata_mismatch:
                raise SessionError(
                    f"{e}\nRebuild the Python module and extension from matching sources. "
                    "Only for a verified matching build, use --allow-metadata-mismatch."
                ) from e
            self.con.execute("SET allow_extensions_metadata_mismatch = true")
            self.con.execute(f"LOAD {sql_literal(self.extension)}")
            self.version_mismatch_bypassed = True

    def close(self) -> None:
        if self.con is not None:
            try:
                self.drop_dataset()
            except Exception:
                pass
            try:
                self.con.close()
            except Exception:
                pass
            self.con = None

    def set_gpu(self, enabled: bool) -> None:
        if self.gpu_available:
            self.con.execute(f"SET gpu_execution = {'true' if enabled else 'false'}")

    # -- datasets -------------------------------------------------------------

    def load_dataset(
        self, ds: Dataset, alias: str, permutation_seed: int | None = None
    ) -> None:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        path = self.db_dir / f"{alias}.duckdb"
        for p in (path, pathlib.Path(str(path) + ".wal")):
            if p.exists():
                p.unlink()
        self.set_gpu(False)
        self.con.execute(f"ATTACH {sql_literal(str(path))} AS {alias}")
        self.con.execute(f"USE {alias}")
        self.con.execute(ds.schema_sql())
        self.con.execute(ds.data_sql(permutation_seed))
        self.con.execute("CHECKPOINT")
        self.current_alias = alias

    def use(self, alias: str) -> None:
        self.con.execute(f"USE {alias}")
        self.current_alias = alias

    def drop_dataset(self, alias: str | None = None) -> None:
        aliases = [alias] if alias else list(self._attached())
        for a in aliases:
            try:
                self.con.execute("USE memory")
                self.con.execute(f"DETACH {a}")
            except Exception:
                pass
            for suffix in ("", ".wal"):
                p = self.db_dir / f"{a}.duckdb{suffix}"
                if p.exists():
                    p.unlink()
        self.current_alias = None

    def _attached(self) -> list[str]:
        rows = self.con.execute(
            "SELECT database_name FROM duckdb_databases() WHERE NOT internal AND database_name NOT IN ('memory','system','temp')"
        ).fetchall()
        return [r[0] for r in rows]

    # -- queries --------------------------------------------------------------

    def begin_query(self, sql: str) -> None:
        """Replace the previous query's evidence before evaluating ``sql``.

        The supervisor attributes a crash to the SQL in the active file, so it must
        name this query even if the worker dies before the first run rewrites it.
        """
        self.evidence = {}
        self.stage = "evaluation"
        if self.evidence_path:
            write_json(self.evidence_path.with_suffix(".observed.json"), self.evidence)
            write_json(
                self.evidence_path,
                {
                    "sql": sql,
                    "stage": self.stage,
                    "status": "pending",
                    "started_at": time.time(),
                },
            )

    def mark_auxiliary(self, operation: str, input_sql: str) -> None:
        """Replace completed query evidence before non-query reducer work."""
        if self.evidence_path:
            write_json(
                self.evidence_path,
                {
                    "operation": operation,
                    "stage": self.stage,
                    "input_sql": input_sql,
                    "status": "running",
                    "started_at": time.time(),
                },
            )

    def begin_fallback_retry(self, reason: str) -> None:
        """Persist the retry path before native code can crash or be killed."""
        self.plan_fallback_reason = reason
        if self.evidence_path:
            state = dict(self.evidence.get("gpu", {}))
            state.update(
                plan_fallback_reason=reason,
                status="running",
                started_at=time.time(),
            )
            write_json(self.evidence_path, state)

    def end_fallback_retry(self) -> None:
        self.plan_fallback_reason = None

    def run(self, sql: str, gpu: bool, timeout: float) -> RunResult:
        import duckdb

        phase = "gpu" if gpu else "cpu"
        state = {
            "sql": sql,
            "phase": phase,
            "stage": self.stage,
            "settings": dict(self.active_settings),
            "status": "running",
            "started_at": time.time(),
            "dataset": self.current_alias,
        }
        if gpu and self.plan_fallback_reason is not None:
            state["plan_fallback_reason"] = self.plan_fallback_reason
        if self.evidence_path:
            write_json(self.evidence_path, state)
        self.set_gpu(gpu)
        timer = threading.Timer(timeout, self._interrupt) if timeout > 0 else None
        start = time.monotonic()
        interrupted = False
        try:
            if timer:
                timer.start()
            cur = self.con.execute(sql)
            rows = cur.fetchall()
            desc = cur.description or []
        except duckdb.InterruptException:
            interrupted = True
            return self._record(
                phase,
                state,
                RunResult(
                    "timeout", elapsed=time.monotonic() - start, error="interrupted"
                ),
            )
        except (
            Exception
        ) as e:  # noqa: BLE001 - every DuckDB error type is a query error here
            msg = str(e)
            if "INTERRUPT" in msg.upper() or "interrupted" in msg.lower():
                return self._record(
                    phase,
                    state,
                    RunResult("timeout", elapsed=time.monotonic() - start, error=msg),
                )
            return self._record(
                phase,
                state,
                RunResult("error", error=msg, elapsed=time.monotonic() - start),
            )
        finally:
            if timer:
                timer.cancel()
            if not interrupted:
                self.set_gpu(False)
        cols = [ColumnInfo(d[0], st.parse_duckdb_type(str(d[1]))) for d in desc]
        return self._record(
            phase,
            state,
            RunResult("ok", ResultSet(cols, rows), elapsed=time.monotonic() - start),
        )

    def _record(
        self, phase: str, state: dict[str, Any], result: RunResult
    ) -> RunResult:
        state.update(status=result.status, elapsed=result.elapsed, error=result.error)
        if result.result is not None:
            hashes = [
                hashlib.sha256(repr(tuple(canonical(v) for v in row)).encode()).digest()
                for row in result.result.rows
            ]
            state.update(
                fingerprint_ordered=hashlib.sha256(b"".join(hashes)).hexdigest(),
                fingerprint_multiset=hashlib.sha256(
                    b"".join(sorted(hashes))
                ).hexdigest(),
                row_count=result.result.row_count,
                columns=[
                    {"name": c.name, "type": str(c.type)} for c in result.result.columns
                ],
                sample_rows=[
                    [repr(v)[:500] for v in row] for row in result.result.rows[:10]
                ],
                sample_note="First 10 rows, values represented as Python literals, truncated to 500 characters.",
            )
        self.evidence[phase] = state
        operations = self.evidence.setdefault("operations", [])
        if self.stage != "reduction" and len(operations) < 32:
            operations.append(state)
        if self.evidence_path:
            write_json(self.evidence_path, state)
            write_json(self.evidence_path.with_suffix(".observed.json"), self.evidence)
        return result

    def _interrupt(self) -> None:
        try:
            self.con.interrupt()
        except Exception:
            pass

    def describe(self, sql: str) -> list[ColumnInfo] | None:
        """Exact output types (CPU side); None when DESCRIBE itself fails."""
        self.set_gpu(False)
        try:
            rows = self.con.execute(f"DESCRIBE {sql}").fetchall()
        except Exception:
            return None
        return [ColumnInfo(r[0], st.parse_duckdb_type(str(r[1]))) for r in rows]

    # -- settings ---------------------------------------------------------------

    def setting_supported(self, name: str) -> bool:
        row = self.con.execute(
            "SELECT value FROM duckdb_settings() WHERE name = ?", [name]
        ).fetchone()
        if row is None:
            return False
        self._setting_defaults.setdefault(name, row[0])
        return True

    def set(self, name: str, value: Any) -> None:
        import re

        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
            raise ValueError(f"invalid setting name: {name}")
        if isinstance(value, str):
            self.con.execute(f"SET {name} = {sql_literal(value)}")
        else:
            self.con.execute(f"SET {name} = {value}")
        self.active_settings[name] = value

    def restore(self, name: str) -> None:
        default = self._setting_defaults.get(name)
        if default is None:
            return
        # Settings are stored as text in duckdb_settings(); numeric defaults round-trip as-is.
        try:
            self.con.execute(f"SET {name} = {int(default)}")
        except Exception:
            self.con.execute(f"SET {name} = {sql_literal(str(default))}")
        self.active_settings.pop(name, None)

    def load_sqlsmith(self) -> bool:
        if self.sqlsmith_loaded:
            return True
        try:
            self.con.execute("LOAD sqlsmith")
        except Exception:
            return False
        self.sqlsmith_loaded = True
        return True

    # -- canary -----------------------------------------------------------------

    def check_interception(self) -> tuple[bool, str]:
        """Prove Sirius intercepts plain SQL on this connection.

        Injects a runtime GPU error via the TEST ONLY option; a query that then fails
        with the injected text went through the GPU operator. Falls back to a probe
        query that Sirius rejects at plan time when the option is unavailable.
        """
        if not self.gpu_available:
            return False, "no extension loaded (cpu-only mode)"
        if self.current_alias is None:
            return False, "no dataset attached"
        table = self.con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE database_name = ? LIMIT 1",
            [self.current_alias],
        ).fetchone()
        if table is None:
            return False, "dataset has no tables"
        probe = f'SELECT count(*) FROM "{table[0]}"'
        if self.setting_supported(CANARY_SETTING):
            try:
                self.set(CANARY_SETTING, "fuzz-canary")
                res = self.run(probe, gpu=True, timeout=30)
            finally:
                self.set(CANARY_SETTING, "")
                self.active_settings.pop(CANARY_SETTING, None)
            if res.status == "error" and "fuzz-canary" in res.error:
                return True, "canary injected error observed"
            return (
                False,
                f"canary not observed (status={res.status}: {res.error[:120]})",
            )
        res = self.run(f'SELECT DISTINCT "k" FROM "{table[0]}"', gpu=True, timeout=30)
        if res.status == "error" and "GPU plan generation failed" in res.error:
            return True, "plan-time rejection observed"
        return (
            False,
            f"probe did not reach Sirius (status={res.status}: {res.error[:120]})",
        )

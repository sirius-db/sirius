# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""One DuckDB shell process per session, driven over pipes.

The shell is the one built next to the Sirius extension (``build/release/duckdb``),
so nothing has to be built besides Sirius and no version metadata can disagree.
Every statement is followed by a marker on each stream, written with the shell's
own dot commands (``.print``, and ``.output /dev/stderr`` for the stderr one) so
the markers never pass through SQL or Sirius; a result (JSON on stdout) and its
error text (stderr) are therefore delimited exactly. Sirius prints a banner on
stdout when it falls back to the CPU; it is removed from the result and kept
as log. A query that overruns its timeout gets the shell killed and restarted,
because the shell exits on SIGINT when stdin is a pipe; a GPU fault takes the
shell down the same way and is reported as a crash. The next statement starts a
fresh shell and re-attaches the datasets, which are files on disk.

Generated tables live in an ATTACHed file-backed database (in-memory tables
never reach the GPU native scan) and are CHECKPOINTed so the scan sees them on
disk. Queries run with ``enable_duckdb_fallback = false`` so plan-time and
runtime fallbacks surface as errors instead of silent CPU runs.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import sqltypes as st
from .artifacts import sql_literal, write_json
from .compare import ColumnInfo, ResultSet
from .config import REPO_ROOT
from .schema_gen import Dataset

CANARY_SETTING = "sirius_test_inject_transparent_gpu_error"
SHELL_ENV = "SIRIUSFUZZ_SHELL"
TEARDOWN_SECONDS = 90.0  # a killed shell can take a while to release a large GPU pool
UNKNOWN_TYPE = st.SqlType("UNKNOWN", "varchar")  # until DESCRIBE supplies the real one
_SENTINEL = "siriusfuzz"
_ERROR_LINE = re.compile(r"^[A-Za-z ]+ Error: ")
_FALLBACK_BANNER = (
    "=============================================\n",
    "Error in Sirius GPU execution, fallback to DuckDB\n",
    "=============================================\n",
)


class SessionError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# locating the shell
# --------------------------------------------------------------------------


def find_shell(explicit: str | None, configured: str | None) -> str:
    """The DuckDB shell to drive: ``--shell``, then ``$SIRIUSFUZZ_SHELL``, then the
    configured repository-relative path, then ``duckdb`` on PATH."""
    tried = []
    for source, candidate in (
        ("--shell", explicit),
        (SHELL_ENV, os.environ.get(SHELL_ENV)),
        ("sirius.shell", configured),
    ):
        if not candidate:
            continue
        path = pathlib.Path(candidate).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_file():
            return str(path)
        tried.append(f"{source}={path}")
    on_path = shutil.which("duckdb")
    if on_path:
        return on_path
    raise SessionError(
        "no DuckDB shell found (tried " + ", ".join(tried) + " and PATH); "
        "build Sirius with pixi run make, which produces build/release/duckdb, "
        "or pass --shell"
    )


def check_shell(shell: str) -> str:
    """The shell's version string; raises with the fix when it cannot run."""
    try:
        out = subprocess.run(
            [shell, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SessionError(f"cannot run the DuckDB shell {shell}: {exc}") from exc
    if out.returncode != 0:
        raise SessionError(
            f"{shell} --version failed (exit {out.returncode}): {out.stderr.strip()}"
        )
    return out.stdout.strip()


def discover_sirius_yaml() -> tuple[str | None, str]:
    """The Sirius YAML Sirius itself would pick up, and where it comes from.

    Mirrors Sirius's order: ``SIRIUS_CONFIG_FILE``, then ``./sirius.yaml``, then
    ``~/.sirius/sirius.yaml``. An empty environment variable counts as unset.
    """
    env = os.environ.get("SIRIUS_CONFIG_FILE", "").strip()
    if env:
        return env, "SIRIUS_CONFIG_FILE"
    candidates = [("current directory", pathlib.Path.cwd() / "sirius.yaml")]
    if "HOME" in os.environ:
        candidates.append(
            ("home directory", pathlib.Path(os.environ["HOME"]) / ".sirius/sirius.yaml")
        )
    for source, path in candidates:
        if path.exists():
            return str(path), source
    return None, "none"


def single_select(sql: str) -> bool:
    """Whether ``sql`` is exactly one SELECT/WITH/FROM statement (comments allowed)."""
    text = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    text = re.sub(r"--[^\n]*", " ", text).strip().rstrip(";").strip()
    if not re.match(r"(?is)^(?:\(\s*)*(select|with|from)\b", text):
        return False
    quote = ""
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == ";":
            return False
    return True


# --------------------------------------------------------------------------
# the shell process
# --------------------------------------------------------------------------


@dataclass
class ShellResult:
    status: str  # ok | error | timeout | crash
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: str = ""  # DuckDB's error text, or the stderr tail after a crash
    log: str = ""  # other stderr lines printed during the statement
    elapsed: float = 0.0
    exitcode: int | None = None


class Shell:
    """A DuckDB shell in batch JSON mode, one statement at a time."""

    def __init__(self, binary: str, log_path: pathlib.Path | None = None):
        self.binary = binary
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.events: queue.Queue = queue.Queue()
        self.counter = 0
        self.tail: collections.deque[str] = collections.deque(maxlen=400)
        self.exitcode: int | None = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.proc is not None:
            self.stop()
        self.events = queue.Queue()
        self.tail.clear()
        self.exitcode = None
        try:
            self.proc = subprocess.Popen(
                [self.binary, "-batch", "-json", "-unsigned"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise SessionError(f"cannot start the DuckDB shell {self.binary}: {exc}")
        for stream, name in ((self.proc.stdout, "out"), (self.proc.stderr, "err")):
            threading.Thread(
                target=self._pump, args=(stream, name, self.events), daemon=True
            ).start()
        # Startup chatter (an extension banner, driver messages) belongs to no statement.
        # A SET is never intercepted by Sirius; a SELECT here would run on the GPU with
        # fallback still enabled.
        first = self.execute("SET enable_progress_bar = false", timeout=300)
        if first.status != "ok":
            raise SessionError(
                f"the DuckDB shell {self.binary} did not start: "
                f"{first.error or first.log or first.status}"
            )

    def _pump(self, stream: Any, name: str, events: queue.Queue) -> None:
        log = None
        if self.log_path and name == "err":
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "a", encoding="utf-8")
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace")
                if log:
                    log.write(line)
                    log.flush()
                events.put((name, line))
        finally:
            if log:
                log.close()
            events.put((name, None))

    def execute(self, sql: str, timeout: float) -> ShellResult:
        if not self.alive:
            return ShellResult(
                "crash", error="shell is not running", exitcode=self.exitcode
            )
        assert self.proc is not None
        self.counter += 1
        tag = f"{_SENTINEL}-{self.counter}"
        stderr_tag = f"{tag}-stderr"
        # Dot commands never reach SQL, so the markers cannot be intercepted by
        # Sirius or fail with it; a SELECT marker did both when GPU execution was on.
        text = (
            f"{sql.rstrip().rstrip(';')}\n;\n"
            f".output /dev/stderr\n.print {stderr_tag}\n.output\n"
            f".print {tag}\n"
        )
        start = time.monotonic()
        try:
            self.proc.stdin.write(text.encode())  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError):
            return self._died(start, [])
        out: list[str] = []
        err: list[str] = []
        got_out = got_err = False
        deadline = start + timeout if timeout > 0 else None
        killed = False
        while not (got_out and got_err):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                if killed:
                    break
                # The shell leaves its input loop on SIGINT when stdin is a pipe, so
                # interrupting would end it anyway; kill, and let the next statement
                # start a fresh one.
                self.kill()
                killed = True
                deadline = time.monotonic() + 5
                continue
            try:
                name, line = self.events.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                if killed:
                    break
                return self._died(start, err)
            if name == "out":
                if line.strip() == tag:
                    got_out = True
                else:
                    out.append(line)
            else:
                self.tail.append(line)
                if line.strip() == stderr_tag:
                    got_err = True
                else:
                    err.append(line)
        elapsed = time.monotonic() - start
        if killed:
            self.exitcode = self.proc.poll()
            return ShellResult(
                "timeout",
                error="killed after exceeding the timeout",
                log="".join(err),
                elapsed=elapsed,
                exitcode=self.exitcode,
            )
        error, log = _split_stderr(err)
        out, banners = _strip_fallback_banners(out)
        if banners:
            log = "\n".join(part for part in (log, "".join(banners).strip()) if part)
        body = "".join(out).strip()
        if body:
            try:
                parsed = json.loads(
                    _normalize_shell_json_nulls(body),
                    object_pairs_hook=lambda pairs: pairs,
                )
            except ValueError as exc:
                return ShellResult(
                    "error", error=f"unreadable shell output: {exc}", log=body[:500]
                )
            columns = [key for key, _ in parsed[0]] if parsed else []
            rows = [tuple(value for _, value in pairs) for pairs in parsed]
            return ShellResult("ok", columns, rows, error, log, elapsed)
        if error:
            return ShellResult("error", error=error, log=log, elapsed=elapsed)
        return ShellResult("ok", [], [], "", log, elapsed)

    def _died(self, start: float, err: list[str]) -> ShellResult:
        """The process ended mid-statement: collect the rest of stderr (the backtrace)."""
        assert self.proc is not None
        self.exitcode = self._reap(10)
        until = time.monotonic() + 2
        while time.monotonic() < until:
            try:
                name, line = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is not None and name == "err":
                self.tail.append(line)
                err.append(line)
        return ShellResult(
            "crash",
            error="".join(err)[-16000:],
            elapsed=time.monotonic() - start,
            exitcode=self.exitcode,
        )

    def _reap(self, grace: float) -> int | None:
        """Wait for the process to end, SIGKILL after ``grace`` seconds; never raises.

        A killed shell keeps running until the driver has released its GPU pool,
        which can take tens of seconds; the next shell must not start before that.
        """
        assert self.proc is not None
        if grace > 0:
            try:
                return self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        self.proc.kill()
        try:
            return self.proc.wait(timeout=TEARDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.proc is None:
            return
        self.exitcode = self._reap(0) if self.proc.poll() is None else self.proc.poll()

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            self.exitcode = self._reap(10)
        else:
            self.exitcode = self.proc.poll()
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _normalize_shell_json_nulls(body: str) -> str:
    """The DuckDB shell prints bare ``NULL`` inside nested LIST/ARRAY JSON values."""
    result: list[str] = []
    quoted = escaped = False
    index = 0
    while index < len(body):
        char = body[index]
        if quoted:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            result.append(char)
        elif (
            body.startswith("NULL", index)
            and (index == 0 or body[index - 1] in "[:, \t\r\n")
            and (index + 4 == len(body) or body[index + 4] in ",]} \t\r\n")
        ):
            result.append("null")
            index += 4
            continue
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _split_stderr(lines: list[str]) -> tuple[str, str]:
    """DuckDB's error message (from its first ``<Type> Error:`` line on) and the rest."""
    for index, line in enumerate(lines):
        if _ERROR_LINE.match(line):
            return "".join(lines[index:]).strip(), "".join(lines[:index]).strip()
    return "", "".join(lines).strip()


def _strip_fallback_banners(lines: list[str]) -> tuple[list[str], list[str]]:
    """Keep Sirius's stdout-only fallback notice out of the shell's JSON result."""
    output: list[str] = []
    banners: list[str] = []
    index = 0
    while index < len(lines):
        if tuple(lines[index : index + len(_FALLBACK_BANNER)]) == _FALLBACK_BANNER:
            banners.extend(_FALLBACK_BANNER)
            index += len(_FALLBACK_BANNER)
        else:
            output.append(lines[index])
            index += 1
    return output, banners


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    status: str  # ok | error | timeout | crash
    result: ResultSet | None = None
    error: str = ""
    elapsed: float = 0.0
    exitcode: int | None = None


class Session:
    def __init__(
        self,
        shell: str,
        extension: str | None,
        sirius_config: str | None,
        db_dir: pathlib.Path,
        worker_id: int,
        cpu_only: bool = False,
        evidence_path: pathlib.Path | None = None,
        log_path: pathlib.Path | None = None,
    ):
        self.shell_path = shell
        self.extension = extension
        self.sirius_config = sirius_config
        self.sirius_config_mode: str | None = None
        self.db_dir = db_dir
        self.worker_id = worker_id
        self.gpu_available = not cpu_only
        self.evidence_path = evidence_path
        self.shell = Shell(shell, log_path)
        self.active_settings: dict[str, Any] = {}
        self.evidence: dict[str, Any] = {}
        self.stage = "evaluation"
        self.current_alias: str | None = None
        self.attached: dict[str, pathlib.Path] = {}
        self._setting_defaults: dict[str, Any] = {}
        self.sqlsmith_loaded = False
        self.duckdb_version: str | None = None
        self.sirius: str = "none"  # built-in | loaded | none
        self.restarts = 0

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        self._configure_sirius()
        # Registers the TEST ONLY fault-injection option used as the interception canary.
        os.environ.setdefault("SIRIUS_ENABLE_TEST_OPTIONS", "1")
        self._start()

    def _configure_sirius(self) -> None:
        if not self.gpu_available:
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
        ambient, source = discover_sirius_yaml()
        if ambient:
            # Make Sirius's own choice explicit so the session records what it used.
            self.sirius_config = ambient
            os.environ["SIRIUS_CONFIG_FILE"] = ambient
            self.sirius_config_mode = f"ambient ({source})"
            return
        self.sirius_config_mode = "builtin_defaults"

    def _start(self) -> None:
        """Start the shell and restore the session state a previous shell had.

        No SELECT runs before GPU execution is switched off: with Sirius built in,
        a query would otherwise run on the GPU with fallback still enabled.
        """
        self.shell.start()
        self.sqlsmith_loaded = False
        present = self._disable_gpu()
        if self.gpu_available:
            if present:
                self.sirius = "built-in"
            elif self.extension:
                self._load_extension()
                if not self._disable_gpu():
                    raise SessionError(
                        f"{self.extension} loaded but has no gpu_execution setting"
                    )
                self.sirius = "loaded"
            else:
                raise SessionError(
                    f"{self.shell_path} has no Sirius built in and no --extension was "
                    "given; use build/release/duckdb from the Sirius build, or pass "
                    "--extension, or --cpu-only"
                )
            # Strict mode: a fallback surfaces as an error instead of a silent CPU run.
            self.execute("SET enable_duckdb_fallback = false")
        self.duckdb_version = self.scalar("SELECT version()")
        for alias, path in self.attached.items():
            self.execute(f"ATTACH {sql_literal(str(path))} AS {alias}")
        if self.current_alias:
            self.execute(f"USE {self.current_alias}")
        for name, value in self.active_settings.items():
            self._set(name, value)

    def _disable_gpu(self) -> bool:
        """Switch GPU execution off; False when the shell has no Sirius in it."""
        result = self.shell.execute("SET gpu_execution = false", timeout=120)
        if result.status == "ok":
            return True
        if "unrecognized configuration parameter" in result.error:
            return False
        raise SessionError(result.error or result.log or f"shell {result.status}")

    def _load_extension(self) -> None:
        try:
            self.execute(f"LOAD {sql_literal(self.extension)}")
        except SessionError as exc:
            hint = ""
            if "built specifically for DuckDB version" in str(exc):
                hint = (
                    " (the shell and the extension come from different DuckDB "
                    "versions; use build/release/duckdb from the same build)"
                )
            raise SessionError(f"cannot load {self.extension}: {exc}{hint}") from exc

    def close(self) -> None:
        if self.shell.alive:
            try:
                self.drop_dataset()
            except Exception:
                pass
        self.shell.stop()

    def set_gpu(self, enabled: bool) -> None:
        if self.gpu_available:
            self.execute(f"SET gpu_execution = {'true' if enabled else 'false'}")

    # -- statements -------------------------------------------------------------

    def _ensure_alive(self) -> None:
        if not self.shell.alive:
            self.restarts += 1
            self._start()

    def execute(self, sql: str, timeout: float = 600.0) -> ShellResult:
        """Run a statement that must succeed (setup, settings, data loading)."""
        self._ensure_alive()
        result = self.shell.execute(sql, timeout)
        if result.status != "ok":
            raise SessionError(
                result.error or result.log or f"shell {result.status}: {sql[:120]}"
            )
        return result

    def query(self, sql: str) -> list[dict[str, Any]]:
        result = self.execute(sql)
        return [dict(zip(result.columns, row)) for row in result.rows]

    def scalar(self, sql: str) -> Any:
        result = self.execute(sql)
        return result.rows[0][0] if result.rows else None

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
        self.execute(f"ATTACH {sql_literal(str(path))} AS {alias}")
        self.attached[alias] = path
        self.execute(f"USE {alias}")
        self.current_alias = alias
        self.execute(ds.schema_sql())
        self.execute(ds.data_sql(permutation_seed))
        self.execute("CHECKPOINT")

    def use(self, alias: str) -> None:
        self.execute(f"USE {alias}")
        self.current_alias = alias

    def drop_dataset(self, alias: str | None = None) -> None:
        aliases = [alias] if alias else list(self.attached)
        for a in aliases:
            if self.shell.alive:
                try:
                    self.execute("USE memory")
                    self.execute(f"DETACH {a}")
                except SessionError:
                    pass
            for suffix in ("", ".wal"):
                p = self.db_dir / f"{a}.duckdb{suffix}"
                if p.exists():
                    p.unlink()
            self.attached.pop(a, None)
        self.current_alias = None

    # -- queries --------------------------------------------------------------

    def begin_query(self, sql: str) -> None:
        """Replace the previous query's evidence before evaluating ``sql``.

        The supervisor attributes a worker death to the SQL in the active file, so
        it must name this query even if the worker dies before the first run.
        """
        self.evidence = {}
        self.stage = "evaluation"
        if self.evidence_path:
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

    def run(self, sql: str, gpu: bool, timeout: float) -> RunResult:
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
        if self.evidence_path:
            write_json(self.evidence_path, state)
        self._ensure_alive()
        self.set_gpu(gpu)
        res = self.shell.execute(sql, timeout)
        if res.status == "ok":
            columns = [ColumnInfo(name, UNKNOWN_TYPE) for name in res.columns]
            out = RunResult("ok", ResultSet(columns, res.rows), elapsed=res.elapsed)
        elif res.status == "error":
            status = "timeout" if "INTERRUPT" in res.error.upper() else "error"
            out = RunResult(status, error=res.error, elapsed=res.elapsed)
        elif res.status == "timeout":
            out = RunResult("timeout", error=res.error, elapsed=res.elapsed)
        else:
            out = RunResult(
                "crash", error=res.error, elapsed=res.elapsed, exitcode=res.exitcode
            )
        if self.shell.alive:
            self.set_gpu(False)
        return self._record(phase, state, out)

    def _record(
        self, phase: str, state: dict[str, Any], result: RunResult
    ) -> RunResult:
        state.update(status=result.status, elapsed=result.elapsed, error=result.error)
        if result.result is not None:
            state["row_count"] = result.result.row_count
        self.evidence[phase] = state
        if self.evidence_path:
            write_json(self.evidence_path, state)
        return result

    def describe(self, sql: str) -> list[ColumnInfo] | None:
        """Exact output types (CPU side); None when DESCRIBE itself fails."""
        if not self.shell.alive:
            return None
        self.set_gpu(False)
        res = self.shell.execute(f"DESCRIBE {sql}", timeout=120)
        if res.status != "ok":
            return None
        rows = [dict(zip(res.columns, row)) for row in res.rows]
        return [
            ColumnInfo(r["column_name"], st.parse_duckdb_type(str(r["column_type"])))
            for r in rows
        ]

    # -- settings ---------------------------------------------------------------

    def setting_supported(self, name: str) -> bool:
        rows = self.query(
            f"SELECT value FROM duckdb_settings() WHERE name = {sql_literal(name)}"
        )
        if not rows:
            return False
        self._setting_defaults.setdefault(name, rows[0]["value"])
        return True

    def _set(self, name: str, value: Any) -> None:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
            raise ValueError(f"invalid setting name: {name}")
        literal = sql_literal(value) if isinstance(value, str) else value
        self.execute(f"SET {name} = {literal}")

    def set(self, name: str, value: Any) -> None:
        self._set(name, value)
        self.active_settings[name] = value

    def restore(self, name: str) -> None:
        default = self._setting_defaults.get(name)
        self.active_settings.pop(name, None)
        if default is None:
            return
        # Settings are text in duckdb_settings(); numeric defaults round-trip as-is.
        try:
            self.execute(f"SET {name} = {int(default)}")
        except (ValueError, SessionError):
            self.execute(f"SET {name} = {sql_literal(str(default))}")

    def load_sqlsmith(self) -> bool:
        if self.sqlsmith_loaded:
            return True
        try:
            self.execute("LOAD sqlsmith")
        except SessionError:
            return False
        self.sqlsmith_loaded = True
        return True

    def sqlsmith_candidates(self, sql: str) -> list[str]:
        rows = self.query(f"SELECT sql FROM reduce_sql_statement({sql_literal(sql)})")
        return [r["sql"] for r in rows if r.get("sql")]

    # -- canary -----------------------------------------------------------------

    def check_interception(self) -> tuple[bool, str]:
        """Prove Sirius intercepts plain SQL in this shell.

        Injects a runtime GPU error via the TEST ONLY option; a query that then fails
        with the injected text went through the GPU operator. Falls back to a probe
        query that Sirius rejects at plan time when the option is unavailable.
        """
        if not self.gpu_available:
            return False, "cpu-only mode"
        if self.current_alias is None:
            return False, "no dataset attached"
        tables = self.query(
            "SELECT table_name FROM duckdb_tables() WHERE database_name = "
            f"{sql_literal(self.current_alias)} LIMIT 1"
        )
        if not tables:
            return False, "dataset has no tables"
        table = tables[0]["table_name"]
        probe = f'SELECT count(*) FROM "{table}"'
        if self.setting_supported(CANARY_SETTING):
            try:
                self.set(CANARY_SETTING, "fuzz-canary")
                res = self.run(probe, gpu=True, timeout=120)
            finally:
                self.set(CANARY_SETTING, "")
                self.active_settings.pop(CANARY_SETTING, None)
            if res.status == "error" and "fuzz-canary" in res.error:
                return True, "canary injected error observed"
            return (
                False,
                f"canary not observed (status={res.status}: {res.error[:120]})",
            )
        res = self.run(f'SELECT DISTINCT "k" FROM "{table}"', gpu=True, timeout=120)
        if res.status == "error" and "GPU plan generation failed" in res.error:
            return True, "plan-time rejection observed"
        return (
            False,
            f"probe did not reach Sirius (status={res.status}: {res.error[:120]})",
        )

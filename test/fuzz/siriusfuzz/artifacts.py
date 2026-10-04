# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Portable evidence and best-effort provenance, without collecting secrets."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from typing import Any

from . import __version__
from .config import REPO_ROOT


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def fingerprint(path: pathlib.Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def command(argv: list[str], cwd: pathlib.Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(extension: str | None) -> dict[str, Any]:
    source = pathlib.Path(__file__).parent
    return {
        "schema_version": 1,
        "harness_version": __version__,
        "harness_files": {p.name: fingerprint(p) for p in sorted(source.glob("*.py"))},
        "checkout": {
            "path": str(REPO_ROOT),
            "revision": command(["git", "rev-parse", "HEAD"]),
            "status": command(["git", "status", "--porcelain", "--untracked-files=no"]),
            "submodules": command(["git", "submodule", "status", "--recursive"]),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "extension": (
            {"path": extension, **fingerprint(pathlib.Path(extension))}
            if extension
            else None
        ),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "SIRIUS_DISABLE",
                "SIRIUS_ENABLE_TEST_OPTIONS",
            )
        },
        "note": "Checkout revisions describe source at run time, not proof of the binary's build revision. The extension hash identifies the binary.",
    }


def runtime_info(session: Any) -> dict[str, Any]:
    import duckdb
    import _duckdb

    return {
        "duckdb_version": duckdb.__version__,
        "duckdb_source_id": getattr(duckdb, "__git_revision__", None),
        "duckdb_module": _duckdb.__file__,
        "duckdb_binary": fingerprint(pathlib.Path(_duckdb.__file__)),
        "metadata_mismatch_allowed": session.allow_metadata_mismatch,
        "metadata_mismatch_bypassed": session.version_mismatch_bypassed,
        "sirius_config": session.sirius_config,
        "sirius_config_mode": session.sirius_config_mode,
        "session_settings": dict(
            session.con.execute(
                "SELECT name, value FROM duckdb_settings() WHERE name IN "
                "('threads', 'TimeZone', 'default_null_order', 'default_order', 'preserve_insertion_order', 'disabled_optimizers', "
                "'expression_evaluator_strategy', 'hash_partition_bytes', 'max_build_hash_table_bytes', 'max_sort_partition_bytes')"
            ).fetchall()
        ),
    }


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"

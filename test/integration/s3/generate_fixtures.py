#!/usr/bin/env python3
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Generate local fixtures for the [s3][integration] tests.

Writes files under ``<out_dir>`` (default:
``test/integration/s3/fixtures/local``) which ``fixtures.sh`` then uploads to
the MinIO container via ``mc``. Fixtures are regenerated deterministically each
run so tests can bit-compare the S3-read bytes against the local copy.

  hello.txt          — 16-byte ASCII blob; HEAD + tiny-range test
  small.parquet      — ~256 rows, 3 columns; bit-equal parquet scan
  medium.parquet     — ~200k rows; multi-range reads and larger scans

Uses DuckDB to write the parquet files since DuckDB is already a hard
dependency of the extension; this avoids adding pyarrow/pandas to the pixi env
for a test-only helper.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


HELLO_BYTES = b"sirius-s3-hello\n"  # exactly 16 bytes


def write_hello(out_dir: Path) -> Path:
    p = out_dir / "hello.txt"
    p.write_bytes(HELLO_BYTES)
    assert p.stat().st_size == 16, f"hello.txt should be 16 bytes, got {p.stat().st_size}"
    return p


def write_parquet_via_duckdb(out_path: Path, sql_select: str) -> Path:
    # Lazy import so the script can still print --help without duckdb installed.
    try:
        import duckdb  # type: ignore
    except ImportError as e:  # pragma: no cover
        sys.stderr.write(
            "error: python `duckdb` module not found; "
            "inside pixi shell run `pip install duckdb` or use the "
            "duckdb-python pixi env.\n"
        )
        raise SystemExit(1) from e

    con = duckdb.connect(":memory:")
    con.execute(f"COPY ({sql_select}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION snappy)")
    con.close()
    return out_path


def write_small_parquet(out_dir: Path) -> Path:
    p = out_dir / "small.parquet"
    # 256 rows, 3 columns with different types — exercises cudf's column decode
    # path enough to catch any byte-level mismatch between local and S3 reads.
    return write_parquet_via_duckdb(
        p,
        """
        SELECT
          i::INTEGER                              AS id,
          (i * 7919)::BIGINT                      AS v,
          ('name_' || lpad(i::VARCHAR, 4, '0'))   AS name,
          (DATE '2024-01-01' + INTERVAL (i) DAY)  AS d
        FROM range(256) t(i)
        """,
    )


def write_medium_parquet(out_dir: Path) -> Path:
    p = out_dir / "medium.parquet"
    # ~200k rows, small row groups so multi-range reads cross boundaries.
    return write_parquet_via_duckdb(
        p,
        """
        SELECT
          i::INTEGER                               AS id,
          (i % 97)::SMALLINT                       AS bucket,
          (random() * 1e9)::BIGINT                 AS v,
          repeat('x', 32)                          AS payload
        FROM range(200000) t(i)
        """,
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "local",
        help="output directory for generated fixtures",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional path to write a sha256 manifest (defaults to <out>/MANIFEST.sha256)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        write_hello(out_dir),
        write_small_parquet(out_dir),
        write_medium_parquet(out_dir),
    ]

    manifest_path = args.manifest or (out_dir / "MANIFEST.sha256")
    with manifest_path.open("w") as f:
        for p in paths:
            f.write(f"{sha256_of(p)}  {p.name}\n")

    for p in paths:
        print(f"  wrote {p} ({p.stat().st_size} bytes)")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

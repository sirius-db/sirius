#!/usr/bin/env python3
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Prepare local fixtures for the [s3][integration] tests.

Writes files under ``<out_dir>`` which the test harness
(``test/cpp/utils/s3_container.*``) then uploads to the MinIO containers from
the host via Sirius's SigV4 signer. Text/binary fixtures are regenerated
deterministically each run, and the standard integration Parquet fixtures are
copied from ``test/cpp/integration/data/parquet`` so S3 tests exercise the same
known TPCH, nested, and edge-case data as the regular GPU integration suite.

  hello.txt        -- 16-byte ASCII blob; HEAD + tiny-range test
  small.bin        -- 20 KiB deterministic binary blob; bit-equal read via factory
  medium.bin       -- 8 MiB deterministic binary blob; multi-range reads
  parquet/*.parquet -- standard integration Parquet fixtures used by semantic tests

The binary blobs are opaque bytes (NOT real parquet) -- the byte-equality
tests in test_s3_integration.cpp only need deterministic, size-known objects.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path


HELLO_BYTES = b"sirius-s3-hello\n"  # exactly 16 bytes
SMALL_SIZE = 20 * 1024  # 20 KiB
MEDIUM_SIZE = 8 * 1024 * 1024  # 8 MiB (medium test needs > 4 MiB)
SMALL_SEED = 0xA17E57
MEDIUM_SEED = 0xBE57ED


def write_hello(out_dir: Path) -> Path:
    p = out_dir / "hello.txt"
    p.write_bytes(HELLO_BYTES)
    assert (
        p.stat().st_size == 16
    ), f"hello.txt should be 16 bytes, got {p.stat().st_size}"
    return p


def write_deterministic_bytes(out_path: Path, size: int, seed: int) -> Path:
    # random.Random seeded with a fixed int produces identical bytes across runs
    # and across Python 3.9+ platforms - good enough for bit-equality tests that
    # don't care about format, only about byte stability.
    rng = random.Random(seed)
    out_path.write_bytes(rng.randbytes(size))
    assert out_path.stat().st_size == size
    return out_path


def copy_parquet_fixtures(source_dir: Path, out_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Parquet fixture directory not found: {source_dir}")

    dst_dir = out_dir / "parquet"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    paths: list[Path] = []
    for src in sorted(source_dir.glob("*.parquet")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        paths.append(dst)
    if not paths:
        raise FileNotFoundError(f"No .parquet fixtures found in {source_dir}")
    return paths


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
    parser.add_argument(
        "--parquet-source",
        type=Path,
        required=True,
        help="directory containing standard integration .parquet fixtures to upload",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "hello.txt",
        "small.bin",
        "medium.bin",
        "small.parquet",
        "MANIFEST.sha256",
    ):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    paths = [
        write_hello(out_dir),
        write_deterministic_bytes(out_dir / "small.bin", SMALL_SIZE, SMALL_SEED),
        write_deterministic_bytes(out_dir / "medium.bin", MEDIUM_SIZE, MEDIUM_SEED),
    ]
    paths.extend(copy_parquet_fixtures(args.parquet_source, out_dir))

    manifest_path = args.manifest or (out_dir / "MANIFEST.sha256")
    with manifest_path.open("w") as f:
        for p in paths:
            f.write(f"{sha256_of(p)}  {p.relative_to(out_dir)}\n")

    for p in paths:
        print(f"  wrote {p} ({p.stat().st_size} bytes)")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

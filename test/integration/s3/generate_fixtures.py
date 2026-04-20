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

  hello.txt      — 16-byte ASCII blob; HEAD + tiny-range test
  small.bin      — 20 KiB deterministic binary blob; bit-equal read via factory
  medium.bin     — 8 MiB deterministic binary blob; multi-range reads
  small.parquet  — 256-row patterned parquet (requires pyarrow); drives the
                   [s3][parquet][integration] test which parses the S3-read
                   bytes with DuckDB and checks semantic correctness.

The binary blobs are opaque bytes (NOT real parquet) — the byte-equality
tests in test_s3_integration.cpp only need deterministic, size-known objects.
Parquet generation is optional: if pyarrow is not installed we skip
small.parquet with a warning, and the parquet-integration test
auto-skips when the object is absent from S3. Install pyarrow (``pip install
pyarrow``) on the host that runs ``make s3-up`` to enable it.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path


HELLO_BYTES = b"sirius-s3-hello\n"  # exactly 16 bytes
SMALL_SIZE = 20 * 1024              # 20 KiB
MEDIUM_SIZE = 8 * 1024 * 1024       # 8 MiB (medium test needs > 4 MiB)
SMALL_SEED = 0xA17E57
MEDIUM_SEED = 0xBE57ED

PARQUET_ROWS = 256
PARQUET_KNUTH = 2654435761          # Knuth's multiplicative hash constant
PARQUET_INT64_MASK = (1 << 63) - 1  # keep v positive so it fits INT64 cleanly


def write_hello(out_dir: Path) -> Path:
    p = out_dir / "hello.txt"
    p.write_bytes(HELLO_BYTES)
    assert p.stat().st_size == 16, f"hello.txt should be 16 bytes, got {p.stat().st_size}"
    return p


def write_deterministic_bytes(out_path: Path, size: int, seed: int) -> Path:
    # random.Random seeded with a fixed int produces identical bytes across runs
    # and across Python 3.9+ platforms — good enough for bit-equality tests that
    # don't care about format, only about byte stability.
    rng = random.Random(seed)
    out_path.write_bytes(rng.randbytes(size))
    assert out_path.stat().st_size == size
    return out_path


def write_patterned_parquet(out_path: Path, num_rows: int = PARQUET_ROWS) -> Path | None:
    """Write a patterned parquet if pyarrow is available; return None otherwise.

    Schema: id INT32, v INT64, s VARCHAR. Values follow a closed-form pattern
    so the C++ test can regenerate expected values without reading the file:
      id = 0..num_rows-1
      v  = (id * 2654435761) & INT64_MAX
      s  = f"row-{id:04d}"

    Compression is snappy (DuckDB/cudf both read it; matches what Sirius's
    prod parquet scan encounters most often).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "  [skip] small.parquet — pyarrow not installed; run"
            " `pip install pyarrow` to enable the [s3][parquet][integration] test",
            file=sys.stderr,
        )
        return None

    ids = list(range(num_rows))
    vs = [((i * PARQUET_KNUTH) & PARQUET_INT64_MASK) for i in ids]
    ss = [f"row-{i:04d}" for i in ids]

    table = pa.table(
        {
            "id": pa.array(ids, type=pa.int32()),
            "v": pa.array(vs, type=pa.int64()),
            "s": pa.array(ss, type=pa.string()),
        }
    )
    pq.write_table(table, out_path, compression="snappy")
    return out_path


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
        write_deterministic_bytes(out_dir / "small.bin", SMALL_SIZE, SMALL_SEED),
        write_deterministic_bytes(out_dir / "medium.bin", MEDIUM_SIZE, MEDIUM_SEED),
    ]
    parquet_path = write_patterned_parquet(out_dir / "small.parquet")
    if parquet_path is not None:
        paths.append(parquet_path)

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

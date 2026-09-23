#!/usr/bin/env python3
"""Parquet inputs for the SQL join and shuffle benchmarks, one file per worker.

Same rules as cascade-tpc-shuttle's bench.cpp (after distributed-join):
  join:    build and probe tables of <rows> per worker, INT64 key + INT64 payload, unique
           build keys (the even numbers of [2*rows*w, 2*rows*(w+1))) in random order, a probe
           key hits the build side with probability 0.3 and is an odd number otherwise.
  shuffle: one INT32 column of <rows> per worker, keys rand % (10 * rows).
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

WORKERS = 2


def write(path: Path, table: pa.Table, pad: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No embedded Arrow schema: a footer key-value pad then grows the file byte for byte.
    with pq.ParquetWriter(path, table.schema, store_schema=False) as writer:
        writer.write_table(table)
        if pad:
            writer.add_key_value_metadata({"pad": "x" * pad})


def equalize(paths: list[Path], tables: list[pa.Table]) -> None:
    """Pad parquet footers so every file has the same byte size.

    StarRocks FileScanNode splits a file whose size exceeds total / instances, and this CN
    only scans whole files. Equal sizes make the two files land on the two CNs whole.
    """
    target = max(p.stat().st_size for p in paths) + 64
    for path, table in zip(paths, tables):
        pad = 32
        for _ in range(50):
            write(path, table, pad)
            size = path.stat().st_size
            if size == target:
                break
            pad = max(1, pad + (target - size))
        else:
            raise SystemExit(f"could not pad {path} to {target} bytes")


def main() -> None:
    out = Path(sys.argv[1])
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    written: dict[str, list[tuple[Path, pa.Table]]] = {}

    def emit(kind: str, path: Path, table: pa.Table) -> None:
        write(path, table)
        written.setdefault(kind, []).append((path, table))

    for w in range(WORKERS):
        rng = np.random.default_rng(w + 1)
        rand_max = 2 * rows
        key_base = rand_max * w
        pay_base = rows * w
        build_keys = key_base + 2 * np.arange(rows, dtype=np.int64)
        rng.shuffle(build_keys)
        hit = rng.random(rows) < 0.3
        pick = rng.integers(0, rows, rows, dtype=np.int64)
        probe_keys = key_base + 2 * pick + np.where(hit, 0, 1)
        payload = pay_base + np.arange(rows, dtype=np.int64)
        emit("build", out / "join" / "build" / f"build_{w}.parquet",
             pa.table({"b_key": build_keys, "b_pay": payload}))
        emit("probe", out / "join" / "probe" / f"probe_{w}.parquet",
             pa.table({"p_key": probe_keys, "p_pay": payload}))
        keys = rng.integers(0, 10 * rows, rows, dtype=np.int64).astype(np.int32)
        emit("shuffle", out / "shuffle" / "t" / f"t_{w}.parquet", pa.table({"k": keys}))
    for files in written.values():
        equalize([p for p, _ in files], [t for _, t in files])
        sizes = {p.name: p.stat().st_size for p, _ in files}
        print(sizes)
    print(f"wrote {WORKERS} workers x {rows} rows under {out}")


if __name__ == "__main__":
    main()

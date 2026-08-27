#!/usr/bin/env python
"""Measure the same polars query under --io-mode lukewarm vs hot.

Reproduces, with plain polars over the same parquet dataset, the only two settings
that differed between the published Engine C numbers and the current ones:

                     BEFORE (08-08 published)   CURRENT (run-abc.sh)
    io_mode          lukewarm                   hot
    iterations       1                          4

Semantics are copied from the installed harness
(cudf_polars/streaming/benchmarks/utils.py, cudf-polars 26.08):

    cold      drop the Linux page cache before EVERY iteration (needs kvikio)
    lukewarm  no cache manipulation at all -- it does not warm anything either (default)
    hot       iteration 0 is a warm-up; only iterations 1+ are the measurement

Two warts this script makes visible rather than hiding:

  * `hot` is not an engine setting. The only io_mode with a runtime code path is
    `cold`; `hot` and `lukewarm` execute identically. utils.py:521 merely VALIDATES
    iterations >= 2 for `hot`, and iteration 0 is still recorded in the jsonl. The
    discard is the consumer's job -- so `--io-mode hot --iterations 1` is refused,
    while `lukewarm --iterations 1` silently reports the first touch as the score.
  * With `lukewarm --iterations 1` there is no iteration 1. The one execution you
    measure is the warm-up: first-touch allocation, cold page cache, worker spin-up.

Usage:
    python io-mode-demo.py                      # runs both protocols, prints the delta
    python io-mode-demo.py --io-mode hot --iterations 4
    python io-mode-demo.py --query q1 --engine gpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

DEFAULT_PATH = Path("/home/prestouser/aocsa/tpch_parquet_sf100")
IO_MODES = ("cold", "lukewarm", "hot")


# --------------------------------------------------------------------------- queries


def scan(path: Path, table: str) -> pl.LazyFrame:
    """Scan the <table>/*.parquet layout the harness uses with --suffix ''."""
    return pl.scan_parquet(path / table / "*.parquet")


def q6(path: Path) -> pl.LazyFrame:
    """Scan-bound: one table, one filter, one sum. Dominated by I/O, so the
    lukewarm/hot gap shows up as page-cache cost more than compute warm-up."""
    line = scan(path, "lineitem")
    return (
        line.filter(
            pl.col("l_shipdate").is_between(
                pl.date(1994, 1, 1), pl.date(1994, 12, 31), closed="left"
            )
            & pl.col("l_discount").is_between(0.05, 0.07)
            & (pl.col("l_quantity") < 24)
        )
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
    )


def q1(path: Path) -> pl.LazyFrame:
    """Group-by over the whole of lineitem. The 2.4x q01 outlier in the published
    Engine C sweep lives here -- it was iteration 0 wearing a warm number's clothes."""
    line = scan(path, "lineitem")
    return (
        line.filter(pl.col("l_shipdate") <= pl.date(1998, 9, 2))
        .with_columns(
            disc_price=pl.col("l_extendedprice") * (1 - pl.col("l_discount")),
        )
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            pl.col("disc_price").sum().alias("sum_disc_price"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


QUERIES = {"q1": q1, "q6": q6}


# --------------------------------------------------------------------------- protocol


@dataclass(frozen=True)
class RunConfig:
    """The subset of the harness's RunConfig that this demo needs."""

    dataset_path: Path
    query: str
    iterations: int
    io_mode: str = "lukewarm"
    engine: str = "cpu"

    def __post_init__(self) -> None:
        # Verbatim from utils.py:521 -- the whole reason the mistake was catchable.
        if self.io_mode == "hot" and self.iterations < 2:
            raise ValueError(
                "--io-mode hot requires at least 2 iterations: "
                "iteration 0 warms the cache, iterations 1+ are the hot measurements."
            )
        if self.io_mode not in IO_MODES:
            raise ValueError(f"io_mode must be one of {IO_MODES}, got {self.io_mode!r}")

    @property
    def measured(self) -> range:
        """Iteration indices that count toward the reported number."""
        return range(1, self.iterations) if self.io_mode == "hot" else range(self.iterations)


@dataclass
class Result:
    config: RunConfig
    durations: list[float] = field(default_factory=list)  # seconds, index == iteration

    @property
    def iter0(self) -> float:
        return self.durations[0]

    @property
    def reported(self) -> float:
        """Median of whatever the protocol says is the measurement."""
        return statistics.median(self.durations[i] for i in self.config.measured)

    def label(self) -> str:
        m = list(self.config.measured)
        return f"{self.config.io_mode}x{self.config.iterations} (reports iter {m})"


def drop_page_cache(path: Path) -> None:
    """utils.py:793, same failure message so the missing dependency reads the same."""
    try:
        import kvikio
    except ImportError as err:
        raise RuntimeError(
            "kvikio is required for cold-run page cache dropping. "
            "Install it or switch to --io-mode lukewarm."
        ) from err
    for f in path.expanduser().rglob("*"):
        if f.is_file():
            kvikio.drop_file_page_cache(f)


def run(config: RunConfig, *, verbose: bool = True) -> Result:
    result = Result(config)
    build = QUERIES[config.query]

    for i in range(config.iterations):
        if config.io_mode == "cold":
            drop_page_cache(config.dataset_path)

        # Rebuild the LazyFrame each iteration: reusing one lets polars serve a
        # cached plan and hides exactly the first-touch cost we are measuring.
        lf = build(config.dataset_path)

        t0 = time.perf_counter()
        if config.engine == "gpu":
            lf.collect(engine="gpu")
        else:
            lf.collect()
        elapsed = time.perf_counter() - t0

        result.durations.append(elapsed)
        if verbose:
            phase = "cold (warm-up)" if i == 0 else "warm"
            counted = "counted" if i in config.measured else "DISCARDED"
            print(f"  iter {i}  {elapsed * 1000:8.1f} ms  {phase:15} {counted}")

    return result


# --------------------------------------------------------------------------- reporting


def compare(path: Path, query: str, engine: str, hot_iterations: int) -> None:
    print(f"dataset : {path}")
    print(f"query   : {query}   engine: {engine}\n")

    print("BEFORE -- lukewarm, iterations=1  (the published protocol)")
    before = run(RunConfig(path, query, iterations=1, io_mode="lukewarm", engine=engine))

    print(f"\nCURRENT -- hot, iterations={hot_iterations}  (run-abc.sh)")
    after = run(RunConfig(path, query, iterations=hot_iterations, io_mode="hot", engine=engine))

    print("\n" + "-" * 68)
    print(f"{'protocol':34} {'reported':>12} {'iter 0':>12}")
    for r in (before, after):
        print(f"{r.label():34} {r.reported * 1000:11.1f}ms {r.iter0 * 1000:11.1f}ms")
    print("-" * 68)

    ratio = before.reported / after.reported
    print(
        f"\nSame code, same data, same engine: {ratio:.2f}x apparent 'speedup' from the\n"
        f"protocol alone. The published number ({before.reported * 1000:.0f} ms) is the "
        f"warm-up; today's\niteration 0 ({after.iter0 * 1000:.0f} ms) is the same "
        f"measurement under a different flag."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, default=DEFAULT_PATH, help="parquet dataset root")
    p.add_argument("--query", choices=sorted(QUERIES), default="q6")
    p.add_argument("--engine", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--io-mode", dest="io_mode", choices=IO_MODES, default=None,
                   help="run one protocol instead of comparing both")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("-o", "--output", type=Path, help="write per-iteration records as JSON")
    args = p.parse_args()

    if not args.path.is_dir():
        p.error(f"dataset not found: {args.path}")

    if args.io_mode is None:
        compare(args.path, args.query, args.engine, args.iterations)
        return

    config = RunConfig(args.path, args.query, args.iterations, args.io_mode, args.engine)
    result = run(config)
    print(f"\n{result.label()}  ->  {result.reported * 1000:.1f} ms")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "io_mode": config.io_mode,
                    "iterations": config.iterations,
                    "query": config.query,
                    "engine": config.engine,
                    "dataset_path": str(config.dataset_path),
                    "records": [
                        {"iteration": i, "duration": d, "counted": i in config.measured}
                        for i, d in enumerate(result.durations)
                    ],
                },
                indent=2,
            )
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

# TPC-H benchmark: engine A (Sirius GPU CN) vs engine B (standalone StarRocks)

## Setup (identical for both, sequential runs on one host)

- Host: 16-core x86_64, 60 GiB RAM, 1× NVIDIA L4 (23 GiB), Ubuntu 24.04.
- Data: TPC-H **SF1** parquet, one file per table at
  `/home/ubuntu/git/sirius/scratch/tpch_sf1/<table>/part.0.parquet` (388 MB total), queried
  via `FILES("path"="file:///…","format"="parquet")` CTEs — **no data loading**; both engines
  scan the same external files (lineitem 155 MB gets byte-split across the two backends by
  the FE in both cases).
- Distribution shape: 1 FE + **2 backend processes** on the same host.
  - **A**: the `demo-multi-cn` FE + 2 Sirius GPU CNs (8 GiB GPU carve-out each + 512 MiB
    exchange arena, nixl/UCX cross-process exchange), `pixi run cluster2`.
  - **B**: stock StarRocks (prebuilt binaries extracted from `starrocks/artifacts-ubuntu`
    Docker image; version recorded below) — 1 FE + 2 CNs/BEs, `mem_limit` capped identically,
    same ports, engine A fully down during B runs (both contend for the same CPUs).
- Queries: the duckdb-parameterized TPC-H Q1–Q22 texts with FILES() CTE preludes (the same
  files the plan-survey used; Q22 substring dialect fix applied).
- Protocol per query: 1 discarded warm-up run + N=3 timed runs, wall-clock ms measured around
  `mysql -e`; median reported. `query_timeout` left at 300 s. Engine A gets a CN restart after
  any mid-execution failure (known no-cancel/GC gap); refusals are recorded as unsupported,
  not timed.
- Fairness caveats recorded up front: A executes on the GPU (that is the comparison's point);
  B's native BE is a mature vectorized CPU engine with years of optimization; SF1 is small
  enough that fixed per-query overheads (fragment dispatch, first-touch) matter as much as
  scan throughput; results are indicative, not a TPC-compliant benchmark.

## Status log

- 2026-08-06/07: harness ready (`bench.sh <label> <runs> [restart_cmd]` → `timings.csv`).
  Engine B binaries: Docker Hub `starrocks/artifacts-ubuntu:3.5.20` pull in progress (direct
  tarball URLs 403). Engine A blockers before its sweep: two-phase avg landed in the
  translator (scalar avg oracle-exact live) but the GROUPED avg merge fragment (merge agg +
  finalize division + SORT) hangs in GPU execution — root-caused
  later the same day (empty exchange partition; fixed in 19d7cca2).

- 2026-08-07: avg stack landed (19d7cca2, 2c535b0e, bd232c40) + merge-cast/conformance
  hardening (bb066e90, 830380f4). Harness committed at
  `experimental/starrocks/benchmarks/tpch/` (c69002ce). A-sweep attempt: q01 ~400 ms,
  q04 ~308 ms, q06 ~190 ms (pass, 3 runs each); q05 loud refusal (cross-fragment ORDER BY
  on a non-leading sort-tuple slot); q02/q03/q07/q08 silent 300 s hangs — under
  investigation (see QUERY-TIMEOUT-ANALYSIS.md when it lands); q09–q22 invalidated by the
  hang cascade (no cancel_plan_fragment → stranded fragments → "No available backends").
  Full A-vs-B table + plot deferred until the hang classes are fixed.
- 2026-08-07 (later): timeout fixes landed (c858e79a failure propagation + cancel stubs,
  4beca977 date-fn casts, 4323197d sort-tuple order; QUERY-TIMEOUT-ANALYSIS.md has the
  full post-fix table). A sweep now: 15 pass / 6 loud refusals / 1 wedge, zero cascade
  (was 3 pass / wipe-out). CSV: /tmp/sirius-tpch-bench/bench/A2/timings.csv. Caveat before
  publishing a table: revenue-shaped sums carry a ~0.1 % deficit (see analysis doc).
- 2026-08-07 (final): third fix wave landed (7bdcd312 grouping-key slot order, a94e8660
  lease decoupling + SIGTERM escalation, 8c23e7e7 decimal hash keys, 1d4428da 1280 MiB
  staging arena, 90750142 harness; CLONE_EXPR + slot-id-fallback translator patches pending
  commit). Full A sweep, 3 timed runs/query: **19/22 run to completion with exact keys,
  counts, and self-consistent ordering; 17/22 additionally hold all values inside the 0.25 %
  tolerance** — q03/q10 revenue values deterministically low beyond the band (task #24).
  Non-pass rows, each reproduced solo with a characterized cause: q02 wedge (engine-thread
  stall, #26), q14 loud refusal (common_slot_map descriptor error — new translator blocker),
  q15 empty-result flake (FP64 equality race, #29; 3/6 empty on a warm cluster). Zero
  cascade, clean teardown (0 leftover processes, 0 MiB GPU). CSV:
  /tmp/sirius-tpch-bench/bench/A4/timings.csv; per-query table + open-issue detail in
  QUERY-TIMEOUT-ANALYSIS.md "Final status".

- 2026-08-07 (final): **20/22 pass** after fe236e8b (q14 common slots). A5 sweep CSV:
  /tmp/sirius-tpch-bench/bench/A5/timings.csv. Warm medians 288 ms (q01) to 1.2 s (q08).
  Outstanding: q02 (#26), q15 flake (#29), and the deferred #24 value deficit.

## Results

(pending — table + plot land here; partial A data preserved at /tmp/sirius-tpch-bench/bench/)

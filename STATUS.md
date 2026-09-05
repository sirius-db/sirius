# Multi-CN performance work

Status: implementation, controlled SF500 comparison, separate Q09/Q21 dispatch diagnostics, profiling and validation are complete.

- Branch: `perf/multi-cn-ingress-packing-transfer`, based on `all22/integration` at `95bec853b684e1510c7ddb3d9becc9b73374e983`.
- Plans 04, 07 and 06 were recovered from `origin/codex/starrocks-performance-plans` at `0515ff75ad364a7c7b754b5addd154c6f2adae0b`; the supplied Ubuntu directory is absent on this machine. The recovered documents are proposals. Current scope and limitations are in [RESULTS.md](RESULTS.md).
- Implemented opt-in owned ingress/credits with physical host reservations; spillable inbound and parked output; independent GPU packing; asynchronous fair transfer/control workers; local dispatch/drain overlap; retry-safe lease and cancellation ownership. Supporting ownership, spill/reload and peer-setup prerequisites were included.
- No changes under `src/legacy/`.

## Measured outcomes

- Exactly two CNs on one physical GB10, each with the original 24 GiB GPU, 8 GiB host, 2 GiB staging and 512 GiB spill limits. SF500 remains the same verified 131.306 GiB Parquet dataset.
- Frozen baseline: 15/22 queries pass all three repetitions; 45 successful attempts, seven failures and 14 warm slots skipped after failure. All seven historical GPU capacity failures reproduced.
- Optimized exchange: 20/22 queries have eligible two-CN successes in all three repetitions. There are 63 raw correctness passes, but Q09's three successes followed one-CN retries and are excluded, leaving 60 eligible samples. One SQL client failure and two skipped warm slots remain. Q05/Q07/Q08/Q17/Q18 are recovered in the controlled two-CN sweep.
- Same 15 successful queries: warm aggregate 170.108 seconds baseline versus 183.034 seconds optimized, a 7.6% time regression. No general throughput speedup claim. Q13 and Q04 are the largest regressions.
- Q21 hits the separate FE 60-second deployment RPC deadline while an inline sender takes about 77 seconds. The 139.224-second failed attempt includes an FE retry. A separate frozen-binary Q21 experiment enables the existing asynchronous sender-dispatch flag and passed all three repetitions (292.920 / 305.074 / 266.183 seconds) with two-CN execution. Q09 also passed all three verified two-CN attempts under that flag (142.703 / 145.256 / 215.895 seconds), without FE retries. Neither follow-up replaces the original failed/degraded samples.
- Cold means fresh application cluster, with uncontrolled OS page cache and transport warmup. One cold plus two warm attempts per arm/query; alternating arm order; independent restart after each failure. Numeric comparison is a duplicate-preserving multiset with relative `1e-6` and absolute `1e-8` tolerance; output order is not verified.

## Validation

- Engine and native NIXL CN builds passed; original and optimized binaries are frozen separately under `build/multi-cn-throughput/` and verified against actual running executable and library mappings.
- CPU CN/translator suite: 454 tests passed. Later PRPC subset: five tests passed, including two new retry regressions.
- C++ repository/staging tests: 25 cases, 20,330 assertions passed.
- Four GPU regressions passed: independent packing during occupied engine queue, query-retirement isolation, null/large-string ingress with early staging reuse, and safe late-handle rejection after context teardown. Explicit test budgets are 4 GiB GPU and 1 GiB host; the initial default-pool setup failure is preserved separately.
- Ten focused harness tests passed for shutdown timeout paths, PID ownership, and asynchronous-dispatch CLI/env/manifest consistency. Fourteen retry-placement tests also passed, including all three real Q09 degradations and the Q21 failure. A log-close error cannot bypass owned-process cleanup; a successful one-CN retry cannot silently qualify as a two-CN benchmark result.
- SF500 smoke Q01/Q06/Q05 passed; cold lazy peer establishment with warmup disabled passed Q01. Main timed sweep uses warmup enabled.
- Changed-file checks, Rust workspace formatting and all 47 primary-and-follow-up cluster telemetry decodes passed. The final comparison has 22 per-query profiles; early missing log-filter metadata remains an explicit warning.

## Evidence and reproduction

- [RESULTS.md](RESULTS.md): outcome, tradeoffs, measurement protocol and implementation limits.
- [MULTI_CN_BENCHMARK.md](scripts/local-gb10/MULTI_CN_BENCHMARK.md): local build, artifact freezing, data/oracle preparation, A/B and profiling commands.
- [Full comparison and per-query profiles](results/multi-cn-throughput-ab/analysis/RESULTS.md), [Q09 verified two-CN follow-up](results/multi-cn-throughput-q09-async-dispatch/PROFILE.md), [credit trace with degraded-retry distinction](results/multi-cn-throughput-credit-profile/PROFILE.md).
- [Controlled A/B artifacts](results/multi-cn-throughput-ab/), [Q21 separate dispatch diagnostic](results/multi-cn-throughput-q21-async-dispatch/), [smoke evidence](results/multi-cn-throughput-smoke/), [cold lazy setup](results/multi-cn-throughput-lazy-session/).
- [Q05 memory/packing profile](results/multi-cn-throughput-smoke/Q05-PROFILE.md), [Q13 regression profile](results/multi-cn-throughput-ab/Q13-REGRESSION-PROFILE.md), [Q03/Q04 profiles](results/multi-cn-throughput-ab/FITTING-QUERY-PROFILE.md), [Q21 failure profile](results/multi-cn-throughput-ab/optimized/q21/FAILURE-PROFILE.md).
- Source mode is default-off. EOS execution gating, synchronous pack completion fencing, explicit oversized-frame failures, bounded session replay storage, and fail-closed peer-epoch/quarantined-WRITE recovery remain documented limits. The full production acceptance matrix and multi-GPU scaling have not been established.

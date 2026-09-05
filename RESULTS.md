# SF500 multi-CN exchange improvements

The controlled two-CN sweep improved completion from **15/22 to 20/22 queries** at the original memory limits. All 20 eligible optimized queries passed one cold and two warm repetitions. Q09 returned correct results only after one-CN retries and is excluded from successful two-CN measurements. The same 15 queries that both versions complete were **7.6% slower in aggregate warm time** with the optimized path. This is a capacity and progress improvement; this machine does not demonstrate a general throughput speedup.

Work is on `perf/multi-cn-ingress-packing-transfer`, based on `all22/integration` commit `95bec853b684e1510c7ddb3d9becc9b73374e983`. The implementation commit is `7412ef4d`.

## Controlled comparison

| Measurement | Frozen baseline | Optimized exchange |
|---|---:|---:|
| Queries passing every repetition with eligible two-CN execution | 15/22 | 20/22 |
| Correct SQL client results | 45 | 63 |
| Eligible successful two-CN samples | 45 | 60 |
| Correct results excluded for one-CN retry | 0 | 3 |
| Failed measured attempts | 7 | 1 |
| Warm slots skipped after failure | 14 | 2 |
| Same 15 queries: cold query-time sum | 172.840 s | 192.157 s |
| Same 15 queries: sum of warm medians | 170.108 s | 183.034 s |
| Successful complete 22-query suite time | Unavailable | Unavailable |

Q05, Q07, Q08, Q17 and Q18 now pass on two CNs, each in all three repetitions. Q09's three correct results followed 60-second FE deployment timeouts and successful retries on only one CN; the other CN was blacklisted. These remain raw correctness passes, but do not qualify as two-CN successes. Checking two registered/live CNs before and after a query did not catch this; the analyzer now audits retry UUIDs and actual fragment participation. Baseline Q21 fails with GPU capacity exhaustion; optimized Q21 reaches a different limit: the FE's **60-second fragment deployment RPC deadline**. Its 139.224-second failed client attempt includes an automatic FE retry and is not a successful query time.

The historical reference also passed 15/22 with the same seven failing queries. Its 15 successful query times sum to 170.255 seconds; it used one attempt per query and retained clusters between successful queries. The fresh frozen A/B comparison above is the primary timing comparison because it controls the new repetition and restart protocol. Historical standalone Sirius's 244.319-second full-suite result uses a different execution path and memory configuration; it is context, not a speedup denominator.

The largest fitting-query regressions are Q13, **16.081 → 23.456 s (+45.9%)**, and Q04, **20.033 → 24.040 s (+20.0%)**, using per-query warm medians. Q13 logs show two large receivers that previously ran mostly sequentially now overlapping on the same physical GPU. Their individual durations increase from roughly 5.4 to 18 seconds. Resource contention is a supported hypothesis, not an isolated causal measurement. Q04 also shows export reload activity. See the dedicated [Q13 profile](results/multi-cn-throughput-ab/Q13-REGRESSION-PROFILE.md) and [Q03/Q04 profiles](results/multi-cn-throughput-ab/FITTING-QUERY-PROFILE.md).

## What changed

| Plan priority | Implemented mechanism | Evidence and remaining limits |
|---|---|---|
| 04: early ingress and receive credits | Reserve actual host evacuation capacity before granting a receive range; fence ingress before releasing staging; register inbound and parked output for GPU/HOST/DISK downgrade and reload; replay-safe ownership and bounded credits. | Q05 processes over 22 GiB received per CN through a 2 GiB arena. Its smoke cluster's observed receive-ledger peak is 122.616 MiB per CN; live, copying and quarantined bytes return to zero. Execution still waits for EOS; publication acknowledgment waits for evacuation completion. |
| 07: independent GPU packing | Owned export provider, dedicated CUDA stream and packing worker, synchronized residency transitions, producer-event ordering, safe query/context retirement. | The GPU regression packs while the engine queue is occupied. Q05 packs 351 frames / 9.738 GiB during a local join. This proves progress outside the engine queue, not simultaneous GPU kernel execution. CUDA event-ticket scheduling remains future work. |
| 06: fair transfer pipeline | Bounded per-peer transfer windows, independent packing/control/cleanup workers, asynchronous peer establishment and transfer polling, source leases held until WRITE completion. | Full two-CN sweep, cold lazy-session test and ownership/fairness tests. There is only one remote destination per CN in this topology, so slow-destination fairness across several remote peers is not established by this benchmark. |

The integration baseline already copied receives out of staging before EOS. The critical added memory mechanism is making exchange ownership spillable and discoverable by the downgrade executor; merely limiting grants would otherwise stall an EOS-gated receiver once persistent GPU storage filled.

The path is opt-in with `SIRIUS_EXCHANGE_OPTIMIZED=1` on both CNs. Default behavior remains unchanged. Oversized frames fail explicitly; batch splitting is not implemented. A peer epoch change or uncertain in-flight WRITE fails closed and may require restarting both CNs. Session replay storage is bounded but does not yet have automatic garbage collection. These are implemented slices of the plans, not completion of their entire production acceptance matrix.

Keep this path opt-in: the evidence supports larger exchanges completing within the memory budget, while fitting workloads can regress on a shared GPU. A GPU-resident ingress fast path with reserved host fallback and scheduling across CNs sharing a GPU are follow-up candidates; neither was added without a separate controlled measurement.

## Measurement conditions

- One NVIDIA GB10 on Ubuntu 24.04 aarch64, 119 GiB shared CPU/GPU memory. Both CNs use physical GPU 0; this is not a multi-GPU fabric experiment.
- Per CN: 24 GiB GPU pool, 8 GiB host pool, 2 GiB separate staging arena, 512 GiB disk spill limit. One FE, exactly two live CNs and no live BEs.
- SF500: 47 Parquet files, 140,988,717,545 bytes (131.306 GiB). Original frozen SQL and typed DuckDB oracles were reused after fingerprint validation, including the SF500 Q11 threshold.
- FE settings: `pipeline_dop=1`, `cbo_cte_reuse_rate=0`. Same FE package and data in both arms. Optimized transfer window 2. Asynchronous sender dispatch was off in the controlled sweep.
- Alternate baseline/optimized order per query, start a fresh cluster for each query/arm, measure one cold then two warm attempts, restart after every failure and preserve skipped slots. Cold means fresh application processes; OS page cache is uncontrolled and NIXL startup warmup precedes timing.
- Timings include SQL client startup, execution, result fetching and transfer. They exclude cluster startup, EXPLAIN, oracle comparison, restarts and post-run profiling. Warm aggregate is the sum of per-query medians of two samples, not one continuous full-suite wall time. Two warm samples do not establish statistical significance or tail latency.
- One client query runs at a time. The sweep measures query completion and service time with two CNs, not concurrent-query throughput under load.
- Comparison preserves duplicate rows, using relative numeric tolerance `1e-6` and absolute tolerance `1e-8`. Integers, text and NULLs match exactly. ORDER BY sequence is not verified. Some floating results differ within tolerance; these are not all exact matches.
- Benchmark harness holds a GPU UUID lock, rejects foreign GPU workloads, records actual CN executable/engine mappings and hashes, and only stops owned processes identified by PID plus start time.

## Follow-up and artifacts

A separate Q21 experiment with `SIRIUS_CN_ASYNC_SENDER_DISPATCH=1` passed all three attempts in **292.920 / 305.074 / 266.183 seconds**, with two-CN execution and zero numeric error in its 100 result rows. See [its profile](results/multi-cn-throughput-q21-async-dispatch/PROFILE.md). It uses the same frozen binaries and budgets and does not replace the original failed attempt. Q09 also passed three verified two-CN attempts without retries under that setting: **142.703 / 145.256 / 215.895 seconds**, with 175 exact result rows each; see [its profile](results/multi-cn-throughput-q09-async-dispatch/PROFILE.md). Every query has now passed on two CNs across the controlled sweep and these two follow-ups, but these different settings cannot be combined into a uniform all-22 suite time. The separate [Q05 credit profile](results/multi-cn-throughput-credit-profile/PROFILE.md) reconciles all 1,440 grant/return pairs from a failed two-CN FE execution, with zero unmatched tokens and zero final live credits. Its later successful FE retry ran on one CN; that diagnostic is excluded from all two-CN performance measurements. The successful two-CN Q05 smoke provides the separate completion evidence in the implementation table. All 44 primary and three later diagnostic cluster directories have parsed telemetry summaries. The [full comparison](results/multi-cn-throughput-ab/analysis/RESULTS.md) links all 22 per-query profiles; [analysis.json](results/multi-cn-throughput-ab/analysis/analysis.json) and [samples.csv](results/multi-cn-throughput-ab/analysis/samples.csv) retain the validation and sample ledgers. Early manifests omitted the explicit log-filter field in 9 baseline and 10 optimized blocks; the report keeps that provenance gap visible instead of inferring a value.

Reproduction commands and build dependencies are in [MULTI_CN_BENCHMARK.md](scripts/local-gb10/MULTI_CN_BENCHMARK.md). The [controlled A/B artifacts](results/multi-cn-throughput-ab/), [Q05 detailed profile](results/multi-cn-throughput-smoke/Q05-PROFILE.md), and [Q21 failure profile](results/multi-cn-throughput-ab/optimized/q21/FAILURE-PROFILE.md) retain plans, per-run outputs, logs, binary identity and correctness evidence. Detailed profiles and raw runtime artifacts remain local and are excluded from Git; the root summary and reproduction guide are versioned.

Validation completed: engine and native CN builds; 454 CPU CN/translator tests; a later five-test PRPC subset including two new retry tests; 25 C++ memory/staging cases with 20,330 assertions; and four targeted GPU regressions covering independent packing, query isolation, null/large-string ingress and context teardown. The 10 focused harness tests and 14 retry-placement tests also pass. Changed-file lint/format and both Rust workspace formatting checks pass. Relevant logs are under `build/multi-cn-throughput/`, `build/multi-cn-noengine-tests.log`, `build/multi-cn-prpc-tests.log`, and `build/multi-cn-gpu-regressions.log`. The original GPU-test default-pool setup failure is preserved separately; the successful regressions use explicit 4 GiB GPU / 1 GiB host test pools.

**00 · Trustworthy measurements and benchmark coverage**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`, reviewed 2026-09-05. Scope: StarRocks CN performance measurements and correctness gates. This path enables trustworthy decisions; it is not itself a claimed engine speedup.

**Problem and evidence**

The Q1/Q6 harness measures client-to-FE wall time and checks one selected result per query. In the source review, temporary fixtures demonstrated four false positives: a later failed run hidden by an earlier match, an earlier wrong answer hidden by a later match, NaN matching a finite number, and unequal integers above 2^53 matching at zero tolerance. The agent smoke test confirms transfer completion but does not inspect payload values. Q1/Q6 do not establish join, large-shuffle, or broadcast performance.

**Code map**

| Source | Responsibility |
|---|---|
| [bench.sh](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/benchmarks/tpch/bench.sh) | Run execution, outcomes, restart handling, end-to-end timing. |
| [compare.py](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/tools/compare.py) and [oracle.py](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/tools/oracle.py) | Typed result validation and oracle generation. |
| [cn-distribution.py](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/scripts/cn-distribution.py) | Per-CN telemetry analysis and explicit idle CNs. |
| [fragment_executor.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/fragment_executor.rs) | Distinguish producer/query labels from receiver-addressed `SenderSlot`. |
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs) and [engine.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/engine.rs) | Transport, queue, pack, and ingestion timestamps. |
| [exchange_staging_arena.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/exec/exchange_staging_arena.cpp) | Actual live/peak bytes and fragmentation counters. |

**Proposed design**

Define one machine-readable run manifest with run ID, query ID, numeric repetition index, phase, expected output, process status, timeout, source commit, binary hash, plan hash, and configuration hash. A successful sweep requires every expected performance sample to execute successfully and match its oracle. Explicit failure-injection samples belong to a separate run class with their own expected outcomes.

Store output types in an oracle sidecar instead of guessing from TSV spelling. Compare integer/count values exactly, decimal values using decimal arithmetic and documented tolerances, and floating values using explicit absolute/relative policies with nonfinite checks. Distinguish NULL from text and compare unordered results as multisets, preserving duplicate counts. Ordered queries retain row order; equal sort-key ties must not require an order SQL does not guarantee.

Emit correlated frame events with query, producer, receiver, exchange, sender, peer, sequence, and lease token. Use local monotonic intervals and CUDA events. Record queue wait separately from device execution and network/control wait. Do not infer one-way latency by subtracting clocks on different machines.

**Implementation slices**

1. **Correctness gate:** refactor the comparator into importable functions and a CLI, add type sidecars, numeric run sorting, manifest-based enumeration, and failure propagation from the sweep. Add the four demonstrated fixtures plus missing output, stale output, NULL/string, duplicate-row, and ordered-tie cases.
2. **Reproducible runs:** assign unique directories per sweep, record actual FE plan and per-CN configuration, and revalidate topology after every restart. Record cold setup explicitly; keep timed warm samples out of a cluster contaminated by a failed run.
3. **Timing instrumentation:** add enqueue/start/finish events around engine requests, pack and receive-copy CUDA intervals, control RPC timings, first/last output, EOS, first consumption, and live memory counters. Keep event schema stable and sample expensive detail.
4. **Benchmark layers:** add payload-validating raw transfer, production pack/RPC/ingest, multi-edge slow-peer, and end-to-end SQL workloads. Add supported hash-join, broadcast, skew, string/null, and ordered-query shapes; verify the FE produced the intended operators.

**Experiments and acceptance**

Use isolated baselines and alternating A/B run blocks on identical topology and data. Choose a sample count from observed variability; ten warm repetitions can be a pilot, not evidence for a reliable p99. Publish all outcomes, medians and distributions, memory high-water marks, copied bytes, and failures. Measure tracing overhead with instrumentation enabled/disabled and lower sampling if it changes the conclusion.

The comparator must reject all demonstrated false positives. Every manifest row must have a validated outcome. Raw transfer must validate a deterministic nonuniform payload. A synthetic delayed engine request must appear as queue time, not NIXL time. Missing or idle CN telemetry must remain visible.

**Rollout and dependencies**

This path has no implementation dependency. Preserve the old CSV export for consumers, but make the strict manifest gate authoritative; do not offer a silent fallback to the old comparator. All other paths use this protocol. Linux/GPU execution is still required for engine measurements; this document does not report new measurements.

**Decisions before implementation**

Choose oracle type-sidecar format, stable event schema, and the smallest supported workload set that covers every proposed optimization. Verify the pinned telemetry APIs rather than introducing a second unrelated trace system.

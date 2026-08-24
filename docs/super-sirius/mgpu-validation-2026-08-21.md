# Multi-GPU Validation — 2026-08-21

## Result

**PASS.** After correcting a test-only CUDA stream/device mismatch, the focused regression, all
five required selectors, both ten-process race loops, and the optional TPC-H validation passed on
four NVIDIA GB200 GPUs.

This was an independent second run on branch `kk/batch-preview-mgpu` at base commit
`b625abdb55c185543b64888492ae68230b1ad79d`. The correction changes no production behavior.

## Correction validated

The original `[mgpu]` failure came from the branch-added retry in
`prepare_for_processing rebinds idle batches to the prepared clones`. The test created one stream
on GPU 0, cloned its batch to GPU 1, and then reused the GPU 0 stream when preparing the retained
GPU 1 clone. The same-space fast path correctly rejected that foreign-device stream during
`rebind_stream`.

The corrected test now:

- constructs the source batch with GPU 0 current and a GPU 0-owned stream;
- performs both GPU 1 prepares with GPU 1 current and a GPU 1-owned stream; and
- declares guards, streams, and batches in an order that preserves stream lifetime through batch
  destruction.

This matches production, where each GPU executor pairs its reservation memory space with a stream
from the same device-bound stream pool. API comments and the pipeline, memory, multi-GPU, and
dynamic-filter documentation now state that contract explicitly.

## Hardware

- 4 x NVIDIA GB200, 189471 MiB each
- NVIDIA driver 580.105.08
- CUDA 13.0
- MIG disabled
- No other GPU processes at validation start

## Build and focused regression

The rebuilt `sirius_unittest` linked successfully. The final scoped pre-commit run passed every
configured hook, including clang-format, codespell, Markdown/link validation, and orphaned-test
checks.

Focused command:

```text
pixi run build/release/extension/sirius/test/cpp/sirius_unittest \
  "prepare_for_processing rebinds idle batches to the prepared clones"
```

Result: `All tests passed (14 assertions in 1 test case)`; exit 0; 6.72 seconds.

## Required invocations

The five processes ran sequentially with exclusive GPU ownership.

| Order | Filter | Result | Catch2 final counts | Wall time |
|---:|---|---|---|---:|
| 1 | `[mgpu]` | PASS | `All tests passed (75381 assertions in 34 test cases)` | 55.14 s |
| 2 | `[dynamic_filter][publication_claim]` | PASS | `All tests passed (6456 assertions in 17 test cases)` | 7.19 s |
| 3 | `[dynamic_filter][publisher]~[integration]` | PASS | `All tests passed (876 assertions in 64 test cases)` | 9.73 s |
| 4 | `[dynamic_filter]~[integration]` | PASS | `All tests passed (8967 assertions in 277 test cases)` | 22.16 s |
| 5 | `[integration][dynamic_filter]` | PASS | `All tests passed (920 assertions in 8 test cases)` | 7.90 s |

Notable activated coverage:

- The corrected clone/retry case passed both focused and within `[mgpu]`.
- Broadcast BUILD_PROBE publication passed its `[broadcast]`, publication-marker, and
  no-`NOT published` assertions.
- The previously unexecuted non-plan-GPU eager-claim expectations passed.
- Cross-device strict replication/reservation, direct Bloom merge, and PARTITION
  freeze/clone/retry passed.
- The three-GPU direct-peer fan-out case ran rather than returning through its hardware gate.
- The forced-host-staging test selected `host_staging` and verified the copied bytes exactly.

`[mgpu]` emitted one data-availability warning: the optional
`/datasets/tpch_parquet_sf1/lineitem.parquet` fixture was absent, so its stress test ran four of
five query shapes. All 34 registered cases completed.

## Race hunting

| Filter | Result | Uniform Catch2 final counts | Per-process wall times |
|---|---|---|---|
| `[mgpu][dynamic_filter]` | 0 failures / 10 | `547 assertions in 10 test cases` | 8.75, 8.37, 8.42, 8.48, 8.36, 8.39, 8.54, 8.50, 8.36, 8.51 s |
| `[dynamic_filter][publisher][concurrency]` | 0 failures / 10 | `172 assertions in 12 test cases` | 6.68, 6.29, 6.16, 6.32, 6.16, 6.23, 6.20, 6.11, 6.23, 6.67 s |

Every one of the twenty independent processes exited 0. No iteration was retried.

## Optional TPC-H validation

The SF1 vendored DuckDB database was tested with Q5, Q7, and Q21 under
`test/cpp/integration/integration-2gpu.yaml`. Each Sirius GPU result was compared byte-for-byte
with the DuckDB CPU result while debug logging was enabled.

| Query | Validation | DuckDB CPU | Sirius GPU | Publication lines | `NOT published` lines |
|---|---|---:|---:|---:|---:|
| Q5 | byte-exact PASS | 0.023153 s | 0.088026 s | 4 | 0 |
| Q7 | byte-exact PASS | 0.015272 s | 0.028308 s | 5 | 0 |
| Q21 | byte-exact PASS | 0.028598 s | 0.036729 s | 5 | 0 |

Overall validation: **3/3 queries passed**. These single-iteration SF1 timings are retained as run
metadata, not as a performance evaluation.

## Scope and remaining coverage

The task-scoped patch modifies one test, two API-comment blocks, and Super Sirius documentation.
No production `.cpp` or `.cu` behavior changed.

Direct peer DMA and the explicitly forced host-staging route were exercised. Because peer DMA
works on GB200, the automatic probe-selected fallback on peer-DMA-broken hardware was not
exercised. Performance evaluation also remains outside this correctness run.

The complete raw outputs, per-process timings, logs, result files, checksums, and original plus
rerun reports remain in the validation workspace under `mgpu_test_evidence/`. They are not part of
this repository report because two individual debug logs exceed 17 MB.

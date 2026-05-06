---
plan: 20-06
phase: 20-scan-manager-pin-tables-port-pr-731-pr-721
type: verdict
status: in-progress  # flips to PASS or PARTIAL after Task 4 sanitizer + tests
requirements: [SM-06, IO-15B, IO-MGPU-02]
created: 2026-05-06
---

# Phase 20.6 Verdict — parquet_split_provider Sirius-side bypass closure

## Summary

Eliminate the kvikio bypass at `parquet_split_provider::run_batch:222` revealed by
20-05 sanitizer trace re-inspection. Plumbing landed in Task 1 (commit `a1a8c68`);
fix landed in Task 2 (commit `fbded29`); strengthened IO-15B grep gate documented in
Task 3 (this file).

The 20-05 sanitizer trace at `/tmp/p20_sanitizer.out` lines 117-138 showed the kvikio
`FileHandle` ctor invoked from a `cudf` file_source factory called by
`parquet_split_provider::run_batch`. The shared_ptr flowed through `parquet_scan_data`
into `sirius_gpu_parquet_scan_operator::read_table_from_metadata`, where it was
consumed by `cudf::io::read_parquet` — making every column-chunk read go through
cudf+kvikio instead of the Phase 19 sirius_datasource (io_uring + per-GPU CUDA-context
binding).

The OLD path (`parquet_scan_task.cpp`) was correctly migrated in Phase 19. The NEW
scan_manager path (PR #731 → Phase 20) was authored without IO framework integration
— we caught the gap via 20-05 sanitizer log forensics.

## Strengthened IO-15B grep gate

The original IO-15 grep (Phase 19) only checked `cucascade_datasource` — it would not
have caught the parquet_split_provider bypass because the new path uses the cudf
factory, not the cucascade adapter. Phase 20.6 strengthens the gate:

```bash
# Strict IO-15: zero cucascade_datasource references
grep -rn "cucascade_datasource" src/ test/ | wc -l
# expected: 0

# Strengthened IO-15B (Phase 20.6): zero cudf bundled file_source factory
# calls outside known-deferred sites
grep -rn "cudf::io::datasource::create" src/ \
  | grep -v "src/op/scan/iceberg_metadata_reader.cpp" \
  | grep -v "src/op/scan/iceberg_scan_task.cpp"
# expected: 0
```

Both gates are in REQUIREMENTS.md; the strengthened gate would have caught Phase 19's
parquet_split_provider gap had it been live.

## Known-deferred sites (IO-MGPU-02)

Two sites still use the cudf bundled file_source factory; they are tracked as
`IO-MGPU-02` for v1.5+ scope:

1. **`src/op/scan/iceberg_metadata_reader.cpp:227`** — iceberg manifest /
   manifest-list reads. Single-GPU at present (planning-time, before any per-task
   scheduling); kvikio's CUDA-context binding poses no correctness risk because
   these reads precede multi-GPU column-chunk dispatch.
2. **`src/op/scan/iceberg_scan_task.cpp:159`** (commentary only) — explanatory note
   that iceberg equality-delete file reads use the cudf factory directly. The actual
   call site is in `read_equality_delete_file`, where multi-GPU residency would
   require similar plumbing to Phase 20.6's parquet_split_provider work.

Per `IO-MGPU-02` (added to REQUIREMENTS.md Future Requirements section in Task 3),
both sites need: (1) plumbing `gpu_ioctxs` into the iceberg metadata path, (2)
constructing per-call `uring_io_object` instances, (3) routing through the
appropriate ioctx by the consumer's preferred device.

## Verification status

| Gate | Command | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| MCP build | `mcp build` | exit 0 | exit 0 (~2-7s incremental) | PASS |
| IO-15 strict | `grep -rn "cucascade_datasource" src/ test/` | 0 | 0 | PASS |
| IO-15B strengthened | `grep -rn "cudf::io::datasource::create" src/ \| grep -v <iceberg sites>` | 0 | 0 | PASS |
| make_datasource present | `grep -n "make_datasource" src/scan_manager/parquet_split_provider.cpp` | ≥ 1 | 2 | PASS |
| uring_io_object present | `grep -n "uring_io_object" src/scan_manager/parquet_split_provider.cpp` | ≥ 1 | 3 | PASS |
| HYG-02 baseline | `grep -rn "rmm::cuda_stream_default" src/ \| wc -l` | ≤ 40 | 40 | PASS |
| Task 4: sanitizer Cluster A | `grep "kvikio" /tmp/p20_06_sanitizer/sanitizer.out` in parquet_split_provider path | 0 frames | TBD | TBD |
| Task 4: Q11 SF1 num_gpus=2 | direct binary | exit 0 | TBD | TBD |
| Task 4: [integration][TPC-H] | mcp unit-tests | 48/48 PASS | TBD | TBD |
| Task 4: [mgpu] continuity | mcp unit-tests | 16/16 PASS | TBD | TBD |

## Phase 20 verdict implication

- If Task 4 sanitizer shows Cluster A (kvikio frames in `parquet_split_provider::run_batch`
  call path) eliminated AND Q11 SF1 num_gpus=2 + [integration][TPC-H] 48/48 + [mgpu] 16/16
  all pass: Phase 20 verdict flips PARTIAL → COMPLETE PASS; SM-06 SF1 closed.
- If Cluster B (16 cucascade host-staging races) persists post-fix but tests pass: Phase
  20 verdict stays PARTIAL with smaller residual; cucascade follow-up scoped narrowly to
  `alloc_and_peer_copy_async` host-staging fallback.
- If Cluster A persists: re-investigation needed; this fix was insufficient.

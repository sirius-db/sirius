# PR #732: Multi-GPU Execution + Upstream Sync — Change Summary

This document explains the changes in PR #732 (`feature/single-node-multi-gpu2` → `dev`) for reviewers and future readers. The PR delivers transparent multi-GPU SQL execution on a single node plus two rounds of upstream sync (cucascade `origin/main` + sirius `origin/dev`).

**Headline numbers (vs `dev`):**
- **114 source files** changed (8 binary docs/images excluded) — `7,309` insertions, `628` deletions
- **438 commits** ahead of `origin/dev` — organized across 5 milestones (v1.1 → v1.5+)
- **Cucascade submodule** bumped from `96bfea1` → `5203de5` (9 commits ahead of upstream `origin/main` HEAD `9ceebaa`); fork branch `fix/pinned-portable-flags` carries the divergence
- **Test suite**: 1,205 unit tests / 41,421,987 assertions PASS on the full `~[.]` filter

## What This PR Delivers

Before this PR, Sirius was a single-GPU GPU-accelerated SQL engine. After this PR:

1. **Transparent multi-GPU execution** — any query can use every GPU on the node. The Sirius optimizer extension routes plain SQL through the GPU engine without user intervention; tasks are scheduled to the GPU where their input data already resides; memory pressure is absorbed by NUMA-aware downgrade to host or disk.
2. **NUMA-aware memory management** — per-GPU `cucascade::memory_space`s with host-tier capacity sized per-NUMA-domain; new HOST-tier pinning support.
3. **Multi-GPU-safe parquet I/O** — kvikio fully removed from the scan path. All file reads route through `sirius_ioctx::make_datasource` (built on cucascade's `idisk_io_backend` + per-GPU `uring_ioctx`), eliminating the kvikio per-FileHandle CUDA-context binding that silently broke multi-GPU residency.
4. **GPU↔Host↔Disk tiered memory** with explicit downgrade machinery and per-thread reservation tracking.
5. **Round-robin pinned-table distribution** across GPU memory spaces (PIN-MGPU-01) with the new upstream HOST-tier pinning path coexisting alongside.
6. **Two upstream sync rounds** absorbing 14 cucascade upstream commits and 14 sirius upstream commits, with every conflict resolved under a documented behavioral-correctness rule.

## Milestone-by-Milestone Breakdown

The PR aggregates five sequential milestones, each containing 4–7 phases. A reviewer can pull a single milestone's commits via `git log <previous-milestone-tip>..<milestone-tip>` for narrower review.

### v1.1 — Multi-GPU Re-integration + Cucascade I/O Migration (Phases 4–7, 2026-04-21)

Built the first multi-GPU foundation on top of cucascade's tiered memory and `idisk_io_backend`. Replaced legacy `GPUBufferManager` with `cucascade::memory_space`s; introduced per-GPU `uring_ioctx` for parquet metadata + chunk reads; established the `sirius_ioctx::make_datasource` abstraction.

Key files: `src/sirius_context.{hpp,cpp}` (per-GPU initialization), `src/io/datasource_factory.{hpp,cpp}` (registry-based datasource resolution), `src/io/uring/uring_reactor.cpp` (driver-level setup wrapped in RAII), `src/include/scan_manager/sirius_scan_manager.hpp` (multi-GPU pin-table accounting).

### v1.2 — Multi-GPU SQL Pipeline Fix (Phases 8–10, 2026-04-28)

Wired the multi-GPU plumbing through the pipeline executor and task scheduler. Phase 11 (a v1.2 patch) closed an AUDIT TEST_CASE attach-path SIGSEGV with a 33-LOC fix split across `cucascade::io_worker` (member-init-order race) and `task_creator` (per-query state lifetime).

Key files: `src/pipeline/gpu_pipeline_executor.cpp`, `src/pipeline/task_scheduler.cpp`, `src/creator/task_creator.{hpp,cpp}`.

### v1.3 — Multi-GPU Distribution (Phases 12–15, 2026-05-01)

Delivered real multi-GPU work distribution for source-pipeline tasks (parquet metadata + GPU parquet scan + downstream operators). The parquet AST filter re-translation fix (`86e821a`) unblocked distribution.

- Phase 12: `vector::at(2)` correctness bug in the small-sort plan path
- Phase 13: Q11 multi-GPU hang/illegal-address — cucascade writer-event stream lineage fix
- Phase 14: SCHED-RR round-robin distribution landed
- Phase 15: Cross-GPU operator-colocation audit

Key files: `src/op/sirius_physical_*.cpp` (operator-side colocation), `src/include/op/scan/duckdb_scan_executor.hpp` (NUMA-aware reservation requests).

### v1.4 — Rebase After DataBatch Changes (Phases 16–21, 2026-05-06)

Largest milestone of the PR. Absorbed cucascade `origin/main` PRs #117 (DataBatch RAII refactor) + #112 + #116 and sirius `origin/dev` PRs #739 (cucascade-compat), #675 (IO Framework), #731 (Scan Manager), #721 (Pin Tables), #733/#734/#735, while preserving every multi-GPU behavior shipped in v1.1+v1.2+v1.3.

- Phase 16: Cucascade rebase + pin recovery (4 group commits on `73d00c4`)
- Phase 17: Sirius `origin/dev` base-layer merge — produced expected DataBatch API breaks
- Phase 18: DataBatch RAII migration (all `batch->get_data()` call sites migrated to RAII accessors; Path A architectural fix — drop R5 lock-and-hold from `gpu_pipeline_task::compute_task`, perform eager memory-space conversion under SHORT-scoped accessors)
- Phase 19: IO Framework adoption (retired `sirius::io::cucascade_datasource`; adopted `sirius::io::sirius_datasource` with per-GPU `uring_ioctx`)
- Phase 20: Scan Manager + Pin Tables port — uncovered + fixed the `parquet_split_provider::run_batch` kvikio-bypass site (SM-06)
- Phase 21: v1.4 ship gate — full REG-01..06 gauntlet PASS

Key files: every operator under `src/op/sirius_physical_*.cpp`, `src/data/host_parquet_representation*.cpp`, `src/include/data/data_batch_utils.hpp`, `src/include/scan_manager/cached_split_provider.hpp`, `src/include/scan_manager/parquet_split_provider.hpp`.

### v1.5+ — Multi-GPU Pinning + Stream Lineage + Upstream Sync (Phases 22–24, 2026-05-08 → 2026-05-13)

Polish + two upstream sync rounds.

- **Phase 22** (PIN-MGPU-01 + Cluster B): Round-robin pinned-table distribution across GPU memory spaces (`PinTableFunction` distributes chunks via `idx % gpu_spaces.size()`); `pinned_entry::chunk_memory_spaces` vector parallel to `data_batches_by_column`; cucascade stream-lineage hardening — collapse producer + DtoH + HtoD onto a single `target_stream` in `alloc_and_peer_copy_async` (closes the Cluster B race seen at SF100 Q11).
- **Phase 22.1** (Remove kvikio): All 7 D-01 bypass sites migrated to `sirius_ioctx::make_datasource(uring_io_object)`. K.1 Cluster A closed (6→0 race blocks). New gates: GATE-22.1-A bypass-grep returns 0; GATE-22.1-B sanitizer Cluster A = 0; GATE-22.1-C SF1 Q11 num_gpus=2 functional.
- **Phase 22.2** (K.6): `cudaSetDevice(-1)` fix in `src/downgrade/downgrade_executor.cpp` — gate `_per_thread_init` on `_space_id.tier == GPU` so HOST-tier downgrade executors don't call `cudaSetDevice` with sentinel `-1`.
- **Phase 22.3** (K.7 NO-REPRO + CTE `_types`): K.7 reclassified — the original SQL fixture used a constant `0.0001` instead of spec-compliant `0.0001/SF`. Shipped a cosmetic CTE planner `_types` cleanup + a new SF10 Q11 mgpu regression test.
- **Phase 23** (First upstream sync round): Rebased cucascade fork onto `bcddb89` (PR #121 portable host memory supersedes our portable-pinning hunks; surgical-split kept 3 ours-only files). Merged sirius `origin/dev` (12 upstream commits) including value AST Phase 2 (`7eeaab4`), task-creation race fix (`7cc7a79`), converter rename (`972cb32`), per-op memory estimate (`e94ad4a`), bytes-to-materialize fix (`5d09a59`). Gap-closure plans `23-06`/`23-07` added cucascade fixes `37df815` (RAII `dst_guard` around HtoD memcpy in `alloc_and_peer_copy_async`) and `9da4047` (`run_p2p_probe_locked` device-context restore) — both needed because the rebased commit `8392c3d` introduced a new `convert_gpu_to_gpu` codepath whose outer `target_guard` does not propagate into the inner host-staging branch.
- **Phase 24** (Second upstream sync round): Rebased onto `9ceebaa` (PR #122 slice host table + PR #124 reconstruct_column STRING fix). Merged sirius `origin/dev` (2 upstream commits) including `ba5ed27` (wire_data_repositories Phase 2 refactor) and `2e197c6` (pin_table tier='host' for host-tier caching). All conflicts resolved with the upstream-as-source-of-truth rule: upstream wins by default; ours preserved only for unique behavior (PIN-MGPU-01, sirius_ioctx kvikio-bypass, Phase 22.3 CTE _types) or re-derived bug fixes against upstream's new code shape.

### Post-sync hardening

A test-suite pollution flake surfaced after Phase 24 — `record_on_failure deduplicates OOM records and keeps max peak` (in `test_gpu_pipeline_task_history.cpp`) leaves pending `cudaFreeAsync` operations un-synchronized at TEST_CASE teardown, which crashed the next `[gpu_pipeline_disk]` test inside `cuMemAllocFromPoolAsync`. The fix lands in `src/memory/sirius_memory_reservation_manager.cpp:~/sirius_memory_reservation_manager()` — a `cudaDeviceSynchronize()` per managed GPU before restoring cudf's device resource, ensuring the cuda_async pool is drained before destruction.

## Changes by Area

### `src/sirius_context.{hpp,cpp}` — Lifecycle + Per-GPU Initialization

- Added `gpu_ioctxs_` field + accessors (Phase 19): per-GPU `uring_ioctx` constructed at `SiriusContext::initialize()`.
- Added `datasource_registry_` member (Phase 22.1): registry-based scheme→datasource resolution, replacing the file-path → cudf-bundled-datasource fallback that silently used kvikio.
- Connection-open/close registers/clears `sirius_state` in DuckDB's `ClientContext::registered_state`.
- New `drain_after_error` machinery preserved through merges to avoid UAF on task_creator queues.

### `src/io/datasource_factory.{hpp,cpp}` — kvikio-free factory

- Strict `create()` (Phase 22.1): registry resolves all schemes or throws — no silent kvikio fallback.
- Bare relative paths normalized to `file:///<absolute>`.
- Test order: `uring_reactor` include must be last (Phase 19 IO-15 ordering invariant).

### `src/op/` and `src/include/op/` — Operators

All GPU physical operators migrated to:
- Path A: SHORT-scoped batch accessors (Phase 18) — no R5 lock-and-hold; `pipelineable_operator_data::prepare_for_processing` performs eager memory-space conversion under brief accessors and returns empty `vector<>`.
- Multi-GPU memory-space awareness via `_min_num_partitions` and `no_history_peak_memory_estimate` on `sirius_physical_partition.hpp` (Phase 23 merge resolution preserves both fields).
- Phase 22.3 cosmetic CTE planner `_types` cleanup in `sirius_plan_cte.cpp`.

### `src/op/scan/` — Scan Path

- `sirius_gpu_parquet_scan_operator`, `parquet_scan_task`, `iceberg_scan_task`, `iceberg_metadata_reader`, `duckdb_scan_task` and `duckdb_scan_executor` all route file reads through per-GPU `sirius_ioctx::make_datasource(io_object)` — no `cudf::io::datasource::create(path)` or `source_info{path}` anywhere in `src/` (GATE-22.1-A invariant).
- NUMA-preference reservation requests via `any_memory_space_in_tier_with_preference` (Phase 23 merge integrates upstream's new `reservation_info` struct with our NUMA logic).

### `src/scan_manager/` — Pin Tables

- `PinTableFunction` distributes parquet chunks round-robin across GPU memory spaces (PIN-MGPU-01).
- `pinned_entry::chunk_memory_spaces` vector parallel to `data_batches_by_column`.
- `cached_split_provider` per-chunk memory_space lookup; host-mode constructor (Phase 24, `2e197c6`) for the new HOST-tier pinning path.
- `parquet_split_provider` rebuild to route through `sirius_datasource` (Phase 20 SM-06 fix); fallback removed (Phase 22.1).

### `src/pipeline/` — Task Scheduling + Pipeline Build

- `task_scheduler.{hpp,cpp}` and `sirius_pipeline_converter.cpp` integrated the upstream `wire_data_repositories` Phase 2 split (Phase 24, `ba5ed27`) which moves plan-time wiring into pure-data `repository_wiring` descriptors emitted by the converter; runtime materialization moves into `materialize_repository_wiring(wirings, manager)` called from `sirius_engine::initialize_internal()`.
- `gpu_pipeline_task` reservation tracking + per-thread state with absl::Cleanup-guarded `reset_stream_reservation`.

### `src/downgrade/downgrade_executor.cpp`

- Phase 22.2 K.6 fix: gate stream pool creation + `per_thread_init` on `_space_id.tier == cucascade::memory::Tier::GPU` so HOST-tier downgrade workers don't call `cudaSetDevice(-1)`.

### `src/memory/sirius_memory_reservation_manager.cpp`

- Constructor: install per-GPU `cucascade::memory_space` as the current cudf device resource; record the previous ref for restoration.
- Destructor: **drain each GPU via `cudaDeviceSynchronize()` before restoring** — required to prevent the `cudaMemPoolDestroy` corruption that the test-suite flake exposed. See the post-sync hardening section above.

### `src/sirius_extension.cpp` + `src/sirius_engine.cpp`

- Engine init: `set_min_num_partitions` invariant + new `no_history_peak_memory_estimate` field both preserved across the Phase 23 `origin/dev` merge (D-16 resolution).
- `drain_after_error` invocation moved to the post-converter site after Phase 24's `ba5ed27` refactor.

### `test/cpp/` — 41 test files changed

- New unit tests: `[mgpu]`, `[mgpu_stress]`, `[multi_gpu_foundation]`, `[pin_mgpu]`, `[pin_table]`, `[pin_table_host]`, `[downgrade_lifecycle]`, `[downgrade_disk]`, `[gpu_pipeline_disk]`, `[gpu_pipeline_task]`, `[datasource_factory]`, `[tpch_sf10]`, `[mgpu-audit]`.
- `integration-2gpu.yaml` for parameterizable num_gpus=2 tests.
- `compare_gpu_vs_cpu` fixture rewrite for transparent execution stats validation.
- `mgpu_test_utils.hpp` shared helpers.
- 1 SF10 Q11 mgpu regression test (Phase 22.3) using spec-compliant `0.0001/SF` fraction.

### `test/scripts/sanitizer_gate_22.sh`

- Phase 22 D-12 / Phase 22.1 GATE-22.1-B sanitizer gate.
- Phase 23-07 hardening: windowed-awk counter to distinguish race-section headers from API-error backtraces (closes a cluster_B false-positive); `P22_SELFTEST=1` mode with synthetic positive+negative test.

### Cucascade submodule (`cucascade` gitlink)

The fork branch `fix/pinned-portable-flags` carries 9 commits ahead of upstream `origin/main` HEAD `9ceebaa`. The submodule pointer moved from the v1.0 baseline to `5203de5` across the PR. Each fork commit is either:
- A bug fix needed on top of upstream (`37df815` dst_guard, `9da4047` probe-device-restore, `1e889d7` same-stream invariant) — candidates for upstream PR
- A behavioral preservation of a fork-only code path (`8392c3d` P2P override + DMA probe, `085d917` stream-lineage writer event API)
- Build/style hygiene (`0c0a4af` io_worker reorder, `89d6a3f` clang-format)

CC-UPSTREAM-01: the fork stays local-only in-repo. Upstream PR submission is handled separately by the project maintainer.

## Test Strategy

Verification ran on each ship gate. The 18-gate gauntlet (the most recent baseline, used by Phase 24's v1.5+ ship gate):

| Gate | Suite filter | Threshold | Result |
|------|-------------|-----------|--------|
| REG-01 | `[mgpu]` | 16/16 cases, ≥79,091 assertions | PASS |
| REG-02 | `[TPC-H][parquet]` | 22/22, ≥36,256 assertions | PASS |
| REG-03 | `[integration][TPC-H]` | 49/49 (post-merge), ≥71,623 assertions | PASS |
| REG-04 | SF100 Q1 num_gpus=2 wall-clock | ≤5.7s | PASS (3.05s) |
| REG-05 | `[mgpu_stress]` 500-iter | exit 0, ≥77,053 assertions | PASS |
| REG-06 Leg 1 | `[multi_gpu_foundation]` functional + memcheck | 7/7 + 38 assertions, 0 violations | PASS |
| REG-06 Leg 2 | `[integration][gpu_execution][parquet][join]` memcheck | 42/42, ≥1.92M assertions, 0 violations | PASS |
| GATE-22.1-A | bypass-grep | 0 hits in `src/` | PASS |
| GATE-22.1-B | sanitizer Cluster A | total = 0 | PASS |
| GATE-22.1-C | SF1 Q11 num_gpus=2 functional | 1/1, ≥9,011 assertions | PASS |
| K.6 NO-REPRO | SF100 Q11 num_gpus=2 | exit 0, 0 cudaSetDevice(-1) errors | PASS |
| K.7 NO-REPRO | `[tpch_sf10]` 4/4 | including tpch_q11_sf10_2gpu | PASS |
| Cluster B same-stream | `sanitizer_gate_22.sh` | total_races=0, cluster_B=0 | PASS |
| HYG-02 | `rmm::cuda_stream_default` count | ≤40 (all in `src/legacy/`) | PASS (40) |
| `[datasource_factory]` | 11/11 | Phase 22.1 strict policy | PASS |
| `[mgpu-audit]` | 6/6 in suite mode | side-benefit from upstream `7cc7a79` | PASS |
| `[pin_table_host]` (new) | 1/1, 51 assertions | upstream `2e197c6` test | PASS |
| Full `~[.]` suite | 1,205 / 1,205 cases | 41,421,987 assertions | PASS |

## Carry-Forward Items (Not in this PR)

- **CC-UPSTREAM-01**: 9 cucascade fork commits eligible for upstream PR submission (handled by the project maintainer, not this PR).
- **PIN-MGPU-02/03**: adaptive pin distribution + HOST-tier path integration with PIN-MGPU-01 round-robin — currently the two pinning paths coexist in parallel.
- **OOM-RETRY-01**: retry budget restoration.
- **IO-MGPU-04**: per-GPU iceberg metadata residency (renamed from IO-MGPU-02 half-closure in Phase 22.1).
- The hidden `[.][integration_disabled]` tests in `test_gpu_execution_tpch.cpp` are pre-existing known-broken edge cases unrelated to this PR; they remain hidden.

## Reviewer's Guide

A reviewer who wants a smaller scope can pull individual milestones:

```bash
# v1.1 only (Phases 4–7)
git log <v1.1-start>..<v1.1-tip>

# v1.4 ship-gate (Phase 21) — just the gauntlet result
git log -1 21-VERDICT.md
```

Highest-leverage code to read:
1. `src/memory/sirius_memory_reservation_manager.cpp` — the post-sync drain fix (12-line addition fixes a suite-pollution flake)
2. `src/sirius_context.{hpp,cpp}` — per-GPU initialization and the datasource registry
3. `src/io/datasource_factory.{hpp,cpp}` — strict kvikio-free factory with scheme registry
4. `src/scan_manager/` — round-robin pinned-table distribution
5. `src/op/sirius_physical_partition.hpp` — `set_min_num_partitions` + `no_history_peak_memory_estimate` (illustrative of the upstream-favored merge style applied in Phase 23/24)
6. `test/scripts/sanitizer_gate_22.sh` — the windowed-awk gate (a fix that protects all future invariant runs from a false-positive class)

Behavioral verification commands:

```bash
# Build (uses MCP project commands; equivalent to make -j)
make

# Full unit suite (excludes hidden intentionally-broken tests)
build/release/extension/sirius/test/cpp/sirius_unittest "~[.]"

# Just the v1.5+ ship-gate invariants
build/release/extension/sirius/test/cpp/sirius_unittest "[mgpu]"
build/release/extension/sirius/test/cpp/sirius_unittest "[TPC-H][parquet]"
build/release/extension/sirius/test/cpp/sirius_unittest "[mgpu_stress]"
build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation]"
build/release/extension/sirius/test/cpp/sirius_unittest "[pin_table_host]"
build/release/extension/sirius/test/cpp/sirius_unittest "[mgpu-audit]"

# Phase 22 sanitizer gate (Cluster B + windowed-awk + P22_SELFTEST)
./test/scripts/sanitizer_gate_22.sh
P22_SELFTEST=1 ./test/scripts/sanitizer_gate_22.sh
```

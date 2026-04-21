# Phase 6: Multi-GPU Gap Closure (Topology, Device Safety, Host Memory, GPU↔GPU Converter) - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the five structural v1.0 gaps that never cleared on `feature/multi-gpu-execution`, now that Phase 5 has shipped the cucascade-backed I/O substrate:

1. **MGPU-01** — Runtime topology discovery via `cucascade::topology_discovery`. `SiriusContext` caches the result and exposes it as the single source of truth for GPU count, per-GPU NUMA domain, and GPU→NUMA map.
2. **MGPU-02** — Single-GPU TPC-H SF10 wall-clock within 5% of current `dev` HEAD baseline (3-run median, same build flags). Proves the topology + per-NUMA allocator work doesn't introduce regression on 1-GPU hosts.
3. **MGPU-03** — Device-guard enforcement audit. `compute-sanitizer --tool memcheck --require-cuda-init` on the multi-GPU test suite reports zero "invalid device" / "context mismatch" errors. Phase 5 already proved cucascade_datasource pass; Phase 6 extends that proof to the full multi-GPU code paths (downgrade executor, converters, pipeline executor per-thread).
4. **MGPU-04** — GPU↔GPU representation converter registered in cucascade converter registry. Phase 6 registers a **host-staged** implementation (GPU0 → host pinned → GPU1) so the registry entry exists and the round-trip correctness test passes. Phase 7 will swap the converter body to direct `cudaMemcpyPeerAsync`.
5. **MGPU-05** — Per-NUMA host memory spaces configured with `numa_region_pinned_host_allocator`. `memory_manager_` owns one host space per NUMA domain; per-domain allocations land on the correct node verified via `/proc/PID/numa_maps`.

**In scope:** topology call + caching, accessor APIs, hand-rolled `cudaGetDeviceCount` / `numa_*` sweep in `src/`, GPU↔GPU converter slot + host-staged body + round-trip test, per-NUMA host memory space construction, single-GPU SF10 regression measurement, compute-sanitizer re-run.

**Out of scope:**
- P2P direct transfer (`cudaMemcpyPeerAsync`) — Phase 7 (MGPU-06).
- Adaptive scan partitioning — Phase 7 (MGPU-07).
- Fixing the Phase-4-deferred `test_downgrade_executor.cpp:813 TODO(MGPU-06)` cross-GPU converter return-leg — Phase 7.
- Test files (`test/`) containing legacy `cudaGetDeviceCount` / `numa_*` calls — sweep is `src/` only.
- Changes to cucascade submodule — Phase 6 consumes the existing `topology_discovery` API from the f47de0b pin.
- Phase-4 baseline regression comparison re-visit (IO-10 deferral from Phase 5 remains deferred).

</domain>

<decisions>
## Implementation Decisions

### Topology Discovery Integration (MGPU-01)
- Call `cucascade::topology_discovery` **inside `SiriusContext::initialize()` early** — before `memory_manager_` construction — because topology drives memory-space layout (per-NUMA host spaces need NUMA count; per-GPU memory spaces need GPU count).
- **Cache once per context lifetime** on `SiriusContext` (mirror the `gpu_io_backends_` cache pattern Plan 05-03 established at `src/sirius_context.cpp` lines 185-204).
- Expose via `SiriusContext::get_topology() const` returning a const ref to the cached struct.
- Log a summary at **`info` level** on startup: GPU count, per-GPU NUMA domain, GPU→NUMA map (explicit per success criterion 1).
- On topology-discovery failure, **fail-hard**: throw from `initialize()`. No fallback to `cudaGetDeviceCount` / `numa_node_of_cpu` (the whole point is to centralize on cucascade).

### Hand-Rolled CUDA/NUMA Sweep (MGPU-01 gate)
- Scope: **all of `src/`** — success criterion 1 requires `grep -rn 'cudaGetDeviceCount\|numa_node_of_cpu' src/` == zero hits outside the cucascade bridge.
- Scope exclusion: `test/` may retain `numa_*` / `cudaGetDeviceCount` as test fixtures; only `src/` is swept.
- Replacement pattern: every callsite routes through `SiriusContext::get_topology()` (or an equivalent accessor built on the cached struct).
- Cucascade-bridge exception: cucascade internally calls `cudaGetDeviceCount`; Sirius consumes its topology result, not the raw API. This exception is documented in the phase SUMMARY.

### GPU↔GPU Converter (MGPU-04)
- Register a **host-staged converter** (GPU0 → pinned host buffer → GPU1) in `converter_registry::initialize()` at `src/sirius_extension.cpp:1053`, alongside the existing tier converters.
- The registered converter satisfies the registration gate (`converter_registry::instance().has_converter(Tier::GPU, Tier::GPU) == true`) and passes the round-trip correctness test (GPU0 data → convert → GPU1 → memcpy back → bytes-equal to input).
- Phase 7 (MGPU-06) will **replace the body** with `cudaMemcpyPeerAsync` when P2P is available, with fallback to host staging when not. The registry slot stays the same; Phase 6 and Phase 7 do not interfere at the registration surface.
- **Leave Phase-4 deferred items in place**: `test_downgrade_executor.cpp:813 TODO(MGPU-06)` + `[.][multi_gpu_transfer]` + `[.][mem_04_p2p_transfer]` hidden tags stay off-by-default until Phase 7.

### Per-NUMA Host Memory Spaces (MGPU-05)
- **One host space per NUMA domain**, allocator = cucascade `numa_region_pinned_host_allocator`.
- `memory_manager_` constructor iterates the cached topology's NUMA list and builds one host space per node.
- Validation: `memory_manager_->get_memory_spaces_for_tier(Tier::HOST).size() == topology.numa_node_count()` + spot-check allocations land on the correct NUMA node via `/proc/PID/numa_maps`.

### Single-GPU Regression Gate (MGPU-02)
- Baseline: current `dev` HEAD (not Phase-5 HEAD — per wording of success criterion 2, "current `dev` HEAD baseline").
- Measurement: **`python3 test/tpch_performance/performance_test.py 10`**, captured in phase SUMMARY.
- Threshold: **5% wall-clock regression** on TPC-H SF10 end-to-end, 3-run median, same build flags as baseline.
- If regression > 5%: block phase; investigate the per-NUMA allocator or topology cache for accidental overhead.

### Device-Guard Audit (MGPU-03)
- Run `compute-sanitizer --tool memcheck --require-cuda-init build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation]"` on the N=2 host.
- Also run `[integration][gpu_execution][parquet][join]` subset (Phase 5 already proved 0 errors there — re-run on Phase-6 HEAD to confirm MGPU-01..05 didn't break device-guard invariants).
- Zero "invalid device" / "context mismatch" errors required.

</decisions>

<code_context>
## Existing Code Insights

### Phase 5 shipped (load-bearing for Phase 6)
- `SiriusContext::initialize()` already has `io_backend_registry_` + `gpu_io_backends_` cache populated under `rmm::cuda_set_device_raii` at `src/sirius_context.cpp:185-204` (Plan 05-03). Topology-discovery hook goes immediately before this block — ensures topology is cached before any memory-space or per-GPU backend construction.
- Teardown ordering: `gpu_io_backends_.clear()` at `~SiriusContext` runs before `memory_manager_->shutdown()` (line 308 → 330). MGPU-04 converter must follow the same ordering to avoid `cudaErrorInvalidResourceHandle`.
- `converter_registry::initialize()` at `src/sirius_extension.cpp:1053` is the register site — Phase 6 adds one more `register_converter(Tier::GPU, Tier::GPU, ...)` call here.

### Cucascade topology API
- Header: `cucascade/include/cucascade/topology/topology_discovery.hpp` (via pinned f47de0b).
- Expected surface: a free function like `cucascade::topology_discovery()` returning a struct with `int gpu_count`, `std::vector<int> gpu_numa_nodes`, `int numa_count`, etc. (exact shape to be confirmed by researcher).
- Internally wraps `cudaGetDeviceCount` + `numa_node_of_cpu` + `nvml*` (as available). Sirius doesn't touch those APIs directly.

### Hand-rolled callsites expected to be swept (researcher to confirm current line numbers)
- `src/sirius_context.cpp` — currently uses `cudaGetDeviceCount` and/or `numa_node_of_cpu` for per-GPU memory-space construction (pre-Phase-6 pattern). Replace with `topology_` struct usage.
- `src/memory/` — any NUMA-aware pinned allocator paths that bypass cucascade.
- `src/op/scan/` — scan-distribution code that currently calls `cudaGetDeviceCount` (if any).
- `src/creator/task_creator.cpp` — per-device routing; may call cuda runtime directly.

### Reference patterns already in the codebase
- **Registry-plus-cache pattern** (Plan 05-03): `io_backend_registry_` + `gpu_io_backends_` on SiriusContext, populated under `rmm::cuda_set_device_raii`, torn down before memory_manager_. Apply the same shape for:
  - `topology_` cache (single struct, populated once)
  - `numa_host_memory_spaces_` cache (vector of shared_ptr, populated per-NUMA)
- **Pure-consumer invariant** (Plans 05-04 + 05-05): downstream plans consume `SiriusContext` accessors without mutating the header. Phase 6 Plans 2+ should honor the same invariant — the plan that owns `src/include/sirius_context.hpp` additions is the sole writer.

### Hardware availability
- N=2 host reachable from this worktree (2 × RTX 6000 Ada, driver 595.58.03) — Phase 5 proved this via actual compute-sanitizer + SF10 runs. Phase 6 MGPU-02 + MGPU-03 gates execute on the same hardware without a separate verification host.
- `test_datasets/tpch_parquet_sf10/` symlinked to `/home/felipe/sirius/test_datasets/tpch_parquet_sf10` for SF10 data access.
- Sirius config env var: `SIRIUS_CONFIG_FILE=/path/to/yaml`. Existing 1-GPU and 2-GPU configs at `/tmp/phase5-validation/sirius-sf10.yaml` and `/tmp/phase5-validation/sirius-2gpu.yaml`.

### Phase-4 deferrals still standing (Phase 7 scope, not Phase 6)
- `test_downgrade_executor.cpp:813` has a `TODO(MGPU-06)` marker for the cross-GPU converter return leg. Phase 6 does NOT fix this — registering the converter slot is orthogonal to fixing the return-leg bug (which requires `cudaMemcpyPeerAsync`).
- `[.][multi_gpu_transfer]` and `[.][mem_04_p2p_transfer]` hidden tests stay off-by-default. Phase 6 may verify MGPU-03 against the non-hidden `[multi_gpu_foundation]` tag; the hidden tags are Phase-7 gate.

</code_context>

<specifics>
## Specific Ideas

- **Plan ordering suggestion (for planner):**
  1. Topology discovery call + caching on SiriusContext (header + impl + startup log). Required first — everything else consumes the cached topology.
  2. Per-NUMA host memory space construction (MGPU-05). Consumes topology; feeds memory_manager_ before any per-GPU backend initialization.
  3. GPU↔GPU converter registration + host-staged implementation + round-trip unit test (MGPU-04).
  4. Hand-rolled CUDA/NUMA sweep across `src/` (MGPU-01 gate). Safe to parallelize with (3) — disjoint file sets.
  5. Single-GPU SF10 regression measurement + compute-sanitizer audit + phase SUMMARY (MGPU-02, MGPU-03). Last wave, checkpoint-gated like Plan 05-06.

- **Topology struct API** (researcher to nail down): expose `int gpu_count() const`, `int numa_count() const`, `int gpu_numa_node(int gpu_id) const`, `std::vector<int> gpus_for_numa(int numa_id) const`. Keeps consumer callsites readable (`topology_.gpu_count()` vs `topology_.gpu_count`).

- **Single-GPU regression gate wording** says baseline = "current `dev` HEAD" — that's distinct from Phase-5's "Phase-4 HEAD" baseline. The dev-HEAD baseline is the *pre-multi-GPU-branch* state. Measurement is only meaningful if the test machine has 1 GPU (or is forced with `CUDA_VISIBLE_DEVICES=0`). The MGPU-02 evidence comes from `performance_test.py 10` on a single GPU of our 2-GPU host, compared against current-dev-HEAD timing.

- **MGPU-03 audit — reuse Phase 5 sanitizer evidence** for the `[parquet][scan]` + `[filter]` + `[join]` subsets (they were green on Phase-5 HEAD; re-run on Phase-6 HEAD). Add `[multi_gpu_foundation]` as the new coverage this phase brings.

- **Host-staged converter implementation path**: follow existing HOST↔GPU converter shape in `cucascade/src/data/representation_converter.cpp`. Chain GPU→HOST followed by HOST→GPU with the source GPU's stream pinning down the H→D hop. Register the composite converter as a single `Tier::GPU → Tier::GPU` entry.

- **Phase-6 research agent should verify**:
  - Exact symbol name / location of `cucascade::topology_discovery` in the pinned f47de0b submodule.
  - The current `dev` HEAD SHA to establish MGPU-02 baseline (likely still `484db35` per Phase 4 init or a newer one; query `git log --oneline main ^HEAD` from feature branch).
  - Actual current callsites of `cudaGetDeviceCount` + `numa_node_of_cpu` in `src/` so Plan ordering reflects reality.
  - Cucascade `numa_region_pinned_host_allocator` signature and where to feed it into the host memory space constructor.

</specifics>

<deferred>
## Deferred Ideas

- **P2P direct `cudaMemcpyPeerAsync` converter body** — Phase 7 (MGPU-06). Phase 6 registers the host-staged slot; Phase 7 swaps the body when P2P is available.
- **Adaptive scan partitioning by available GPU memory** — Phase 7 (MGPU-07).
- **Cross-GPU converter return-leg fix** (`test_downgrade_executor.cpp:813 TODO(MGPU-06)`) — Phase 7.
- **Hidden multi-GPU tests** (`[.][multi_gpu_transfer]`, `[.][mem_04_p2p_transfer]`) — stay off-by-default until Phase 7 closes the return-leg bug.
- **Phase-4-vs-Phase-5 SF10 regression comparison** (Phase 5's IO-10) — still deferred per user directive 2026-04-21 ("we don't need to run any comparisons, let's just make sure everything is working, we can optimize later").
- **`test/` directory hand-rolled CUDA/NUMA sweep** — test fixtures may legitimately need these; only `src/` is Phase-6 scope.
- **Per-backend NUMA affinity tuning** (pinning the pipeline_io_backend's pinned host buffers to the GPU's NUMA domain) — out of v1.1 scope; may surface as optimization if telemetry shows cross-NUMA traffic.
- **Cucascade upstream changes** (new allocators, new topology fields) — out of scope; Phase 6 consumes the f47de0b pinned API as-is.

</deferred>

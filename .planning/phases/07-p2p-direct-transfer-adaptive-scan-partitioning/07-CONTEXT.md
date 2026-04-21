# Phase 7: P2P Direct Transfer + Adaptive Scan Partitioning - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the two final v1.1 requirements that complete the Phase-3 Plan 2 work never landed on `feature/multi-gpu-execution`:

1. **MGPU-06** — GPU↔GPU data transfer uses `cudaMemcpyPeerAsync` directly (no host staging) when P2P access is available. Detection + branching lives inside the existing cucascade Tier::GPU→Tier::GPU converter. Falls back cleanly to host staging when `cudaDeviceCanAccessPeer` returns false. The Phase-4-deferred GPU1→GPU0 return-leg bug at `test_downgrade_executor.cpp:813 TODO(MGPU-06)` is fixed here. Hidden tests `[.][multi_gpu_transfer]` + `[.][mem_04_p2p_transfer]` are un-hidden once green.
2. **MGPU-07** — Scan batch distribution across GPUs is proportional to available GPU memory (not round-robin). Distribution logic lives in `src/op/scan/`; queries `memory_manager_->get_available_bytes(device_id)` per batch (real-time, user choice). Validated by an asymmetric-memory test that pre-loads GPU 0 to ~80% capacity and asserts batch-count skew ≥ 2× matching the free-memory ratio within 10%.

**In scope:**
- `cudaDeviceCanAccessPeer` detection + `cudaDeviceEnablePeerAccess` once-per-pair at `SiriusContext::initialize()` for every `(src, dst)` pair where peer access is available
- `cudaMemcpyPeerAsync` path added inside cucascade's existing GPU↔GPU converter body (or via Sirius-registered override if the existing body is Sirius-controlled — researcher to confirm)
- Host-staged fallback retained for non-P2P hosts; explicit test via `CUDA_VISIBLE_DEVICES` forcing restricted peer visibility
- Fixing the GPU1→GPU0 return-leg (`test_downgrade_executor.cpp:813 TODO(MGPU-06)`) — most likely root cause is the same device-guard / context-mismatch class MGPU-03 fixed on the forward leg; revisit after P2P lands
- `src/op/scan/` batch-distribution refactor from round-robin to memory-weighted
- Per-batch real-time `get_available_bytes` query (Area 2 Q2 user override)
- Un-hiding `[.][multi_gpu_transfer]`, `[.][mem_04_p2p_transfer]`, `[.][mem_05_scan_distribution]` tags
- Extending `test_gpu_execution_locality` with P2P + adaptive-scan scenarios; Catch2-v2 `WARN+return` on single-GPU hosts
- nsys trace evidence showing zero host-stage allocations on the P2P path + `cudaMemcpyPeerAsync` appearing in the trace
- Relaxed bandwidth gate (≥ 1.5× host-staged baseline) per success criterion 1 — measured on N=2 host, documented

**Out of scope:**
- Changes to cucascade submodule — Phase 7 consumes the f47de0b pinned surface. If cucascade's built-in converter body is the right place for the P2P branch and it isn't already doing this, that's a Sirius-side override via `register_converter(Tier::GPU, Tier::GPU, sirius_p2p_converter_factory)` + `unregister_converter` of the cucascade built-in. Research to confirm feasibility.
- Optimization work beyond "it runs, it's measurably faster, fallback works" — no micro-tuning of batch sizes, stream counts, etc.
- Multi-node or cross-socket topologies — single-node N=2 only
- Heterogeneous GPU support (still out of scope per PROJECT.md)
- Changes to the Phase-5 cucascade_datasource path or the Phase-6 topology/NUMA path — Phase 7 consumes them
- Any work on legacy `namespace duckdb` (frozen)
- Phase-5 / Phase-6 SF10 regression comparisons — user directive 2026-04-21 stands

</domain>

<decisions>
## Implementation Decisions

### P2P Direct Transfer (MGPU-06)
- **Branch inside cucascade's existing GPU↔GPU converter body** (the one Phase 6 verified). Use `cudaDeviceCanAccessPeer(src, dst)` to decide P2P vs host staging. If the cucascade built-in body isn't user-modifiable in Sirius (most likely — cucascade is a submodule), register a Sirius-side override via `converter_registry::instance().register_converter(Tier::GPU, Tier::GPU, sirius_p2p_aware_factory)` after first unregistering the cucascade built-in. Researcher confirms feasibility.
- **Fix the GPU1→GPU0 return-leg bug here** (`test_downgrade_executor.cpp:813 TODO(MGPU-06)`). The `[.][multi_gpu_transfer]` and `[.][mem_04_p2p_transfer]` hidden tests become non-hidden (`[multi_gpu_transfer]` without the dot) once green.
- **Enable peer access once at `SiriusContext::initialize()`** for every `(src, dst)` pair where `cudaDeviceCanAccessPeer` returns true. Cache pair-enable state on SiriusContext so the converter path doesn't re-enable per transfer.
- **No-P2P fallback**: force no-peer-access via `CUDA_VISIBLE_DEVICES` or topology masking in test, verify host-staged path still works (converter routes through pinned host buffer, correct data, no error).

### Adaptive Scan Partitioning (MGPU-07)
- **Batch distribution lives in `src/op/scan/`** — replace the round-robin assignment with memory-weighted via `memory_manager_->get_available_bytes(device_id)`. Exact function name to be confirmed by researcher; may already exist as `memory_space->available_bytes()` or similar.
- **Query frequency: per batch, real-time** (user override of the "query once per plan" recommendation). Trade-off acknowledged: more responsive to runtime memory pressure but can oscillate under concurrent scans. Mitigation: if oscillation surfaces in validation, consider a small hysteresis window (e.g., snap to 10% buckets) but only if needed.
- **Validation tolerance: batch-count ratio matches free-memory ratio within 10%** (per success criterion 3).
- **Asymmetric-memory test setup**: pre-load GPU 0 to ~80% capacity with a dummy allocation from `memory_space`, then run a scan task spanning ≥16 batches. Assert batch-count skew ≥ 2× between GPU 0 and GPU 1 AND ratio matches free-memory ratio within 10%.

### Re-enabling Hidden Tests & Regression Gate
- **Un-hide**: `[.][multi_gpu_transfer]` → `[multi_gpu_transfer]`, `[.][mem_04_p2p_transfer]` → `[mem_04_p2p_transfer]`, `[.][mem_05_scan_distribution]` → `[mem_05_scan_distribution]`. Tag hierarchy stays the same; just the `.` (hide marker) is removed.
- **Integration test scope**: extend `test_gpu_execution_locality` with P2P + adaptive-scan scenarios. Single-GPU hosts emit `WARN+return` per Catch2-v2 convention (documented project idiom in Phase 4/5).
- **P2P bandwidth validation**: nsys trace on the round-trip test shows zero pinned-host staging allocations for the P2P path + `cudaMemcpyPeerAsync` call confirmed. Relaxed bandwidth gate per success criterion 1: P2P throughput ≥ 1.5× host-staged baseline, measured on N=2 host. Document measurement in phase SUMMARY.
- **Host-staged fallback test**: explicit test with restricted peer visibility (e.g., force `cudaDeviceCanAccessPeer` to return false via topology mask or env var). Converter must route through host staging without error; correct data; no crash.

</decisions>

<code_context>
## Existing Code Insights

### What Phase 6 shipped (load-bearing for Phase 7)
- Cucascade's built-in `Tier::GPU → Tier::GPU` converter is registered in `converter_registry` at extension load time (verified by Plan 06-03 test `converter_registry exposes gpu_to_gpu converter after initialize() (MGPU-04)`). Body is cucascade's internal peer-async implementation at `cucascade/src/data/representation_converter.cpp:1464`.
- Forward-leg round-trip GPU0→GPU1 bytes-equal confirmed on N=2 host (Plan 06-03 Task 2, hidden `[mgpu_04_round_trip]`, PASS).
- Return-leg GPU1→GPU0 is the known bug (still failing per Phase-4 deferred marker). Phase 7 MGPU-06 closes it.
- Topology cache at `SiriusContext::get_hw_topology()` exposes GPU count, per-GPU NUMA, GPU↔GPU pair enumeration — needed for the peer-access enable loop.
- Device-guard correctness (MGPU-03 from Phase 6): unchecked `cudaSetDevice` sites in Super Sirius are fixed. Phase 7's new peer-access + peer-copy sites should follow the same `cudaError_t err = ...; if (err != cudaSuccess) { SPDLOG_ERROR(...); std::terminate(); }` idiom.

### What Phase 5 shipped (consumed here)
- Per-GPU `cucascade::idisk_io_backend` cache at `SiriusContext::gpu_io_backends_` — not directly consumed by Phase 7 but the pattern (per-GPU cache keyed by device_id, populated under `rmm::cuda_set_device_raii`, cleared before `memory_manager_->shutdown()`) is the template Phase 7 can mirror for the peer-access enable cache.

### Expected migration sites (researcher to verify and update line numbers)
- **`src/sirius_context.cpp`** — add the peer-access enable loop in `SiriusContext::initialize()` after topology + io_backend init. Likely near line 200 alongside existing per-GPU initialization. Cache shape: `std::unordered_set<std::pair<int,int>, hash>` of enabled pairs, or a sorted vector. Keep small (N=2 host: at most 2 pairs).
- **Cucascade converter override** — if Sirius needs to register its own P2P-aware factory via `register_converter(Tier::GPU, Tier::GPU, ...)`, the registration site is `src/sirius_extension.cpp:1053` (same location as Phase 5's io_backend_registry + Phase 6's converter tests). Requires `unregister_converter` of the cucascade built-in first — research to confirm the registry API exposes unregister.
- **`src/op/scan/` batch distribution** — researcher to identify the exact round-robin assignment callsite. Most likely in a scan task factory or scheduler. Phase 5's `task_creator.cpp` threads the `preferred_device_id` (local_state first, global_state fallback) — the batch-distribution change may live adjacent to this.
- **`test/cpp/downgrade/test_downgrade_executor.cpp:813`** — the `TODO(MGPU-06)` marker. Phase 7 closes this by fixing the return-leg bug and removing the comment.
- **`test/cpp/integration/test_gpu_execution_locality.cpp`** — extend with P2P + adaptive-scan scenarios.

### Phase 7 environmental constraints
- N=2 host available (2× RTX 6000 Ada, driver 595.58.03, CUDA 13.2). Sandbox fallback required for GPU-touching commands per Phase 5/6 precedent.
- `cudaDeviceCanAccessPeer(0, 1)` on RTX 6000 Ada via PCIe: confirmed-possible, researcher to verify on this specific host (may depend on PCIe topology / IOMMU config).
- No NVLink on RTX 6000 Ada consumer cards — P2P is PCIe-based. Bandwidth gate (1.5× vs host staging) is realistic but not impressive.
- nsys binary: `/usr/local/cuda-13.0/bin/nsys` expected available (research confirms).

### Reference patterns already in the codebase
- Per-GPU cache idiom (Phase 5): Plan 05-03 established `gpu_io_backends_` on SiriusContext. Phase 7 mirrors for `peer_access_enabled_pairs_`.
- Pure-consumer invariant (Phase 5): downstream plans consume SiriusContext accessors without mutating the header. Phase 7's `src/op/scan/` changes should be pure consumers of the adaptive-scan query API exposed by the SiriusContext / memory_manager.
- Plan-wave + checkpoint structure (Phase 5/6): validation plan is checkpoint-gated; atom plans parallelize on disjoint file scope.

### Known deferrals that Phase 7 CLOSES
- Phase 4: `test_downgrade_executor.cpp:813 TODO(MGPU-06)` — return-leg bug
- Phase 4/5/6: `[.][multi_gpu_transfer]`, `[.][mem_04_p2p_transfer]`, `[.][mem_05_scan_distribution]` hidden tags
- Phase-4-vs-Phase-X SF10 regression comparison — STILL DEFERRED per user directive 2026-04-21

</code_context>

<specifics>
## Specific Ideas

- **Plan ordering suggestion** (planner — verify feasibility):
  1. Peer-access detection + enable loop + cache at `SiriusContext::initialize()` (MGPU-06 infra).
  2. P2P converter body — either Sirius-side override or cucascade-built-in branch (MGPU-06 core) + return-leg bug fix.
  3. Adaptive scan batch distribution in `src/op/scan/` (MGPU-07).
  4. Tests — un-hide existing hidden tags + extend `test_gpu_execution_locality` + asymmetric-memory test for MGPU-07.
  5. Phase validation (nsys trace, bandwidth measurement, fallback test, compute-sanitizer re-run) + SUMMARY. Checkpoint-gated like 05-06 / 06-04.

- **Per-batch real-time memory query** (user override): document the oscillation risk in Plan 3 notes. If validation shows thrashing under concurrent scans, the fix is a 10%-bucket snap or a 100ms cache — NOT reverting to "once per plan" which loses the responsiveness the user wanted.

- **Return-leg bug triage first**: before writing the P2P converter body, the researcher + executor should reproduce the GPU1→GPU0 return-leg failure and characterize it. Phase 6 MGPU-03 already fixed the two Super-Sirius `cudaSetDevice` callsites — if the return-leg bug was caused by one of those, it may already be fixed. Quick verification step: run `[.][multi_gpu_transfer]` on current HEAD before Plan 2. If it passes, Plan 2 scope shrinks to "add P2P branch + un-hide".

- **Bandwidth measurement approach**: not per-host tuning. Run the hidden round-trip test under nsys twice (pre-P2P / post-P2P), extract `cudaMemcpyPeerAsync` duration + bytes via nsys export, compute effective bandwidth. Compare; document. If P2P < 1.5× host-staged, file upstream cucascade issue + document in SUMMARY, do NOT block phase (per Phase-5/6 deferral precedent).

- **nsys invocation**: `nsys profile --trace=cuda,nvtx --cudabacktrace=all -o /tmp/phase7-p2p.nsys-rep build/release/extension/sirius/test/cpp/sirius_unittest "[mgpu_04_round_trip]"`. Export + grep for `cudaMemcpyPeerAsync` vs `cudaMemcpyAsync`.

- **Un-hide mechanics**: sed across the affected test files to remove leading `.` in tag definitions. Sanity check: `grep -c '\[\\.' test/cpp/` should drop by 3 after the sweep.

- **Asymmetric-memory test fixture**: use `cucascade::memory::device_memory_resource::allocate` directly with an 80%-GPU-0 size to reserve without CUDA kernel activity. Release at test-fixture teardown.

</specifics>

<deferred>
## Deferred Ideas

- **Cucascade upstream changes** — if a Sirius-side override is infeasible, file upstream issue to make the GPU↔GPU converter factory pluggable.
- **Batch-distribution hysteresis / bucket-snap** — only if per-batch real-time memory query causes oscillation in validation. Default: no mitigation.
- **NVLink P2P** — out of scope (host has PCIe-only consumer cards); if a production DGX/HGX host later runs Phase 7 code, bandwidth gate should exceed 1.5× trivially.
- **Scan distribution by join-key / hash partition** — PROJECT.md OPT-03 is a v2.0 item.
- **Per-NUMA scan distribution** — separate optimization; Phase 7 distributes by GPU free-memory only.
- **Phase-X vs Phase-Y SF10 regression** — still deferred per user directive 2026-04-21.
- **P2P over RDMA / multi-node** — v2.0 scope.
- **Release notes / user-facing docs for the P2P capability** — milestone-exit work, not Phase 7.

</deferred>

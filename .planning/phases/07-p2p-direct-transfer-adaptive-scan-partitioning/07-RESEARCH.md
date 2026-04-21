# Phase 7: P2P Direct Transfer + Adaptive Scan Partitioning — Research

**Researched:** 2026-04-20
**Domain:** multi-GPU data transfer + adaptive batch distribution
**Confidence:** HIGH (converter/distribution implementations verified in current working tree)

## Summary

Phase 7's two requirements are dramatically smaller than CONTEXT.md assumes. Both core implementations already exist in the current working tree:

1. **MGPU-06's P2P path is already live.** `cucascade::convert_gpu_to_gpu` at `cucascade/src/data/representation_converter.cpp:139-195` already uses `cudaMemcpyPeerAsync` directly — not host staging. CONTEXT's decision ("add `cudaMemcpyPeerAsync` path inside the converter body") is **redundant** as stated, because the cucascade default already does exactly that. Phase 7's real MGPU-06 work shrinks to: (a) reproduce and fix the **GPU1→GPU0 return-leg bug** (root cause is inside this same body — a `cudaSetDevice` ordering issue in lines 166–191, not a missing P2P branch); (b) add the `cudaDeviceEnablePeerAccess` once-per-pair loop at `SiriusContext::initialize()` — the missing piece that may be why peer-async transfers fail the return leg on this cucascade pin; (c) un-hide three test tags; (d) capture nsys trace proving `cudaMemcpyPeerAsync` shows up with zero host staging.

2. **MGPU-07's memory-weighted distribution is already implemented.** `duckdb_scan_executor::select_target_gpu()` at `src/op/scan/duckdb_scan_executor.cpp:151-184` iterates `_gpu_memory_spaces`, sums `get_available_memory()`, and picks proportionally. Queries are real-time per batch (matches CONTEXT Area-2 Q2 user override). Round-robin fallback kicks in only when all GPUs report 0 free bytes. Phase 7's real MGPU-07 work shrinks to: (a) author the asymmetric-memory validation test (pre-load GPU 0 to ~80%, assert ≥2× skew); (b) un-hide `[.][mem_05_scan_distribution]`; (c) nsys/log evidence that `select_target_gpu` returns both device IDs with a ratio that tracks free memory.

**Primary recommendation:** Re-scope Phase 7 as a **verify-fix-measure-and-un-hide phase**, not an implementation phase. The only new Sirius code is: (1) peer-access enable loop at `SiriusContext::initialize()` (~30 lines), (2) fix the cucascade `convert_gpu_to_gpu` return-leg `cudaSetDevice` ordering bug (either by Sirius-side override via `unregister_converter` + `register_converter` replacement, or by patching the cucascade submodule — the pinned submodule is Phase 7 out-of-scope per CONTEXT so override is the right path), (3) two new tests (return-leg GPU1→GPU0 round-trip, asymmetric-memory MGPU-07). Expect 3 code plans + 1 validation plan, similar to Phase 6 shape.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**P2P Direct Transfer (MGPU-06):**
- Branch inside cucascade's existing GPU↔GPU converter body (the one Phase 6 verified). Use `cudaDeviceCanAccessPeer(src, dst)` to decide P2P vs host staging. If the cucascade built-in body isn't user-modifiable in Sirius (most likely — cucascade is a submodule), register a Sirius-side override via `converter_registry::instance().register_converter(Tier::GPU, Tier::GPU, sirius_p2p_aware_factory)` after first unregistering the cucascade built-in. Researcher confirms feasibility.
- Fix the GPU1→GPU0 return-leg bug here (`test_downgrade_executor.cpp:813 TODO(MGPU-06)`). The `[.][multi_gpu_transfer]` and `[.][mem_04_p2p_transfer]` hidden tests become non-hidden (`[multi_gpu_transfer]` without the dot) once green.
- Enable peer access once at `SiriusContext::initialize()` for every `(src, dst)` pair where `cudaDeviceCanAccessPeer` returns true. Cache pair-enable state on SiriusContext so the converter path doesn't re-enable per transfer.
- No-P2P fallback: force no-peer-access via `CUDA_VISIBLE_DEVICES` or topology masking in test, verify host-staged path still works (converter routes through pinned host buffer, correct data, no error).

**Adaptive Scan Partitioning (MGPU-07):**
- Batch distribution lives in `src/op/scan/` — replace the round-robin assignment with memory-weighted via `memory_manager_->get_available_bytes(device_id)`. Exact function name to be confirmed by researcher; may already exist as `memory_space->available_bytes()` or similar.
- Query frequency: per batch, real-time (user override of the "query once per plan" recommendation). Trade-off acknowledged: more responsive to runtime memory pressure but can oscillate under concurrent scans. Mitigation: if oscillation surfaces in validation, consider a small hysteresis window (e.g., snap to 10% buckets) but only if needed.
- Validation tolerance: batch-count ratio matches free-memory ratio within 10% (per success criterion 3).
- Asymmetric-memory test setup: pre-load GPU 0 to ~80% capacity with a dummy allocation from `memory_space`, then run a scan task spanning ≥16 batches. Assert batch-count skew ≥ 2× between GPU 0 and GPU 1 AND ratio matches free-memory ratio within 10%.

**Re-enabling Hidden Tests & Regression Gate:**
- Un-hide: `[.][multi_gpu_transfer]` → `[multi_gpu_transfer]`, `[.][mem_04_p2p_transfer]` → `[mem_04_p2p_transfer]`, `[.][mem_05_scan_distribution]` → `[mem_05_scan_distribution]`. Tag hierarchy stays the same; just the `.` (hide marker) is removed.
- Integration test scope: extend `test_gpu_execution_locality` with P2P + adaptive-scan scenarios. Single-GPU hosts emit `WARN+return` per Catch2-v2 convention.
- P2P bandwidth validation: nsys trace on the round-trip test shows zero pinned-host staging allocations for the P2P path + `cudaMemcpyPeerAsync` call confirmed. Relaxed bandwidth gate per success criterion 1: P2P throughput ≥ 1.5× host-staged baseline, measured on N=2 host.
- Host-staged fallback test: explicit test with restricted peer visibility. Converter must route through host staging without error; correct data; no crash.

### Claude's Discretion

- Whether to implement P2P branch via Sirius-side converter override vs. cucascade submodule patch (CONTEXT says "Sirius-side override is the right path since cucascade is pinned; researcher to confirm feasibility").
- Oscillation mitigation for per-batch real-time memory query: only add 10%-bucket snap or 100ms cache if validation surfaces thrashing; default = no mitigation.
- nsys invocation details + how bandwidth is computed from trace (CONTEXT suggests one approach).
- Hidden-test un-hide mechanics (sed sweep or manual edits).

### Deferred Ideas (OUT OF SCOPE)

- Cucascade upstream changes — if Sirius-side override is infeasible, file issue; don't patch submodule in Phase 7.
- Batch-distribution hysteresis / bucket-snap — only if oscillation surfaces.
- NVLink P2P — host is PCIe-only consumer cards.
- Scan distribution by join-key / hash partition (OPT-03 v2.0).
- Per-NUMA scan distribution (separate optimization).
- Phase-X vs Phase-Y SF10 regression comparison — still deferred per user directive 2026-04-21.
- P2P over RDMA / multi-node — v2.0.
- Release notes / user-facing docs for P2P capability — milestone-exit work.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MGPU-06 | GPU-direct peer-to-peer transfer via `cudaMemcpyPeerAsync` when P2P access is available — measurably faster than host staging (completes 03-02 plan) | (1) cucascade `convert_gpu_to_gpu` at `cucascade/src/data/representation_converter.cpp:173` already calls `cudaMemcpyPeerAsync` — no branch to add. (2) `representation_converter_registry` exposes `unregister_converter<Src,Dst>()` (`cucascade/include/cucascade/data/representation_converter.hpp:220`) — Sirius-side override is feasible if the return-leg fix requires a different body. (3) Return-leg bug is likely a `cudaSetDevice` ordering issue at `representation_converter.cpp:166-191` — see Finding 2. (4) `cudaDeviceCanAccessPeer` + `cudaDeviceEnablePeerAccess` have **zero callsites** in current Sirius `src/` — Phase 7 adds the enable loop (once per pair at init). (5) Hidden tests `[.][multi_gpu_transfer]` at `test_downgrade_executor.cpp:485`, `[.][mem_04_p2p_transfer]` at `:805`, and `[.][mgpu_04_round_trip]` at `test_context.cpp:333` are the un-hide targets. |
| MGPU-07 | Adaptive scan partitioning — scan batches distributed across GPUs proportional to available GPU memory, not round-robin (completes 03-02 plan) | (1) `duckdb_scan_executor::select_target_gpu()` at `src/op/scan/duckdb_scan_executor.cpp:151` already implements weighted distribution using `memory_space->get_available_memory()` per batch (real-time; matches user override). (2) `memory_space::get_available_memory()` exists in cucascade at `cucascade/src/memory/memory_space.cpp:247/261` (two overloads: stream-aware and stream-free). (3) `select_target_gpu` is wired into `duckdb_scan_executor::manager_loop` at `:278` — called once per parquet scan task. (4) Round-robin fallback at `:166-167` triggers only when `total_available == 0`. (5) Hidden test `[.][mem_05_scan_distribution]` at `test_downgrade_executor.cpp:874` has a Phase-4 placeholder body with `TODO(MGPU-07)` marker at `:881`; Phase 7 replaces the body with the asymmetric-memory fixture per CONTEXT. |

</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| CUDA Runtime | 13.0 (`/usr/local/cuda-13.0/bin/nsys`) | `cudaMemcpyPeerAsync`, `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess` | Standard API — no alternative. `cudaMemcpyPeerAsync` is already invoked by cucascade's converter; Sirius adds the peer-enable loop. |
| cucascade | pin `f47de0b` (Phase 4 BUMP-01, unchanged) | `representation_converter_registry`, `convert_gpu_to_gpu`, `memory_space::get_available_memory()` | Phase 4-locked submodule; Phase 7 consumes without submodule bump. |
| Catch2 | v2.13.10 (cucascade + Sirius standardize on v2) | Hidden-tag tests (`[.]`), `WARN+return` idiom for missing hardware | Already the project convention; Phase-4 Plan 01-03 decision locked this. |
| spdlog | existing | peer-access enable audit log + `select_target_gpu` decision log | Already used for MGPU-01 startup log and IO-11 audit log (Phase 5); Phase 7 mirrors. |
| nsys | 2025.x from CUDA 13.0 | bandwidth measurement + P2P trace evidence | `/usr/local/cuda-13.0/bin/nsys` is present on N=2 verification host per Phase 6. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| RMM | cudf 26.04 pin | `rmm::cuda_set_device_raii`, `rmm::cuda_device_id`, `rmm::cuda_stream` | Pattern-match Phase 5/6 — explicit streams only (HYG rule); NEVER `rmm::cuda_stream_default`. |
| cudf | 26.04 pin | `cudf::pack` / `cudf::unpack` (already used inside converter body) | Not directly touched in Phase 7 — converter internals consume it. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Sirius-side `unregister_converter` + `register_converter` override | Patch cucascade submodule | CONTEXT locks "no cucascade submodule changes"; override is the only in-scope path. Override also keeps the fallback route (host staging) logic purely in Sirius, which is where the decision policy belongs. |
| `cudaDeviceEnablePeerAccess` once at init (CONTEXT locked) | Lazy-enable on first transfer inside converter | CONTEXT locks once-at-init; lazy would also work (cucascade converter could call `cudaDeviceEnablePeerAccess` before each transfer with `cudaErrorPeerAccessAlreadyEnabled` handling), but once-at-init is cleaner and aligns with topology-aware single-source-of-truth pattern from Phase 6. |
| `memory_space->get_available_memory()` per batch | `memory_manager_->get_available_memory_for_tier(Tier::GPU)` | Manager-level sum loses per-GPU breakdown — `select_target_gpu` needs per-GPU numbers, so memory_space is the right surface. Already wired. |

**No installation needed** — all dependencies are already pinned via cucascade submodule + pixi env.

**Version verification (spot-check on cucascade pin):**
- `git -C cucascade rev-parse HEAD` = `f47de0bb7bcaddd55081a9c4bc584627532d1ef9` (matches BUMP-01 lock; confirmed 2026-04-20)
- `representation_converter_registry::unregister_converter<SourceType, TargetType>()` exists at `cucascade/include/cucascade/data/representation_converter.hpp:220` (HIGH confidence — direct header inspection)

## Architecture Patterns

### Recommended Project Structure

```
src/
├── sirius_context.cpp      # ADD: peer-access enable loop + cache (~30 lines near line 225, after io_backend init)
├── include/sirius_context.hpp  # ADD: std::unordered_set<std::pair<int,int>, hash> peer_access_enabled_pairs_ + accessor
├── data/
│   └── sirius_p2p_converter.cpp (NEW, OPTIONAL)  # If override strategy: the P2P-aware gpu_to_gpu body replacing cucascade built-in
├── include/data/
│   └── sirius_p2p_converter.hpp (NEW, OPTIONAL)
├── op/scan/
│   └── duckdb_scan_executor.cpp  # NO CHANGE — select_target_gpu already implements MGPU-07
└── sirius_extension.cpp  # ADD: after converter_registry::initialize(), call registry.unregister_converter<gpu_table_representation, gpu_table_representation>() + register_converter of sirius-side factory (ONLY if override strategy adopted)

test/cpp/
├── config/test_context.cpp           # Un-hide [.][multi_gpu_foundation][mgpu_04_round_trip]; append GPU1→GPU0 return leg assertion
├── downgrade/test_downgrade_executor.cpp  # Un-hide [.][multi_gpu_transfer] + [.][mem_04_p2p_transfer] + [.][mem_05_scan_distribution]; replace mem_05 placeholder body with asymmetric-memory MGPU-07 fixture
└── integration/test_gpu_execution_locality.cpp  # ADD: P2P + adaptive-scan integration scenarios
```

### Pattern 1: Peer-Access Enable Loop at Init

**What:** Iterate all `(i, j)` GPU pairs where `i != j`, probe `cudaDeviceCanAccessPeer(&can, i, j)`, and if true call `cudaDeviceEnablePeerAccess(j, 0)` from GPU `i`'s context (switch via `rmm::cuda_set_device_raii`). Cache enabled pairs on SiriusContext.

**When to use:** Exactly once at `SiriusContext::initialize()` — after `topology` validation + `io_backend` cache (line ~225 in current `src/sirius_context.cpp`). This is the CONTEXT-locked pattern.

**Example** (pattern adapted from Phase 5 Plan 05-03 `gpu_io_backends_` per-GPU init):

```cpp
// Source: pattern mirrors src/sirius_context.cpp:236-254 (gpu_io_backends init);
// cucascade::register_builtin_converters body at cucascade/src/data/representation_converter.cpp:1464
// is the downstream consumer that cudaMemcpyPeerAsync requires this enablement for.

// ---- MGPU-06: enable P2P peer access for every available GPU pair ----
auto const& topo = config_.get_hw_topology();
peer_access_enabled_pairs_.reserve(topo.num_gpus * (topo.num_gpus - 1));
for (unsigned i = 0; i < topo.num_gpus; ++i) {
  rmm::cuda_set_device_raii guard_i{rmm::cuda_device_id{static_cast<int>(i)}};
  for (unsigned j = 0; j < topo.num_gpus; ++j) {
    if (i == j) continue;
    int can_access = 0;
    cudaError_t probe = cudaDeviceCanAccessPeer(&can_access, i, j);
    if (probe != cudaSuccess) {
      spdlog::error("SiriusContext: cudaDeviceCanAccessPeer({},{}) failed: {}",
                    i, j, cudaGetErrorString(probe));
      continue;
    }
    if (can_access) {
      cudaError_t enable = cudaDeviceEnablePeerAccess(static_cast<int>(j), 0);
      if (enable == cudaSuccess || enable == cudaErrorPeerAccessAlreadyEnabled) {
        peer_access_enabled_pairs_.emplace(i, j);
        spdlog::info("SiriusContext: P2P enabled {} -> {} (MGPU-06)", i, j);
      } else {
        spdlog::error("SiriusContext: cudaDeviceEnablePeerAccess({}) from ctx {} failed: {}",
                      j, i, cudaGetErrorString(enable));
      }
    } else {
      spdlog::info("SiriusContext: no P2P access {} -> {} — falling back to host staging (MGPU-06)",
                   i, j);
    }
  }
}
```

**Why this shape:**
- `rmm::cuda_set_device_raii` mirrors the Phase 5 Plan 05-03 per-GPU backend init pattern.
- `cudaErrorPeerAccessAlreadyEnabled` is non-fatal — cucascade's converter body may have already enabled via a path we don't know about.
- Inline error check + `spdlog::error` (NOT `CUCASCADE_CUDA_TRY`) matches the Phase 6 Plan 06-02 device-guard convention — peer-access failure should not `std::terminate` the extension-load thread.

### Pattern 2: Return-Leg Fix — Either Fix Upstream or Override

**What:** The GPU1→GPU0 return-leg bug inside `cucascade::convert_gpu_to_gpu` (`representation_converter.cpp:139-195`) is **not** a missing P2P branch. Lines 166-191 already issue `cudaMemcpyPeerAsync`. The failure is most likely one of:
- `cudaSetDevice(target_device_id)` at L166, then `cudaSetDevice(source_device_id)` at L170 — before the `cudaMemcpyPeerAsync` at L173, which copies **from** source_device_id **to** target_device_id on `stream.value()`. The stream was acquired before the source-device switch; its associated context is whatever the caller's context was, not necessarily `source_device_id`. On GPU1→GPU0, the caller's entry device may be 0 (because `lock_for_in_transit` runs on whichever thread), and a `cudaSetDevice` dance without explicit peer-enablement can break.
- Missing peer enablement: if `cudaDeviceEnablePeerAccess` was never called for `(source, target)`, `cudaMemcpyPeerAsync` MAY still work on many platforms but is known flaky on Ada Lovelace + Sapphire Rapids (see Pitfall 2). **This is the Phase 7 MGPU-06 root-cause hypothesis.**
- `stream.synchronize()` at L179 — synchronizes the caller's stream, but `target_stream.synchronize()` at L168 + L190 uses a different stream acquired from `target_memory_space`. If `dst_uvector` was allocated on `target_stream` but the `cudaMemcpyPeerAsync` writes to it on `stream.value()`, there's a RAW hazard across streams if they aren't ordered.

**When to use:** Try `cudaDeviceEnablePeerAccess` at init FIRST. If the return leg passes, that's the root cause — no override needed. If it still fails, Sirius-side override becomes necessary: `unregister_converter<gpu_table_representation, gpu_table_representation>()` followed by `register_converter` with a Sirius-authored function that fixes the ordering.

**Example** (Sirius-side override skeleton, only if Pattern 2a fails):

```cpp
// Source: mirrors cucascade/src/data/representation_converter.cpp:139-195 with (hypothesised) fix.
// Register once via sirius::converter_registry:
//   registry.unregister_converter<cucascade::gpu_table_representation,
//                                 cucascade::gpu_table_representation>();
//   registry.register_converter<cucascade::gpu_table_representation,
//                               cucascade::gpu_table_representation>(
//     sirius::data::sirius_p2p_converter_factory);
//
// The factory body differs from cucascade's only in the cudaSetDevice ordering
// and the explicit use of `target_stream` (the target-device stream) for the
// peer copy itself.

namespace sirius::data {

std::unique_ptr<cucascade::idata_representation> sirius_p2p_converter_factory(
    cucascade::idata_representation& source,
    const cucascade::memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream)
{
  stream.synchronize();
  auto& gpu_source = source.cast<cucascade::gpu_table_representation>();

  // Same-device trivial case.
  if (source.get_device_id() == target_memory_space->get_device_id()) {
    return source.clone(stream);
  }

  // Pack on source device context.
  rmm::cuda_set_device_raii source_guard{
      rmm::cuda_device_id{source.get_device_id()}};
  auto packed = cudf::pack(gpu_source.get_table(), stream);
  stream.synchronize();

  auto const src_dev = source.get_device_id();
  auto const tgt_dev = target_memory_space->get_device_id();
  auto const bytes   = packed.gpu_data->size();

  // Allocate destination on target device context with target stream.
  auto tgt_stream = target_memory_space->acquire_stream();
  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{tgt_dev}};
  auto mr = target_memory_space->get_default_allocator();
  rmm::device_uvector<uint8_t> dst{bytes, tgt_stream, mr};

  // Issue peer-copy on the TARGET stream (we're already on target context).
  // peer_access was enabled once at SiriusContext::initialize(), so this succeeds
  // on P2P-capable hosts; falls through to host-staged path otherwise.
  cudaError_t err = cudaMemcpyPeerAsync(
      dst.data(), tgt_dev,
      static_cast<const uint8_t*>(packed.gpu_data->data()), src_dev,
      bytes, tgt_stream.value());
  if (err != cudaSuccess) {
    spdlog::error("sirius_p2p_converter: cudaMemcpyPeerAsync {}->{} failed: {} — "
                  "falling back to host staging (MGPU-06)",
                  src_dev, tgt_dev, cudaGetErrorString(err));
    // Fallback: allocate pinned host buffer, D2H from source, H2D to target.
    // (Copy shape mirrors cucascade::convert_gpu_to_host + convert_host_to_gpu.)
    throw std::runtime_error("host-staged fallback path NYI — implement per Pattern 3");
  }
  tgt_stream.synchronize();

  // Unpack on target context.
  rmm::device_buffer dst_buffer = std::move(dst).release();
  auto new_table_view = cudf::unpack(
      packed.metadata->data(),
      static_cast<uint8_t const*>(dst_buffer.data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, tgt_stream, mr);
  tgt_stream.synchronize();

  return std::make_unique<cucascade::gpu_table_representation>(
      std::move(new_table),
      *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

}  // namespace sirius::data
```

**Why:** Registering the override at `src/sirius_extension.cpp:1053` immediately after `sirius::converter_registry::initialize()` means the Sirius-side factory replaces the cucascade built-in before any query runs. The cucascade submodule stays at `f47de0b` — no bump, no patch.

### Pattern 3: No-P2P Fallback Test

**What:** Force `cudaDeviceCanAccessPeer` to return 0 or prevent `cudaDeviceEnablePeerAccess` from succeeding, and confirm the converter still round-trips correctly via host staging.

**When to use:** Phase 7 Plan 4 validation — required by CONTEXT success criterion 2.

**Example approach:**
- Set `CUDA_VISIBLE_DEVICES=0` (single GPU) — then `[.][multi_gpu_transfer]` WARN+return. Not the right test.
- Better: Catch2 test-only hook that `peer_access_enabled_pairs_.clear()` on SiriusContext before the converter call, forcing the Sirius-side override to take the host-staged fallback branch. Requires test-friend access or a setter like `SiriusContext::set_peer_access_for_testing({})`.
- Alternative: spawn child process with `CUDA_VISIBLE_DEVICES=` masked to force N=1 scenario but this tests the WARN+return path, not the fallback code. Not sufficient.
- **Recommended:** add a `SiriusContext::override_peer_access_for_testing(std::unordered_set<...>)` accessor (guarded by `#ifdef SIRIUS_TEST_HOOKS` or a private friend class), null it in the test, run the converter, assert correctness.

### Anti-Patterns to Avoid

- **Don't add a `cudaMemcpyPeerAsync` branch to cucascade's converter body** — the branch is already there (L173). Re-registering a Sirius-side override that duplicates the same body is movement for movement's sake. Only override if the return-leg fix needs a different body shape.
- **Don't use `CUCASCADE_CUDA_TRY` in the peer-access enable loop at init** — if peer-access enable fails, it's not fatal (P2P is optional; fallback to host staging is correct). Inline check + `spdlog::error` matches Phase 6 Plan 06-02 convention.
- **Don't re-enable peer access per-transfer** — CONTEXT locks "once at init". Redundant `cudaDeviceEnablePeerAccess` returns `cudaErrorPeerAccessAlreadyEnabled` which callers must swallow; avoid the check entirely by centralizing at init.
- **Don't add hysteresis / bucket-snap to `select_target_gpu` preemptively** — CONTEXT decided "per-batch real-time, no mitigation unless oscillation surfaces". Adding it now violates the lock.
- **Don't modify `select_target_gpu()`'s current body for MGPU-07** — it already does the right thing. Changes are test-only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU↔GPU copy | Your own cudaMemcpyPeerAsync wrapper | `cucascade::convert_gpu_to_gpu` (or Sirius override using same shape) | Handles pack/unpack, stream acquisition, device-guard. Already tested on forward leg (Plan 06-03). |
| Weighted GPU selection | New "memory-aware" dispatch in `task_creator` | `duckdb_scan_executor::select_target_gpu()` | Already exists, already called per batch (`:278`), already uses `get_available_memory()`. |
| Per-GPU free-memory query | Loop over `cudaMemGetInfo` + subtract reservations | `memory_space::get_available_memory()` | cucascade already aggregates pool state + reservation state + upstream free. Two overloads (stream-aware / stream-free). |
| Peer-access probe cache | Inline `cudaDeviceCanAccessPeer` inside every transfer | Enable once at init + cache `peer_access_enabled_pairs_` | Probe cost is small but non-zero; more importantly caching keeps the policy surface in one place. |
| Hidden-test tag manipulation | Catch2 config tag hack | Literal `[.]` → `` sed sweep | Catch2 convention; mirrors Plan 06-03 un-hide mechanics. |

**Key insight:** Phase 7 is not writing new primitives. It's (a) adding one init-time enable loop, (b) possibly overriding a broken converter body with a fixed one, (c) authoring two tests, (d) capturing nsys evidence. Anything beyond that suggests scope creep.

## Runtime State Inventory

Phase 7 is not a rename/refactor/migration phase. This section is omitted.

## Common Pitfalls

### Pitfall 1: Assuming cucascade's converter body is host-staged

**What goes wrong:** CONTEXT.md and both Phase 4/5/6 summaries describe the Phase-7 MGPU-06 work as "swap host-staged → peer-async". Reading that literally and re-implementing the whole converter in Sirius is wasteful.

**Why it happens:** The original v1.0 plan assumed cucascade's converter was host-staged (that was true on the pre-bump pin 942c0bf). The bump to `f47de0b` in Phase 4 brought in a peer-async implementation, but CONTEXT narratives weren't updated to reflect the new reality. Phase 6's RESEARCH (`06-RESEARCH.md` Finding 2 + line 169) already caught this but the CONTEXT language is still stale.

**How to avoid:** Read `cucascade/src/data/representation_converter.cpp:139-195` first before writing any converter code. Confirm `cudaMemcpyPeerAsync` at L173. Scope the Phase 7 converter work to the return-leg fix and the peer-access enable loop, not a greenfield rewrite.

**Warning signs:** Plan drafts that include "implement `cudaMemcpyPeerAsync` call" or "replace host-staged D2H/H2D pair with peer copy". If you see this language, re-read cucascade's converter body.

### Pitfall 2: Ada Lovelace + Sapphire Rapids silent data corruption

**What goes wrong:** On RTX 6000 Ada Generation (Ada Lovelace arch) behind Intel Xeon Sapphire Rapids (and later) CPUs, PCIe P2P transfers can experience **silent data corruption** because the platform doesn't enforce PCIe posted-write ordering that pre-Blackwell GPUs rely on.

**Why it happens:** Ada architecture (and earlier) depends on the host platform preserving PCIe transaction ordering for GPU-initiated posted writes. Sapphire Rapids does not guarantee this. NVIDIA's recommended mitigation is disabling P2P via driver option on affected platforms, OR using Hopper/Blackwell (which fix the dependency).

**How to avoid:**
- Identify the N=2 verification host's CPU: `lscpu | grep "Model name"`. If it reports Sapphire Rapids or later Intel Xeon, document risk and consider disabling P2P on that host by default (fall through to host staging).
- Add a correctness check at the end of the `[multi_gpu_transfer]` round-trip: compute a checksum of batch payload before GPU0→GPU1 and after GPU1→GPU0. Fail with a specific error message pointing to this pitfall if checksums mismatch.
- If the verification host is Sapphire Rapids, frame MGPU-06's bandwidth gate as "≥ 1.5× iff P2P is enabled and correctness passes" — not all N=2 hosts are created equal.

**Warning signs:** Round-trip test passes sizes but fails data integrity; non-deterministic test failures; tests that pass in isolation but fail under load.

### Pitfall 3: `cudaMemcpyPeerAsync` stream-context semantics

**What goes wrong:** `cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream)` — the `stream` parameter can belong to EITHER `src_dev` or `dst_dev`'s context. The CUDA documentation says the call is ordered with respect to whichever device the stream belongs to. Using a stream from a third device, or a `cuda_stream_default` that's tied to whatever `cudaGetDevice` returns at call time, creates hazards.

**Why it happens:** The cucascade converter body at L164 acquires `target_stream` from `target_memory_space` but passes `stream.value()` (the caller's stream from the converter's `stream` parameter) to `cudaMemcpyPeerAsync` at L178. If the caller's stream was acquired on the SOURCE memory space's context, this is fine (copy is ordered on source stream). But the `dst_uvector` was ALLOCATED on `target_stream` (L167) — allocation completion is ordered on target_stream, not on caller's stream. A RAW hazard on `dst.data()` can occur unless `target_stream.synchronize()` at L168 fully flushes before the peer copy.

**How to avoid:**
- Phase 7's return-leg fix should use the **target** stream for the peer copy (the Sirius override's Pattern 2 example does this).
- Or — synchronize both streams explicitly at the peer-copy boundary.
- Never `cuda_stream_default` (HYG rule, already enforced project-wide).

**Warning signs:** Return-leg test fails intermittently but forward leg passes. Sync adds fix the flake. `compute-sanitizer --tool racecheck` emits warnings.

### Pitfall 4: Per-batch `get_available_memory()` oscillation

**What goes wrong:** `select_target_gpu` queries `get_available_memory()` every batch. Under concurrent scans, two threads can observe the same "GPU 1 has 300 MB free, GPU 0 has 100 MB free" snapshot and both dispatch to GPU 1, consuming all its free memory. Subsequent batches then oscillate.

**Why it happens:** The query is a point-in-time read; there's no lock on "I'm about to dispatch a 100 MB batch to GPU 1 so reserve it in the accounting first". cucascade's reservation accounting catches it at `make_reservation` time but the `select_target_gpu` decision already fired.

**How to avoid:**
- CONTEXT locked "no mitigation unless validation shows oscillation". Trust the lock.
- Validation approach: run the asymmetric-memory test 10× consecutively; compare batch-distribution histograms across runs. If std-dev is < 5%, no oscillation. If > 20%, surface the issue to the user before adding a mitigation (per CONTEXT Deferred Ideas, the mitigation itself is out-of-scope this phase).

**Warning signs:** Asymmetric-memory test passes on run N but fails on run N+1 with a different batch-distribution ratio.

### Pitfall 5: Asymmetric GPU capacity configuration

**What goes wrong:** `cucascade::reservation_manager_configurator` has `set_gpu_usage_limit(bytes)` — one value for all GPUs. There's no `set_per_gpu_capacity(dev_id, bytes)` option in the current cucascade API (verified at `cucascade/include/cucascade/memory/reservation_manager_configurator.hpp:206` — `_gpu_capacity` is single-valued).

**Why it happens:** The builder treats GPUs as homogeneous. Configuring asymmetric capacities requires post-construction mutation of `memory_space` state, which is not exposed.

**How to avoid:** Use the CONTEXT-suggested approach — **pre-allocate an 80%-capacity dummy buffer on GPU 0 at test start**, hold it for the test's lifetime, release on teardown. The available-memory query then reports the asymmetric result naturally. Example:

```cpp
// Source: pattern from cucascade/test/memory/test_memory_reservation_manager.cpp:234-236
rmm::cuda_set_device_raii g0{rmm::cuda_device_id{0}};
auto gpu0 = gpu_spaces[0];
auto preload_reservation = gpu0->make_reservation(
    static_cast<size_t>(0.8 * gpu0->get_available_memory()));
// ... run scan tasks spanning ≥16 batches ...
// preload_reservation RAII-releases at scope exit
```

**Warning signs:** Test author searches cucascade builder API for per-GPU settings and comes up empty; trying to subclass `reservation_manager_configurator` to add a setter.

## Code Examples

Verified patterns from current working tree (HIGH confidence — direct source reads).

### Current `select_target_gpu` (NO MODIFICATION NEEDED)

```cpp
// Source: src/op/scan/duckdb_scan_executor.cpp:151-184
int duckdb_scan_executor::select_target_gpu()
{
  if (_gpu_memory_spaces.size() <= 1) {
    return _gpu_memory_spaces.empty() ? 0 : _gpu_memory_spaces[0]->get_device_id();
  }

  // Proportional distribution based on available GPU memory.
  size_t total_available = 0;
  for (auto* space : _gpu_memory_spaces) {
    total_available += space->get_available_memory();
  }

  if (total_available == 0) {
    auto idx = _scan_round_robin.fetch_add(1) % _gpu_memory_spaces.size();
    return _gpu_memory_spaces[idx]->get_device_id();
  }

  auto counter      = _scan_round_robin.fetch_add(1);
  size_t target     = counter % total_available;
  size_t cumulative = 0;
  for (auto* space : _gpu_memory_spaces) {
    cumulative += space->get_available_memory();
    if (target < cumulative) {
      SIRIUS_LOG_DEBUG("Scan executor: distributing scan batch to GPU {} (available: {} bytes)",
                       space->get_device_id(), space->get_available_memory());
      return space->get_device_id();
    }
  }
  return _gpu_memory_spaces.back()->get_device_id();
}
```

**Phase 7 implication:** MGPU-07's implementation is already shipped. Phase 7 writes the **test**, not the implementation. The distribution log at `:177-179` is the evidence trail for the MGPU-07 audit-log pattern (mirrors Phase 5/6 pattern).

### Current `convert_gpu_to_gpu` in cucascade (UNCHANGED; reference shape for the Sirius override)

```cpp
// Source: cucascade/src/data/representation_converter.cpp:139-195 (cucascade f47de0b)
std::unique_ptr<idata_representation> convert_gpu_to_gpu(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  stream.synchronize();
  auto& gpu_source = source.cast<gpu_table_representation>();

  if (source.get_device_id() == target_memory_space->get_device_id()) {
    return source.clone(stream);
  }

  auto packed_data = cudf::pack(gpu_source.get_table(), stream);

  auto const target_device_id = target_memory_space->get_device_id();
  auto const source_device_id = source.get_device_id();
  auto const bytes_to_copy    = packed_data.gpu_data->size();
  auto mr                     = target_memory_space->get_default_allocator();

  auto target_stream = target_memory_space->acquire_stream();

  CUCASCADE_CUDA_TRY(cudaSetDevice(target_device_id));
  rmm::device_uvector<uint8_t> dst_uvector(bytes_to_copy, target_stream, mr);
  target_stream.synchronize();
  CUCASCADE_CUDA_TRY(cudaSetDevice(source_device_id));

  // cudaMemcpyPeerAsync is already here — no branch to add.
  CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(dst_uvector.data(),
                                         target_device_id,
                                         static_cast<const uint8_t*>(packed_data.gpu_data->data()),
                                         source_device_id,
                                         bytes_to_copy,
                                         stream.value()));
  stream.synchronize();
  CUCASCADE_CUDA_TRY(cudaSetDevice(target_device_id));
  rmm::device_buffer dst_buffer = std::move(dst_uvector).release();
  auto new_metadata = std::move(packed_data.metadata);
  auto new_gpu_data = std::make_unique<rmm::device_buffer>(std::move(dst_buffer));
  auto new_table_view =
    cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, target_stream, mr);
  target_stream.synchronize();
  CUCASCADE_CUDA_TRY(cudaSetDevice(source_device_id));

  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space));
}
```

**Phase 7 implication:** This body is what ships. The return-leg bug hypothesis: lines 166-170 leave the thread on `source_device_id` with `target_stream` (which is associated with target context) being used at L178 via `stream.value()` — not `target_stream.value()`. The `cudaMemcpyPeerAsync` succeeds but the `dst_uvector` might not be fully visible to target context before unpack at L186-188 if the cross-stream synchronization is incomplete. Forward leg works because source context was target context's entry state; return leg flips the asymmetry.

**Fix options:**
1. Sirius-side override using `target_stream` for the peer copy itself (see Pattern 2 earlier).
2. Or: verify Phase-7's peer-access enable loop alone fixes the return leg. If yes, no override needed.

### Converter Registry Override Registration

```cpp
// Source: src/sirius_extension.cpp:1053 + cucascade/include/cucascade/data/representation_converter.hpp:220
// Register Sirius P2P override ONLY if Pattern 2 return-leg fix needs it.

void LoadInternal(ExtensionLoader& loader)
{
  // ... existing ...
  sirius::converter_registry::initialize();
  // At this point cucascade::register_builtin_converters has registered the peer-async
  // convert_gpu_to_gpu. If Phase 7 needs the Sirius override:
  #ifdef SIRIUS_USE_P2P_OVERRIDE
  auto& registry = sirius::converter_registry::get();
  auto removed = registry.unregister_converter<
      cucascade::gpu_table_representation,
      cucascade::gpu_table_representation>();
  if (!removed) {
    spdlog::warn("sirius: expected cucascade built-in GPU->GPU converter to be registered "
                 "before override but unregister reported false (MGPU-06)");
  }
  registry.register_converter<
      cucascade::gpu_table_representation,
      cucascade::gpu_table_representation>(
      &sirius::data::sirius_p2p_converter_factory);
  #endif
  // ... existing ...
}
```

**Macro gate:** `SIRIUS_USE_P2P_OVERRIDE` only compiled-in if Step 1 of Plan 2 validates that the peer-access enable loop alone doesn't fix the return leg. Keep it conditional so Phase 7 can ship with the minimal change if cucascade's body is already correct post-enable.

### Peer-Access Enable Cache Declaration

```cpp
// Source: pattern mirrors src/include/sirius_context.hpp:204-205 gpu_io_backends declaration.
// Add to SiriusContext private members:

struct peer_pair_hash {
  size_t operator()(std::pair<int, int> const& p) const noexcept {
    return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
  }
};

std::unordered_set<std::pair<int, int>, peer_pair_hash> peer_access_enabled_pairs_;

// Public accessor (after get_gpu_io_backends at line 166):
[[nodiscard]] bool is_peer_access_enabled(int src, int dst) const noexcept {
  return peer_access_enabled_pairs_.count({src, dst}) > 0;
}
```

**Teardown:** `peer_access_enabled_pairs_.clear()` in `SiriusContext::terminate()` — matches the gpu_io_backends teardown pattern. No `cudaDeviceDisablePeerAccess` needed; CUDA cleans up at process exit.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cucascade host-staged `convert_gpu_to_gpu` (v1.0 baseline from pin 942c0bf) | Peer-async `cudaMemcpyPeerAsync` body | cucascade PR / pin `f47de0b` adopted in Phase 4 BUMP-01 | Phase 7 MGPU-06's "swap body" narrative is obsolete. Only the enable loop + return-leg fix remain. |
| Round-robin scan distribution (v1.0 MEM-05 open item) | Memory-weighted `select_target_gpu` in `duckdb_scan_executor` | Phase 2 v1.0 commit `5e8e9b7` (preserved through Phase 4 PORT-04) | Phase 7 MGPU-07's "implement distribution" narrative is obsolete. Phase 7 writes the test only. |
| `cudaDeviceEnablePeerAccess` loop in v1.0 SiriusContext (commit `dd86dd0`) | Deferred to Phase 7 per Plan 04-03 SUMMARY | Phase 4 Plan 04-03 decision | Phase 7 re-adds the loop. This is the only net-new Sirius code. |

**Deprecated/outdated narratives to ignore:**
- Any CONTEXT/PLAN text that says "swap host-staged to peer-async" — the swap happened at Phase 4 BUMP-01. Current code is peer-async.
- Any text that implies `duckdb_scan_executor::select_target_gpu()` uses round-robin in the common path — it uses round-robin only as fallback when all GPUs report 0 free bytes.

## Open Questions

1. **Is the GPU1→GPU0 return-leg bug fixed by the peer-access enable loop alone?**
   - What we know: the return leg fails on Phase-6 HEAD (Plan 04-05 Task 2 confirmed). Peer-access was NEVER enabled in Phase 4/5/6 — so `cudaMemcpyPeerAsync` inside cucascade's converter may be running without the peer-access bit set. On some driver versions this still succeeds via an internal fallback path; on others (or in combination with the Ada Lovelace Sapphire Rapids issue — Pitfall 2), it corrupts data or errors on the return leg specifically.
   - What's unclear: did v1.0's `dd86dd0` enable-loop (deferred to Phase 7) make the peer-async body function correctly on v1.0's HW? If yes, Phase 7's enable loop alone is the whole fix.
   - **Recommendation:** Plan 7's Wave 1 first step is a **probe plan**: add the peer-access enable loop (Pattern 1), rebuild, run `[.][mem_04_p2p_transfer]` on N=2 host. If PASS, Plan 2's scope shrinks to "un-hide tests + nsys + bandwidth evidence". If FAIL, Plan 2 expands to add the Sirius override (Pattern 2).

2. **Does the N=2 host `6f7e4c9-lcedt` have a Sapphire-Rapids-class CPU that risks silent P2P corruption (Pitfall 2)?**
   - What we know: 2× RTX 6000 Ada (Ada Lovelace generation — affected GPU arch per NVIDIA forum).
   - What's unclear: CPU model — Phase 6 validation evidence doesn't include `lscpu` output.
   - **Recommendation:** Plan 7's Task 1 (infrastructure prep) runs `lscpu | grep -i "model name"` on `6f7e4c9-lcedt` and documents. If Sapphire Rapids or later Xeon SP, add a checksum-based data-integrity assertion to the round-trip test (not just size-in-bytes equality) — see the Phase-7 CONTEXT success criterion 2 gate.

3. **Does `cudaDeviceCanAccessPeer(0, 1)` return 1 on `6f7e4c9-lcedt`?**
   - What we know: PCIe-only (no NVLink on RTX 6000 Ada consumer cards). PCIe P2P on RTX 6000 Ada **is possible** per NVIDIA docs but depends on motherboard, BIOS (IOMMU off or passthrough), and driver config (595.58.03 is a late-2025 driver — should support).
   - What's unclear: actual return value on this specific host.
   - **Recommendation:** Plan 7's Task 1 runs a 5-line standalone probe first. If can_access==0 in both directions, Phase 7 MGPU-06's P2P gate becomes documentation-only ("host doesn't support P2P — fallback path is the tested path; bandwidth gate of 1.5× N/A"). Phase still ships because fallback correctness is still testable.

4. **Does Sirius need a test-only hook to force the fallback path (Pattern 3 / CONTEXT §Decisions no-P2P fallback test)?**
   - What we know: CONTEXT locks the requirement — "force no-peer-access via `CUDA_VISIBLE_DEVICES` or topology masking in test".
   - What's unclear: `CUDA_VISIBLE_DEVICES=0` reduces to 1 GPU and triggers WARN+return (doesn't test the fallback code); masking peer-access at runtime requires a Sirius-owned kill-switch.
   - **Recommendation:** Plan 4 adds `SiriusContext::disable_peer_access_for_testing()` (private + test-friend, or under `#ifdef SIRIUS_TEST_HOOKS`). Test flips it, runs converter, asserts correctness, restores.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CUDA 13.0 runtime | `cudaMemcpyPeerAsync`, `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess` | ✓ (pixi env) | 13.2 driver 595.58.03 on N=2 host | — |
| cucascade submodule | Converter + memory_space APIs | ✓ | Pinned `f47de0b` | — (submodule bump out of scope) |
| nsys | Bandwidth measurement, trace evidence | ✓ | `/usr/local/cuda-13.0/bin/nsys` on N=2 host (per Phase 6) | — |
| compute-sanitizer | Regression check for new peer-enable path | ✓ | 2025.3.1.0 at `/usr/local/cuda-13.0/bin/compute-sanitizer` (per Phase 6 SUMMARY §"Phase 7 unblockers") | — |
| N=2 GPU host | Real P2P execution | ✓ | `6f7e4c9-lcedt`, 2 × RTX 6000 Ada | — (single-GPU hosts use Catch2 `WARN+return`) |
| `numactl` | `/proc/PID/numa_maps` + `numactl --show` for MGPU-05 evidence reuse | ✓ | on N=2 host | — |
| `lscpu` | CPU model identification for Pitfall 2 | ✓ | standard | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None — all Phase 7 needs are already in place from Phase 4/5/6.

**Current worktree host (`6f7e4c9-lcedt` validation happens on a different machine):** This worktree has no NVIDIA driver (`nvidia-smi` returns "couldn't communicate with the NVIDIA driver") and reports single-NUMA `node 0`. All GPU-touching validation must run on the N=2 verification host, same as Phase 5 Plan 05-06 and Phase 6 Plan 06-04. Use the `mcp__project-commands__run_command` pattern with `dangerouslyDisableSandbox` when the task binary needs GPU access.

## Project Constraints (from CLAUDE.md)

Extracted directives Phase 7 plans must satisfy:

| Directive | Source | Phase 7 Impact |
|-----------|--------|----------------|
| Build via `mcp__project-commands__run_command`, NOT `pixi run` / `make` directly | CLAUDE.md + MEMORY.md | All Plan 7 build/test invocations go through MCP. |
| No `rmm::cuda_stream_default` | CLAUDE.md + MEMORY.md | The peer-access enable loop + Sirius override MUST use `rmm::cuda_stream` or a stream acquired from `memory_space->acquire_stream()`. Pre-commit gate should include `grep -c 'cuda_stream_default' <touched files> == 0`. |
| Super Sirius only (`namespace sirius`) | CLAUDE.md §Super Sirius | The Sirius-side override factory lives under `namespace sirius::data` or `namespace sirius`, NOT `namespace duckdb`. |
| cuCascade API for all disk I/O + tier conversion | CLAUDE.md §Constraints | Phase 7's converter override goes through the cucascade `representation_converter_registry` — no hand-rolled dispatch. |
| `/module-context` before implementation | CLAUDE.md §"Loading Library Context" | Before Plan 2 executor work, run `/module-context MGPU-06 P2P converter override using cucascade representation_converter_registry + cudaMemcpyPeerAsync`. |
| Feature branches for GSD work | MEMORY.md | Phase 7 work stays on current feature branch `feature/single-node-multi-gpu2` — do NOT merge to dev until milestone closes. |
| Test framework: Catch2 v2, `WARN+return` idiom (not SKIP) | CLAUDE.md §Testing + Plan 01-03 decision | Phase 7 tests follow the existing convention. |
| Feature-branch / no-verify commits under Wave parallel execution | Phase 4-6 pattern | Continue this pattern for Phase 7 Wave 1 parallel code plans. |

## Sources

### Primary (HIGH confidence — direct source reads)

- `cucascade/src/data/representation_converter.cpp:139-195` — `convert_gpu_to_gpu` body with explicit `cudaMemcpyPeerAsync` at L173
- `cucascade/src/data/representation_converter.cpp:1461-1506` — `register_builtin_converters` registering `convert_gpu_to_gpu` at L1464-1465
- `cucascade/include/cucascade/data/representation_converter.hpp:97-245` — `representation_converter_registry` full class including `register_converter`, `has_converter`, `unregister_converter`, `clear`
- `cucascade/include/cucascade/memory/reservation_manager_configurator.hpp:45-226` — builder API (`set_gpu_usage_limit`, `use_host_per_gpu`, `use_host_per_numa`); confirmed no per-GPU capacity setter — Pitfall 5
- `cucascade/include/cucascade/memory/topology_discovery.hpp:1-100` — `system_topology_info` with `num_gpus`, `num_numa_nodes`, `gpus[]` list
- `cucascade/src/memory/memory_space.cpp:247,261` — `memory_space::get_available_memory()` (stream-aware and stream-free overloads) confirmed
- `src/op/scan/duckdb_scan_executor.cpp:151-184` — `select_target_gpu()` memory-weighted distribution already implemented
- `src/op/scan/duckdb_scan_executor.cpp:278` — `manager_loop` calls `select_target_gpu()` per parquet task (real-time query per batch)
- `src/sirius_context.cpp:168-254` — current `SiriusContext::initialize()` including topology fail-hard (L184-196), per-NUMA host-space assertion (L210-225), and per-GPU io_backend init (L235-254) — the location to insert peer-access enable loop is between L225 and L235
- `src/include/sirius_context.hpp:190-217` — private members block where `peer_access_enabled_pairs_` goes; `io_backend_registry_` and `gpu_io_backends_` reference pattern at L204-205
- `src/include/data/sirius_converter_registry.hpp:1-99` — Sirius singleton around cucascade's `representation_converter_registry`
- `src/sirius_extension.cpp:1044-1065` — `LoadInternal` including `converter_registry::initialize()` at L1053 (the override registration site)
- `src/pipeline/gpu_pipeline_executor.cpp:54-72` — Phase 6 device-guard pattern (inline `cudaError_t` check + `spdlog::error` in `noexcept` lambda) — the convention Phase 7 follows for peer-access error reporting
- `test/cpp/downgrade/test_downgrade_executor.cpp:485` — `[.][multi_gpu_transfer]` round-trip test (un-hide target)
- `test/cpp/downgrade/test_downgrade_executor.cpp:805-872` — `[.][mem_04_p2p_transfer]` placeholder with `TODO(MGPU-06)` at L813 (un-hide target; body already round-trips)
- `test/cpp/downgrade/test_downgrade_executor.cpp:874-921` — `[.][mem_05_scan_distribution]` placeholder with `TODO(MGPU-07)` at L881 (un-hide + replace body with asymmetric-memory fixture)
- `test/cpp/config/test_context.cpp:333` — `[.][multi_gpu_foundation][mgpu_04_round_trip]` forward-leg test (un-hide target; optionally append return leg)
- `test/cpp/integration/test_gpu_execution_locality.cpp:205-229` — `[.][data_locality][multi_gpu]` (CONTEXT says "extend with P2P + adaptive-scan scenarios")
- `.planning/phases/06-multi-gpu-gap-closure-*/06-RESEARCH.md` Finding 2 — earlier confirmation that cucascade converter is peer-async (L111-129)
- `.planning/phases/06-multi-gpu-gap-closure-*/06-SUMMARY.md` §"Phase 7 unblockers" — N=2 host details, nsys location, compute-sanitizer version
- `.planning/phases/04-cucascade-bump-v1-0-re-integration/04-05-SUMMARY.md` §"Deferred Issues" — 2 of 5 hidden tags fail on GPU1→GPU0 return leg (the Phase 7 regression anchor)

### Secondary (MEDIUM confidence — web search + cross-reference)

- NVIDIA Developer Forums — Ada Lovelace PCIe P2P behavior: posted-write ordering dependency + Sapphire Rapids silent corruption warning (see Pitfall 2). Cross-referenced with NVIDIA driver documentation but not verified on specific-CPU-model basis for `6f7e4c9-lcedt`.
- `https://forums.developer.nvidia.com/t/cudamemcpypeerasync-behavior-for-different-hardware/292239` — behavioral differences between NVLink and non-NVLink systems

### Tertiary (LOW confidence — not verified)

- Exact CPU model of `6f7e4c9-lcedt` (needs `lscpu` on host — Open Question 2)
- Actual `cudaDeviceCanAccessPeer(0, 1)` return value on `6f7e4c9-lcedt` (needs runtime probe — Open Question 3)
- Whether the Sirius-side converter override is actually needed, or if the peer-access enable loop alone fixes the return-leg bug (Open Question 1 — requires execution)

## Metadata

**Confidence breakdown:**

- **Standard stack:** HIGH — direct header/source inspection on current working tree + cucascade pin `f47de0b`
- **Architecture patterns:** HIGH — patterns mirror Phase 5 Plan 05-03 and Phase 6 Plan 06-02 which are verified shipped
- **Pitfalls:**
  - Pitfall 1 (stale narrative): HIGH — verified by direct cucascade source read
  - Pitfall 2 (Sapphire Rapids): MEDIUM — NVIDIA-documented but specific to host CPU model which is not yet probed
  - Pitfall 3 (stream-context semantics): MEDIUM — derived from CUDA API docs + inspection of cucascade converter body; not verified by experiment
  - Pitfall 4 (oscillation): MEDIUM — logically sound but not observed in current codebase
  - Pitfall 5 (asymmetric GPU capacity): HIGH — direct configurator header inspection
- **Code examples:** HIGH — all are verbatim from source or trivial extensions of verbatim patterns
- **Environment availability:** HIGH for N=2 host (Phase 5/6 evidence) + current worktree (direct probes)
- **Open questions:** explicitly flagged as needing runtime verification on N=2 host before Phase 7 Plan 2 writes the override

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (30 days — stable submodule pin; CUDA API stable; the only volatile bit is whether cucascade gets a patch upstream that changes the converter body, which is out-of-scope this phase)

**Sources (for the "Sources:" protocol requirement of WebSearch):**
- [cudaMemcpyPeerAsync behavior for different hardware — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/cudamemcpypeerasync-behavior-for-different-hardware/292239)
- [NVIDIA RTX 6000 Ada Generation — NVIDIA](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)
- [NVIDIA Data Center GPU Driver 570.195.03 release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-570-195-03/index.html)

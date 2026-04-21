# Phase 6: Multi-GPU Gap Closure — Research

**Researched:** 2026-04-21
**Domain:** Multi-GPU runtime topology, device-guard safety, per-NUMA pinned host memory, GPU↔GPU converter
**Confidence:** HIGH
**dev HEAD baseline SHA:** `484db3509c395646a7c5cfd0543e860fa2e9cd9b` (confirmed via `git rev-parse origin/dev` — identical to the Phase 4 init SHA `484db35 Add extension-ci-tools distribution workflow (#621)`)
**cucascade pin:** `f47de0bb7bcaddd55081a9c4bc584627532d1ef9` (confirmed via submodule `HEAD`)

## Summary

Phase 6 closes MGPU-01..05 on `feature/single-node-multi-gpu2` rebased on `dev@484db35`. The unique finding that reshapes the plan: **the five "structural gaps" listed in the CONTEXT.md are partially closed already** — topology discovery runs in `sirius_config.cpp:267`, per-NUMA host allocation is the cucascade default (`make_default_host_memory_resource` returns a `numa_region_pinned_host_memory_resource`), GPU↔GPU conversion is already registered (`register_builtin_converters` installs a `cudaMemcpyPeerAsync`-based `convert_gpu_to_gpu`), and per-thread device guards already exist on every Super-Sirius GPU entry point. Phase 6 is primarily an **audit + enforcement + logging + test** phase, not a "stand up from scratch" phase. Three specific code deltas remain: (1) collapse the confusing dual path between `SiriusContext::get_hw_topology()` and `config_.get_hw_topology()` into one authoritative accessor + add the required startup log, (2) replace the unguarded `cudaSetDevice(device_id)` calls in `gpu_pipeline_executor.cpp:58` and `downgrade_executor.cpp:61` with CUDA-TRY variants (or document why the current pattern is safe on N=2), (3) explicitly invoke `use_host_per_numa()` in the default configurator path so MGPU-05's "one host per NUMA" holds even when the user does not set `host.numa_id` in YAML.

**Primary recommendation:** Do NOT re-plumb topology discovery or re-register the GPU↔GPU converter. Audit-and-log for MGPU-01, assert-and-log for MGPU-05, add enforcement + regression test for MGPU-03 on the existing code, measure wall-clock for MGPU-02, and add explicit converter-registration + round-trip tests to MGPU-04 (the hidden-test failure from Phase 4 is a device-guard bug in the GPU→GPU round-trip path, not a "missing converter" bug).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Topology Discovery Integration (MGPU-01)**
- Call `cucascade::topology_discovery` inside `SiriusContext::initialize()` early — before `memory_manager_` construction — because topology drives memory-space layout (per-NUMA host spaces need NUMA count; per-GPU memory spaces need GPU count).
- Cache once per context lifetime on `SiriusContext` (mirror the `gpu_io_backends_` cache pattern Plan 05-03 established at `src/sirius_context.cpp` lines 185-204).
- Expose via `SiriusContext::get_topology() const` returning a const ref to the cached struct.
- Log a summary at **`info` level** on startup: GPU count, per-GPU NUMA domain, GPU→NUMA map (explicit per success criterion 1).
- On topology-discovery failure, **fail-hard**: throw from `initialize()`. No fallback to `cudaGetDeviceCount` / `numa_node_of_cpu`.

**Hand-Rolled CUDA/NUMA Sweep (MGPU-01 gate)**
- Scope: all of `src/` — success criterion 1 requires `grep -rn 'cudaGetDeviceCount\|numa_node_of_cpu' src/` == zero hits outside the cucascade bridge.
- Scope exclusion: `test/` may retain these as test fixtures; only `src/` is swept.
- Replacement pattern: every callsite routes through `SiriusContext::get_topology()` (or an equivalent accessor built on the cached struct).
- Cucascade-bridge exception: cucascade internally calls `cudaGetDeviceCount`; Sirius consumes its topology result, not the raw API.

**GPU↔GPU Converter (MGPU-04)**
- Register a **host-staged converter** (GPU0 → pinned host buffer → GPU1) in `converter_registry::initialize()` at `src/sirius_extension.cpp:1053`, alongside the existing tier converters.
- The registered converter satisfies the registration gate (`converter_registry::instance().has_converter(Tier::GPU, Tier::GPU) == true`) and passes the round-trip correctness test.
- Phase 7 (MGPU-06) will replace the body with `cudaMemcpyPeerAsync` when P2P is available.
- Leave Phase-4 deferred items in place: `test_downgrade_executor.cpp:813 TODO(MGPU-06)` + hidden tags stay off-by-default until Phase 7.

**Per-NUMA Host Memory Spaces (MGPU-05)**
- One host space per NUMA domain, allocator = cucascade `numa_region_pinned_host_allocator`.
- `memory_manager_` constructor iterates the cached topology's NUMA list and builds one host space per node.
- Validation: `memory_manager_->get_memory_spaces_for_tier(Tier::HOST).size() == topology.num_numa_nodes` + spot-check via `/proc/PID/numa_maps`.

**Single-GPU Regression Gate (MGPU-02)**
- Baseline: current `dev` HEAD.
- Measurement: `python3 test/tpch_performance/performance_test.py 10`, captured in phase SUMMARY.
- Threshold: 5% wall-clock regression on TPC-H SF10 end-to-end, 3-run median, same build flags as baseline.

**Device-Guard Audit (MGPU-03)**
- Run `compute-sanitizer --tool memcheck --require-cuda-init build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation]"` on the N=2 host.
- Also run `[integration][gpu_execution][parquet][join]` subset.
- Zero "invalid device" / "context mismatch" errors required.

### Claude's Discretion

- Exact struct shape of the `SiriusContext::get_topology()` return — reuse existing `cucascade::memory::system_topology_info` vs wrap in a Sirius-side accessor struct (specifics section of CONTEXT mentioned `gpu_count()`, `numa_count()`, `gpu_numa_node(int)`, `gpus_for_numa(int)` helpers).
- Whether to add a Sirius-side MGPU-04 round-trip unit test, and whether it runs in `[multi_gpu_foundation]` (already present) or a new tag.
- Which of the existing `cudaSetDevice` callsites in the Super-Sirius path should be upgraded to `rmm::cuda_set_device_raii` or `CUCASCADE_CUDA_TRY` vs left as-is with audit-log justification.

### Deferred Ideas (OUT OF SCOPE)

- P2P direct `cudaMemcpyPeerAsync` converter body — Phase 7 (MGPU-06). *(See "Finding 2" below — cucascade `f47de0b` already implements this; the Phase 7 work is Sirius-side orchestration + fallback, not the converter body.)*
- Adaptive scan partitioning by available GPU memory — Phase 7 (MGPU-07).
- Cross-GPU converter return-leg fix (`test_downgrade_executor.cpp:813 TODO(MGPU-06)`) — Phase 7.
- Hidden multi-GPU tests (`[.][multi_gpu_transfer]`, `[.][mem_04_p2p_transfer]`) — stay off-by-default until Phase 7 closes the return-leg bug.
- Phase-4-vs-Phase-5 SF10 regression comparison (Phase 5's IO-10) — still deferred.
- `test/` directory hand-rolled CUDA/NUMA sweep — only `src/` in Phase-6 scope.
- Per-backend NUMA affinity tuning of `pipeline_io_backend`'s pinned host buffers.
- Cucascade upstream changes — Phase 6 consumes the `f47de0b` pinned API as-is.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MGPU-01 | Runtime topology discovery via cucascade `topology_discovery` — GPU count, NUMA domains, GPU↔NUMA mapping | Already called at `src/sirius_config.cpp:267-268`, cached in `sirius_config::_hw_topology`, exposed via `SiriusContext::get_hw_topology()` (header line 117) and `sirius_config::get_hw_topology()`. CONTEXT locks "fail-hard on discovery failure" (currently a no-op when `discover()` returns false — hardcoded `_hw_topology{.num_gpus = 1}` default). Startup info log missing; hand-rolled sweep is the net-new gate. |
| MGPU-02 | Single-GPU TPC-H SF10 wall-clock within 5% of current `dev` HEAD | CONTEXT says `performance_test.py 10`; BUT that script wraps queries in `call gpu_processing(...)` (legacy path). The Super-Sirius path is exercised by `test/tpch_performance/run_tpch_parquet.sh sirius 10` + `tpch_queries/gpu/q*.sql` (use `gpu_execution`). Phase 5 measured via `/tmp/phase5-validation/sirius-sf10.yaml` + 1-GPU config. See Pitfall 1 below. |
| MGPU-03 | Device-guard enforcement on every execution thread — compute-sanitizer memcheck 0 errors | Existing device-guard pattern documented in comment block at `test/cpp/config/test_context.cpp:110-148`. Live code sites: `gpu_pipeline_executor.cpp:58`, `downgrade_executor.cpp:61` (raw `cudaSetDevice`, return values discarded); `gpu_pipeline_executor.cpp:65` and `downgrade_executor.cpp:52` use `rmm::cuda_set_device_raii`; per-GPU io_backend construction (`sirius_context.cpp:192`) uses `rmm::cuda_set_device_raii`. Hidden-test failure at `test_downgrade_executor.cpp:813` fails on GPU1→GPU0 return leg — likely a device-guard gap in the `convert_gpu_to_gpu` call chain inside `batch_lock_utils.hpp:87` (the caller's stream belongs to the source GPU, not the target). |
| MGPU-04 | GPU↔GPU representation converter registered in cucascade converter registry | **Already registered** — `cucascade::register_builtin_converters` at `cucascade/src/data/representation_converter.cpp:1461-1465` registers `convert_gpu_to_gpu`, which uses `cudaMemcpyPeerAsync` directly (not host-staged). Sirius calls this via `converter_registry::initialize()` at `src/sirius_extension.cpp:1053` → `sirius::converter_registry::initialize` → `cucascade::register_builtin_converters(*instance_)`. Existing assertion test at `test/cpp/config/test_context.cpp:241-255`. **The "host-staged body" in CONTEXT §Decisions is a mis-statement of the current state** — see Finding 2. |
| MGPU-05 | Per-NUMA host memory spaces configured with `numa_region_pinned_host_allocator` | **Already wired by default** — `cucascade::memory::make_default_host_memory_resource` at `cucascade/src/memory/common.cpp:36-40` returns `numa_region_pinned_host_memory_resource(numa_node_id)`. The `reservation_manager_configurator` has `.use_host_per_numa()` and `.use_host_per_gpu()` builder methods; current Sirius YAML-path / default-path uses whichever the user sets (`apply_defaults` in `sirius_config.cpp:271-285` does NOT call either — it relies on the configurator's default, which creates a single host space per `host_capacity`). MGPU-05 net-new work: call `.use_host_per_numa()` explicitly in `apply_defaults` and validate count-matches-numa-count. |
</phase_requirements>

## Finding 1 — Topology discovery is already wired; the gap is logging + enforcement

**What exists (src/sirius_config.cpp:265-269):**
```cpp
sirius_config::sirius_config()
{
  cucascade::memory::topology_discovery discovery;
  if (discovery.discover()) { _hw_topology = discovery.get_topology(); }
}
```

**What exists (src/include/sirius_context.hpp:117-120):**
```cpp
[[nodiscard]] const cucascade::memory::system_topology_info& get_hw_topology() const noexcept
{
  return config_.get_hw_topology();
}
```

**What is missing per CONTEXT locks:**
1. **Fail-hard on discovery failure.** Current code silently retains the default `_hw_topology{.num_gpus = 1}` (sirius_config.hpp:111). CONTEXT locks: throw from `initialize()` when `discover()` returns false.
2. **Info-level startup log summarizing topology.** No log currently emitted.
3. **Hand-rolled sweep gate.** Need to prove `grep -rn 'cudaGetDeviceCount\|numa_node_of_cpu' src/` returns zero hits **outside the cucascade bridge and the legacy `src/cuda/` path** (see Finding 5).
4. **Ordering.** CONTEXT says "call topology discovery early, BEFORE memory_manager_ construction." Today the discovery runs in `sirius_config`'s constructor (which happens when `SiriusContextExtensionCallback::read_config_file_if_exists` instantiates a default `sirius_config`), long before `SiriusContext::initialize()` is ever called. The CONTEXT ordering constraint is already honored — what's missing is the explicit, visible call in `SiriusContext::initialize()` to re-validate the cached topology and to emit the info log.

**Plan implication:** MGPU-01's "add topology discovery call" is a misnomer. The actual plan item is: **validate the cached topology at `SiriusContext::initialize()` entry, throw if `num_gpus == 0`, emit info-level summary, add the src/ sweep as a grep gate.** Do NOT add a second `topology_discovery` call — that would re-query NVML/sysfs unnecessarily.

## Finding 2 — GPU↔GPU converter body is ALREADY `cudaMemcpyPeerAsync`, not host-staged

**Source:** `cucascade/src/data/representation_converter.cpp:137-195` (reviewed line-by-line)

```cpp
// Line 139-195: convert_gpu_to_gpu implementation
std::unique_ptr<idata_representation> convert_gpu_to_gpu(...) {
  stream.synchronize();
  auto& gpu_source = source.cast<gpu_table_representation>();
  if (source.get_device_id() == target_memory_space->get_device_id()) {
    return source.clone(stream);
  }
  auto packed_data = cudf::pack(gpu_source.get_table(), stream);
  auto target_stream = target_memory_space->acquire_stream();
  CUCASCADE_CUDA_TRY(cudaSetDevice(target_device_id));
  rmm::device_uvector<uint8_t> dst_uvector(bytes_to_copy, target_stream, mr);
  target_stream.synchronize();
  CUCASCADE_CUDA_TRY(cudaSetDevice(source_device_id));
  CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(dst_uvector.data(),
                                         target_device_id,
                                         static_cast<const uint8_t*>(packed_data.gpu_data->data()),
                                         source_device_id,
                                         bytes_to_copy,
                                         stream.value()));
  stream.synchronize();
  // ... unpack on target device ...
}

// Line 1461-1465: registration
void register_builtin_converters(representation_converter_registry& registry) {
  registry.register_converter<gpu_table_representation, gpu_table_representation>(convert_gpu_to_gpu);
  // ... HOST↔GPU, HOST↔HOST, GPU↔DISK, DISK↔HOST, HOST_FAST variants ...
}
```

**Sirius already consumes this** via `src/include/data/sirius_converter_registry.hpp:52`:
```cpp
instance_ = std::make_unique<registry_type>();
cucascade::register_builtin_converters(*instance_);     // <-- registers gpu_to_gpu (peer-async)
sirius::register_parquet_converters(*instance_);
```

And the existing assertion test at `test/cpp/config/test_context.cpp:241-255` (tag `[multi_gpu_foundation]`) passes today:
```cpp
TEST_CASE("converter_registry has gpu_to_gpu converter (MEM-03)", "[multi_gpu_foundation]") {
  sirius::converter_registry::reset_for_testing();
  sirius::converter_registry::initialize();
  auto& registry = sirius::converter_registry::get();
  bool has_gpu_to_gpu = registry.has_converter<cucascade::gpu_table_representation,
                                                cucascade::gpu_table_representation>();
  REQUIRE(has_gpu_to_gpu);
}
```

**Plan implication:** The CONTEXT §Decisions language "Register a host-staged converter (GPU0 → pinned host buffer → GPU1)" reflects an outdated assumption. The actual Phase 6 work is:
- **DO NOT** register a second GPU↔GPU converter — that would throw `std::runtime_error("already exists")` per `representation_converter.cpp` dedup check.
- **DO** add an explicit round-trip correctness test (GPU0 → convert → GPU1 → convert back → bytes-equal) in `[multi_gpu_foundation]` on the N=2 host.
- **DO** diagnose and close the hidden-test GPU1→GPU0 return-leg bug (from Phase 4's `test_downgrade_executor.cpp:813 TODO(MGPU-06)`) — CONTEXT says this is Phase 7 scope. **Research recommends this stay Phase 7** because the return-leg bug is almost certainly inside cucascade's `convert_gpu_to_gpu` or in the upstream stream/event plumbing, which requires either a cucascade upstream PR or a Sirius-side wrapper that fixes the device/stream at call time. The Phase-4 deferral is correctly scoped.
- The **Phase 7 MGPU-06 work** is therefore NOT "swap host-staged → peer-async" (the body is already peer-async) but "replace or wrap `convert_gpu_to_gpu` to fix the GPU1→GPU0 return-leg device-guard bug AND add a P2P-vs-host-staging policy for GPUs that lack `cudaDeviceCanAccessPeer == 1`."

**Recommendation for planner:** Update CONTEXT.md (or mirror the update in the phase SUMMARY) to correct the "register a host-staged converter" wording. The MGPU-04 gate as stated ("registry entry exists + round-trip test passes") is achievable without any new converter code.

## Finding 3 — Per-NUMA host memory resource is the cucascade default

**Source:** `cucascade/src/memory/common.cpp:36-40`:
```cpp
std::unique_ptr<rmm::mr::device_memory_resource> make_default_host_memory_resource(
  int numa_node_id, [[maybe_unused]] size_t capacity)
{
  return std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(numa_node_id);
}
```

`numa_region_pinned_host_memory_resource` (cucascade/include/cucascade/memory/numa_region_pinned_host_allocator.hpp:29) is constructed with a `numa_node` int and derives from `rmm::mr::device_memory_resource` with both `cuda::mr::host_accessible` and `cuda::mr::device_accessible` properties. The impl is in `cucascade/src/memory/numa_region_pinned_host_allocator.cpp` and uses `libnuma` (`find_library(NUMA_LIB numa REQUIRED)`).

**Builder API** (`cucascade/include/cucascade/memory/reservation_manager_configurator.hpp:109-112`):
```cpp
builder_reference& use_host_per_gpu();       // one host space per GPU device (shares numa_id)
builder_reference& use_host_per_numa();      // one host space per distinct NUMA node
```

**Current Sirius behavior:**
- `src/sirius_config.cpp:271-285` (`apply_defaults`) builds a configurator with default settings (neither `use_host_per_gpu` nor `use_host_per_numa` explicitly) and calls `builder.build(_hw_topology)`. Depending on cucascade's default policy, the host-space-creation policy defaults to `_host_creation_policy{}` (line 217 of configurator header — the default-constructed variant holds `bind_cpu_to_gpu_numa`, which is `use_host_per_numa`).
- `spaces.yaml` / `configurator.yaml` test fixtures explicitly set `numa_id` per-space.
- Plan 04-03 showed `config_.get_hw_topology().gpus[i].numa_node` already flows into `downgrade_executor_config.preferred_numa_node` (src/sirius_context.cpp:242-251).

**Plan implication:** MGPU-05 is essentially **already satisfied by the cucascade defaults**. The net-new work is:
1. **Explicitly** call `.use_host_per_numa()` in `apply_defaults` so the intent is visible at the Sirius layer (not relying on cucascade's private default).
2. Add an init-time assertion + log: `host_spaces.size() == topology.num_numa_nodes` (or at least `host_spaces.size() >= 1` when num_numa_nodes is `0` on non-NUMA hosts).
3. Add `/proc/PID/numa_maps` spot-check evidence to the phase SUMMARY, on the N=2 real-hardware host.

## Finding 4 — Device-guard coverage in Super Sirius is already near-complete; enforcement is the gap

**Evidence (comment block at test/cpp/config/test_context.cpp:110-148):**
This comment block is a live audit of GPU thread entry points in Super Sirius. It documents:
1. `gpu_pipeline_executor::get_per_thread_init()` → `cudaSetDevice(device_id)` (line 58)
2. `gpu_pipeline_executor::manager_loop()` → `rmm::cuda_set_device_raii` (line 65)
3. `downgrade_executor` per-thread init → `cudaSetDevice(device_id)` (line 61)
4. `downgrade_executor::start()` stream-pool construction → `rmm::cuda_device_id` (line 52)
5. `sirius_memory_reservation_manager` constructor (via cucascade) → `rmm::cuda_set_device_raii`
6. `duckdb_scan_executor` — host-side only, GPU uploads go through converters
7. Legacy `src/cuda/*.cu` — `namespace duckdb` legacy path, not Super Sirius

**Raw `cudaSetDevice` callsites in `src/` (full list):**
| File | Line | Context | Return check? |
|------|------|---------|---------------|
| `src/pipeline/gpu_pipeline_executor.cpp` | 58 | per-thread init | NO (return value discarded) |
| `src/downgrade/downgrade_executor.cpp` | 61 | per-thread init | NO (return value discarded) |
| `src/cuda/allocator.cu` | 65, 84, 118, 121, 126, 129 | **Legacy `namespace duckdb`** | Mixed (some gpuErrchk'd) |
| `src/cuda/communication.cu` | 97, 100, 115, 122, 138, 145, 160, 164 | **Legacy `namespace duckdb`** | NO |

**Raw `cudaGetDeviceCount` callsites in `src/`:** exactly **one**, at `src/cuda/allocator.cu:70` (legacy path).

**Raw `numa_node_of_cpu` / `numa_available`:** **zero hits in `src/`**.

**Plan implication:** The MGPU-01 "sweep" gate is simpler than CONTEXT implies. Only two files have hits that need a decision:
- **Super-Sirius path (in scope for this phase):** none. All `cudaSetDevice` calls are derived from `memory_space->get_device_id()` (a per-executor captured value), which is safe against multi-threaded context drift but silently drops errors. MGPU-03 improvement: wrap both in `CUCASCADE_CUDA_TRY` or `gpuErrchk`, or convert to `rmm::cuda_set_device_raii` for symmetry with the adjacent `manager_loop`.
- **Legacy path (out of v1.1 scope per PROJECT.md):** `src/cuda/*` is `namespace duckdb` legacy code. Phase 5's HYG-02 left these untouched as "frozen-path hygiene debt" per the Phase 5 SUMMARY §Deferred Items. The sweep grep should exclude `src/cuda/` or the phase SUMMARY must document why those hits are acceptable.

**Recommendation:** The sweep gate should be phrased as: "`grep -rn 'cudaGetDeviceCount\|numa_node_of_cpu' src/` returns zero hits **in Super-Sirius files** (i.e. excluding `src/cuda/`)." Document the `src/cuda/` exclusion in the phase SUMMARY.

## Finding 5 — Dev HEAD baseline + single-GPU measurement tool

**dev HEAD SHA:** `484db3509c395646a7c5cfd0543e860fa2e9cd9b` ("Add extension-ci-tools distribution workflow (#621)"). `git fetch origin dev && git rev-parse origin/dev` confirms this — same SHA that Phase 4 used as its baseline anchor.

**`dev..HEAD`:** 30 commits (Phase 4 + Phase 5 work + current Phase 6 smart-discuss commit). A 1-GPU run of the current branch is the "post" reading for MGPU-02.

**Tool choice (diverges from CONTEXT):** The CONTEXT specifies `python3 test/tpch_performance/performance_test.py 10`. That script wraps queries in `call gpu_processing(...)` (see `performance_test.py:28-32`), which hits the **legacy `namespace duckdb` path** — not Super Sirius. It will NOT exercise the topology / device-guard / host-space code paths that Phase 6 is validating, so its timings are not meaningful for MGPU-02 as scoped.

**Correct tool:** `test/tpch_performance/run_tpch_parquet.sh sirius 10 $(seq 1 22)`, which executes the `tpch_queries/gpu/q*.sql` files that contain `call gpu_execution(...)`. This is the tool Phase 5 actually used — evidence lives in `/tmp/phase5-validation/sf10-phase5.log` and `sf10-phase5-v2.log`.

**Plan recommendation:** Either (a) update the plan to use `run_tpch_parquet.sh` with the `gpu_execution` path, or (b) modify `performance_test.py:32` to say `gpu_execution` instead of `gpu_processing`. **Recommend (a)** because Phase 5's absolute SF10 numbers in `05-06-MULTIGPU-VALIDATION.md` (1-GPU Q1=1.273s, Q6=0.233s, Q12=0.717s) are the direct predecessor the Phase 6 reading compares against — using the same shell script makes the comparison apples-to-apples.

**Baseline capture note:** The CONTEXT locks "baseline = current `dev` HEAD" — which means MGPU-02 requires running `run_tpch_parquet.sh sirius 10` on `484db35` itself as the "before" reading, then on current HEAD as the "after" reading. If the dev HEAD run is skipped (because Phase 4/5 already captured SF10 timings on the multi-GPU branch), the comparison is relative to a different baseline than what the requirement specifies. **Recommend: use Phase-5 SF10 timings as the baseline** and document that choice in the phase SUMMARY, rather than re-checking out `484db35` for an independent measurement. This is also consistent with the user directive on 2026-04-21 ("we don't need to run any comparisons, let's just make sure everything is working").

## Finding 6 — The "host-staged" language in CONTEXT §MGPU-04 is inconsistent with the reference pointer

CONTEXT §code_context "Host-staged converter implementation path" points to `cucascade/src/data/representation_converter.cpp` as the reference for the host-staged shape. That file implements `convert_gpu_to_gpu` as **direct peer async** (see Finding 2). The `convert_gpu_to_host` at line 200 is the host-staged shape, but it operates on GPU→HOST, not GPU→GPU. There is no host-staged GPU↔GPU template in cucascade as written.

**Two interpretations:**
- **CONTEXT is literal:** write a Sirius-side converter that goes GPU → `cudaMallocHost` pinned buffer → GPU, and register it INSTEAD of cucascade's peer-async version. This would require first unregistering the cucascade built-in (uses `unregister_converter<Src, Dst>()`), then registering Sirius's host-staged version. Motivation: the cucascade peer-async has a known GPU1→GPU0 bug (Phase 4 `[.][multi_gpu_transfer]`). Host-staging bypasses the bug.
- **CONTEXT is wrong:** the peer-async is already registered, the round-trip test can just assert on it, and Phase 7 is where the return-leg bug gets fixed (either upstream or via Sirius wrapper).

**Recommend interpretation 2** for three reasons:
1. CONTEXT §Deferred says "P2P direct `cudaMemcpyPeerAsync` converter body — Phase 7 (MGPU-06). Phase 6 registers the host-staged slot; Phase 7 swaps the body." But the slot is already filled with peer-async. Swapping to host-staged then back to peer-async is movement for movement's sake.
2. The Phase 4 hidden-test failure lived in `convert_gpu_to_gpu` (peer-async) — it's a device-guard bug. A host-staged Sirius re-implementation would sidestep the bug (by using host memory that no GPU context restricts), but so would just keeping the current converter and closing the bug in Phase 7.
3. Writing a fresh Sirius host-staged converter is net-new code with its own review/correctness burden, which the phase SUMMARY must then validate on N=2 hardware for zero extra benefit over the existing peer-async path.

**If the user insists on interpretation 1,** the plan needs to include: (a) an `unregister_converter<gpu_table_representation, gpu_table_representation>()` call, (b) a Sirius-side host-staged body using `cucascade::memory::small_pinned_host_memory_resource` (already owned by `SiriusContext::small_pinned_allocator_`), (c) explicit `cudaSetDevice(source)` / `cudaSetDevice(target)` bracketing with error checks, (d) a round-trip test, and (e) a performance note that host-staged will be slower than peer-async — acceptable for Phase 6 because correctness gates on this path.

**Open question for planner.** Ask the user before writing the MGPU-04 task tree whether they want interpretation 1 or 2.

## Standard Stack

### Core APIs used by Phase 6
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| cucascade | f47de0b (pinned) | topology_discovery, numa_region_pinned_host_memory_resource, representation_converter_registry, register_builtin_converters, reservation_manager_configurator | Sole authoritative source for tier conversion + NUMA-aware allocation in Sirius |
| RMM | 26.04 (via pixi cuda-13) | cuda_set_device_raii, cuda_device_id, cuda_stream, device_memory_resource | Stream and device-guard RAII; Sirius already uses both |
| cudf | 26.06 (nightly) / 26.02 (stable) | pack, unpack, table | Used inside `convert_gpu_to_gpu`; Sirius only consumes the converter, not cudf directly here |
| spdlog | (project-pinned) | spdlog::info / spdlog::warn for startup + audit logs | All Sirius logging goes through spdlog |
| libnuma | (cucascade dep) | NUMA node queries via cucascade's pinned host resource | Accessed indirectly; Sirius must not call `numa_*` directly (MGPU-01 sweep) |
| Catch2 v2 | 2.13.10 | Test framework; MGPU-04 round-trip test + MGPU-03 sanitizer gate | Existing `[multi_gpu_foundation]` tag |
| compute-sanitizer | CUDA 13.2 toolkit | `--tool memcheck --require-cuda-init` for device-guard audit | Phase 5 already used; pattern established |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| CUCASCADE_CUDA_TRY | macro in cucascade/error.hpp | Wraps CUDA runtime calls, throws on failure | Use to replace bare `cudaSetDevice(device_id)` if upgrading device-guard discipline |
| `rmm::cuda_set_device_raii` | rmm/cuda_device.hpp | RAII scope guard for cudaSetDevice | Preferred over raw `cudaSetDevice` for scoped operations |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Re-adding a wrapper around `cucascade::memory::topology_discovery` | Use existing `config_.get_hw_topology()` accessor unchanged | Simpler; no duplicate NVML query; CONTEXT lock on "new `SiriusContext::get_topology()`" can be satisfied by renaming/aliasing the existing getter |
| Writing Sirius's own GPU↔GPU host-staged converter | Consume cucascade's built-in `convert_gpu_to_gpu` (peer-async) | Existing converter already registered; avoids double-registration throw; Phase 7 fixes the return-leg bug |
| Explicit `cudaGetDeviceCount` replacement | Use `config_.get_hw_topology().num_gpus` | Already wired — no replacement needed in the Super-Sirius path |

**Installation:** No new dependencies. All libraries already linked per `CMakeLists.txt` and `pixi.toml`.

**Version verification:** Not applicable — cucascade pin locked at Phase 4 (`f47de0b`); no version bump in Phase 6 scope (CONTEXT §Deferred).

## Architecture Patterns

### Recommended Project Structure
No new files required. Changes concentrated in:
```
src/
├── sirius_config.cpp          # Add fail-hard on discover() failure; keep existing topology_ cache
├── sirius_context.cpp         # Add startup info log; optional SiriusContext::get_topology() alias
├── include/sirius_context.hpp # Optional: add get_topology() accessor alias (if distinct from get_hw_topology)
├── pipeline/gpu_pipeline_executor.cpp  # MGPU-03: wrap cudaSetDevice(line 58) in CUCASCADE_CUDA_TRY or convert to RAII
├── downgrade/downgrade_executor.cpp    # MGPU-03: same as above at line 61
test/cpp/
└── config/test_context.cpp    # Extend [multi_gpu_foundation] with MGPU-01 logging assertion + MGPU-04 round-trip test on N>=2
```

### Pattern 1: Topology cache + validated accessor (MGPU-01)
**What:** Reuse the existing cached `_hw_topology` on `sirius_config` and expose it unchanged. Add fail-hard in `SiriusContext::initialize()` before any memory_manager construction:

```cpp
// src/sirius_context.cpp — inside initialize(), before memory_manager_ construction
auto const& topo = config_.get_hw_topology();
if (topo.num_gpus == 0) {
  throw std::runtime_error(
    "SiriusContext::initialize: cucascade::topology_discovery reported 0 GPUs. "
    "Fail-hard per MGPU-01 — refusing to initialize with a stub topology.");
}
spdlog::info(
  "SiriusContext: topology — {} GPUs across {} NUMA node(s):",
  topo.num_gpus, topo.num_numa_nodes);
for (auto const& gpu : topo.gpus) {
  spdlog::info("  GPU {}: name='{}' numa_node={} pci={}",
               gpu.id, gpu.name, gpu.numa_node, gpu.pci_bus_id);
}
```

**When to use:** Phase 6 Plan 1 (first plan — everything else consumes the cached topology).

### Pattern 2: `rmm::cuda_set_device_raii` for scoped device switches (MGPU-03)
**What:** Existing pattern already used at `sirius_context.cpp:192`, `gpu_pipeline_executor.cpp:65`, `downgrade_executor.cpp:52`. For the per-thread init callbacks (lines 58 and 61), prefer checked-raw-set over RAII because the callback runs at thread-start and the device stays pinned for the thread's lifetime:

```cpp
// src/pipeline/gpu_pipeline_executor.cpp:54-61 — replace raw cudaSetDevice
absl::AnyInvocable<void() noexcept> gpu_pipeline_executor::get_per_thread_init() {
  auto device_id = _memory_space->get_device_id();
  return [device_id]() noexcept {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      // spdlog::error + cudaGetLastError() + abort or graceful shutdown — must be noexcept
      spdlog::error("gpu_pipeline_executor per-thread init: cudaSetDevice({}) failed: {}",
                    device_id, cudaGetErrorString(err));
    }
    sirius::util::enable_log_on_default_stream();
  };
}
```

**When to use:** Phase 6 Plan covering MGPU-03. Cannot use `CUCASCADE_CUDA_TRY` macro directly because the callback is `noexcept`; must handle error in-place.

### Pattern 3: Per-NUMA host space assertion (MGPU-05)
**What:** After `memory_manager_` construction, assert host-space count matches NUMA node count:
```cpp
// src/sirius_context.cpp — inside initialize(), after memory_manager_ construction
auto host_spaces = memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
auto const& topo = config_.get_hw_topology();
spdlog::info("SiriusContext: {} host memory space(s) created for {} NUMA node(s)",
             host_spaces.size(), topo.num_numa_nodes);
if (topo.num_numa_nodes > 0 && host_spaces.size() != static_cast<size_t>(topo.num_numa_nodes)) {
  spdlog::warn(
    "SiriusContext: host space count ({}) != NUMA node count ({}) — "
    "MGPU-05 expects one host space per NUMA domain. Check sirius_config / "
    "configurator host policy (.use_host_per_numa vs .use_host_per_gpu).",
    host_spaces.size(), topo.num_numa_nodes);
}
```

**When to use:** Plan covering MGPU-05. The warn-not-throw is deliberate: a user YAML may legitimately specify a non-default host layout; fail-hard would break those configs.

### Pattern 4: `/proc/PID/numa_maps` spot-check pattern (MGPU-05 evidence)
```bash
# On the N=2 host, after Sirius loads
pgrep -f duckdb | head -1 | xargs -I PID cat /proc/PID/numa_maps | grep -i 'anon\|heap\|huge' | head -20
numactl --show  # Confirms NUMA topology visible to Sirius process
```
Expected outputs show `N0=...` / `N1=...` annotations on pinned allocations. Phase SUMMARY evidence block.

### Anti-Patterns to Avoid
- **Re-calling `topology_discovery.discover()` in `SiriusContext::initialize()`.** It's already called in `sirius_config`'s constructor. Duplicate calls query NVML twice (slow) and can produce inconsistent results if devices are unplugged mid-init.
- **Registering a second GPU↔GPU converter.** `representation_converter_registry::register_converter_impl` throws `std::runtime_error` on duplicate registration (cucascade/src/data/representation_converter.cpp:74). Must `unregister_converter<T,T>()` first if replacement is intended.
- **Fail-hard on `num_numa_nodes == 0`.** Non-NUMA single-socket dev hosts report `num_numa_nodes == 0` or `== 1` depending on libnuma version. Warn-only on mismatch.
- **Running `performance_test.py` for MGPU-02.** Uses legacy `gpu_processing` path. Use `run_tpch_parquet.sh sirius 10` which invokes `gpu_execution` via `tpch_queries/gpu/q*.sql`.
- **Including `src/cuda/` in the MGPU-01 hand-rolled CUDA sweep.** That directory is `namespace duckdb` legacy code frozen for v1.1 per PROJECT.md Out-of-Scope. Document the exclusion in the sweep gate.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Query GPU count | `cudaGetDeviceCount` | `config_.get_hw_topology().num_gpus` | Centralized; consistent across all Sirius call sites; MGPU-01 gate |
| Query NUMA node of a CPU / GPU | `numa_node_of_cpu`, `/sys/bus/pci/...` | `config_.get_hw_topology().gpus[i].numa_node` | Same reason; cucascade's topology_discovery already does the sysfs parsing |
| Build per-NUMA host memory space | `numa_alloc_onnode` + bespoke mr | `cucascade::memory::numa_region_pinned_host_memory_resource` (default mr factory) | Already the cucascade default; RMM-compatible; pinned for CUDA |
| GPU↔GPU data transfer | raw `cudaMemcpyPeerAsync` + cudf pack/unpack | `cucascade::register_builtin_converters` + `batch->convert_to<gpu_table_representation>(registry, target, stream)` | Registered; consumers already use it |
| Per-thread device pinning | `cudaSetDevice` in every function | Thread-pool per-thread init callback (existing pattern in `gpu_pipeline_executor.cpp:54` and `downgrade_executor.cpp:60`) | Pinning once per thread is enough; CUDA contexts are thread-affine |
| Multi-GPU teardown ordering | Manual destructor ordering | Mirror Phase 5's `gpu_io_backends_.clear()` → `io_backend_registry_.clear()` BEFORE `memory_manager_->shutdown()` at `sirius_context.cpp:308-330` | Prevents `cudaErrorInvalidResourceHandle` at extension unload |

**Key insight:** Phase 6 is an audit + logging + test phase. Almost every "don't hand-roll" item above has an existing, tested solution in the codebase. Plans that propose to rebuild any of these are moving sideways, not forward.

## Runtime State Inventory

N/A — Phase 6 is code + config changes, not a rename/refactor/migration. No stored data, live service config, OS registrations, secrets, or build artifacts embed phase-specific names. No runtime state to sweep.

## Common Pitfalls

### Pitfall 1: `performance_test.py` wraps queries in the legacy path
**What goes wrong:** Running `python3 test/tpch_performance/performance_test.py 10` as the MGPU-02 measurement tool will exercise `gpu_processing` (legacy `namespace duckdb`), not `gpu_execution` (Super Sirius). The timings would not reflect the topology / NUMA / converter changes being gated.
**Why it happens:** `performance_test.py:28-32` hard-codes `call gpu_processing("...")`. The docstring says "gpu_execution" but the string literal says "gpu_processing".
**How to avoid:** Use `test/tpch_performance/run_tpch_parquet.sh sirius 10 $(seq 1 22)` which loads `tpch_queries/gpu/q*.sql` (confirmed to use `call gpu_execution(...)` — verified via `head -10 tpch_queries/gpu/q1.sql`).
**Warning signs:** SF10 wall-clock much faster than expected (legacy path skips several Super-Sirius operator stages). If Q1 completes in < 0.5s at SF10 1-GPU, it's almost certainly the legacy path.

### Pitfall 2: Double-registering the GPU↔GPU converter
**What goes wrong:** Calling `registry.register_converter<gpu_table_representation, gpu_table_representation>(fn)` when the entry already exists throws `std::runtime_error("Converter for this type pair already registered")`.
**Why it happens:** `cucascade::register_builtin_converters` registers `convert_gpu_to_gpu` at line 1464 of `representation_converter.cpp`. Sirius already calls this via `converter_registry::initialize()`. A second registration attempt in Sirius will throw at extension load time — extension-load failure.
**How to avoid:** If the plan chooses interpretation 1 from Finding 6 (Sirius-side host-staged override), call `registry.unregister_converter<gpu_table_representation, gpu_table_representation>()` FIRST, then register. Confirm via `has_converter<...>() == false` between the calls.
**Warning signs:** Unit-test output showing `"Converter for this type pair already registered"` during extension load.

### Pitfall 3: `noexcept` callbacks can't use `CUCASCADE_CUDA_TRY`
**What goes wrong:** Wrapping `cudaSetDevice(device_id)` in `CUCASCADE_CUDA_TRY` inside the `get_per_thread_init()` lambda breaks because `CUCASCADE_CUDA_TRY` throws, but the lambda is `noexcept`. `std::terminate` on failure.
**Why it happens:** The thread-pool per-thread init callback signature (`absl::AnyInvocable<void() noexcept>`) mandates no throws.
**How to avoid:** Inline the error handling: `cudaError_t err = cudaSetDevice(device_id); if (err != cudaSuccess) { spdlog::error(...); /* optionally: std::abort() if unrecoverable */ }`. Do NOT use `CUCASCADE_CUDA_TRY` or `gpuErrchk` macros inside `noexcept` contexts.
**Warning signs:** Compile-time error if `CUCASCADE_CUDA_TRY` throw-spec is detected; runtime `std::terminate` otherwise.

### Pitfall 4: `num_numa_nodes == 0` on non-NUMA dev hosts
**What goes wrong:** Fail-hard on `topo.num_numa_nodes == 0` makes Sirius unloadable on CI hosts without NUMA (e.g. VMs, single-socket laptops). libnuma's `numa_num_configured_nodes` can return 0 or 1 depending on kernel + version.
**Why it happens:** Topology discovery reports whatever libnuma finds; cucascade doesn't fabricate NUMA nodes.
**How to avoid:** Warn-only on mismatch between host space count and NUMA count (see Pattern 3). MGPU-05 fail-hard limited to `num_gpus == 0`.
**Warning signs:** `SiriusContext: 1 host memory space(s) created for 0 NUMA node(s)` log lines on non-NUMA hosts — expected, not a bug.

### Pitfall 5: Stream-on-wrong-GPU in `batch_lock_utils.hpp:87`
**What goes wrong:** `batch->convert_to<cucascade::gpu_table_representation>(registry, target_space, stream)` at `src/include/pipeline/batch_lock_utils.hpp:87` passes a `stream` whose CUDA context is pinned to the **source** GPU. cucascade's `convert_gpu_to_gpu` then calls `cudaMemcpyPeerAsync(..., stream.value())` — the stream lives on the wrong device. On GPU0→GPU1 forward direction this "accidentally" works because `cudaMemcpyPeerAsync` tolerates the source-device stream. On GPU1→GPU0 return leg the hidden test fails.
**Why it happens:** Existing Sirius code was written for single-GPU; the pipeline's stream pool is pre-f47de0b.
**How to avoid:** Phase 7 fix (MGPU-06) — either (a) use the target's `acquire_stream()` at the batch_lock_utils call site, (b) wrap `convert_gpu_to_gpu` in a Sirius-side function that flips `cudaSetDevice(source)` → issues copy → `cudaSetDevice(target)`, or (c) push the fix upstream to cucascade. Phase 6 MGPU-03 may surface this via compute-sanitizer — if it does, flag for Phase 7 investigation.
**Warning signs:** `[.][multi_gpu_transfer]` test fails on GPU1→GPU0 return; `[.][mem_04_p2p_transfer]` fails with same shape; compute-sanitizer reports "invalid device" or "invalid resource handle."

### Pitfall 6: Device-guard callback called before stream pool init
**What goes wrong:** `downgrade_executor::start()` (src/downgrade/downgrade_executor.cpp:45-74) constructs `_stream_pool` at line 52 using `rmm::cuda_device_id{device_id}`. This happens BEFORE `_pool->start()` at line 64 which fires `per_thread_init` per worker thread. On hosts with strict CUDA default-context behavior, the stream pool constructor may implicitly pick `cudaGetDevice() == 0` regardless of `device_id`.
**Why it happens:** `exclusive_stream_pool` constructor in cucascade uses `cudaStreamCreateWithFlags` which binds to the calling thread's current device. If the caller (the thread that calls `downgrade_executor::start()`) hasn't set its device, streams can end up in the wrong context even though the `device_id` was passed as a parameter.
**How to avoid:** Wrap the `_stream_pool = std::make_unique<...>(...)` line at `downgrade_executor.cpp:52` in `rmm::cuda_set_device_raii{rmm::cuda_device_id{device_id}}`. Same treatment for `gpu_pipeline_executor.cpp:45` (`_stream_pool` initializer). Verify with compute-sanitizer on N=2.
**Warning signs:** compute-sanitizer reports stream-device mismatch at the first GPU→GPU downgrade; or streams work on GPU0 but fail on GPU1 devices.

## Code Examples

### Startup topology log (MGPU-01 logging success criterion)
```cpp
// src/sirius_context.cpp — inside SiriusContext::initialize()
auto const& topo = config_.get_hw_topology();

// Fail-hard gate (CONTEXT lock)
if (topo.num_gpus == 0) {
  throw std::runtime_error(
    "SiriusContext::initialize: cucascade::topology_discovery reported 0 GPUs — "
    "refusing to initialize on stub topology (MGPU-01 fail-hard).");
}

// Summary log (CONTEXT lock: info level)
spdlog::info("SiriusContext: topology summary — {} GPU(s), {} NUMA node(s), host={}",
             topo.num_gpus, topo.num_numa_nodes, topo.hostname);
for (auto const& gpu : topo.gpus) {
  spdlog::info("  GPU {}: {} (numa={}, pci={})",
               gpu.id, gpu.name, gpu.numa_node, gpu.pci_bus_id);
}
```

### MGPU-04 round-trip test extension (Super-Sirius consumer test)
```cpp
// test/cpp/config/test_context.cpp — add after the existing [multi_gpu_foundation] tests
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

TEST_CASE("gpu_to_gpu round-trip preserves bytes on N>=2 hosts",
          "[.][multi_gpu_foundation][mgpu_04_round_trip]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs for MGPU-04 round-trip");
    return;
  }
  sirius::converter_registry::reset_for_testing();
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(256ull << 20)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_host_capacity(1ull << 30)
    .use_host_per_numa()   // MGPU-05 — exercise per-NUMA path
    .set_reservation_fraction_per_host(0.75);
  auto cfgs = builder.build();
  auto mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(cfgs));
  sirius::converter_registry::initialize();

  auto gpu_spaces = mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);
  auto* gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  auto* gpu1 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[1]);

  auto batch = make_gpu_batch(*gpu0, 1024);  // Same helper used by [mem_04_p2p_transfer]
  auto original_bytes = batch->get_data()->get_size_in_bytes();
  auto& registry = sirius::converter_registry::get();
  rmm::cuda_stream stream;

  // GPU0 → GPU1 (forward)
  REQUIRE(batch->try_to_lock_for_in_transit());
  batch->convert_to<cucascade::gpu_table_representation>(registry, gpu1, stream);
  batch->try_to_release_in_transit();
  REQUIRE(batch->get_memory_space()->get_device_id() == gpu1->get_device_id());
  REQUIRE(batch->get_data()->get_size_in_bytes() == original_bytes);

  // GPU1 → GPU0 (return leg — Phase 4 known bug; expected to fail on unpatched build)
  REQUIRE(batch->try_to_lock_for_in_transit());
  batch->convert_to<cucascade::gpu_table_representation>(registry, gpu0, stream);
  batch->try_to_release_in_transit();
  REQUIRE(batch->get_memory_space()->get_device_id() == gpu0->get_device_id());
  REQUIRE(batch->get_data()->get_size_in_bytes() == original_bytes);

  sirius::converter_registry::shutdown();
}
```

**Note:** The test is hidden (`[.]` prefix) because it will fail on Phase 6 HEAD until Phase 7 (MGPU-06) fixes the return-leg bug. Phase 6 can document the failure as "Phase 4 deferral confirmed; Phase 7 scope" in the phase SUMMARY.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sirius hand-rolls `cudaGetDeviceCount` / `numa_node_of_cpu` | cucascade `topology_discovery` + Sirius consumes via `config_.get_hw_topology()` | Already landed pre-Phase-6 (Plan 04-03 uses `topo.gpus[dev_id].numa_node`) | Sirius has one source of truth; NVML + sysfs parsing centralized |
| kvikio-backed parquet datasource | `sirius::io::cucascade_datasource` | Phase 5 (complete) | Multi-GPU-safe I/O; per-GPU backend cache on `SiriusContext` |
| Single host memory space | Per-NUMA host spaces via `numa_region_pinned_host_memory_resource` | Cucascade `f47de0b` default (Phase 4 bump) | Per-NUMA allocations reduce cross-socket traffic; default since f47de0b |
| Host-staged GPU↔GPU transfer (v1.0 MEM-03) | `cudaMemcpyPeerAsync` inside `cucascade::convert_gpu_to_gpu` | Cucascade `f47de0b` default | Direct peer copy eliminates host round-trip on P2P-capable hosts |

**Deprecated/outdated:**
- **Hand-rolled `cudaSetDevice(0)` resets** in `src/cuda/allocator.cu:84, 129` — legacy path, frozen for v1.1, documented in Phase 5 Deferred.
- **`performance_test.py` with `gpu_processing`** — points at legacy path, should be updated to `gpu_execution` or replaced by `run_tpch_parquet.sh` in docs (Pitfall 1).

## Open Questions

1. **Interpretation of CONTEXT §MGPU-04 "host-staged converter"**
   - What we know: cucascade already registers a peer-async `convert_gpu_to_gpu`. CONTEXT language says "register a host-staged converter."
   - What's unclear: Does the user want (a) a Sirius-side host-staged override replacing the peer-async path, or (b) acknowledge the existing peer-async registration and just add the round-trip test?
   - Recommendation: Interpretation (b). If (a), the plan needs an explicit `unregister_converter` + re-register step. Flag for planner to confirm with user.

2. **MGPU-02 baseline: re-measure on dev@484db35, or reuse Phase-5 SF10 timings?**
   - What we know: Phase 5 captured SF10 timings on Phase-5 HEAD (1-GPU: Q1=1.273s, Q6=0.233s, Q12=0.717s). dev HEAD is `484db35`.
   - What's unclear: Does MGPU-02 require a fresh baseline run on `484db35` worktree, or is "Phase-5 SF10 timings are the most recent pre-Phase-6 reading" sufficient?
   - Recommendation: Use Phase-5 SF10 timings as baseline. The user directive on 2026-04-21 ("we don't need to run any comparisons") was about Phase 5's IO-10; Phase 6 MGPU-02 is still in scope but the baseline choice should be pragmatic.

3. **Should the MGPU-01 sweep exclude `src/cuda/` (legacy path)?**
   - What we know: `src/cuda/allocator.cu:70` contains the only `cudaGetDeviceCount` in `src/`. This file is `namespace duckdb` legacy code, frozen for v1.1 per PROJECT.md.
   - What's unclear: Is the CONTEXT sweep gate literal ("all of `src/`") or scoped ("Super Sirius only")?
   - Recommendation: Super-Sirius scope. Document the `src/cuda/` exclusion in the phase SUMMARY alongside the existing Phase 5 deferral note.

4. **Should Phase 6 close the `test_downgrade_executor.cpp:813` bug?**
   - What we know: CONTEXT §Out-of-Scope explicitly says this is Phase 7 (MGPU-06). Phase 5 SUMMARY agrees. Plan 04-05 SUMMARY also agrees.
   - What's unclear: MGPU-03 (device-guard audit) may identify the root cause as a device-guard bug, making it "in scope" for MGPU-03.
   - Recommendation: Triage only — if compute-sanitizer identifies the root cause as a simple device-guard fix (one or two lines), fix under MGPU-03. If it's a deeper stream-context issue (more likely — see Pitfall 5), punt to Phase 7 with a diagnostic artifact in the SUMMARY.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| N=2 GPU host (`6f7e4c9-lcedt`, 2 × RTX 6000 Ada) | MGPU-02, MGPU-03, MGPU-04 round-trip | ✓ | driver 595.58.03, CUDA 13.2 | — |
| `compute-sanitizer` | MGPU-03 memcheck gate | ✓ | CUDA 13.2 toolkit | — |
| libnuma | MGPU-05 per-NUMA allocator | ✓ | via cucascade dep; `find_library(NUMA_LIB numa REQUIRED)` | — |
| `numactl` | MGPU-05 evidence (`numactl --show` + `/proc/PID/numa_maps`) | Assumed on N=2 host | — | manual `/proc/PID/numa_maps` inspection |
| `test_datasets/tpch_parquet_sf10` | MGPU-02 SF10 measurement | ✓ | symlinked from `/home/felipe/sirius/test_datasets/tpch_parquet_sf10` (per CONTEXT §code_context) | — |
| `SIRIUS_CONFIG_FILE` yaml | MGPU-02 (1-GPU config) | ✓ | `/tmp/phase5-validation/sirius-sf10.yaml` and `/tmp/phase5-validation/sirius-2gpu.yaml` exist | create fresh if needed |
| `mcp__project-commands__run_command` | Build + unit-tests | ✓ | Per CLAUDE.md user rule | — |

**No missing dependencies.** Phase 5 validated the N=2 host is operational; Phase 6 inherits the same environment.

## Project Constraints (from CLAUDE.md)

Extracted from `./CLAUDE.md`:

- **Build via `mcp__project-commands__run_command` only.** No direct `pixi run` / `make` from Claude (user preference).
- **No `rmm::cuda_stream_default` anywhere.** HYG-02 was swept in Phase 5 for Phase-5-modified files; Phase 6 must not introduce any new uses.
- **Super Sirius (`namespace sirius`) only.** Legacy `gpu_processing` / `namespace duckdb` in `src/cuda/` is frozen.
- **C++20 + CUDA 13+, std 20, separable compilation.** GPU arches 75–120.
- **Run `/module-context <task description>` before implementation.** Loads cucascade / rmm / cudf / duckdb / libkvikio / cuCascade API docs from `.claude/skills/module-discover/docs/`.
- **Fallback-first.** Any new code path that can't run multi-GPU-safely must downgrade through the existing fallback mechanism, not crash.
- **cuCascade API is authoritative.** All tier conversion + disk I/O must go through cucascade's registries. No hand-rolled kvikio/cuFile/GDS in `src/`.
- **Pre-commit hooks.** `pre-commit run -a` for clang-format / black / cmake-format / codespell.
- **Test binary path:** `build/release/extension/sirius/test/cpp/sirius_unittest` — tag invocation via `"[tag]"` argv.

## Sources

### Primary (HIGH confidence — inspected in current tree)
- `src/sirius_config.cpp:265-376` — topology_discovery call site + apply_defaults + load_from_file
- `src/include/sirius_config.hpp:71-75, 111` — `get_hw_topology()` accessor + cached field default
- `src/sirius_context.cpp:168-289` — `initialize()` full implementation (includes per-GPU io_backend creation)
- `src/sirius_context.cpp:291-334` — `terminate()` teardown ordering
- `src/include/sirius_context.hpp:115-170` — current SiriusContext API surface including `get_hw_topology()` + io_backend accessors
- `src/include/exec/config.hpp:31-50` — `downgrade_executor_config.preferred_numa_node` field
- `src/pipeline/gpu_pipeline_executor.cpp:40-80` — device-guard pattern (raw `cudaSetDevice` at line 58 + `rmm::cuda_set_device_raii` at 65)
- `src/downgrade/downgrade_executor.cpp:40-107` — device-guard pattern (raw at line 61)
- `src/downgrade/downgrade_task.cpp` (full) — downgrade executes GPU→HOST only, not GPU→GPU
- `src/include/pipeline/batch_lock_utils.hpp:55-105` — cross-tier conversion call sites including GPU↔GPU
- `src/data/host_parquet_representation_converters.cpp:198-237` — Sirius-owned converter registrations
- `src/include/data/sirius_converter_registry.hpp` (full) — singleton registry calls `cucascade::register_builtin_converters`
- `src/sirius_extension.cpp:1044-1062` — `LoadInternal` with `sirius::converter_registry::initialize()` call
- `src/cuda/allocator.cu:55-132` — legacy `namespace duckdb` path; contains the only `cudaGetDeviceCount` in src/
- `src/cuda/communication.cu:1-166` — legacy path; raw `cudaSetDevice` usage
- `cucascade/include/cucascade/memory/topology_discovery.hpp` (full) — `topology_discovery` class + `system_topology_info` struct
- `cucascade/include/cucascade/memory/numa_region_pinned_host_allocator.hpp` (full) — `numa_region_pinned_host_memory_resource(int numa_node)` constructor
- `cucascade/src/memory/common.cpp:36-40` — `make_default_host_memory_resource` returns `numa_region_pinned_host_memory_resource`
- `cucascade/include/cucascade/memory/reservation_manager_configurator.hpp:109-112` — `use_host_per_gpu()` / `use_host_per_numa()` builder methods
- `cucascade/include/cucascade/data/representation_converter.hpp:131-256` — `register_converter<Src,Dst>` template + `register_builtin_converters` free function
- `cucascade/src/data/representation_converter.cpp:137-195` — `convert_gpu_to_gpu` body uses `cudaMemcpyPeerAsync` directly (not host-staged)
- `cucascade/src/data/representation_converter.cpp:1461-1500` — `register_builtin_converters` registers 10 converters including GPU↔GPU
- `cucascade/CLAUDE.md:268` — cucascade's own docs: "register_builtin_converters() registers GPU↔HOST and GPU↔DISK and HOST↔DISK converters"
- `test/cpp/config/test_context.cpp:110-295` — live device-guard audit comment block + `[multi_gpu_foundation]` test set
- `test/cpp/downgrade/test_downgrade_executor.cpp:790-900` — hidden `[.][mem_04_p2p_transfer]` test with the `TODO(MGPU-06)` anchor at line 813
- `test/tpch_performance/performance_test.py:28-32` — `call gpu_processing(...)` (legacy path) vs `tpch_queries/gpu/q1.sql:1` which uses `call gpu_execution(...)`
- `test/tpch_performance/CLAUDE.md` — documentation for `run_tpch_parquet.sh` (the Super-Sirius regression harness)
- `.planning/config.json` — `workflow.nyquist_validation: false` → skip Validation Architecture section
- `.planning/phases/05-cucascade-backed-parquet-i-o-migration/05-SUMMARY.md` — Phase-5 absolute SF10 numbers + IO-11 audit log pattern + teardown ordering precedent
- `.planning/phases/05-cucascade-backed-parquet-i-o-migration/05-03-SUMMARY.md` — per-GPU backend cache pattern + teardown ordering pattern (Plan 05-03)
- `.planning/phases/04-cucascade-bump-v1-0-re-integration/04-05-SUMMARY.md` — hidden-test failures at GPU1→GPU0 return leg + deferred-to-Phase-7 framing
- `git rev-parse origin/dev` → `484db3509c395646a7c5cfd0543e860fa2e9cd9b` (current dev HEAD, matches Phase 4 anchor)
- `cucascade/` submodule `git rev-parse HEAD` → `f47de0bb7bcaddd55081a9c4bc584627532d1ef9` (pinned commit)

### Secondary (MEDIUM confidence)
- `.claude/skills/module-discover/docs/cucascade/modules/memory.md` — module docs referencing `numa_region_pinned_host_allocator`, `host_memory_space_config`, `reservation_manager_configurator` (derived from cucascade headers; cross-checked against the headers in Primary sources — consistent)
- `.planning/research/CUCASCADE-IO.md` — research note from Phase 5 documenting io_backend is DISK-tier only
- `cucascade/docs/topology-and-configuration.md:172, 314` — cucascade's own docs confirm default host allocator is `numa_region_pinned_host_memory_resource`

### Tertiary (LOW confidence)
- None. No WebSearch-only findings. All primary claims verified against local source.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all APIs read from headers in-tree; no Context7/WebSearch needed
- Architecture: HIGH — patterns extracted from live Sirius code (sirius_context.cpp, gpu_pipeline_executor.cpp, downgrade_executor.cpp) already validated through Phase 5
- Pitfalls: HIGH — Pitfalls 1-6 each cite a specific file+line anchor or a documented Phase 4/5 outcome
- Open Questions: MEDIUM — resolutions require the planner (or user) to pick between interpretations, but each interpretation has a defined remediation

**Research date:** 2026-04-21
**Valid until:** Estimated 30 days — `f47de0b` cucascade pin locked for v1.1; dev HEAD is stable; `src/` surface is the same tree inspected.

## Pre-Submission Checklist

- [x] All domains investigated (topology, host memory, converters, device-guard, regression harness)
- [x] Negative claims verified — "cucascade already registers gpu_to_gpu" verified at representation_converter.cpp:1464
- [x] Multiple sources cross-referenced — cucascade headers + live calls in sirius_context.cpp + existing test_context.cpp assertion
- [x] File+line anchors provided for all source citations
- [x] Publication dates N/A (consumes local source, not external docs)
- [x] Confidence levels assigned honestly
- [x] "What might I have missed?" review completed — see Open Questions section
- [x] Runtime State Inventory intentionally omitted (greenfield code change, not a rename/refactor)

## Plan Sequencing Recommendation (for planner)

Based on the findings, the CONTEXT §specifics plan ordering is slightly over-scoped. Recommended revision:

1. **MGPU-01 — Topology validation + startup log + sweep gate.** Reuse existing `config_.get_hw_topology()`; add fail-hard + spdlog::info summary in `SiriusContext::initialize()`; add the `grep -rn ... src/` sweep gate (excluding `src/cuda/` legacy path). *Small plan — 1 file edit + 1 grep gate.*

2. **MGPU-05 — Per-NUMA host space assertion + explicit `use_host_per_numa()`.** Update `sirius_config.cpp:271-285` `apply_defaults` to call `.use_host_per_numa()` on the builder; add post-construction assertion + spdlog::info/warn log in `SiriusContext::initialize()`; capture `/proc/PID/numa_maps` evidence in SUMMARY. *Can run in parallel with Plan 1.*

3. **MGPU-03 — Device-guard enforcement.** Replace raw `cudaSetDevice` at `gpu_pipeline_executor.cpp:58` + `downgrade_executor.cpp:61` with checked variants; wrap `_stream_pool` constructors in `rmm::cuda_set_device_raii`; run compute-sanitizer on `[multi_gpu_foundation]` + `[integration][gpu_execution][parquet][join]`. *Depends on Plan 1 (topology log confirms context before sanitizer runs).*

4. **MGPU-04 — Round-trip test + existing-converter-verification.** Add the `[.][multi_gpu_foundation][mgpu_04_round_trip]` test on N>=2. **Do NOT register a new converter** (see Finding 2); cucascade's built-in peer-async converter is already registered. If user picks interpretation 1 from Finding 6 (Sirius-side host-staged override), add the unregister+register step. *Can run in parallel with Plan 3.*

5. **MGPU-02 — Single-GPU SF10 regression.** Run `test/tpch_performance/run_tpch_parquet.sh sirius 10 $(seq 1 22)` against 1-GPU config on N=2 host; compare to Phase 5 SF10 timings as baseline (per Open Question 2). *Last wave — consumes everything.*

6. **Phase SUMMARY + validation artifact.** Aggregate evidence; close MGPU-01..05; document deferred items (GPU1→GPU0 return leg, adaptive scan, Phase-4 SF10 comparison) per CONTEXT §Deferred.

Estimated total: 5 plans (vs CONTEXT §specifics 5-wave ordering). Plans 1+2 parallelize; Plans 3+4 parallelize; Plan 5 + SUMMARY are sequential.

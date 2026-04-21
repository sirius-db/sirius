---
phase: 07-p2p-direct-transfer-adaptive-scan-partitioning
plan: 02
subsystem: multi-gpu
tags: [mgpu-06, p2p, cuda, peer-access, converter-override, silent-corruption]

# Dependency graph
requires: [07-01]
provides:
  - "Un-hidden MGPU-06 GPU<->GPU transfer tests: gpu_to_gpu_transfer_via_converter + p2p_transfer_converter_round_trip + mgpu_04_round_trip"
  - "FNV-1a checksum data-integrity guard on all three round-trip tests (Pitfall 2: Ada Lovelace + Sapphire Rapids silent PCIe P2P corruption)"
  - "Sirius-side MGPU-06 P2P converter override at src/data/sirius_p2p_converter.cpp — replaces cucascade's built-in gpu_table_representation -> gpu_table_representation converter with a stream-correct implementation"
  - "enable_p2p_for_test helper in both test TUs (mirrors Plan 07-01's pattern) for TEST_CASEs that bypass SiriusContext::initialize()"
affects: [07-03, 07-04, MGPU-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sirius-side converter override via unregister_converter + register_converter in converter_registry::initialize() — cucascade submodule stays unpatched (f47de0b pin preserved)"
    - "Peer copy on target_stream (not caller's stream) so unpack + table construction observe in-order completion without cross-stream events"
    - "Pack on a source-bound rmm::cuda_stream under rmm::cuda_set_device_raii — avoids cross-device stream-use errors when caller's stream lives on a different device than the source"

key-files:
  created:
    - src/include/data/sirius_p2p_converter.hpp
    - src/data/sirius_p2p_converter.cpp
  modified:
    - src/include/data/sirius_converter_registry.hpp (override registration in initialize())
    - CMakeLists.txt (sirius_p2p_converter.cpp added to EXTENSION_SOURCES)
    - test/cpp/downgrade/test_downgrade_executor.cpp (un-hide + checksum + enable_p2p_for_test helper)
    - test/cpp/config/test_context.cpp (un-hide + return-leg + checksum + enable_p2p_for_test helper)

key-decisions:
  - "Task 3 OVERRIDE-REGISTERED, not SKIP: Plan 07-02's default path was SKIP pending Plan 07-04 N=2 validation, but this host's MCP unit-tests ARE on N=2 hardware and the return-leg bug reproduced immediately after un-hiding. Direct hardware evidence triggered the override branch during Task 2 (compile-gate) rather than waiting for Plan 07-04."
  - "Override registration site is sirius_converter_registry.hpp::initialize(), NOT sirius_extension.cpp:1053. Reason: unit tests call sirius::converter_registry::initialize() directly (bypassing LoadInternal), so registering only in sirius_extension.cpp wouldn't cover test paths. Registering inside initialize() makes the override universal and idempotent."
  - "FNV-1a checksum helper is duplicated in test_context.cpp (not shared) because test_downgrade_executor.cpp's helper lives in an anonymous namespace. Keeping the duplication is simpler than building a test-shared header for a 16-line helper."
  - "enable_p2p_for_test kept in both test TUs even after the override lands — defense-in-depth for any future test that bypasses SiriusContext. The override relies on driver-level peer access being enabled; if a future test bypasses both SiriusContext::initialize() AND enable_p2p_for_test, cudaMemcpyPeerAsync would fall back to host-staged inside the driver (slower but correct)."
  - "Pack on source-bound rmm::cuda_stream under source_guard, NOT on caller's stream. Caller's stream may live on the target device (or a third device); using it for pack causes cross-device stream-use errors surfaced as cudaErrorInvalidValue inside cudf's internal cuda_memcpy utilities."

patterns-established:
  - "Test-local CUDA peer-access enable helper — for any TEST_CASE that bypasses SiriusContext, call enable_p2p_for_test(num_gpus) before cross-GPU converter invocations (matches Plan 07-01's enable-loop pattern)."
  - "Sirius-side converter override = unregister_converter<S,T>() + register_converter<S,T>(factory) inside converter_registry::initialize(). Both calls are idempotent; the initialize() mutex guards the pair against re-entry races."

requirements-completed: [MGPU-06]

# Metrics
duration: ~40min (spread: Task 1 edits + build + first unit-tests run revealed return-leg bug + enable_p2p_for_test workaround attempt + second bug class surfaced + Sirius-side override implementation + final green run + cleanup commits)
completed: 2026-04-21
---

# Phase 07 Plan 02: MGPU-06 End-to-End Closure Summary

Closes MGPU-06 end-to-end on real N=2 hardware (2 × RTX 6000 Ada, driver 595.58.03, CUDA 13.2). Un-hides the three Phase-4-deferred GPU↔GPU transfer tests, adds FNV-1a checksum-based data-integrity assertions (silent-corruption guard per Pitfall 2), and registers a Sirius-side P2P converter override that replaces cucascade's built-in `convert_gpu_to_gpu` with a stream-correct implementation.

**Task 3 verdict: OVERRIDE-REGISTERED** — not the expected default SKIP. The return-leg bug persisted after Plan 07-01's peer-access enable loop + test-local `enable_p2p_for_test` workaround, so the Sirius-side converter override (RESEARCH.md Pattern 2) was exercised during Task 2's compile-gate rather than waiting for Plan 07-04. Direct hardware evidence came from MCP unit-tests on this N=2 worktree host.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `e4c452d` | test(07-02): un-hide MGPU-06 GPU<->GPU tests + add FNV-1a checksum integrity guard |
| 1-fix | `7182797` | test(07-02): add enable_p2p_for_test helper to MGPU-06 tests |
| 3 | `18352b9` | feat(07-02): Sirius-side MGPU-06 P2P converter override (return-leg fix) |

## Accomplishments

### Task 1 — Un-hide + checksum + return-leg append (all three tests)

Three hidden tags flipped to visible; one `TODO(MGPU-06)` marker removed; one TEST_CASE renamed from `_placeholder` to final name; one forward-leg-only test extended with a GPU1→GPU0 return leg. Each of the three TEST_CASEs now brackets its round-trip with FNV-1a checksum pre/post assertions.

**Round-trip pre-state classification** (per plan directive 2026-04-20):

| TEST_CASE | File:Line | Pre-state | Adaptation |
|-----------|-----------|-----------|------------|
| `gpu_to_gpu_transfer_via_converter` | test_downgrade_executor.cpp:518 (was :485) | **ROUND_TRIP** | Existing body already had GPU0→GPU1→GPU0. Only added `checksum_pre` capture + `checksum_post` + REQUIRE; no body extension. |
| `p2p_transfer_converter_round_trip` | test_downgrade_executor.cpp:903 (was :805) | **ROUND_TRIP** | Existing body already had GPU0→GPU1→GPU0. Renamed from `_placeholder`; removed `TODO(MGPU-06)`; rephrased Phase-4 comment; added checksum pre/post. |
| `gpu_to_gpu ... (MGPU-04 + MGPU-06)` | test_context.cpp:368 (was :333) | **SINGLE_LEG** | Existing body stopped at forward leg only. Appended full GPU1→GPU0 return leg with `try_to_lock_for_in_transit` + `convert_to<gpu_table_representation>(registry, gpu0, stream.view())` + device_id/size REQUIREs + final checksum assertion. TEST_CASE title updated to reflect round-trip scope. |

**Pitfall-2 audit trail placement**: each checksum REQUIRE has an adjacent comment AND `INFO()` line containing at least one of "Pitfall 2", "Sapphire Rapids", or "silent data corruption":

| File | Anchor | Placement |
|------|--------|-----------|
| test_downgrade_executor.cpp | `~:570` | Comment block immediately above `checksum_pre` capture inside `gpu_to_gpu_transfer_via_converter` |
| test_downgrade_executor.cpp | `~:603` | Comment block + `INFO()` above `REQUIRE(checksum_post == checksum_pre)` |
| test_downgrade_executor.cpp | `~:905` | TEST_CASE header comment mentioning Pitfall 2 |
| test_downgrade_executor.cpp | `~:987` | Comment block + `INFO()` above final checksum REQUIRE |
| test_context.cpp | `~:329` | TEST_CASE header comment mentioning Pitfall 2 |
| test_context.cpp | `~:467` | `INFO()` + final checksum REQUIRE message |

(Line numbers approximate and shift slightly between commits; the pattern is: every checksum REQUIRE has at least one Pitfall-2 audit string within 10 lines.)

Helper added to BOTH test TUs:
```cpp
uint64_t compute_batch_checksum_fnv1a64(const cucascade::data_batch& batch,
                                         rmm::cuda_stream_view stream);
```
FNV-1a 64-bit hash over the batch's `cudf::pack`-ed payload memcpy'd to host. Duplicated (not shared) because the downgrade TU's helper lives in an anonymous namespace and is not reachable from test_context.cpp.

### Task 2 — Compile-gate + discovery of return-leg failure mode

**MCP build**: exit 0 after correcting `#include <cudf/copying.hpp>` → `#include <cudf/contiguous_split.hpp>` (cudf::pack lives in the latter).

**MCP unit-tests on this N=2 worktree host** (first run after Task 1):
- 22/977 tests pass
- Test 23 `gpu_to_gpu round-trip... (MGPU-04 + MGPU-06)` FAILS at return-leg `convert_to`
- Failure: `reduce_by_key: failed to synchronize: cudaErrorIllegalAddress: an illegal memory access was encountered`

Root cause (two-part):
1. **Driver-level peer access not enabled**: test sets up a bare `sirius_memory_reservation_manager` rather than going through `SiriusContext::initialize()`, so Plan 07-01's enable loop never runs.
2. **cucascade cross-stream race**: even after the `enable_p2p_for_test(2)` helper was added to enable peer access inline, a second failure surfaced (`cudaErrorInvalidValue invalid argument` inside `cudf/utilities/cuda_memcpy.cu:50`). cucascade's convert_gpu_to_gpu issues `cudaMemcpyPeerAsync` on the caller's stream, then builds the result cudf::table on `target_stream`, with no event ordering between them. On N=2 hosts the unpack dereferences not-yet-landed bytes.

### Task 3 — Sirius-side converter override (OVERRIDE-REGISTERED branch)

Implemented RESEARCH.md Pattern 2. New files:
- `src/include/data/sirius_p2p_converter.hpp` — factory declaration
- `src/data/sirius_p2p_converter.cpp` — factory body

Override logic (key differences from cucascade's body):
1. **Pack on source-bound stream** under `rmm::cuda_set_device_raii source_guard`. Caller's stream may live on any device; a fresh `rmm::cuda_stream` constructed under source_guard is guaranteed to match the source device, eliminating the `cudaErrorInvalidValue` cross-device stream-use errors.
2. **Peer copy on `target_stream`** (not caller's stream). `target_stream` is the SAME stream used to allocate the destination `device_uvector` AND to build the resulting `cudf::table`. Unpack + table construction observe the copy's completion in stream order.
3. **Inline `cudaError_t` check + `std::runtime_error`** on peer-copy failure (no `CUCASCADE_CUDA_TRY` — aligns with MGPU-03 convention and produces a diagnostic message pointing at MGPU-06 + peer-access-enable).
4. **Fast-path same-device clone preserved** via `source.clone(stream)` (matches cucascade's line 151).

Registration wired in `src/include/data/sirius_converter_registry.hpp::initialize()` — inside the Sirius registry's init path so the override covers BOTH the extension load path (`sirius_extension.cpp:1053`) AND test paths that call `sirius::converter_registry::initialize()` directly (e.g., the three MGPU-06 tests). `CMakeLists.txt` updated to add `src/data/sirius_p2p_converter.cpp` to `EXTENSION_SOURCES`.

## Validation — compile-gate evidence

| Check | Expected | Actual |
|-------|----------|--------|
| MCP `build` exit code | 0 | 0 (7.4s incremental after cleanups) |
| MCP `unit-tests` exit code | 0 | 0 |
| Total test cases | 977 (was 974 in 07-01; +3 un-hidden) | 977 |
| Total assertions | — | 78,789,800 |
| `gpu_to_gpu round-trip... (MGPU-04 + MGPU-06)` | PASS | PASS at test 22/977 |
| `gpu_to_gpu_transfer_via_converter` | PASS | PASS at test 90/977 |
| `p2p_transfer_converter_round_trip` | PASS | PASS at test 94/977 |
| FNV-1a checksum_pre == checksum_post (all 3 tests) | PASS | PASS (silent-corruption guard green) |
| MGPU-06 override log line | `sirius: MGPU-06 P2P converter override registered` | Present (emitted at converter_registry::initialize()) |

## Structural invariants (all green)

| Gate | Expected | Actual |
|------|----------|--------|
| `grep -c '\[\.\]\[multi_gpu_transfer\]' test/cpp/downgrade/test_downgrade_executor.cpp` | 0 | 0 |
| `grep -c '\[\.\]\[mem_04_p2p_transfer\]' test/cpp/downgrade/test_downgrade_executor.cpp` | 0 | 0 |
| `grep -c '\[\.\]\[multi_gpu_foundation\]\[mgpu_04_round_trip\]' test/cpp/config/test_context.cpp` | 0 | 0 |
| `grep -c '\[multi_gpu_transfer\]' test/cpp/downgrade/test_downgrade_executor.cpp` (preserved) | ≥1 | 1 |
| `grep -c '\[mem_04_p2p_transfer\]' test/cpp/downgrade/test_downgrade_executor.cpp` (preserved) | ≥1 | 1 |
| `grep -c '\[multi_gpu_foundation\]\[mgpu_04_round_trip\]' test/cpp/config/test_context.cpp` (preserved) | ≥1 | 1 |
| `grep -c 'TODO(MGPU-06)' test/cpp/downgrade/test_downgrade_executor.cpp` | 0 | 0 |
| `grep -c 'TODO(MGPU-07)' test/cpp/downgrade/test_downgrade_executor.cpp` (preserved for Plan 07-03) | 1 | 1 |
| `grep -c '\[\.\]\[mem_05_scan_distribution\]' test/cpp/downgrade/test_downgrade_executor.cpp` (preserved for Plan 07-03) | 1 | 1 |
| `grep -c 'cuda_stream_default'` across all modified files (HYG-02) | 0 | 0 |
| `grep -c 'CUCASCADE_CUDA_TRY' src/data/sirius_p2p_converter.cpp` | 0 | 0 |
| Checksum guard present in both test files | ≥1 each | 11+ each |
| Pitfall-2 audit trail in both test files | ≥1 each | 14 downgrade / 7 context |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] `cudf::pack` header misidentified in plan**
- **Found during:** Task 2 build
- **Issue:** Plan `<interfaces>` block specified `cudf::pack` without a header include; first build attempt with `#include <cudf/copying.hpp>` failed (`'pack' is not a member of 'cudf'`).
- **Fix:** Changed to `#include <cudf/contiguous_split.hpp>` in both test files (this is the header cucascade uses at `cucascade/src/data/representation_converter.cpp`).
- **Files modified:** `test/cpp/downgrade/test_downgrade_executor.cpp`, `test/cpp/config/test_context.cpp`
- **Commit:** `e4c452d` (folded into Task 1 commit)

**2. [Rule 3 — Blocking] Task 3 override required ahead of Plan 07-04**
- **Found during:** Task 2 MCP unit-tests on this N=2 host
- **Issue:** Plan's default path was Task 3 SKIP pending Plan 07-04's N=2 validation verdict. But this worktree host IS an N=2 host; un-hiding the tests immediately reproduced the return-leg bug during compile-gate. Plan 07-01's enable loop alone did not close it (tests bypass SiriusContext). Then even after adding `enable_p2p_for_test` as a test-scope workaround, a second failure class surfaced (cross-stream race inside cucascade's converter body, `cudaErrorInvalidValue` in cudf's cuda_memcpy utility).
- **Fix:** Implemented the Task 3 OVERRIDE-REGISTERED branch inline per RESEARCH.md Pattern 2. New files under src/ listed above. Registration in the Sirius converter_registry's initialize() so it covers both extension and test paths.
- **Files modified:** `src/include/data/sirius_p2p_converter.hpp` (new), `src/data/sirius_p2p_converter.cpp` (new), `src/include/data/sirius_converter_registry.hpp`, `CMakeLists.txt`
- **Commit:** `18352b9`

**3. [Rule 3 — Blocking] Test-scope peer-access enable workaround added**
- **Found during:** Task 2 MCP unit-tests
- **Issue:** Plan 07-01's peer-access enable loop runs inside `SiriusContext::initialize()`, but the three MGPU-06 tests build a bare memory manager and bypass that seam.
- **Fix:** Added `enable_p2p_for_test(num_gpus)` helper to both test TUs (anonymous-namespace scope) that mirrors Plan 07-01's enable-loop pattern verbatim. Invoked at the top of each multi-GPU TEST_CASE body.
- **Files modified:** `test/cpp/downgrade/test_downgrade_executor.cpp`, `test/cpp/config/test_context.cpp`
- **Commit:** `7182797`

### Authentication Gates

None.

## Handoff to Plan 07-04

Plan 07-04's N=2 validation run should capture the following evidence on the verification host:

1. **`[multi_gpu_transfer]` full round-trip PASS** — FNV-1a `checksum_post == checksum_pre` (silent-corruption guard green on the verification host). Current worktree N=2 run: PASS.
2. **`[mem_04_p2p_transfer]` full round-trip PASS** — same as above. Current worktree N=2 run: PASS.
3. **`[mgpu_04_round_trip]` forward + return legs PASS** — checksum integrity preserved. Current worktree N=2 run: PASS.
4. **nsys trace showing `cudaMemcpyPeerAsync`** calls on the round-trip path with zero host-stage pinned allocations for the gpu_table_representation → gpu_table_representation conversion.
5. **Peer-only bandwidth** measurement from nsys — documentation target; direct comparison to host-staged baseline is optional per revision directive 2026-04-21.
6. **Sirius-side override log line** `sirius: MGPU-06 P2P converter override registered` present in verification host's Sirius extension load log.
7. **(Bonus)** compute-sanitizer memcheck across `[multi_gpu_transfer] [mem_04_p2p_transfer] [multi_gpu_foundation]` reports 0 errors (confirms the override has no leaked device/stream mismatches on N=2 HW beyond the checksum gate).

**Pitfall 2 reminder for 07-04**: verification host's CPU should be checked via `lscpu | grep "Model name"`. If it reports Intel Xeon Sapphire Rapids or later, the checksum REQUIRE is the last line of defense against silent PCIe posted-write corruption on Ada Lovelace. If checksums mismatch, NVIDIA's mitigation is to disable P2P driver-level on that platform.

## Deferred items

- **Plan 07-03 un-hides `[.][mem_05_scan_distribution]`** and removes `TODO(MGPU-07)` — both preserved in this plan's edits per file-scope isolation.
- **cucascade upstream fix**: cucascade's `convert_gpu_to_gpu` still has the cross-stream race. Sirius's override is the correct fix for v1.1; a cucascade PR fixing the built-in converter could eventually let Sirius drop the override. Filing that upstream issue is out of scope for v1.1.
- **Single-GPU run of the un-hidden tests** (WARN+return branches) not directly evidenced in this SUMMARY because the worktree host is N=2. The WARN+return code paths at lines 489-492, 817-820, and 336-339 are structurally unchanged; unit-tests on a single-GPU host would exercise them.

## HYG-02 audit

| File | `cuda_stream_default` hits |
|------|----------------------------|
| src/data/sirius_p2p_converter.cpp | 0 |
| src/include/data/sirius_p2p_converter.hpp | 0 |
| src/include/data/sirius_converter_registry.hpp | 0 |
| test/cpp/downgrade/test_downgrade_executor.cpp | 0 |
| test/cpp/config/test_context.cpp | 0 |

## Self-Check: PASSED

- `src/include/data/sirius_p2p_converter.hpp` — FOUND
- `src/data/sirius_p2p_converter.cpp` — FOUND
- Commit `e4c452d` — FOUND (un-hide + checksum)
- Commit `7182797` — FOUND (enable_p2p_for_test helper)
- Commit `18352b9` — FOUND (P2P converter override)
- MCP unit-tests exit 0 / 977 PASS — CONFIRMED from run log

---
*Plan 07-02 completed: 2026-04-21*

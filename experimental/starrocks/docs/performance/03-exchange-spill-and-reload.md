**03 · Spillable exchange repositories and reload**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: let parked exchange data outlive fragment execution without escaping memory accounting or becoming impossible to export. Measurement prerequisite: [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md). Remote lease ownership is supplied by [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md).

**Current behavior**

Streaming fragments allocate input/output repositories separately so output survives ordinary query-window cleanup. The downgrade executor discovers candidates through registered per-query repository managers. Export and output row counting require GPU residency. Simply putting exchange data into the existing repository type does not establish spill discovery, lifetime, or reload.

**Code map**

| Source | Responsibility |
|---|---|
| [streaming_fragment.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/exec/streaming_fragment.cpp#L100) and [streaming_fragment.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/exec/streaming_fragment.hpp) | Exchange repository creation and lifetime. |
| [data_repository_manager_registry.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/data/data_repository_manager_registry.hpp) | Query-scoped manager registration and retirement. |
| [downgrade_executor.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/downgrade/downgrade_executor.cpp#L223) | Candidate enumeration and safe conversion. |
| [sirius_context.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_context.cpp) | Query lifecycle cleanup and resource ownership. |
| [sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_ffi.cpp#L724) and [lib.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/rust/crates/sirius/src/lib.rs) | Reload-aware export and residency-independent row counts. |
| [parked_registry.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/parked_registry.rs) | Destination claims and FE-query retirement. |

**Proposed ownership model**

Introduce an exchange-lifetime registration handle, separate from a single fragment's execution window. Map the FE query identity to the relevant Sirius execution IDs explicitly; do not assume they are interchangeable. The handle retains repositories while any parked destination, ingress operation, reload, pack job, or consumer owns data. Registration makes candidates discoverable; it must not cause ordinary `QueryEnd` to clear output still needed by another fragment.

Keep schema, exact row count, logical byte size, sequence, and producing identity with a batch independent of its GPU/host/disk representation. Export asks for a read reservation, requests reload if needed, waits asynchronously for residency, and only then packs. Completion/error paths drop the reservation once. Preserve batch writer events and make conversion mutually exclusive with readers.

**Implementation slices**

1. **Lifetime audit and registration seam:** enumerate every exchange repository, including result-fragment input, and document who creates, registers, clears, and destroys it. Add an explicit lifetime handle using supported cuCascade interfaces; extend the registry if it cannot register external repositories safely.
2. **Metadata preservation:** capture row counts at production/import, keep them through spill/reload, and replace GPU-only counting for exchange cardinality. Preserve “unknown” as distinct from zero.
3. **Spill and reload:** wire idle exchange data into downgrade selection. Add export/relay readiness tickets that reserve enough destination space before reload and yield while it is unavailable. Do not reload all parked output eagerly.
4. **Retirement:** stop new claims on cancellation, drain active conversions/readers, then unregister and free. Test normal fragment cleanup while downstream data remains live, and concurrent retirement of a different query.

**Validation**

Extend [test_streaming_fragment.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/test/cpp/exec/test_streaming_fragment.cpp), [test_sirius_ffi_fragment.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/test/cpp/exec/test_sirius_ffi_fragment.cpp), and the downgrade lifecycle tests. Force GPU-to-host and host-to-disk transitions, export after reload, count rows while spilled, cancel during each conversion, and run two queries where only one retires. Include strings/nulls and empty outputs.

Acceptance: every exchange byte has one accountable owner; the downgrade sweep can discover eligible parked data; output survives the producing fragment's cleanup; final retirement frees it. Exactly one destination consuming a broadcast-like shared representation must not invalidate other owners. A small active fragment must make progress while a large idle exchange is spilled.

**Performance experiment**

Hold output volume fixed and sweep compute capacity and receiver delay. Report GPU/host/disk occupancy, spill/reload bytes, conversion time, query latency, and pack wait. Compare a memory-fitting control and a pressure case. No-pressure regressions matter: metadata and registration must not impose a heavyweight scan on every batch.

**Rollout and decisions**

Start with explicit exchange registration behind an opt-in path, then enable reload-aware export. Rollback changes admission for new queries; existing handles retain their original lifetime rules until drained. Do not re-enable an old export path for data already spilled.

Before implementation, load the repository's module-context documentation for cuCascade/RMM and verify external-repository registration APIs. Choose whether a dedicated exchange registry or a longer-lived query manager best preserves cleanup invariants; that choice must be resolved before early ingress relies on spill.

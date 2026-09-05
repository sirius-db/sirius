**01 · Retry-safe leases and transport recovery**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: preserve usable staging capacity and predictable recovery after connection failures, query cancellation, and peer restarts. Depends on [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md) for regression evidence.

**Current behavior and code map**

A lease request carries only a length. The receiver allocates before replying. A cached PRPC connection can fail after that allocation and trigger a new attempt, allocating twice. A failed WRITE or unpublished batch is not associated with its query at the receiver, so ordinary query cleanup cannot find it.

| Source | Planned change |
|---|---|
| [nixl-exchange-proto.patch](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/patches/nixl-exchange-proto.patch) | Versioned capabilities, allocation token, ownership, and release/query-status messages. |
| [prpc_client.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/prpc_client.rs#L66) | Method-specific retry policy and stable logical request identity. |
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L1442) | Owned grant ledger, validation, release, and retirement. |
| [local_exchange.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/local_exchange.rs#L319) | Durable duplicate/publication/EOS accounting through the retry horizon. |
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs#L673) | Sender lease guards, transfer completion, cleanup, and canary ownership. |
| [exchange_staging_arena.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/exec/exchange_staging_arena.hpp) | Keep byte allocation separate from distributed ownership; expose safe diagnostics. |

**Proposed protocol and ownership**

Names here are design proposals, not existing interfaces. A `LeaseToken` includes receiver process epoch plus an allocation identity. A grant request includes sender epoch, query, producer, receiver, exchange, sender ordinal, frame sequence, requested length, and a stable request ID. Repeating a request returns the same grant only when all immutable request fields match; disagreement fails explicitly.

Use states such as granted, transfer-active, published, copying/reading, reclaimable, and quarantined. Host bookkeeping and device completion are separate. Publication acknowledges durable receiver ownership, not permission to overwrite destination memory. Local source reuse follows confirmed transfer-read completion; receive reuse follows last-reader or copy-out completion.

Keep tombstones for completed allocations and sender EOS across the retry window. Bound ledger/tombstone memory using explicit session retirement and admission limits. Removing arbitrary oldest live entries is not valid reclamation. Canary allocations use the same identity rules even though they have no query.

**Implementation slices**

1. **Protocol negotiation:** add a lease-protocol capability/epoch handshake and schema tests. Retain existing wire fields where useful, but reject unsupported semantics when the new mode is requested. Updating the checked-in patch and generated-binding inputs is part of this slice.
2. **Idempotent grant ledger:** atomically deduplicate grants; validate sizes and owner identity; retain a grant if its response is lost. Add allocation-count tests using a fake arena.
3. **Publication and release lifecycle:** enforce legal state transitions, terminal-frame replay handling, query ownership at grant time, and idempotent release. Replace blanket RPC retry with policies appropriate to each method.
4. **Failure/quiescence:** add explicit cancellation/status exchange, peer-epoch invalidation, and deferred reclamation. Verify what the pinned NIXL request destructor/abort APIs guarantee. If transfer quiescence cannot be proven, quarantine the range until safe peer/session teardown; a TTL must not free a possibly active WRITE target.

**Validation and benchmarks**

Inject loss before/after grant, after WRITE completion, before/after publication processing, after EOS dispatch, and during release replies. Repeat each request with identical and conflicting fields. Cover zero-length metadata frames, canaries, cancellation before FE receiver registration, and peer restart at the same address.

Acceptance: duplicate requests allocate once; a fresh epoch never inherits stale addresses; completed cancellation returns live bytes to baseline; a subsequent query succeeds. For unresolved transfers, telemetry must show the exact quarantined bytes and bounded admission behavior. No double release or memory reuse before device quiescence is acceptable.

Measure steady-state ledger/RPC overhead at small payloads and recovery latency under a controlled fault rate. The primary gain is avoided capacity loss and restart time, not a promised bandwidth improvement.

**Rollout and alternatives**

Gate new protocol use on peer capability, with explicit homogeneous-cluster deployment first. Rollback applies to new sessions only after all new-protocol allocations drain. Disabling retries alone is an emergency mitigation for duplicate allocation, but does not reclaim a grant whose reply was lost. Do not call that the complete fix.

**Open decisions**

Resolve NIXL quiescence guarantees, bounded tombstone retirement, and the receiver ledger's lifetime across CN shutdown. The plan assumes in-memory ownership; durable restart recovery would need a separate design and cannot make an old GPU address valid.

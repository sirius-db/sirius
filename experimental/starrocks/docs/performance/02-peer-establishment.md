**02 · Nonblocking peer establishment**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: remove reciprocal cold-peer waits and isolate cold-peer latency from healthy transfer progress. Depends on [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md); integrates with [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md) for canary/epoch recovery.

**Problem and code map**

Warmup runs in the background after the agent starts. The CN becomes schedulable after its BRPC listener starts, before every peer is established. A query can still enter the lazy blocking metadata RPC on the transport thread. Its peer may need that same thread to answer a reciprocal request.

| Source | Planned responsibility |
|---|---|
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs#L554) | Replace blocking `ensure_session` with a nonblocking lookup/request transition. |
| [warmup.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport/warmup.rs) | Reuse discovery as a producer of setup requests, not a separate competing setup mechanism. |
| [main.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/main.rs#L205) | Distinguish listener/process readiness from destination session readiness. |
| [prpc_client.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/prpc_client.rs) | Control-worker connection and timeout behavior. |
| [tunable.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/tunable.rs) | Validated bounded setup concurrency and retry parameters, names to be chosen. |

**Proposed design**

Maintain one per-peer session state: absent, discovering, handshaking, installing, probing, ready, or failed/backoff. A query encountering a non-ready peer records a bounded continuation instead of blocking the transport owner. Warmup and query-driven setup coalesce onto the same state. Limit pending bytes and setup requests per peer and CN.

A control worker performs remote metadata RPCs. The transport owner imports metadata and posts agent work. Return control-worker results as messages carrying the expected peer/session epoch so stale completion cannot overwrite a newer session. A canary may itself need asynchronous control/transfer progression; until that path is asynchronous, isolate its bounded waits and measure them. Completion of metadata exchange alone is not proof that a canary passed.

Do not simply wait for all peers before advertising the CN alive: FE discovery depends on peers registering and reporting endpoints. Either retain process liveness with destination-specific scheduling, or define a separate discovery phase whose progress does not depend on query readiness.

**Implementation slices**

1. **State-machine seam:** extract session setup coordination from the NIXL-specific calls; add deterministic fake-control/fake-agent tests for simultaneous requests, retries, shutdown, and stale completions.
2. **Shared setup path:** route both background discovery and lazy query setup through the coordinator. Remove outgoing blocking metadata RPCs from the transport thread and deduplicate repeated setup attempts.
3. **Query continuation:** resume a parked destination only after agent import and probe success. Fail that query edge with its concrete setup error on deadline; leave unrelated peers available.
4. **Lifecycle/observability:** expose peer-ready counts, setup states, cold wait time, attempts, and epoch changes. Preserve the existing allocation/registration path checks. Make shutdown stop setup admission, resolve pending tickets, then join workers.

**Correctness and progress checks**

Run an immediate bidirectional exchange on a cold two-CN cluster, a simultaneous four-CN all-to-all start, delayed FE discovery, a late peer, a restarted peer at the same endpoint, and an unavailable peer alongside a healthy one. Include warmup disabled: query-driven setup must still progress correctly. Ensure duplicate metadata imports and stale probe replies cannot corrupt the active session.

Acceptance: transport metadata requests remain serviceable while outbound setup waits; one failed peer does not prevent a healthy peer's setup/traffic; every waiting query gets completion or a bounded error. No thread should wait on work that only it can service.

**Performance experiment**

Report first-query latency separately from steady state, and break cold time into discovery, metadata RPC, import, canary, and queued wait. Compare startup control traffic and CPU use at 2/4/8 CNs. Setup is nominally O(peers²) for an all-to-all cluster; avoid eagerly probing unused peers at large scale unless measurements justify it.

**Rollout and decision gate**

Use an opt-in coordinator mode first and retain background discovery as a compatibility aid. Roll back before new session creation; do not swap active agent registrations mid-transfer. Accept the path when cold/late-peer stalls disappear under the deterministic scenarios with no warm-path regression beyond measured noise.

The unresolved choice is a small bounded blocking control pool versus an async client. Choose from measured setup fan-out and existing runtime constraints; neither option may block the agent owner on reciprocal remote work.

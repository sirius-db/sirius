# Engine-side stack research

Reviewed tree: /home/ubuntu/sirius-wt/s500-mig at 7610840c, relative to upstream/dev
98fe1a84335616b8674eec7e9ed2594073503fd0. This is a static review; no engine code
was changed and the C++/Rust/GPU test suites were not run.

## Big picture

~~~mermaid
flowchart LR
  FE[StarRocks FE plan] --> TR[Translator]
  TR --> P[Substrait / sirius_stream_id]
  P --> F[FFI Context and Fragment]
  F --> I[STREAMING_SOURCE]
  I --> G[Super Sirius GPU pipeline]
  G --> O[STREAMING_SINK]
  O --> A[chunked_pack / staging arena]
  A --> N[NIXL GPU write]
  N --> S[InboundStore deep copy]
  S --> I
~~~

The staging arena is a short-lived, registered wire region, not the exchange queue.
Engine-local queues are spillable cucascade data-batch repositories. A remote frame is
copied into ordinary pool memory before its receiver is runnable, then the staging lease
returns immediately.

## PR map and landing status

| PR | Commit on this branch | What it establishes |
|---|---|---|
| [#1094](https://github.com/sirius-db/sirius/pull/1094) | historical 23a76741..367fa9d6 | original streaming source operator plus exchange channel; merged 2026-07-13 |
| [#1320](https://github.com/sirius-db/sirius/pull/1320) | historical 4a2d8cd6 | repository-backed source, sender-aware EOS, and producer-error plane; merged 2026-08-11 |
| [#1479](https://github.com/sirius-db/sirius/pull/1479) | historical cfac11dc | partitioned streaming sink over output repositories; merged 2026-08-17 |
| [#1480](https://github.com/sirius-db/sirius/pull/1480) | historical 42ed7f3a | stream_session, ID-addressed router; merged 2026-08-18 |
| [#1481](https://github.com/sirius-db/sirius/pull/1481) | historical stack | streaming fragment builder/runner and Fragment FFI; merged 2026-08-24 |
| [#1693](https://github.com/sirius-db/sirius/pull/1693) | 1e61c16c | fixed-address CUDA staging allocator |
| [#1694](https://github.com/sirius-db/sirius/pull/1694) | f1e8fb17 | optional stream cardinality for DuckDB planning |
| [#1696](https://github.com/sirius-db/sirius/pull/1696) | 94a77836 | StarRocks-compatible byte-range ownership |
| [#1700](https://github.com/sirius-db/sirius/pull/1700) | a22235e1 | range-aware Parquet ingest; partial reads bypass pins |
| [#1697](https://github.com/sirius-db/sirius/pull/1697) | 98661d7d | FFI-owned lowering transaction |
| [#1702](https://github.com/sirius-db/sirius/pull/1702) | 14386a77 | Rust bindings for Context/Fragment |
All current drafts are already merged into the branch integration ancestry. The post-draft
production exchange commits are: 92e91adf (FFI byte ranges), b7452f67
(packed leases and declared cardinality), d88265aa (translator byte-range split emission),
85da8f6f (park local output), 37b48a46 (receiver-first rendezvous), 86b3f9a6
(PRPC client), 36072ef8 (NIXL agent/arena canary), 9e547c89 (GPU-to-GPU transfer),
49d23e8a (peer warmup), 698312a1 (inbound pool copy), b00334cb (claim sequence before
staging), 5e71059c (concurrent PRPC), and 37360a1b/c91d0a84 (cancellation/failure report).
  [*] --> Waiting
  Waiting --> HasData: push
  HasData --> Waiting: pull drains; sender open
  Waiting --> EOS: all distinct senders close
  HasData --> EOS: pull drains after final close
  Waiting --> Poison: fail
  HasData --> Poison: fail
  Poison --> Poison: try_pull rethrows
~~~

batch_stream has sender-set EOS, poison-dominant failure, and persistent source wakeups.
STREAMING_SOURCE never blocks task creation; waiting drops the request and an on-data hook
re-nominates it. STREAMING_SINK closes output only on pipeline finalization.

There is no stream credit or channel-level backpressure: see
[streaming-sessions.md](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/docs/super-sirius/streaming-sessions.md#L356).
Pressure belongs to cuCascade tiers/spilling, scan admission, and an arena that fails loudly
when live leases exceed capacity.

## FFI and DuckDB lowering

At bind time, sirius_stream_source(id) gets schema from stream_bind_catalog; physical planning
replaces it with STREAMING_SOURCE, rather than running the table function
([stream_plan_bindings.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/stream_plan_bindings.cpp#L60)).
The catalog rejects two plan leaves on one stream, avoiding an orphaned receiver
([stream_bind_catalog.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/stream_bind_catalog.cpp#L92)).

PR 1694 supplies the cardinality callback
([stream_plan_bindings.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/stream_plan_bindings.cpp#L68)).
Without a declaration it returns null, keeping DuckDB's old one-row estimate.

Fragment build first commits a setup transaction for types/views, then acquires the query
lifecycle and lowers the Substrait plan. PR 1697 makes lower_substrait own a ClientContext
transaction when none is active
([sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L91)).
Repository/engine/session declaration order is a lifetime invariant
([streaming_fragment.hpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/exec/streaming_fragment.hpp#L112)).
Run failure poisons outputs before rethrowing
([streaming_fragment.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/streaming_fragment.cpp#L201)).

## Staging arena and transport

~~~mermaid
sequenceDiagram
  participant SF as sender Fragment
  participant SA as sender arena
  participant RA as receiver arena
  participant IS as InboundStore
  participant RS as receiver stream
  SF->>SA: lease(total plus 8 MiB); chunked_pack
  SA->>RA: NIXL GPU write into receiver-issued lease
  RA->>RA: claim(sender, seq) before payload touch
  RA->>IS: unpack and deep-copy to pool
  IS->>RA: release lease immediately
  IS->>RS: keep ticket until receiver runs
~~~

The arena is deliberately cudaMalloc outside RMM, with optional cross-host fabric VMM mode.
It uses 256-byte contiguous aligned leases and an address-ordered coalescing free list
([exchange_staging_arena.hpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/exec/exchange_staging_arena.hpp#L26)).
export_packed needs payload plus 8 MiB packing slack; a zero-row batch takes no lease
([sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L869)).
push_packed validates schema and deep-copies out of the lease
([sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L949)).

CN code claims a frame sequence before copying/releasing it, preventing a BRPC replay from
touching a reused offset
([compute_node_service.rs](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/compute_node_service.rs#L797)).
It copies inbound frames before receiver dispatch to keep the arena from retaining shuffle
backlog ([compute_node_service.rs](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/compute_node_service.rs#L841)).

## Byte-range scans and pins

~~~mermaid
flowchart LR
  LF[Substrait LocalFiles start,length] --> EX[range extraction]
  EX --> CS[per-plan ClientContext state]
  CS --> PG[physical plan claims file]
  PG --> PI[Parquet ingestible]
  PI --> OW[row-group start containment]
  OW --> PR[stats prune] --> RD[cuDF read]
~~~

A range owns a row group when its StarRocks-style start offset is within start through
start-plus-length; a group crossing the end belongs to the range containing its start.
Plan state ensures paths claim ranges once and every range is consumed
([substrait_scan_ranges.hpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/planner/substrait_scan_ranges.hpp#L32)).
Ownership runs before stats pruning
([parquet_gpu_ingestible.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/op/scan/parquet_gpu_ingestible.cpp#L822)).

Partial-file scans cannot use whole-file pins. PR 1717 records canonical source file provenance
per pinned chunk, allowing a pin of A,B,C to serve A,C. The integration test covers GPU/host,
subset/exact/superset, zone-map pruning, and the empty-result sentinel
([test_pin_table_file_subset.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/test/cpp/integration/test_pin_table_file_subset.cpp#L196)).

## Confirmed review findings

### P2: malformed byte-range arithmetic wraps instead of rejecting the split

[row_groups_in_byte_range](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/op/scan/parquet_byte_range.cpp#L56)
uses unsigned end equals start plus length without overflow validation. The FFI accepts
LocalFiles start/length unchanged
([substrait_scan_ranges.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/planner/substrait_scan_ranges.cpp#L87)).
This is a malformed-input validation gap rather than a practical normal-FE data-loss path:
Parquet row-group offsets are signed, so start near UINT64_MAX cannot own a real group. It still
has a concrete unsafe input: start = 4 and length = UINT64_MAX wraps end to 3, excluding a valid
row group starting at 8 even though the unwrapped interval contains it. The LocalFiles protobuf
values are accepted without a range check. Existing tests cover ordinary tilings/boundaries, not
overflow ([test_parquet_byte_range.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/test/cpp/scan/test_parquet_byte_range.cpp#L138)).

Fix: reject length greater than UINT64_MAX minus start in extraction or ownership and require
invalid_input_exception in a regression test. Treat this as P2 until the FE can emit such a range.

### P1: InboundStore stage races Context destruction on a raw GPU-memory-space pointer

Rust makes InboundStore Send plus Sync and documents that late calls error
([lib.rs](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/rust/crates/sirius/src/lib.rs#L540)).
The following interleaving is permitted and confirms the UAF:

1. Thread A enters stage, locks State, copies raw gpu_space, and unlocks
   ([sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L435)).
2. Thread B drops Context. close takes that mutex, clears staged state and sets gpu_space null
   ([sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L176)); Context then
   destroys its implementation/memory manager
   ([sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L316)).
3. A continues outside the lock and dereferences stale gpu_space at
   [line 451](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L451). The later null check
   is after the dereference.

Fix: close must wait for an in-flight-stage guard, or stage must retain a shared owner of the
memory space. Add a controlled two-thread test paused after stage's first lock; destruction
must wait or stage must return the documented error.

### P2: checked-in NIXL proto patch fails whitespace validation

git diff --check upstream/dev...HEAD was run. It reports trailing whitespace at lines 8, 52,
and 56 of experimental/starrocks/patches/nixl-exchange-proto.patch. Clean/regenerate the
patch before review-ready status.

## Validation

| Evidence | Present | Run in this review |
|---|---:|---:|
| arena allocator tests | yes | no |
| FFI transaction test | yes | no |
| stream catalog/fragment tests | yes | no |
| byte-range/ingest tests | yes | no |
| pinned-file-subset GPU/host integration | yes | no |
| Rust packed/inbound tests | yes | no |
| git diff --check upstream/dev...HEAD | n/a | yes; fails on three whitespace errors |

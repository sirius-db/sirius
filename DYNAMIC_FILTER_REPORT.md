# Multi-partition dynamic filters: implemented design and engineering handoff

**Date:** 2026-08-12
**Status:** Implemented behind a default-off switch with a per-GPU Bloom policy cap; single-GPU focused validation and TPC-H SF1000 evaluation complete; multi-GPU runtime validation deferred
**Base:** PR #1277 head `49de08e8`, integrated by merge commit `bdeaa56b`
**Audience:** Engineers and reviewers extending, validating, or enabling the feature

## 1. Conclusion

Sirius should use one globally complete Bloom filter per eligible join key for a hash-partitioned build. The filter is built incrementally from the original build batches before hash scatter. On multiple GPUs, each producing GPU builds an equal-geometry partial, the partials are bitwise-OR reduced on a deterministic root GPU, and the complete result is replicated before it is published through the existing dynamic-filter channels.

This is implemented as an extension of PR #1277. It is controlled by `enable_dynamic_filter_multi_partition`, which defaults to `false` and is subordinate to the existing `enable_dynamic_filter` master switch. Bloom construction is additionally bounded by `max_dynamic_filter_bloom_bytes_per_gpu`, default 256 MiB per producing join on each GPU. A controlled TPC-H SF1000 study measured 5.93% lower paired-block suite runtime than one-shot-only mode and 9.31% lower equal-query cohort runtime for the three true multi-build-batch publications. The switch remains off by default because the result used an activation-oriented single-GPU configuration and multi-GPU runtime evaluation is still outstanding. See `DYNAMIC_FILTER_TPCH_SF1000_EVAL.md` for the complete methodology and qualification.

Partition-specific filters delivered to `CONCAT` are not the primary architecture. They would become available after probe partitioning and scatter, require the consumer to reproduce or carry the exact partition identity, and would not fit the current global channel semantics without a new filter kind and routing contract. The global Bloom reuses the existing post-scan consumer, preserves one membership predicate for the whole join key, and keeps the exact hash join authoritative.

## 2. Scope and implementation status

| Area | Implemented behavior | Validation status |
|---|---|---|
| Feature control | Master switch plus default-off multi-partition subordinate switch in YAML and SQL | Unit and SQL tests pass |
| Bloom admission | Per-join, per-GPU allocator-accounted bit-array cap across all Bloom keys; fail-open all-or-none skip | Boundary, one-shot, accumulator, and planned-policy tests pass |
| Build snapshot | Validated `complete_build_snapshot` with exact pre-scatter IDs, global row count, and partition count frozen at the build `PARTITION` FULL barrier | Focused summary/validation and integration coverage |
| Single GPU | Incremental global Bloom with one `add` per exact build contribution | Focused GPU tests pass |
| Multiple GPUs | One full-geometry partial per producing GPU, deterministic-root OR, strict replication, then fan-out | Compiles; runtime evaluation deferred |
| Existing one-shot path | Preserved for whole single-partition and broadcast builds | Existing PR #1277 behavior retained |
| Consumer placement | Existing dynamic-filter channel and post-scan/join-edge consumers | No new pre-partition or serve-side checkpoint |
| Incremental filter kinds | Bloom membership for supported INT32/INT64 keys | Unit tests pass |
| Zone maps and exact sets | Existing one-shot policy only; not accumulated across partitions | Deliberately out of scope |

## 3. Why a global Bloom is the right first design

Correctness requires a membership filter to represent every build key that can match the filtered probe side. A source partition Bloom alone is incomplete. Publishing source partials into the existing channel would be wrong because filters in that channel are combined as predicates, while build partials need an internal OR union.

A globally complete Bloom has the following properties:

- Every build contribution uses the same layout derived from the exact global row count.
- Bloom insertion and Bloom union are both idempotent bit operations.
- A single-GPU build is the degenerate case with no reduction.
- Multi-GPU execution adds only the reduction stage; the publish and consume contracts stay unchanged.
- Consumers either see a complete immutable filter or see no filter and pass rows through.
- False positives only preserve extra rows. The exact join still decides the result.

A `CONCAT`-local design can be revisited for already partitioned and colocated data if global reduction or replica memory is later measured as the bottleneck. It is not a good default for the current Sirius pipeline because it forfeits scan-side and pre-exchange pruning and couples filtering to partition routing.

## 4. Configuration contract

The effective multi-partition setting is:

```text
enable_dynamic_filter &&
enable_dynamic_filter_multi_partition
```

| `enable_dynamic_filter` | `enable_dynamic_filter_multi_partition` | Result |
|---|---|---|
| false | false | No dynamic-filter discovery or publication |
| false | true | No dynamic-filter discovery or publication; the master switch dominates |
| true | false | PR #1277 one-shot publication only; hash-partitioned multi-partition builds remain skipped |
| true | true | One-shot behavior plus global Bloom accumulation for eligible non-broadcast builds with more than one partition |

All three controls are accepted under `sirius.operator_params` and as DuckDB SQL settings. The multi-partition switch defaults to false; `max_dynamic_filter_bloom_bytes_per_gpu` defaults to 256 MiB. A value of 0 disables Bloom construction without disabling exact IN-lists, zone maps, discovery, or the authoritative join.

The cap applies to both the whole-build one-shot path and multi-partition accumulation. For one join it sums the allocator-aligned bit-array footprint of every Bloom candidate on one GPU, using the exact global build row count and overflow-safe arithmetic. It is not multiplied by GPU count or divided by partition count. Equality is admitted for representable footprints; if the complete candidate set exceeds the cap or its aligned footprint is not representable, every Bloom candidate is skipped before allocation so admitted-key order cannot determine which predicate survives.

## 5. Build snapshot and contribution identity

The build `PARTITION` input is a FULL pipeline barrier. After the partition strategy is known and before the first source batch is removed, `sirius_physical_partition::try_freeze_complete_build` freezes a move-owned `complete_build_snapshot`:

- the exact original data-batch ID set;
- the exact global build row count;
- the selected number of hash partitions.

`sirius_physical_partition::try_freeze_complete_build` succeeds only when the source pipeline is finished, every snapshotted representation is GPU resident, and row-count accumulation does not overflow. It then calls `complete_build_snapshot::try_create`, which validates a non-empty unique ID set, a partition count greater than one, and a global row count representable as `std::size_t`. Zero build rows are valid. Moving the snapshot invalidates its source; `dynamic_filter_publication_session::try_arm` rejects invalid or moved-from values before claiming the session or incrementing `publication_attempts`.

`pipelineable_operator_data` captures the original task-input ID when the repository batch is popped. Cross-GPU preparation may replace that batch with a fresh-ID physical clone, and retry can retain the clone, but each call to `sirius_physical_partition::execute` contributes the unchanged task-input ID and the admitted build-key columns before `gpu_partition_impl` scatters the table. Hash partition outputs are never treated as independent logical contributions.

This design also handles strategies whose sizing decision is driven by the sibling partition: arming occurs on the build `PARTITION` only after the shared strategy is fixed and its own FULL source is complete.

## 6. Publication lifecycle

The hash-join-owned `dynamic_filter_publication_session` owns the arbitration state:

```text
                         +-> PUBLISHING -> FINISHED
                         |                `-> FAILED
OPEN --one-shot claim---+
  |
  +--successful accumulator initialization-> ACCUMULATING -> FINISHED
  |                                                          `-> FAILED
  +--failed accumulator initialization-------------------------------> FAILED
  |
  `--finalize unclaimed--------------------> CLOSED
```

The one-shot and accumulator paths claim `OPEN` under the session mutex, so they cannot both publish. `publication_attempts` increments when a one-shot claim succeeds or when accumulated-claim initialization begins; accumulator-construction failure therefore records both an attempt and a failure. Arming installs the exact-ID accumulator and changes the state to `ACCUMULATING`. A complete accumulator moves that state to `FINISHED`; a construction, reduction, replica, or incomplete-finalization failure moves it to `FAILED`. GPU construction, insertion, synchronization, reduction, replication, and fan-out run without the session mutex or `op_state_mutex`. Source-not-resident and build-not-whole diagnostics increment only while the session remains `OPEN`.

`sirius_physical_hash_join::on_finalize_operator` delegates to `dynamic_filter_publication_session::finalize_or_abort`, which never manufactures a filter. It closes an unclaimed `OPEN` window or aborts an `ACCUMULATING` window that is still missing an expected contribution.

`dynamic_filter_publication_session` folds each terminal outcome exactly once. An accumulator-construction failure has no outcome and records failure directly. This preserves the existing counters for attempts, success, failure, policy decisions, filters built, drained targets, and accepted pushes.

## 7. Accumulator contract

`dynamic_filter_accumulator` owns immutable expected IDs, in-flight IDs, completed IDs, per-device partials, and the publication outcome.

For each contribution:

1. Reject an ID outside the frozen expected set.
2. Return a duplicate result for an ID already in flight or complete.
3. Reserve the ID as in flight under the coordinator mutex.
4. Lock only that producing GPU's partial.
5. Preflight every active key ordinal and runtime type before inserting any key from the batch.
6. Lazily allocate full-global-geometry Bloom partials on a durable stream owned by the producing GPU memory space, complete their asynchronous initialization, and then insert all active key columns on the supplied task stream.
7. Synchronize the stream before moving the ID from in flight to complete.
8. Publish only when the completed ID set exactly equals the expected ID set.

Different GPUs use different partial mutexes and can insert concurrently. Contributions on the same GPU serialize. Coordinator state is not held during ordinary insertion, but the final contribution performs reduction, replication, and fan-out while holding the accumulator coordinator mutex. That simple exactly-once policy is acceptable for a default-off first implementation, but its scheduling cost should be measured.

A runtime type mismatch aborts the entire optional publication. The all-key preflight prevents a contribution from updating one key before discovering that another key is incompatible. Previously completed private partials remain unreachable and are never fanned out. When the session commits any terminal result, it drops its accumulator reference; an in-flight caller's local shared reference keeps the accumulator valid until that call returns.

## 8. Single-GPU completion

On one GPU, the device partial already represents the union of every expected build contribution. If the deterministic root has received contributions, its partial is reused directly. If it has not, an empty full-geometry root is created before reduction.

After the final contribution is synchronized, the root filter is replicated to every planned consumer device and then pushed through the immutable target bindings. On a one-GPU plan, replication is already satisfied by the root source replica.

The accumulator intentionally emits Bloom membership only. Existing one-shot publication continues to select small raw IN lists, hash IN lists, Blooms, and optional zone maps under its existing policy.

## 9. Multi-GPU completion

The publish plan sorts replica spaces by physical device ID. The first, lowest-ID planned GPU is the deterministic reduction root. Correctness does not depend on where contributions ran.

For every active key:

1. Ensure the root owns an empty or contributed full-global-geometry Bloom.
2. Visit every non-root producing partial.
3. Validate key type, Bloom variant, source/root device association, and exact word extent.
4. Copy at most 4 MiB of Bloom words at a time into root-device scratch.
5. OR the chunk into the root Bloom on the same root stream.
6. Reuse the scratch for the next ordered chunk.
7. Synchronize the root stream before treating the union as complete.
8. Release reduction scratch and clear all non-root partial vectors.
9. Strictly replicate the complete root Bloom to every device in the immutable replica plan.
10. Fan out the filters only after every required replica is available.

The transfer helper uses the same direct-peer or bounded host-staging path as PR #1277 replica construction. Chunk copies and OR kernels are ordered on the root stream. Each active root filter owns at most 4 MiB of device reduction scratch; with multiple join keys, the temporary bound is 4 MiB per active key until the root synchronization completes.

Strict replication is intentionally atomic at the logical publication level. If any planned device replica is unavailable, the accumulator aborts and no target receives the filter, even if some private replicas were created successfully. This is conservative but prevents partial cross-device visibility in the first implementation.

## 10. Correctness and failure behavior

The implementation maintains these invariants:

1. No filter is visible before all exact expected build IDs complete.
2. Every producing partial uses identical global Bloom geometry and hash semantics.
3. Producer partials are OR-reduced internally; independently complete filters remain conjunctive at consumers.
4. Published filters and replicas are immutable.
5. Duplicate contributions never advance completion twice.
6. Missing, unknown, incompatible, or failed contributions cannot produce a filter.
7. The exact hash join remains present and authoritative.
8. Optional-filter absence is a safe pass-through.
9. A validated zero-row snapshot completes successfully without constructing or publishing a Bloom.
10. Once publication completes, a late duplicate or invalid contribution cannot replace its terminal success.

Construction exceptions are caught at the accumulator boundary and close the publication attempt without fan-out. A best-effort stream drain runs after one-shot construction, accumulated insertion, or root-reduction failure before source or private storage can be released; the one-shot path preserves the original exception after its drain. Mandatory abort and terminal accounting happen before best-effort logging, so logging failures cannot escape contribution or finalization cleanup or replace the original one-shot exception. An incomplete build at operator finalization is recorded as a failed publication. A target set that has already drained completes successfully with no filter; because accumulated policy is evaluated when the session is armed, that terminal result may retain deterministic policy counters alongside the drained-target counter. Terminal resolution is read atomically from the accumulator, so a throwing in-flight caller cannot replace another caller's completed publication or discard an aborted outcome's counters.

A CUDA failure that also makes ordinary query execution unusable can still surface later through the normal execution path. The dynamic-filter layer does not claim to recover a poisoned device or stream; it guarantees only that an incomplete optional predicate is not published.

## 11. Memory and lifetime

For an allocator-accounted Bloom bit array of `m` bytes, `D` producing GPUs, `S` planned consumer GPUs, and `K` active keys:

- private producer memory is at most approximately `D * K * m`;
- final replica memory is approximately `S * K * m`;
- root reduction scratch is at most `K * 4 MiB`;
- non-root partial vectors are released before strict final replication;
- transfer staging uses the existing planned host memory space.

All allocations use the GPU allocators and replica/staging spaces captured by the immutable publish plan. Device guards make destruction and reduction device-correct. In particular, cuCO filter storage is constructed on a durable memory-space stream rather than an executor task stream, because its deleter retains the construction stream. Initialization is synchronized before the first task-stream insertion; contribution and failure paths drain task or reduction streams before private storage can be released.

A terminal session transition releases its accumulator reference. Contributions that were already in flight retain a local shared reference until they return, so failure cleanup cannot invalidate active work; successful filters remain alive through their target channels. This prevents a failed optional publication from retaining private Bloom partials for the rest of the hash join's lifetime.

Before Bloom construction, the producer requires `K * m <= max_dynamic_filter_bloom_bytes_per_gpu` on each GPU. The check uses division rather than overflowing multiplication and skips all `K` Bloom candidates when it fails. Destination replicas additionally acquire explicit scoped reservations. The one-shot source Bloom and per-device accumulator partials allocate through CuCascade's reservation-aware default GPU allocator without a separate up-front reservation; they remain allocator-accounted and hard-limit checked.

The policy cap is not a query-wide reservation ledger. Concurrent joins each receive the full per-join allowance, and transient root-reduction scratch (up to `K * 4 MiB`) is outside the cap. A future global admission service or shared scratch buffer may be warranted if concurrent publication pressure is measured.

## 12. Consumer behavior and placement

This change does not add a new probe checkpoint. The completed global Bloom is pushed into the channels and sparse key bindings introduced by PR #1277. Membership evaluation therefore follows the existing post-decode dynamic-filter operator at eligible scans and join-edge endpoints.

Consumers remain opportunistic and do not wait for publication. A split that snapshots its channel before the filter arrives passes through; a later split may observe the complete filter. There is never a partial Bloom in a channel.

The earlier design evaluation found that placement can dominate filter-construction cost. Serve-side or pre-partition filtering may produce substantially more value than a late post-scan mask, but those placements require separate readiness, capability, and duplicate-application work. They are not silently included in this implementation.

## 13. Implementation map

| Responsibility | Files |
|---|---|
| Default and YAML option | `src/include/sirius_config.hpp`, `src/sirius_config.cpp` |
| SQL option and setter | `src/sirius_extension.cpp` |
| Effective planner gate | `src/planner/sirius_plan_comparison_join.cpp` |
| Snapshot validation, session arbitration, terminal statistics, exact-ID accumulator, and fan-out | `src/include/op/dynamic_filter/dynamic_filter_publisher.hpp`, `src/op/dynamic_filter/dynamic_filter_publisher.cpp` |
| PARTITION freezing and pre-scatter contribution | `src/include/op/sirius_physical_partition.hpp`, `src/op/sirius_physical_partition.cpp` |
| Hash-join source readiness, routing, and session facade | `src/include/op/sirius_physical_hash_join.hpp`, `src/op/sirius_physical_hash_join.cpp` |
| Empty Bloom, incremental add, strict replicas, chunked OR | `src/include/op/dynamic_filter/sirius_dynamic_filter.hpp`, `src/cuda/sirius_dynamic_bloom_filter.cu` |
| Configuration tests | `test/cpp/config/test_config.cpp` |
| Snapshot, accumulator, session, and reduction tests | `test/cpp/operator/test_dynamic_filter_publisher.cpp` |
| Hash-join facade and build-delivery tests | `test/cpp/operator/test_dynamic_filter_publication_claim.cpp` |
| SQL and result-parity integration | `test/cpp/integration/test_gpu_execution_dynamic_filter_sip.cpp` |

## 14. Validation completed on this machine

The release build completed with:

```bash
pixi run ninja -C build/release -t clean sirius_extension sirius_loadable_extension sirius_unittest
# cleaned 1,204 target files from the copied build tree
SCCACHE_RECACHE=1 pixi run make
# exit 0 after 1,210 steps in 483.1 seconds
pixi run make
# final exit 0 after 300 steps in 147.3 seconds
```

Focused validation completed:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[partition_snapshot]"
# 87 assertions in 5 test cases

pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[publisher]~[accumulator]~[publication_session]~[snapshot]"
# 242 assertions in 19 test cases

pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[accumulator]"
# 135 assertions in 12 test cases

pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[publication_session]"
# 189 assertions in 14 test cases

pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[publication_claim]"
# 144 assertions in 9 test cases

pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[physical_partition]"
# 1,720 assertions in 25 test cases

pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[batch_lock_utils]"
# 36 assertions in 9 test cases

pixi run build/release/extension/sirius/test/cpp/sirius_unittest "gpu_execution - derived-build and build-block routes preserve results" -c "a multi-partition build obeys the subordinate switch"
# 166 assertions in 1 test case
```

Together these commands executed 2,719 assertions across 94 selected test-case executions. The counts intentionally include overlap: `[accumulator]` and `[publication_claim]` rerun two hash-join facade cases, while three PARTITION claim cases run under `[partition_snapshot]`, `[publication_claim]`, and `[physical_partition]`.

The executed focused runs cover structural and moved-from snapshot validation, direct accumulator rejection, PARTITION summarization, zero-row completion, exact-ID accounting, policy budgets, object lifetime, drained targets, one-shot-versus-accumulator mutual exclusion, exceptional one-shot stream drain with original-exception preservation, no-throw diagnostic cleanup, source-residency retry, real-repository freeze-before-pop and fail-open behavior, terminal statistics, strict replication failure, bounded contribution races, both finalize-versus-final-contribution winners, and late-invalid and terminal-result exception races. The physical PARTITION selection also preserves existing operator behavior.

The integration test forces a non-broadcast multi-partition build. With the subordinate switch off, it preserves PR #1277 one-shot-only behavior and records the build-not-whole skip. With the subordinate switch on and the master on, it verifies result parity and observes producer, membership-filter, successful-publication, and pushed-filter counters. With the master off, the subordinate switch alone creates no producer.

The full test suite was not run. The broad selections discovered the two-GPU clone and preparation tests, but those branches warned and returned before device-1 work because only one GPU was visible. Their code compiled; physical multi-GPU execution was not exercised.

## 15. Deferred multi-GPU validation

Multi-GPU construction is implemented but must be evaluated on the designated multi-GPU machine before it is treated as production validated. The minimum matrix is:

- two or more GPUs with partition counts below, equal to, and above the GPU count;
- skewed inputs, an empty contribution on one GPU, and a root GPU with no contribution;
- concurrent contributions on distinct GPUs;
- nonzero and noncontiguous visible device IDs;
- direct P2P transfer and forced host-staging fallback;
- producer and consumer device sets with different activity;
- duplicate, missing, unknown, and type-mismatched contributions under multi-GPU scheduling;
- injected root copy, OR, synchronization, and required-replica failures;
- output equivalence with both dynamic-filter switches disabled for every eligible join orientation;
- allocator pressure from multiple simultaneous joins and verification that private partials and scratch are released;
- reduction bytes, reduction latency, final replication latency, and probe batches missed before publication;
- a performance comparison on selective and unselective workloads, including TPC-H, before changing the default.

The key multi-GPU acceptance criterion is not merely that filters are pushed. Every inserted build key must probe true on every advertised device replica, no channel may observe a source partial, failures must publish nothing, and query results must match the filter-off baseline.

## 16. Known limitations and rollout decision

The current implementation intentionally has these limits:

- multi-partition accumulation supports Bloom membership for INT32 and INT64 keys only;
- it does not incrementally build small IN lists, hash IN lists, or zone maps;
- consumers are opportunistic and can miss a filter that publishes after their checkpoint;
- final reduction, replication, and fan-out run on the last contribution's path;
- root selection is deterministic lowest device ID rather than topology- or load-aware;
- reduction is a linear root gather, not a tree or collective;
- strict all-device replication can abandon a filter when only one target device fails;
- the Bloom cap is per join rather than query-wide, excludes reduction scratch, and does not explicitly pre-reserve source partials;
- the current counters do not break out per-device insert, reduction, and replication time;
- the completed SF1000 performance evidence is single-GPU and uses deliberate 1 GB (1,000,000,000 bytes) partition and CONCAT sizes to expose the path; production-natural activation and performance remain unmeasured.

Therefore `enable_dynamic_filter_multi_partition` must remain false by default. Enable it explicitly for correctness validation and targeted performance work. The SF1000 study shows that the architecture can pay for itself, most clearly on q10, but q8 was neutral and q16 was block-variable. Consider default enablement only after the deferred multi-GPU matrix and production-natural single-GPU configuration pass, with representative workloads showing benefit or at least neutrality after construction, reduction, replication, and missed-readiness costs are included.

## 17. Final engineering judgment

The implemented architecture is the correct base for single- and multi-GPU Sirius: freeze exact pre-scatter build identity, construct one global-layout Bloom per key, OR producer partials, prepare complete replicas, and publish atomically through the existing channel.

It preserves PR #1277, does not put dynamic-filter ownership into `CONCAT`, and keeps single- and multi-GPU semantics identical except for reduction. The controlled SF1000 result supports retaining this implementation, while the default-off subordinate switch remains the appropriate rollout boundary until production-natural and physical multi-GPU evidence are available.

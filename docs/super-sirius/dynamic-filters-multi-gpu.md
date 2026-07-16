# Dynamic Filters — Multi-GPU Publication

> **Status: implemented and revalidated on 2026-07-06.** Dynamic-filter
> consumers remain nonblocking and are safe on multiple GPUs. The producer builds each
> filter once, materializes its compact representation on every active probe GPU
> (copying finalized membership storage or reconstructing exact zone scalars),
> and publishes one immutable logical filter only after every successful
> device-local replica is ready. TPC-H Q2 passes on physical GPUs 1
> and 2. In the final pinned-host SF300 Q1-Q22 A/B, the sum of warm per-query
> medians fell from **13.7013505 s to 8.2939465 s: 39.466212% faster
> (1.651970x)**. All 22 result files were byte-identical.
> See [dynamic-filters.md](dynamic-filters.md) for the general feature.

## Summary

The crash was a device-ownership bug, not a cuDF filtering bug. The hash-join
build created a `cuco::static_set`, `cuco::bloom_filter`, or zone-map
`cudf::scalar` on its GPU and published a device-agnostic `shared_ptr`. Probe
scans are load-balanced across all active GPUs, so another GPU could dereference
the producer's device pointer. The asynchronous fault surfaced later as:

```text
copy_if failed on 2nd step: cudaErrorIllegalAddress
```

The fix gives every filter explicit device identity and records which
device-local replicas are ready. Consumers select by their memory-space device
ID; they never dereference remote filter storage. When the filter kind's
best-effort policy permits a target omission, that GPU skips the optional
filter; failures outside that policy propagate.

The publication and application contracts are distinct:

- For the producing join's immediate probe input, build-side CONCAT
  synchronously completes the publication attempt before its probe data-scan
  execution begins.
- A base scan reached transitively through an intervening join can execute
  earlier. It snapshots whatever fully ready filters are visible at its
  reader and post-decode checkpoints under normal scheduler order.
- A scan never waits for a dynamic filter and never assumes one was emitted.
- An empty or policy-gated publication and an allocation-unavailable local
  replica are safe pass-through cases; the authoritative join guarantees
  correctness.
- For IN-list/Bloom replicas, serious CUDA construction, transfer, or
  synchronization failures propagate and fail the producing task/query. The
  current zone-map target-clone path instead catches, logs, and omits a failed
  target replica.

Probe metadata parsing and prefetch preparation are independent of publication.
For an immediate probe, the actual `read_parquet`/decode task starts after the
ordered build-port publication attempt has returned. A transitive target may
start sooner: it selects zone maps immediately before `read_parquet` and selects
membership filters in the following post-decode operator. See
[Transitive scan targets and publication timing](dynamic-filters.md#transitive-scan-targets-and-publication-timing).

## Reproduction and diagnosis

The original integration reproducer was run with physical GPUs 1 and 2:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
  build/release/extension/sirius/test/cpp/sirius_unittest \
  "gpu_execution - TPC-H Query 2 parquet"
```

Before the fix, the two-GPU section failed with `cudaErrorIllegalAddress`; the
single-GPU section passed. Q2 is a reliable trigger because its selective join
publishes membership filters consumed by probe scan tasks on both logical
devices.

The unsafe ownership was:

| Filter | Source storage | Pre-fix failure |
|---|---|---|
| IN-list | `cuco::static_set` slots | `contains` on GPU B read GPU A's slots |
| Bloom | `cuco::bloom_filter` words | `contains` on GPU B read GPU A's words |
| Zone map | min/max `cudf::scalar`s | GPU B's AST literal referenced GPU A storage |

The error is reported by `cudf::apply_boolean_mask`/`copy_if`, but that is only
the next synchronizing operation. The invalid access originates in the earlier
membership or AST kernel.

### Hash-join broadcast research

Sirius never copies one opaque cuco hash table across GPUs. A cuco hash table
cannot be copied as an opaque object because its wrapper contains device
pointers and allocator/stream state. Instead, the broadcast small-build-table
join (`_broadcast`) replicates the small build *table* — the raw build batch —
to every GPU and lets each GPU build its own hash table from it (see
[operators.md](operators.md); `sirius_physical_partition.{cpp,hpp}`,
`sirius_physical_hash_join.{cpp,hpp}`). Large / hash-partitioned joins still
distribute partitions and migrate/co-locate their data batches, one hash table
per partition pinned to `partition_idx % num_gpus`.

Dynamic filters use the same "copy a flat, finalized representation and
reconstruct the pointer-bearing owner on the destination" pattern, but for the
*filter*, not the hash table. They implement it entirely in Sirius; they do not
copy a cuco wrapper and do not call or modify CuCascade's representation-transfer
code.

## Implemented design

### 1. Freeze routing and placement in an immutable publish plan

The planner creates a `dynamic_filter_publish_plan` before constructing the
physical hash join. It contains:

- every probe channel and its probe-column mapping;
- whether zone maps may be emitted;
- the per-key build-key-domain estimates used by the publication gates; and
- non-owning placement handles for the context's active GPU memory spaces.

Each `dynamic_filter_replica_space` pairs a non-null CuCascade GPU
`memory_space` with the NUMA-selected CuCascade HOST `memory_space` used for
staging when peer DMA is unavailable. The plan validates both tiers and
sorts/deduplicates placements by the GPU's actual device ID. The completed plan
is moved into the hash join's
`const _dynamic_filter_plan`. Runtime publication can observe that a channel has
drained, but it cannot add targets, rediscover devices, or change policy. This
also avoids assuming that active devices are numbered `0..N-1`, which matters
with explicit GPU selections and `CUDA_VISIBLE_DEVICES` remapping.

The handle is deliberately non-owning. The Sirius memory manager owns each
space, its default allocator, and its CUDA stream pool. That manager must outlive
the publish plan and every filter replica materialized from it; all GPU filter
uses must finish before filter destruction. This was already required by the
replicas' non-owning RMM allocator references. Carrying the memory space makes
the allocator-and-stream lifetime contract explicit in the placement type.

The exact-set/Bloom choice uses the minimum L2 size across the planned devices.
The exact IN-list is selected only when it fits the least-capable probe GPU;
otherwise the publisher selects the much smaller Bloom, which may itself exceed
L2 for a sufficiently large build. The decision never inherits whichever GPU
first queried `cudaDevAttrL2CacheSize`.

### 2. Keep publication local and exactly once

`dynamic_filter_publisher` is a translation-unit-local helper in
`sirius_physical_hash_join.cpp`. It consumes the immutable plan, the join's key
metadata, and one materialized build view; it is not a shared scheduler service
or a mutable routing registry.

The build-port hook offers that view as soon as a concat-folded `BUILD_PROBE`
build batch arrives, acquiring the batch's read-only accessor before routing it
so the GPU representation stays pinned against downgrade until publication
completes. In a single-partition join exactly one build batch arrives. In a
**broadcast** join every partition holds the *full* replicated build, so each
partition's concat_all-folded batch is a complete build side and races this hook
on its own GPU; the `OPEN -> PUBLISHING` compare-exchange lets exactly one win
(the first to arrive publishes and replicates; the others fall out at the CAS
before doing any filter work). A genuinely hash-partitioned (non-broadcast)
multi-partition build still disables pushdown — each partition holds only a
slice, so no single batch could emit a complete filter. The hook and
finalization arbitrate through a publication state machine independent of the
hash-table build state:

```text
OPEN --claim--> PUBLISHING --success--> FINISHED
                         `--exception--> FAILED
OPEN --finalize without a claim-------> CLOSED
```

Only an `OPEN -> PUBLISHING` compare/exchange may claim the work. Finalization
only closes an unclaimed window; it never manufactures a filter from released
state. `op_state_mutex` protects the short eligibility/finalization checks, but
is never held while reducing keys, building filters, copying replicas, or
synchronizing CUDA work.

At the early build-port site, a stream borrowed from the build memory space
first waits on the build representation's writer event. The fallback also
switches from the worker stream to a stream from that same durable pool after
the hash-table build is drained. Persistent cuDF/cuCO filter storage may retain
its allocation stream for eventual asynchronous deallocation, so it must not
retain a worker stream whose executor can be torn down earlier. Publication
remains independent of a probe batch and preserves the existing join task state
machine.

### 3. Expose replication as a producer-only capability

Consumer semantics remain on `sirius_dynamic_filter`: kind, availability, AST
lowering, and/or mask application. Device materialization is a separate
`sirius_device_replicable` capability with
`replicate_to_devices(span<dynamic_filter_replica_space const>)`. The local
publisher passes the immutable plan's placements after construction and before
channel fan-out; concrete filters retain completed replicas, not a second copy
of routing policy. A missing capability is an invariant failure rather than a
silent cross-device publication. Scan consumers never invoke it or know how a
representation is copied.

The publisher synchronizes the construction stream, invokes the capability for
each built filter, and only then calls `push_filter`. Every filter kind treats
a per-target failure — reservation denial, cloning, the copy itself, or the
completion synchronize — as best-effort replica unavailability: it is logged
and that target's replica is omitted. Successful replicas are published without
weakening the authoritative join.

### 4. Build once, copy finalized storage

The producer builds the source filter normally and synchronizes its construction
stream once. It does not retain the build-key column and does not rebuild or
rehash keys on every GPU.

- **IN-list:** create an identical target `cuco::static_set`, verify its
  capacity, then copy `capacity * sizeof(KeyT)` bytes from `static_set::data()`.
- **Bloom:** create the same policy and block extent on the target, verify the
  extent, then copy `block_extent * words_per_block * sizeof(word_type)` bytes.
- **Zone map:** read each bound as its exact host type and construct target-owned
  scalars. This preserves `INT64`, timestamp, decimal, and string semantics; it
  does not round bounds through `double`.

For every target, the planner pairs its GPU memory space with a NUMA-local HOST
memory space (falling back to the first Sirius HOST space when topology is
unknown). Replication selects the GPU and obtains both `acquire_stream()` and
`get_default_allocator()` from that GPU space. The stream is a non-owning view
into the space's managed stream pool; no persistent private stream is created.
The source representation is already the source GPU's ready local replica; no
same-device copy is submitted for it. Peer-DMA copies to remote targets are
submitted before the publisher waits on any target stream, so transfers to
three or more GPUs can overlap. The publisher retains each destination object
through this completion pass and adds it to the ready set only after its stream
completes. Replica destruction selects the owning CUDA device and releases
cuCO/scalar objects while both documented memory spaces are still alive.

```mermaid
flowchart LR
    subgraph SOURCE["Source GPU"]
        KEYS["Build keys"]
        FILTER["One finalized filter<br/>source representation is the local replica"]
        KEYS --> FILTER
    end

    REPLICATE["sirius_device_replicable<br/>replicate_to_devices"]
    PEER["Peer-capable target GPU<br/>cudaMemcpyPeerAsync<br/>enqueue only"]
    HOST["Target's CuCascade<br/>fixed_size_host_memory_resource<br/>borrowed pinned blocks"]
    STAGED["Non-peer target GPU<br/>batched D2H → H2D"]
    ZONE["Zone-map target GPU<br/>exact typed target-owned scalars"]
    SKIP["Best-effort target failure<br/>optional replica omitted"]
    COMPLETE["Completion pass<br/>synchronize target streams"]
    CHANNEL["push one immutable logical filter<br/>into target channels"]
    CONSUMER["Each scan supplies its memory-space device ID<br/>and selects only the local replica"]

    FILTER --> REPLICATE
    FILTER -->|"ready source replica"| COMPLETE
    REPLICATE -->|"IN-list / Bloom<br/>verified peer route"| PEER
    REPLICATE -->|"IN-list / Bloom<br/>peer route unavailable"| HOST
    HOST --> STAGED
    REPLICATE -->|"zone map"| ZONE
    REPLICATE -.->|"membership target failure"| SKIP
    ZONE -.->|"target clone exception"| SKIP
    PEER --> COMPLETE
    STAGED --> COMPLETE
    ZONE --> COMPLETE
    COMPLETE --> CHANNEL --> CONSUMER
```

Copies to different peer-capable targets are enqueued before the completion
pass, so three-or-more-GPU fan-out can overlap. HOST staging remains synchronous
inside the copy helper because the borrowed blocks must not return to their pool
while DMA is still using them.

### 5. Use the Sirius-owned replica-transfer path

`detail::enqueue_replica_copy` in
`src/cuda/dynamic_filter_replica_transfer.cu` owns the byte-copy policy and
returns the selected route:

1. Sirius calls CuCascade's shared, cached empirical peer-DMA probe for the
   ordered `(source, destination)` pair instead of maintaining a second probe.
2. Matching CuCascade's established GPU-to-GPU converter, a verified pair uses
   `cudaMemcpyPeerAsync` directly on the destination replica stream. Sirius
   enables ordinary peer access once during context initialization; neither
   converter performs a per-allocation memory-pool permission query. An
   unexpected enqueue failure propagates instead of changing routes.
3. Otherwise the adapter borrows the minimum number of pre-pinned fixed blocks
   from the target's planned CuCascade
   `fixed_size_host_memory_resource`. Because the blocks are noncontiguous, it
   emits one copy descriptor per block but submits them as two driver batches:
   D2H on a pooled stream acquired from the source GPU space, one source-stream
   completion barrier, then H2D on the destination replica stream followed by a
   destination-stream completion barrier. The two dependent legs cannot share
   one CUDA batch. On toolkits before CUDA 12.8 the same descriptors fall back
   to individual asynchronous copies. The borrowed blocks return to the pool
   before the helper returns, matching CuCascade's converter lifetime policy.

The dynamic-filter code never calls `cudaHostAlloc`/`cudaFreeHost` and does not
modify CuCascade; it uses CuCascade's public probe and existing Sirius-owned HOST
memory-space resource. Source writes are already complete before the helper is
called. Local and peer-DMA routes return after enqueue; the concrete filter
retains those replicas and synchronizes all target streams only after fan-out
submission is complete. HOST staging completes inside the helper because its
borrowed blocks cannot return to the pool while H2D DMA is in flight. Host-pool
exhaustion is allocation unavailability and may omit that optional replica.
CUDA enqueue/synchronization and invariant failures propagate; they do not
silently change routes or allow probing against uncertain replica state.

### 6. Publish one ready immutable snapshot

The claimed publisher constructs and completes all possible replicas, then fans
the same immutable logical filter into the planned channels. Consumers can
therefore remain lock-light and never observe an in-flight replica as available.

Publishing the source replica before remote replicas was deliberately rejected.
It would not advance an immediate probe—the synchronous build-port hook must
still return first—and, although it could help an already-running transitive
split on the source GPU, it would expose a logical filter whose replica set was
still mutating. A caller on another GPU would safely pass through and would not
train the gate, but it could miss optional pruning. The per-filter ready-snapshot
rule keeps each published object immutable and avoids per-device publication
generations. A transitive target may still race the producer's successive
`push_filter` calls and observe a safe subset of the fully ready filters.

### 7. Select locally at the consumer

The scan paths pass their memory-space device ID into dynamic-filter lowering and
mask computation:

- parquet AST merge selects device-local zone-map scalars;
- post-decode membership apply selects the local static set/Bloom;
- `is_available_on_device` is checked before either path.

There is no remote kernel dereference and no consumer-side synchronization with
the producer.

The scan-level `applicable()` fast path is lock-free. On an actual membership
apply, per-filter keep-ratio lookups/first-record updates use the ratio-map
mutex, and the post-mask scan-level decision uses a separate small mutex while
re-reading its state. This makes `ACTIVE` terminal: a stale unselective task
from an older filter generation cannot race a selective task on another GPU and
disable a filter already proven useful.

## Why this is performant

Publication adds `O(filter_size * (GPU_count - 1))` transfer work per join, not
one rehash/rebuild per GPU. A verified pair uses one direct peer-DMA leg; only an
unusable pair pays the HOST-staged D2H/H2D fallback. The steady-state probe remains
entirely device-local: the exact set is selected to fit the smallest active L2
where possible, and the Bloom stays the compact fallback. No key column is
retained, no scan is pinned to the build GPU, and scan parallelism is unchanged.

The copy happens on the producer's publication path. Direct copies to all
peer-capable targets are enqueued before the completion pass, allowing their DMA
legs to overlap on three or more GPUs. For an immediate probe, publication
completion is upstream of data-scan execution, so replica latency is on the
probe-start critical path. For a transitive target, earlier work may proceed
unfiltered while replication is in progress; replica latency instead delays
filter availability and reduces the number of splits it can prune. The scheduler
does not reorder work to minimize that window. The representations are compact
and copied rather than rebuilt; for a two-GPU query this is one remote replica
per emitted filter.

`memory_space::acquire_stream()` may return a pooled stream that already has
work queued, so the publication wait can occasionally include earlier work on
that stream. This is a cold-path head-of-line tradeoff: it can delay publication
and therefore the ordered immediate-probe start or the coverage of a transitive
target, while the managed pool avoids per-filter stream creation/destruction. It
is not a consumer-side wait or a correctness hazard. The end-to-end A/B below
includes this behavior.

## Validation

### Correctness and device ownership

On physical NVIDIA GB200 GPUs 1 and 2 (`CUDA_VISIBLE_DEVICES=1,2`, exposed as
logical devices 0 and 1):

- Full dynamic-filter suite: **242 assertions in 67 test cases, all passed**.
  This includes IN-list, Bloom, and zone-map remote replicas, the forced
  fixed-HOST-pool route, and the concurrent sticky-ACTIVE gate regression.
- Original TPC-H Q2 integration reproducer: **831 assertions, all passed**.
- The release loadable extension and unit-test targets build successfully.

One focused low-level test verifies that a filter object without a replica for a
requested device returns no mask; this exercises API safety and is not a model
of production scheduling, because production publishes only after replication.
The tests also verify that replicas survive destruction of the borrowed
build-key column, that Bloom replication has no false negatives, and that the
explicit pinned-host fallback transfers the exact bytes. Their source
and destination allocations use streams and allocators from a two-GPU Sirius
memory manager, which is deliberately declared before—and destroyed after—the
filters to exercise the documented pool-lifetime contract.

### Performance

The final measurement used SF300 parquet, grouped mode, physical GPUs 1 and 2,
five iterations per query, and `--pin host`. Host pinning occurs outside the
timed region. Zone maps were disabled, so this isolates membership-filter
publication and application. The OFF and ON configurations differ materially
only in `enable_dynamic_filter_pushdown` and their output paths.

For each query, iteration 0 was discarded and the median of iterations 1-4 was
taken. The global TPC-H metric is the sum of those 22 medians:

| Two-GPU pinned-host SF300 | Dynamic filters off | Dynamic filters on | Improvement | Speedup |
|---|---:|---:|---:|---:|
| Sum of Q1-Q22 warm medians | 13.7013505 s | 8.2939465 s | **39.466212%** | **1.651970x** |

Per-query medians:

| Query | Off (s) | On (s) | Improvement |
|---|---:|---:|---:|
| Q1 | 0.5032265 | 0.4830505 | +4.009% |
| Q2 | 0.3523525 | 0.1512455 | +57.076% |
| Q3 | 0.6793430 | 0.5083335 | +25.173% |
| Q4 | 0.3422540 | 0.2919185 | +14.707% |
| Q5 | 0.7750700 | 0.4428915 | +42.858% |
| Q6 | 0.2415565 | 0.2214595 | +8.320% |
| Q7 | 0.8857810 | 0.4226995 | +52.279% |
| Q8 | 0.9209665 | 0.5385575 | +41.523% |
| Q9 | 1.6049180 | 0.5737445 | +64.251% |
| Q10 | 0.8152175 | 0.5586015 | +31.478% |
| Q11 | 0.2668315 | 0.1410555 | +47.137% |
| Q12 | 0.4276590 | 0.3572310 | +16.468% |
| Q13 | 0.3523750 | 0.3220650 | +8.602% |
| Q14 | 0.2516855 | 0.2315255 | +8.010% |
| Q15 | 0.2416645 | 0.2215090 | +8.340% |
| Q16 | 0.2728810 | 0.2426490 | +11.079% |
| Q17 | 1.4642370 | 0.4680155 | +68.037% |
| Q18 | 0.9712030 | 0.4177245 | +56.989% |
| Q19 | 0.5283170 | 0.5334245 | -0.967% |
| Q20 | 0.3545820 | 0.2753625 | +22.342% |
| Q21 | 1.2680195 | 0.7196875 | +43.243% |
| Q22 | 0.1812100 | 0.1711950 | +5.527% |

Both runs contain exactly 110 timing rows, and every ON/OFF `result.txt` pair is
byte-identical. The requested global multi-GPU `>=10%` gate is exceeded by
29.47 percentage points. This is an ON-vs-OFF result at a fixed two-GPU count in
the warm pinned-host regime; it is not a one-to-two-GPU scaling or cold-storage
claim. Q19 regresses 1.0%, so the gain is global rather than universal per query.

## Code map

- Filter API and device-aware ownership:
  `src/include/op/sirius_dynamic_filter.hpp`
- IN-list/Bloom storage replication:
  `src/cuda/sirius_dynamic_in_list_filter.cu`,
  `src/cuda/sirius_dynamic_bloom_filter.cu`
- Exact typed zone-map replication:
  `src/op/sirius_dynamic_filter.cpp`
- Producer device discovery and publication:
  `src/planner/sirius_plan_comparison_join.cpp`,
  `src/op/sirius_physical_hash_join.cpp`
- Consumer device selection:
  `src/include/op/scan/dynamic_filter_gate.hpp`,
  `src/op/scan/dynamic_filter_merge.cpp`,
  `src/op/scan/parquet_gpu_ingestible.cpp`,
  `src/op/scan/sirius_physical_dynamic_filter.cpp`
- Sirius-owned replica-transfer policy:
  `src/include/op/dynamic_filter_replica_transfer.hpp`,
  `src/cuda/dynamic_filter_replica_transfer.cu`
- Focused regression:
  `test/cpp/operator/test_sirius_dynamic_filter_mgpu.cpp`,
  `test/cpp/scan/test_dynamic_filter_merge.cpp`

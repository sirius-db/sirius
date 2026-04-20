# Super Sirius Batch Cache Manager — Design Document

**Status:** Draft v3, for review
**Scope:** Single-node batch cache manager with cluster-ready extension points
**Target language:** C++20
**Supersedes:** the `cache_level` enum in `op/scan/config.hpp` and the ad-hoc wrapping logic in `parquet_scan_task.cpp`

---

## 1. System Context and Assumptions

- **Workload.** OLAP, columnar analytics over largely static tables. Plan re-execution and retries are common; the same chunk is read many times across queries. Cross-query reuse is therefore valuable, not incidental.
- **Hardware.** One host, one or more GPUs, NUMA-aware host memory. All GPUs on the node run inside a single Super Sirius process and share the process-global cache manager described here. GPUs on the node may or may not be NVLink-connected; the design does not assume NVLink.
- **Existing substrate.** Everything described in §1–§7 of `data-batch-caching.md`: the `cucascade::data_batch` abstraction, swappable `idata_representation`s, `cached_shared_representation<T>` for shared ownership, cucascade's host pool and reservation manager, and the byte-range cache used by the parquet scan. This design sits on top of those primitives; it does not replace any of them.
- **Transfer substrate.** A GPU-to-GPU transfer capability is assumed to exist behind a backend-agnostic abstraction. The abstraction — not a specific product — is the dependency. Acceptable backends include NIXL, a homegrown shim over `cudaMemcpyPeer` and a host-bounce fallback, or a future libkvikio-based transport with extensions. No specific backend is mandated.
- **Concurrency regime.** Many pipelines may run concurrently across multiple GPUs. Multiple pipelines may look up the same chunk at the same time. The hot-path lookup must not serialize on a global mutex.
- **Memory regime.** GPU VRAM is scarce and contended with kernels. Host memory is abundant, but pinned host memory is scarce. Compressed parquet bytes are the cheapest tier to retain. cucascade is the source of truth for memory accounting; the cache manager acquires reservations from it, never raw allocations.
- **Workload characteristic that matters most.** Tables are effectively read-only between refreshes. Each table carries a coarse version stamp — an **epoch** — that increments when the table is reloaded; individual rows are never mutated in place. This means invalidation is a rare, whole-table event tied to epoch bumps, not a per-write event. The invalidation code path can be slow without hurting the steady state. See the glossary (§16) for the full definition.

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Replace the global `cache_level` knob with per-batch, per-tier decisions driven by runtime signals (size, reuse probability, tier occupancy, producing-operator cost).
- Enable **cross-query and cross-pipeline reuse** for scan batches via a stable content key.
- Enable **cross-GPU reuse** on a single node: a cached payload on device 0 is directly reachable by a consumer on device 1, via a pluggable transfer engine.
- Extend caching beyond scans to **intermediate operator outputs** (join build sides, CTE materializations, aggregation hash tables) through a single uniform API.
- Make **admission, eviction, tier migration, and prefetch** pluggable without editing the manager.
- Integrate cleanly with cucascade's reservation manager so cache occupancy participates in global memory accounting and can be released under pressure.
- Provide observability hooks (hit/miss/eviction counters per tier, per table, per policy).

### 2.2 Non-goals (this revision)

- **Cross-node cache fetches.** Remote node lookup, RDMA, directory service — all out of scope. The design is single-node, with hooks to extend later.
- **Disk / spill tier owned by the manager.** cucascade supports it; this revision does not add a manager-driven disk tier. Spill under pressure is delegated to cucascade's reservation mechanism.
- **Full query-planner integration.** The planner will eventually consume the manager's lookup API and emit prefetch hints, but cache-aware query planning itself is a separate design effort.
- **Replacing cucascade allocation.** The manager acquires reservations; it does not implement its own allocator.
- **Schema evolution / epoch invalidation machinery.** The hooks are present (see §4 and §11); a concrete broadcast mechanism is out of scope.
- **Mandating a specific transfer backend.** The transfer-engine abstraction is part of this design; which backend ships in v1 is a separate decision (see Open Questions).

---

## 3. Executive Summary — What Is Reused from the Current Codebase

The central claim of this design is that Super Sirius already has the right *primitives* for a non-copying, tier-aware batch cache. What is missing is a **policy layer** sitting above those primitives. The table below states the reuse position precisely.

| Concept in this design         | Primitive it rides on                                         | Status          |
|--------------------------------|---------------------------------------------------------------|-----------------|
| Unit of caching                | `cucascade::data_batch` with swappable `idata_representation` | Reused as-is    |
| Non-copying shared ownership   | `cached_shared_representation<T>`                             | Reused as-is    |
| GPU tier payload               | `gpu_table_representation`                                    | Reused as-is    |
| Host-decoded tier payload      | `host_data_representation`                                    | Reused as-is    |
| Host-parquet tier payload      | `host_parquet_representation`                                 | Reused as-is    |
| Byte-range tier payload        | `cache_ranges` + `prefetched_data_source`                     | Reused, promoted to first-class tier |
| Tier conversion                | cucascade's `registry.convert<T>(...)`                        | Reused as-is    |
| Host memory allocation         | `fixed_size_host_memory_resource`, `fixed_multiple_blocks_allocation` | Reused as-is |
| Memory accounting              | `cucascade::memory_reservation_manager`                       | Reused; manager acquires reservations |
| Pipeline-local identity        | existing `batch_id`                                           | Reused as-is    |
| Cross-query identity           | **new** `cache_key`                                           | New             |
| Per-entry metadata + index     | **new** `cache_entry` inside sharded index                    | New             |
| Policy layer                   | **new** admission / eviction / migration / prefetch           | New             |
| Transfer across devices        | **new** `transfer_engine` abstraction                         | New, pluggable backend |
| Public manager API             | **new** `cache_manager` (pimpl)                               | New             |

What deprecates:

- The `cache_level` enum in `op/scan/config.hpp` is retired. Its three non-`NONE` values are expressible as `(admission_policy, migration_policy)` pairs at manager construction time.
- The ad-hoc `wrap_batch_data` / `cache_decoded_table` branching in `parquet_scan_task.cpp` is replaced by uniform `manager.lookup()` / `manager.insert()` calls.

Nothing that operators currently rely on at the representation level changes. The `cached_shared_representation<T>` wrapper is still how bytes get shared between consumers; the manager just becomes one more holder of those wrappers.

---

## 4. Cluster-Readiness Hooks

"Cluster-ready" in this document means: every decision the cluster version would need to make differently is already behind a concept or a hook in the single-node design. Turning on a cluster tier later is adding implementations, not restructuring the manager.

Four extension points are carved out specifically to carry this property:

| Hook                   | Single-node behavior (today)                          | Cluster behavior (later)                                             |
|------------------------|-------------------------------------------------------|----------------------------------------------------------------------|
| `key_codec`            | Local hash (e.g., xxhash / wyhash) of chunk identity  | Feeds a consistent-hash ring to derive both key and owning node      |
| `placement_policy`     | Always returns "local"                                | Returns `(local \| foreign)` based on consistent-hash ownership      |
| `invalidation_source`  | No-op subscription                                    | Subscribes to cluster epoch broadcast (gossip / pub-sub / periodic poll) |
| `transfer_engine`      | Intra-node D↔D via NIXL / `cudaMemcpyPeer` / host bounce | Inter-node D↔D via RDMA / GPUDirect-over-fabric, same API           |

The `transfer_engine` row is the most important one for reading the rest of this document. It means that GPU-to-GPU movement — already needed within a single node for cross-GPU sharing — is routed through an abstraction whose *shape* does not change when a second node appears. Adding cross-node support is adding a backend and enriching the device descriptor with a `node_id`, not introducing a parallel transfer code path next to the existing one.

---

## 5. Key Design Decisions

1. **Two-level identity.** A *content key* for cross-query and cross-pipeline reuse; a *batch handle* (the existing `batch_id`) for in-pipeline identity. The cache is indexed on the content key.
2. **One entry per chunk, with per-entry tier tracking.** The cache maintains a single logical index (sharded internally) mapping content keys to cache entries. There is exactly one `cache_entry` per cached chunk. Each entry records which tiers currently hold its payload; the same entry can have payload resident at multiple tiers simultaneously (e.g., GPU and host-parquet), but there is never more than one entry for a given key.
3. **Type-erased policies.** Admission, eviction, tier migration, and prefetch are each behind a C++20 concept, wrapped once at manager construction via a small type-erasure shim. The public `cache_manager` class is not a template.
4. **Cross-GPU cache sharing is first-class.** A single cache entry can serve consumers on any GPU on the node. Movement between devices goes through the pluggable `transfer_engine`.
5. **Sharded manager with lock-free lookup and locked slow path.** Shards are keyed by a hash of the content key, independent of GPU or pipeline. Lookups never take a lock; insertion, eviction, and migration take one shard's slow-path mutex.
6. **Pimpl at the manager boundary.** The public header is stable; policies and internal data structures can change without recompiling operator code.
7. **Reservation-owned, not allocation-owned.** The manager holds cucascade reservations at each tier, not raw buffers. Cache memory competes fairly with execution memory.
8. **Cluster hooks are present and trivial.** The four extension points in §4 are real concepts with no-op implementations today. The single-node design already has the shape the cluster version needs.

---

## 6. Design Rationale

### 6.1 Why two-level identity

A pipeline-local `batch_id` is sufficient within one execution but useless across executions — re-running the same query produces new `batch_id`s for the same underlying bytes. A content key closes that loop:

```
content_key := hash(
    table_id, partition_id, row_group_id,
    sorted(column_group_ids),
    projection_fingerprint,
    pushdown_filter_fingerprint,
    epoch
)
```

Each field is a `uint64_t` or an integer that contributes to the final hash; the fingerprints in particular are themselves `uint64_t` hashes of canonicalized forms — a sorted column-id list for the projection, a normalized expression tree for the pushdown filter. See the glossary for precise definitions. The key point: the cache key is a fixed-size value that fully identifies the bytes a batch will hold, independent of which pipeline produced it.

On cache hit, a new `data_batch` with a fresh `batch_id` is handed out, wrapping a shallow clone of the cached representation. The caller never sees that two `batch_id`s are backed by the same bytes — which is correct, because that is exactly what `cached_shared_representation<T>` is for.

A consequence worth stating plainly: **content key correctness is the caller's responsibility.** If a scan task forgets to fold the pushdown filter into the filter fingerprint, the cache will hand out wrong bytes to queries with different predicates. The manager does not inspect payload to verify keys. In practice this means the scan task canonicalizes its projection (sort the column id list) and its filter (normalize the expression tree) at batch-materialization time and hashes both, feeding the results into the `chunk_identity` that produces the key. Both canonicalization steps are local, stateless, and cheap relative to the scan itself.

### 6.2 Why type-erased policies

If admission / eviction / migration / prefetch were template parameters, the manager type would be `cache_manager<A, E, M, P>` and every operator that talks to it would need either to be templated the same way or to go through a virtual base. That is exactly the abstraction-leak we want to avoid.

Type erasure gives us a single non-template `cache_manager` class in the public header, concept-constrained policy authors (bugs become compile errors), and small bounded overhead — one indirection per policy call, and policy calls are not on the hot lookup path. They fire on admission, eviction tick, migration tick, and prefetch tick, all of which are already off the fast path. The cost is that policies can't be inlined into the manager. That is acceptable because, again, they are not hot.

### 6.3 Why one entry per chunk (not a cache hierarchy)

The natural mental model from CPU cache design is L1 → L2 → L3 → DRAM, each level a separate cache with its own tag array. That model does not apply here. In this design, tiers are *states of one cached thing*, not separate caches. A hot chunk may be resident at GPU, host-decoded, and host-parquet simultaneously, and there is still exactly one `cache_entry` for it; the entry tracks all three populated tiers internally. The index maps key → entry, and the entry's structure (an array of tier slots) records where the payload currently lives.

This has several load-bearing consequences that the rest of the design depends on:

- **"Cache miss" has one meaning:** the key is not in the index. If the key is in the index, the manager will serve from whichever tier is populated, doing any necessary conversion internally and transparently. A consumer never sees a "tier miss" surface as a lookup miss — they see a slightly slower hit.
- **Refcount is one number, not per-tier.** Every consumer reading a chunk — regardless of which tier served their payload — increments the same atomic counter on the one entry. The entry can't be evicted while any consumer is reading, across all tiers.
- **Tier-level eviction is cheap.** Demoting a chunk's GPU copy just clears that one tier slot and releases its GPU reservation. Other tier slots on the same entry are untouched. The entry itself only goes away when all tier slots are empty.
- **Cross-GPU sharing has no coordination cost.** Two consumers on different GPUs find the same entry (the sharding-by-key guarantees this), share one refcount, and see the same tier slots. If there were separate entries per tier or per device, every cross-device operation would need cross-entry synchronization.

### 6.4 Why sharded + lock-free hot path (and sharded by key, not by GPU)

A single mutex around the cache index is the wrong answer for a GPU analytics engine: lookups fire on every batch emit, from every pipeline, and serializing them kills throughput. A reader-writer lock is not much better; write amplification from admission and eviction updates is not rare.

A per-GPU or per-pipeline shard is also wrong, because one of our stated goals is cross-GPU and cross-query reuse. If two pipelines on different GPUs look up the same table, they should hit the same entry — not different shards that each hold their own copy. Sharding by producer would defeat the point.

The right answer is **sharding by content-key hash**, independent of GPU or pipeline. Two lookups of the same key always land on the same shard. Lookups on unrelated keys land on different shards with high probability and never contend. Details in §8.

### 6.5 Why pimpl at the manager boundary

The public `cache_manager` header is included from dozens of operator translation units. A heavy header that drags in cucascade, cuDF, parquet internals, and every policy type would hurt compile times and expose internal types across ABI boundaries we want to keep stable. Pimpl is the right tool: public header declares `class cache_manager { std::unique_ptr<impl> _p; ... };`, and the cpp file includes everything. Policy types flow through the public API as `std::unique_ptr<admission_policy>` (the type-erased wrapper), not as templates, so the header stays thin.

### 6.6 Why reservation-owned

If the manager owns raw allocations, two systems (the manager and cucascade) make decisions about the same memory and they will disagree under pressure. If the manager owns *reservations*, cucascade remains the single source of truth for memory accounting and the cache becomes a participant in the same budget as execution.

Concretely: an admission decision is "acquire a reservation of size N at tier T from cucascade; if granted, insert; if denied, decline." An eviction is "release reservation." cucascade's existing pressure signals drive the eviction policy.

### 6.7 Why a pluggable transfer engine (and why single-node needs it)

Cross-GPU cache sharing requires moving payload between devices. Hand-coding `cudaMemcpyPeer` inside the manager would work today but bakes in three assumptions we don't want to bake in: (1) the transport is always CUDA P2P, (2) reachability is binary, (3) there is no node boundary to cross. All three are wrong on realistic hardware and on a future cluster.

Routing all device-to-device movement through a backend-agnostic `transfer_engine` fixes this cheaply. The manager calls `transfer_engine.move(src_desc, dst_desc)` and the engine picks the path — NVLink P2P if available, PCIe P2P if not, host-bounce otherwise. Backend choice is deferred: we can ship with a homegrown shim, adopt NIXL, or build on libkvikio later, without the manager changing. When the cluster tier arrives, the same API handles inter-node movement; only the device descriptor widens to include a `node_id` and a new backend implementation handles RDMA / GPUDirect-over-fabric. **This is strictly more valuable than a single-node-only transfer path**, at the cost of one indirection on cross-device hits.

---

## 7. Architecture

### 7.1 Layer diagram

```
┌───────────────────────────────────────────────────────────────────┐
│ Operators (scan, join build, aggregation, CTE, ...)               │
│   - produce batches                                               │
│   - call cache_manager::lookup() / insert() / pin()               │
└───────────────────────────────┬───────────────────────────────────┘
                                │  public API (pimpl boundary)
┌───────────────────────────────▼───────────────────────────────────┐
│ cache_manager  (ONE per process)                                  │
│  ┌─────────────┬─────────────┬─────────────┬────────────────────┐ │
│  │ shard 0     │ shard 1     │ ...         │ shard N-1          │ │
│  │ ┌─────────┐ │ ┌─────────┐ │             │ ┌────────────────┐ │ │
│  │ │ index   │ │ │ index   │ │             │ │ index          │ │ │
│  │ │ entries │ │ │ entries │ │             │ │ entries        │ │ │
│  │ └────┬────┘ │ └────┬────┘ │             │ └────────┬───────┘ │ │
│  │ per-shard eviction-policy state + slow-path mutex            │ │
│  └──────┼─────────────┼─────────────────────┼────────────────────┘ │
│         │             │                     │                      │
│  ┌──────▼─────────────▼─────────────────────▼──────┐               │
│  │ policies (type-erased, concept-constrained)     │               │
│  │   admission | eviction | migration | prefetch   │               │
│  └──────────────────────┬──────────────────────────┘               │
│  ┌──────────────────────▼──────────────────────────┐               │
│  │ transfer_engine (type-erased, pluggable backend)│               │
│  │   NIXL | cudaMemcpyPeer shim | libkvikio+ | ... │               │
│  └──────────────────────┬──────────────────────────┘               │
└──────────────────────────┼────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│ cucascade reservation manager                                     │
│   - tier-aware reservations (GPU, host, pinned host, disk)        │
│   - pressure signals fed back into eviction / migration policy    │
└───────────────────────────────────────────────────────────────────┘

Tiers held by a cache_entry (one or more populated simultaneously):
  gpu_table | host_decoded | host_parquet | byte_range
```

### 7.2 One entry per chunk, with per-entry tier tracking (the central organizing principle)

The cache has **one** index. The index (sharded internally for concurrency) maps `cache_key → cache_entry*`. There is exactly one `cache_entry` per cached chunk. Each entry records, via a fixed-size array of tier slots, every tier that currently holds the payload.

```
shard N
  │
  index:  cache_key ──► cache_entry*
                           │
                           ▼
                  ┌──────────────────────────────────────┐
                  │ cache_entry  (exactly one per chunk) │
                  │   key                                │
                  │   refcount (atomic)                  │
                  │   hit_count (atomic)                 │
                  │   tier_slots[gpu_table]    → maybe   │
                  │   tier_slots[host_decoded] → maybe   │
                  │   tier_slots[host_parquet] → maybe   │
                  │   tier_slots[byte_range]   → maybe   │
                  │   policy bookkeeping (opaque)        │
                  └──────────────────────────────────────┘
```

The tier slots are independent: any subset may be populated at any time. An entry with three slots populated holds three physically separate allocations of the chunk's data — one on GPU, one decoded on host, one as compressed parquet bytes on host — each backed by its own cucascade reservation. All three belong to the same `cache_entry`; none is a "copy of another entry." This is what makes cheap demotion possible: when the migration policy demotes the GPU copy under pressure, it clears that one slot and releases its reservation, leaving the entry alive at the lower tiers for future re-promotion.

Two direct consequences worth naming:

- **Lookup semantics.** A lookup returns either "no such entry" (the key is not in any shard's index) or "entry found." When an entry is found, the manager serves it from whichever tier is populated, regardless of which tier the consumer would have preferred. Tier shortfall is *never* a miss at the lookup API level.
- **Entry identity is tier-agnostic and device-agnostic.** Two consumers on different GPUs holding handles to "the same cached thing" hold handles to the same `cache_entry`. The entry's GPU-tier slot may live on one specific device, but the entry itself is a shared object.

### 7.3 Object model (public types)

Three value types form the public API. Everything else is implementation.

- **`cache_key`** — content-key hash plus (optionally) a pointer to an interned chunk identity for debugging. Cheap to copy.
- **`cache_handle`** — RAII. Returned by `lookup()` and `insert()`. Holds a pin on the entry; destructor unpins. Exposes the produced `data_batch` and the tier it was served from (for diagnostics).
- **`cache_manager`** — one instance per process. Pimpl. Thread-safe.

Internal, not in the public header: `cache_entry`, `tier_slot`, `shard`, the policy wrapper implementations, the transfer-engine wrapper.

### 7.4 Request flow — hot path (lookup, hit)

The lookup hot path is the walk a consumer takes on a hit, starting from a `cache_key` and a consumer device id and ending with a `cache_handle` holding a `data_batch`. It is lock-free in steady state.

**Step 1 — Route to shard.**

```cpp
uint32_t shard_idx = key.hash() & (N_shards - 1);
shard& s = shards[shard_idx];
```

Arithmetic and an array index. No lock.

**Step 2 — Index lookup within the shard.**

```cpp
cache_entry* entry = s.index.find(key);   // concurrent hash map read
if (entry == nullptr) return cache_handle{};
```

The concurrent hash map (folly / libcuckoo / equivalent) allows concurrent readers without a mutex. A null return is the index-miss case: the cache has nothing for this key. The caller treats this as a true miss and produces the batch itself.

**Step 3 — Pin the entry.**

```cpp
uint32_t prior = entry->refcount.fetch_add(1, std::memory_order_acquire);
if (prior == 0 && entry->state.load() == state::pending_eviction) {
    entry->refcount.fetch_sub(1, std::memory_order_release);
    return cache_handle{};   // evictor got there first; report miss
}
```

Single atomic RMW. From here the entry is guaranteed alive until the handle is destroyed. The pending-eviction guard is the race-safety cover for the rare case that an evictor has marked the entry for removal between `find` and `fetch_add`.

**Step 4 — Walk tier slots, cheapest first.**

The tier preference order is fixed and baked into the manager:

```
gpu_table_local   → entry has gpu_table on consumer's device D
gpu_table_peer    → entry has gpu_table on some device D' ≠ D
host_decoded      → entry has a decoded host representation
host_parquet      → entry has compressed parquet bytes in host memory
byte_range        → entry has raw parquet byte ranges in host memory
```

The walk is a linear scan of at most 5 positions inside a single `std::array<tier_slot, num_tiers>` on the entry. Each check is a pointer load.

**Step 5 — Serve from the chosen tier.**

Each tier has a serving function that produces a `data_batch` bound to consumer device D.

- *`gpu_table_local`:* cheapest path. Shallow-clone the `cached_shared_representation<gpu_table_representation>`, wrap in a fresh `data_batch` with a new `batch_id`, done. No bytes moved, no kernels launched.
- *`gpu_table_peer`:* the entry's GPU slot is on device `D'`. Hand off to the transfer engine (§10.6):

  ```cpp
  transfer_descriptor src{D', slot.representation};
  transfer_descriptor dst{D, /* allocate via cucascade */};
  auto moved_rep = transfer_engine.move(src, dst);
  ```

  What happens to `moved_rep` after the consumer is done with it — whether it becomes an additional populated slot on the entry (materialize-and-cache) or is bound to the handle's lifetime (materialize-and-forget) — is an open question for team review. Both options and their tradeoffs are described in §7.5.

- *`host_decoded`:* cucascade's `registry.convert<gpu_table_representation>(host_slot.representation, D)` produces the GPU form on device D. This is exactly the H→D conversion the scan task already performs today; the manager invokes it instead.
- *`host_parquet`:* decode + H→D via cucascade's conversion registry, onto D.
- *`byte_range`:* parquet reader sourced from `prefetched_data_source` → decode → H→D onto D. Scan-produced entries only.

**Step 6 — Record the hit and notify the eviction policy.**

```cpp
entry->last_access_ns.store(now(), std::memory_order_relaxed);
entry->hit_count.fetch_add(1, std::memory_order_relaxed);
eviction_policy.on_hit(entry);               // may be lock-free, may enqueue
shard.stats.hits.fetch_add(1, std::memory_order_relaxed);
```

Whether `on_hit` is lock-free (LFU counter bump) or takes the shard mutex (LRU list move) is the policy's choice, not the manager's. Policies that need consistency take the lock; policies that tolerate eventual consistency do not.

**Step 7 — Return the handle.**

The `cache_handle` captures the entry pointer (for `unpin` at destruction), the produced `data_batch`, and the tier that served it. Destroying the handle does a single `entry->refcount.fetch_sub(1, release)`. The `data_batch`'s own `shared_ptr` keeps the underlying representation alive for as long as the caller uses it, independent of the handle's lifetime — two lifetimes, one handle, per the `cached_shared_representation<T>` semantics that already exist.

Properties of this walk worth calling out:

- **The shard is only touched in step 2.** Once the entry pointer is in hand, steps 3–7 touch the entry, the transfer engine, cucascade, and the eviction policy — never the shard's mutex.
- **The walk cost scales with the number of tiers, not the cache size.** Hash lookup is constant-time on average; tier walk is at most five pointer loads.
- **Tier preference order is fixed, not a policy.** The migration policy decides which tiers to populate. The walk order is baked into the manager. Making it pluggable is a future extension, not a v1 concern.

### 7.5 Request flow — cross-device hit (sub-case of hot path)

The cross-device case is worth its own subsection because it is the one hot-path scenario that invokes an external subsystem (the transfer engine) and because the caching policy for the transferred payload is currently an open design question.

When the tier walk in §7.4 step 4 selects `gpu_table_peer`:

```
consumer on device D
        │
        │  lookup(key, D) → entry
        │
        ▼
entry.tier_slots[gpu_table] populated on device D' ≠ D
        │
        ▼
transfer_engine.move({D', representation}, {D, new_reservation})
        │
        │   backend picks path:
        │     NVLink P2P  (NVLink-connected pair)
        │     PCIe P2P    (P2P-capable, no NVLink)
        │     host bounce (no P2P, or cross-node in future)
        │
        ▼
moved_rep on device D
        │
        ▼
data_batch wrapping moved_rep → caller
```

**Open question — what happens to `moved_rep` after the transfer?**

Two sub-options, deliberately left open for team review:

*Option A — materialize-and-cache.* The transferred payload becomes an **additional** populated GPU slot on the entry, tagged with device D. (Depending on implementation, the entry may need to hold multiple per-device GPU slots; see §10.5.) The next consumer on D hits locally (zero transfer). Costs a GPU-tier reservation per additional sharing device. Under sustained many-GPU load, a single hot entry may end up with one GPU slot per participating device.

*Option B — materialize-and-forget.* The transferred payload is bound to the returned `cache_handle`'s lifetime. When the handle is destroyed and no other consumer on D is active, the transferred copy is released. Next cross-device consumer on D pays the transfer cost again. Much cheaper in VRAM; re-transfers on every access.

A defensible v1 could be "materialize-and-forget with an opt-in materialize-and-cache when the migration policy says the entry is hot enough." This keeps the common case cheap and promotes to multi-device residency only for entries that earn it. But this should be debated before committing — hence its place in §15.

Both options are purely changes to the step that follows the transfer engine's `move()` call. Neither affects the transfer engine API, the manager's public API, or any other part of the design.

### 7.6 Request flow — slow path (insertion, eviction, migration)

The slow path takes the shard mutex.

**Insertion (produce).** A scan task or operator has produced a batch:

1. Compute the content key.
2. Call `manager.insert(key, batch, produce_context)`.
3. The manager routes to the shard; takes the shard's slow-path mutex.
4. Consult the admission policy: `on_produce(key, produce_context, tier_occupancy) → admission_decision`. If the decision is not-admit, release the lock, return an empty handle.
5. If admit, request a reservation from cucascade at the decided tier. If granted, either create a new `cache_entry` (if the key is absent) or populate a missing tier slot on the existing entry (if the key is already present at a different tier), and return a handle.
6. If the reservation is denied due to pressure, consult the eviction policy's `select_victims(tier, bytes_needed)`, release victims' reservations, and retry once. A second denial is reported to the admission policy, which may downgrade to a cheaper tier or give up.

**Eviction.** Triggered by admission retry (above) or by a pressure signal from cucascade. The eviction policy selects a list of victims for a given tier; the manager releases each victim's reservation and clears the tier slot. If all of an entry's tier slots become empty, the entry is marked `pending_eviction` and removed from the index once its refcount is 0.

**Migration.** Runs on a separate cadence — either a timer tick or a cucascade pressure crossing. The migration policy returns a plan: a list of `(entry, from_tier, to_tier)` triples. For each, the manager converts the payload from the source tier to the target tier (via cucascade's conversion registry), acquires a target-tier reservation, populates the target tier slot, and optionally clears the source. Demotion (GPU → host_decoded, host_decoded → host_parquet) is the common case; promotion (host_parquet → gpu_table on repeated hits) is less common but supported.

All three operations lock one shard at a time. Cross-shard coordination is not needed — each entry lives in exactly one shard, and the migration / eviction decisions are all entry-local.

---

## 8. Concurrency Model

### 8.1 What a shard is (and is not)

A shard is an internal partition of the manager's single logical index. It exists solely to limit contention: by partitioning the index N ways, N threads can do slow-path operations in parallel before any two collide on the same shard's mutex. Sharding is about contention, not capacity or locality.

**One cache manager per process.** On a single node with multiple GPUs, all GPUs share one `cache_manager` instance. That single manager holds N shards internally; shards are not separate managers, not separate processes, and not tied to any GPU.

A shard contains:
- A concurrent hash map of `cache_key → cache_entry*`, holding only keys that hash into this shard.
- Per-shard eviction-policy state (e.g., this shard's portion of an LRU list).
- A slow-path mutex protecting insertion, removal, and eviction-policy updates.
- Per-shard atomic counters for observability.

A shard does **not** contain:
- Any GPU payload. GPU-tier `tier_slot`s live inside `cache_entry`s; the shard is only the index that finds the entry.
- Any device affinity. An entry whose key hashes into shard 5 may have its GPU-tier payload on any device.
- Manager-wide policy state. Admission, migration, and prefetch policies are manager-wide (one instance each). Only eviction state is per-shard, because it is entry-keyed and entries are partitioned by shard.

Shard count is configurable; default is 64 (power of two so `hash & (N-1)` replaces `hash % N`). In practice the right value is "enough that the expected number of concurrent slow-path operations rarely collides on a shard." 64 handles dozens of pipelines across multiple GPUs comfortably.

### 8.2 Refcount and safe eviction

`cache_entry::refcount` is an atomic counter, device-agnostic and tier-agnostic, incremented by `lookup` (hit) and `insert` (successful admit) and decremented by `cache_handle` destruction. An entry with `refcount > 0` must not be evicted.

The eviction-safety protocol:

1. Eviction candidate is selected by the eviction policy while holding the shard mutex.
2. The candidate's `state` is CAS'd from `live` → `pending_eviction`.
3. Its `refcount` is read. If nonzero, the CAS is reverted and the candidate is skipped.
4. If zero, the entry is removed from the shard's index. Reservations are released.
5. The entry object itself is retired via a short quiescence period (a per-shard epoch counter) to protect against the rare race where a `lookup` read the pointer before the eviction CAS but had not yet done the `fetch_add`.

The quiescence period is short because the race window is bounded by the length of the hot-path between `index.find` and `refcount.fetch_add` — a handful of instructions. A full RCU or hazard-pointer scheme is overkill; a per-shard two-epoch scheme suffices.

The full cost of pinning on the hot path is one atomic RMW (fetch_add on hit) and one atomic decrement (on handle destruction). No allocation, no locking.

### 8.3 Per-shard locking discipline

Locks taken by the manager, with scope:

| Lock                              | When taken                     | Scope                     |
|-----------------------------------|--------------------------------|---------------------------|
| Shard slow-path mutex             | Insert, evict, migrate one entry | Exactly one shard         |
| Eviction-policy internal (if any) | Inside `on_hit` for some policies | Typically the same shard  |
| cucascade reservation acquire     | During insert / migrate        | cucascade-internal        |

No lock is taken by the lookup hot path. No lock crosses shards. Migration across tiers within an entry takes the one shard's mutex — `tier_slot` writes are serialized per entry because the entry is within one shard.

### 8.4 Back-pressure from cucascade

cucascade publishes per-tier pressure signals. The manager subscribes at construction and forwards events to the migration and eviction policies. When cucascade forces a release (hard cap breach), it calls back into the manager, which selects victims via the eviction policy and releases reservations. This integrates the cache into cucascade's existing spill path rather than fighting it — the manager is a well-behaved tenant of cucascade's budget.

---

## 9. Lifecycle and Interaction Patterns

### 9.1 Scan pattern

```
scan_task produces or is about to produce a raw host_parquet_representation
    │
    ▼
compute cache_key from (table, partition, row_group, projection, filter, epoch)
    │
    ▼
handle = manager.lookup(key, consumer_device=scan_gpu)
    │
    ├── hit  → use handle.batch(); handle pinned for pipeline stage
    │
    └── miss → scan_task materializes the batch (existing path)
              handle = manager.insert(key, batch, produce_ctx{SCAN, size, est_reuse})
              if handle is empty (not admitted), use the batch directly
              else use handle.batch() for downstream consumers
```

The existing `cache_level` enum is deleted. Its three non-`NONE` values are expressible by manager configuration:

| Old `cache_level` | New equivalent                                                   |
|-------------------|------------------------------------------------------------------|
| `NONE`            | `admission = never_admit`                                        |
| `TABLE_GPU`       | `admission = always_admit`, `migration = static(gpu_table)`      |
| `TABLE_HOST`      | `admission = always_admit`, `migration = static(host_decoded)`   |
| `PARQUET`         | `admission = always_admit`, `migration = static(host_parquet)`   |

No per-query knob is needed. Scans uniformly go through `lookup`/`insert`.

### 9.2 Intermediate output pattern (join build side)

```
join operator builds hash table on build side
    │
    ▼
compute cache_key from (join_op_id, partition_id, structural_fingerprint, epoch)
    │
    ▼
handle = manager.insert(key, batch, produce_ctx{JOIN_BUILD, size, est_reuse=1..N})
    │
    ▼
probe side gets handle.batch()
```

If the same query re-plans or the same build side is reused across queries (a dim table joined against multiple facts), subsequent plans `lookup()` the key and skip the build phase entirely. The structural fingerprint in the key is what makes this work — it captures the build-side's parameters without depending on pipeline identity.

### 9.3 Cross-GPU reuse pattern

```
pipeline P0 on GPU 0 scans table T
    │
    ├── computes cache_key k for row group R
    ├── manager.insert(k, batch, ...)  → GPU-tier slot populated on device 0
    └── downstream consumes batch from device 0

pipeline P1 on GPU 1 scans the same table T, same row group R
    │
    ├── computes the same cache_key k (deterministic from chunk identity)
    ├── manager.lookup(k, consumer_device=1)
    │        │
    │        ├── routes to same shard as P0's insert (sharding is by key)
    │        ├── finds entry; tier walk:
    │        │     gpu_table_local (on dev 1)  → MISS
    │        │     gpu_table_peer  (on dev 0)  → HIT
    │        │
    │        └── transfer_engine.move({0, rep}, {1, new_reservation})
    │              → returns data_batch on dev 1
    └── downstream consumes on device 1
```

The key observation: **P0 and P1 landed on the same shard because the content key is the same; the shard made no decision about devices; the transfer engine handled the rest.** This is the behavior that flows from the design decisions in §5 composing correctly. If shards had been per-GPU, P1 would not have found P0's entry.

### 9.4 Prefetch pattern

```
query planner or prefetch policy emits keys to warm
    │
    ▼
manager.prefetch_hint(key, reason)
    │
    ▼
prefetch policy may ignore, enqueue, or immediately issue
    │
    ▼
background loader reads (from parquet or warm tier) and calls insert()
    with produce_ctx{PREFETCH}, which the admission policy may treat with
    lower priority than foreground produces (policy-defined)
```

---

## 10. Per-Component Design

### 10.1 `cache_key` and the key codec

```cpp
// cache/cache_key.hpp
namespace sirius::cache {

struct chunk_identity {
  uint64_t table_id;
  uint64_t partition_id;
  uint32_t row_group_id;
  std::span<const uint32_t> column_group_ids;   // sorted, unique
  uint64_t projection_fingerprint;              // uint64 hash of canonicalized projection
  uint64_t filter_fingerprint;                  // uint64 hash of canonicalized pushdown predicate
  uint64_t epoch;                               // table version stamp
};

class cache_key {
 public:
  cache_key() = default;

  template <typename Codec>
  static cache_key make(const chunk_identity& id, const Codec& codec);

  uint64_t hash() const noexcept { return _hash; }
  bool operator==(const cache_key&) const noexcept = default;

 private:
  uint64_t _hash = 0;
  const chunk_identity* _debug_identity = nullptr;  // optional, for observability
};

template <typename T>
concept key_codec = requires(const T& c, const chunk_identity& id) {
  { c(id) } -> std::convertible_to<uint64_t>;
};

}  // namespace sirius::cache
```

`projection_fingerprint` and `filter_fingerprint` are `uint64_t` hashes of canonicalized forms of the projection and pushdown predicate respectively — not the raw structures themselves. Canonicalization is the caller's responsibility and is cheap: sort the column id list before hashing; normalize the filter expression tree (consistent operator ordering, constant folding) before hashing. Two queries with semantically identical projections or filters produce identical fingerprints; two queries that differ must produce different fingerprints. See the glossary for the full definition and rationale for the `uint64_t` size.

The default codec is a local xxhash / wyhash of the identity tuple. In a cluster version, the codec would feed into a consistent-hash ring and determine both key and owning node. Same concept, same call site, different implementation.

For intermediate operator outputs, `chunk_identity` is reused: `table_id` becomes a synthetic operator-output id, `partition_id` becomes the partition within that output, and the projection / filter fingerprints encode the operator's structural parameters. Single key type across all producers.

### 10.2 `cache_handle`

```cpp
// cache/cache_handle.hpp
namespace sirius::cache {

class cache_handle {
 public:
  cache_handle() = default;
  cache_handle(cache_handle&&) noexcept;
  cache_handle& operator=(cache_handle&&) noexcept;
  ~cache_handle();                                    // unpins

  cache_handle(const cache_handle&) = delete;         // a handle IS the pin
  cache_handle& operator=(const cache_handle&) = delete;

  explicit operator bool() const noexcept { return _entry != nullptr; }

  cucascade::data_batch batch() const;                // payload on consumer device
  tier_id served_from() const noexcept;               // diagnostics
  const cache_key& key() const noexcept { return _key; }

 private:
  friend class cache_manager;
  cache_entry* _entry = nullptr;
  cache_key _key;
};

}  // namespace sirius::cache
```

The handle is the only thing keeping an entry from being evicted. A missing `unpin` is a refcount leak; an early `unpin` (handle destroyed while a kernel still reads from the `data_batch`) is a use-after-free on GPU. Operator glue code should co-locate handle and batch in the same stack frame.

The `data_batch` returned by `batch()` may outlive the handle in principle — its own shared_ptr keeps the payload alive — but callers are strongly encouraged to treat the handle's lifetime as the batch's lifetime for clarity.

### 10.3 `cache_manager`

```cpp
// cache/cache_manager.hpp
namespace sirius::cache {

struct cache_manager_config {
  uint32_t shard_count = 64;

  std::size_t gpu_budget_bytes          = 0;   // 0 = cucascade decides
  std::size_t host_decoded_budget_bytes = 0;
  std::size_t host_parquet_budget_bytes = 0;

  std::unique_ptr<admission_policy>    admission;
  std::unique_ptr<eviction_policy>     eviction;
  std::unique_ptr<migration_policy>    migration;
  std::unique_ptr<prefetch_policy>     prefetch;       // may be null

  std::unique_ptr<key_codec_erased>    key_codec;
  std::unique_ptr<placement_policy>    placement;      // trivial default
  std::unique_ptr<invalidation_source> invalidation;   // trivial default
  std::unique_ptr<transfer_engine>     transfer;       // required
};

class cache_manager {
 public:
  cache_manager(cache_manager_config cfg,
                cucascade::memory_reservation_manager& rmgr);
  ~cache_manager();
  cache_manager(cache_manager&&) noexcept;
  cache_manager& operator=(cache_manager&&) noexcept;

  // Hot path. Returns empty handle on miss. Consumer device id routes
  // cross-device resolution to the transfer engine as needed.
  [[nodiscard]] cache_handle lookup(const cache_key& key,
                                    int consumer_device_id) noexcept;

  // Produce path. Admission policy decides; reservations flow through cucascade.
  [[nodiscard]] cache_handle insert(const cache_key& key,
                                    cucascade::data_batch batch,
                                    produce_context ctx);

  void prefetch_hint(const cache_key& key, prefetch_reason reason);

  // Drop entries matching predicate. Used by invalidation_source.
  void invalidate_if(std::function<bool(const cache_key&)> pred);

  cache_stats_snapshot stats() const;

 private:
  class impl;
  std::unique_ptr<impl> _p;
};

}  // namespace sirius::cache
```

`produce_context` is a small struct carrying the signals the admission policy needs (producer kind, estimated reuse, batch size, current-tier hint). Keeping this in one struct lets fields be added without churning the admission-policy concept.

### 10.4 Policy concepts (uniform pattern)

Each of the four policies follows the same type-erasure pattern: a concept stating the interface, a public wrapper class that erases the impl type via an internal `concept_t` / `model<T>` pair. Only admission is shown in full; the others mirror it.

```cpp
// cache/policy/admission.hpp
namespace sirius::cache {

struct admission_decision {
  bool admit;
  tier_id initial_tier;
  tier_set secondary_tiers;      // optional populate-simultaneously hints
};

template <typename T>
concept admission_policy_impl = requires(T& p,
                                         const cache_key& k,
                                         const produce_context& ctx,
                                         const tier_occupancy& occ) {
  { p.on_produce(k, ctx, occ) } -> std::same_as<admission_decision>;
};

class admission_policy {
 public:
  template <admission_policy_impl T>
  explicit admission_policy(T impl)
      : _self(std::make_unique<model<T>>(std::move(impl))) {}

  admission_decision on_produce(const cache_key& k,
                                const produce_context& ctx,
                                const tier_occupancy& occ) {
    return _self->on_produce(k, ctx, occ);
  }

 private:
  struct concept_t {
    virtual ~concept_t() = default;
    virtual admission_decision on_produce(const cache_key&,
                                          const produce_context&,
                                          const tier_occupancy&) = 0;
  };
  template <typename T>
  struct model final : concept_t {
    T inner;
    explicit model(T v) : inner(std::move(v)) {}
    admission_decision on_produce(const cache_key& k,
                                  const produce_context& ctx,
                                  const tier_occupancy& occ) override {
      return inner.on_produce(k, ctx, occ);
    }
  };
  std::unique_ptr<concept_t> _self;
};

}  // namespace sirius::cache
```

The other three follow the same pattern:

| Policy            | Key method(s)                                                                                          |
|-------------------|--------------------------------------------------------------------------------------------------------|
| `admission_policy`| `on_produce(key, produce_context, tier_occupancy) → admission_decision`                                |
| `eviction_policy` | `on_hit(entry_ref)`; `on_insert(entry_ref)`; `select_victims(tier, bytes_needed) → victim_list`        |
| `migration_policy`| `on_pressure(tier, pressure_level) → migration_plan`                                                   |
| `prefetch_policy` | `on_hint(key, reason)`; `tick() → prefetch_list`                                                       |

Each policy receives a small per-entry "bookkeeping slot" (opaque `std::byte[16]` for eviction, `std::byte[8]` for migration) inside `cache_entry` to avoid a side map. LRU uses it for list pointers, LFU for a counter, ARC for segment membership, cost-aware eviction for a reload-cost estimate. The manager treats the slot as opaque; the policy casts it.

### 10.5 Tier model and `cache_entry`

```cpp
enum class tier_id : uint8_t {
  gpu_table,
  host_decoded,
  host_parquet,
  byte_range,
};

struct tier_descriptor {
  tier_id      id;
  int          memory_space;
  std::size_t  cost_to_reload_ns;     // order-of-magnitude estimate
  std::size_t  cost_to_promote_ns;
};

struct tier_slot {
  std::shared_ptr<void> representation;         // erased shared_ptr<T>
  void (*emit)(const std::shared_ptr<void>&, int consumer_device,
               cucascade::data_batch&);         // tier-specific serve fn
  cucascade::reservation_handle reservation;
  int device_id = -1;                           // only meaningful for gpu_table
  std::atomic<uint64_t> last_access_ns;
};

struct cache_entry {
  cache_key key;
  std::atomic<uint32_t> refcount;
  std::atomic<uint32_t> hit_count;
  std::atomic<uint8_t>  state;                  // live | pending_eviction

  std::array<tier_slot, num_tiers> tiers;

  std::byte eviction_bookkeeping[16];
  std::byte migration_bookkeeping[8];
};
```

Notes:

- `tier_slot::device_id` is the opaque device descriptor introduced in §4. In the cluster version it becomes a wider struct `{node_id, device_id}`; for v1 it is a single `int` holding the local device ordinal, meaningful only for the `gpu_table` tier.
- The `emit` function pointer is how tier-specific serving logic plugs into the uniform `data_batch` production path. Each tier registers its emit function when the manager is constructed. This is a small internal trait dispatch, not a user-facing extension point.
- Multi-tier residency on one entry follows naturally: if the migration policy populates both `gpu_table` on device 0 and `host_parquet`, the entry simply has both slots filled. Refcount and policy bookkeeping remain per-entry (not per-slot).
- The `tiers` array assumes at most one GPU slot per entry. If the cross-device caching question (§7.5) is resolved in favor of materialize-and-cache with per-device residency, `tiers[gpu_table]` will need to become a small vector of per-device slots; the rest of the structure is unaffected.

### 10.6 Transfer engine

The transfer engine is the pluggable GPU-to-GPU (and later node-to-node) movement abstraction. It is a type-erased wrapper around a concept, identical in pattern to the policy wrappers.

```cpp
// cache/transfer/engine.hpp
namespace sirius::cache {

// Opaque descriptor of a payload's location. Widens in cluster version.
struct transfer_descriptor {
  int device_id;
  std::shared_ptr<void> representation;   // erased shared_ptr<T>
  tier_id tier;
  // Future: int node_id;
};

struct transfer_result {
  std::shared_ptr<void> representation;   // on the destination
  // Diagnostics: path taken (p2p, nvlink, host_bounce, rdma, ...)
};

template <typename T>
concept transfer_engine_impl = requires(T& t,
                                        const transfer_descriptor& src,
                                        const transfer_descriptor& dst) {
  { t.move(src, dst) } -> std::same_as<transfer_result>;
  { t.can_reach(src, dst) } -> std::same_as<bool>;
};

class transfer_engine {
 public:
  template <transfer_engine_impl T>
  explicit transfer_engine(T impl)
      : _self(std::make_unique<model<T>>(std::move(impl))) {}

  transfer_result move(const transfer_descriptor& src,
                       const transfer_descriptor& dst) {
    return _self->move(src, dst);
  }
  bool can_reach(const transfer_descriptor& src,
                 const transfer_descriptor& dst) {
    return _self->can_reach(src, dst);
  }

 private:
  struct concept_t {
    virtual ~concept_t() = default;
    virtual transfer_result move(const transfer_descriptor&,
                                 const transfer_descriptor&) = 0;
    virtual bool can_reach(const transfer_descriptor&,
                           const transfer_descriptor&) = 0;
  };
  template <typename T>
  struct model final : concept_t { /* ... */ };
  std::unique_ptr<concept_t> _self;
};

}  // namespace sirius::cache
```

Backend implementations we anticipate (none mandated for v1):

- **`cuda_p2p_transfer`** — homegrown. NVLink / PCIe P2P via `cudaMemcpyPeerAsync`, host-bounce via pinned host allocation + two H↔D copies. Ships with no external dependencies beyond CUDA.
- **`nixl_transfer`** — thin adapter over NIXL's API. Lets NVIDIA's inference transfer library handle path selection. Adds a dependency on NIXL.
- **`kvikio_transfer`** — libkvikio-based, with extensions for D↔D beyond what kvikio covers today. Speculative; a backend we may build if neither of the above fits.
- **Future: `rdma_transfer`** — inter-node, GPUDirect-over-fabric. Not in this revision.

The choice of backend is a v1 decision separate from adopting this design (see Open Questions).

### 10.7 Invalidation source

```cpp
template <typename T>
concept invalidation_source_impl = requires(T& s,
    std::function<void(std::function<bool(const cache_key&)>)> cb) {
  { s.subscribe(std::move(cb)) };
};
```

Manager hands the source a callback at startup. When the source detects an invalidation event (single-node: nothing; cluster: epoch bump on gossip channel), it fires the callback with a predicate. The default implementation is a no-op. **This is the hook cluster epoch broadcast plugs into.**

### 10.8 Placement policy

```cpp
enum class placement { local, foreign };

template <typename T>
concept placement_policy_impl = requires(T& p, const cache_key& k) {
  { p.where(k) } -> std::same_as<placement>;
};
```

Today's implementation always returns `local`. In a cluster version, it consults the consistent-hash ring. Foreign keys are not cached locally — the manager treats them as a miss on insert, and lookups for foreign keys are expected to be handled by the planner routing the fragment to the owning node.

---

## 11. Interaction with cucascade Reservations

- The manager acquires a `reservation_handle` from cucascade when it admits an entry at a tier. The handle is stored in the `tier_slot` and released on eviction or migration-demotion.
- cucascade's `memory_reservation_manager` publishes per-tier pressure signals. The manager subscribes and forwards them to both the migration policy (for proactive demotion) and the eviction policy (for reactive release).
- When cucascade forces a release (hard cap breach), it calls back into the manager. The manager invokes the eviction policy's `select_victims()` at the pressured tier, releases reservations, and acknowledges cucascade. The cache participates in cucascade's existing spill path rather than racing against it.
- Pinned host memory is a sub-budget of the host tier. The admission policy sees per-tier occupancy via `tier_occupancy` and can decline promotion to `host_decoded` if pinned budget is tight.

The interface between the manager and cucascade is narrow and symmetric: the manager asks for reservations; cucascade asks for releases.

---

## 12. Example Implementations (shipped in the box)

### 12.1 Admission policies
- `always_admit` — matches old `cache_level != NONE` behavior.
- `size_threshold_admit` — admit if `estimated_reuse × size_saved > threshold`.
- `never_admit` — matches old `cache_level == NONE`.

### 12.2 Eviction policies
- `lru_eviction` — per-tier LRU list; bookkeeping slot holds list pointers.
- `lfu_decay_eviction` — LFU with exponential decay; bookkeeping holds counter + last-decay timestamp.
- `arc_eviction` — ARC; bookkeeping holds segment id.
- `cost_aware_eviction` — victim = entry minimizing `reload_cost × (1 − recency_weight)`.

### 12.3 Migration policies
- `static_cache_level_migration` — backward-compat shim; never migrates.
- `demote_on_pressure` — GPU→host_decoded at GPU 80%; host_decoded→host_parquet at host 85%.
- `promote_on_hit_migration` — promotes to hotter tier after K consecutive hits.

### 12.4 Prefetch policies
- `null_prefetch` — ignores hints.
- `history_replay_prefetch` — warms chunks accessed in the last K queries.

### 12.5 Transfer engine backends
- `cuda_p2p_transfer` — `cudaMemcpyPeerAsync` + host-bounce fallback. No external deps.
- `nixl_transfer` — NIXL adapter. Adds a NIXL dep.
- `kvikio_transfer` — libkvikio-based, speculative. Extensions needed beyond current kvikio.

Backward compatibility for existing `cache_level` call sites is achieved by a mechanical mapping (see §9.1). No per-call-site shim is needed if the whole codebase flips in one change.

---

## 13. Observability

```cpp
struct tier_stats {
  std::size_t entries;
  std::size_t bytes;
  std::size_t reservation_bytes;
  uint64_t hits;
  uint64_t misses;
  uint64_t admissions;
  uint64_t admission_denials;
  uint64_t evictions;
  uint64_t promotions;
  uint64_t demotions;
  uint64_t cross_device_serves;     // served via transfer engine
};

struct cache_stats_snapshot {
  std::array<tier_stats, num_tiers> per_tier;
  uint64_t total_entries;
  std::vector<uint64_t> per_shard_entries;   // skew diagnostics
  std::map<std::string, std::string> policy_diagnostics;
  std::map<std::string, std::string> transfer_engine_diagnostics;
};
```

Counters are per-shard atomics, aggregated on snapshot. Per-table / per-producer breakdowns are opt-in (require a side map keyed by `(table_id, producer_kind)`, maintained on the slow path only).

---

## 14. Key Decisions and Tradeoffs

| Decision                                          | Benefit                                                         | Cost                                                             |
|---------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------|
| Two-level identity                                | Cross-query reuse; legacy `batch_id` untouched                  | Caller must compute a correct content key                        |
| One entry per chunk, per-entry tier tracking      | Tier shortfall is transparent; no hierarchy coherence to manage | Entry struct carries per-tier slots regardless of fill           |
| Type-erased policies                              | One non-template `cache_manager`; hot-swap policies             | One indirect call per policy op (off hot path)                   |
| Sharded + lock-free lookup                        | Scales with concurrent pipelines; cross-GPU reuse works         | Eviction-policy state must be per-shard                          |
| Pimpl at manager boundary                         | Stable ABI; fast compiles                                       | One pointer chase per manager call                               |
| Cross-GPU sharing via pluggable transfer engine   | Cluster-ready by construction; backend-agnostic                 | One indirection on cross-device hit; abstraction must cover unknown future backends |
| Reservation-owned (no raw alloc)                  | Fair competition with execution memory                          | Coupled to cucascade; depends on its pressure API                |
| Byte-range tier as first-class                    | Cheapest fallback before re-read                                | Only meaningful for scan-produced entries                        |
| Cluster hooks present but trivial                 | Cluster version is additive, not a rewrite                      | Small complexity cost on single-node today                       |

---

## 15. Open Questions

1. **Which concurrent hash map?** `folly::ConcurrentHashMap`, `libcuckoo`, or a hand-rolled open-addressed map. Dependency footprint and performance characteristics differ. Benchmark with realistic key distribution before deciding.

2. **Cross-device materialization: cache or forget?** (See §7.5.) When a lookup from device D serves from a GPU-tier slot on device D′, does the transferred copy become an additional populated slot on the entry (materialize-and-cache) or does it live only for the handle's lifetime (materialize-and-forget)? Both have merit. A hybrid — forget by default, promote to cache when the migration policy flags the entry as hot enough — is plausible but should be debated. The answer affects whether `cache_entry::tiers[gpu_table]` needs to become a small vector of per-device slots.

3. **Transfer engine backend for v1.** Homegrown `cudaMemcpyPeer` shim, NIXL adapter, libkvikio-based, or a placeholder pending benchmark. The API is fixed; the backend is not. A v1 could ship with the homegrown shim and add others later without manager changes.

4. **Intermediate-output key structural fingerprint.** Should the key include a full structural fingerprint of the producing operator (enables reuse across semantically equivalent but syntactically different plans) or just its id (safer, no false-positive reuse)? Leaning structural, with strict-equality fallback, but worth confirming.

5. **Prefetch admission priority under contention.** Strictly lower than foreground produces, or fair-share? Affects whether prefetch can starve under sustained load. Suggest: strictly lower by default, configurable.

6. **Epoch invalidation granularity.** Table-level invalidation is an O(index) walk. Partition-level needs a secondary index. What is the expected invalidation frequency? If rare, the walk is fine; if not, build the secondary index.

7. **Ref-leak detection.** A caller dropping a `cache_handle` while retaining the returned `data_batch` is a lifetime bug (the batch survives but the pin is released, so the underlying entry may be evicted and migrated under the caller). Debug-build assertion, compile-time tether, or documentation only? Leaning: debug-build assertion plus a lint in CI for the obvious pattern.

8. **Cost model inputs.** `tier_descriptor::cost_to_reload_ns` is static today. Measuring reload latency at runtime and feeding it to cost-aware eviction is strictly better but adds complexity. v1 or v2?

9. **Migration path for existing call sites.** Big-bang flip of all `cache_level` call sites to the new API, or coexistence for a release cycle? Big-bang is cleaner; coexistence is less risky.

10. **Fingerprint width.** `uint64_t` fingerprints for projection and filter give a birthday-bound collision probability of roughly 1 in 30 billion for a million-entry cache. Acceptable for v1. If a deployment needs stronger guarantees, the fix is either 128-bit fingerprints or a defense-in-depth structural compare on fingerprint match. Not a v1 concern unless the workload exposes a correctness risk.

---

## 16. Glossary

- **Batch** — a `cucascade::data_batch`; a unit of pipeline data flow holding one representation.
- **Batch id** — the pipeline-local identity Super Sirius already assigns to each batch. Not a cache key.
- **Cache entry** — internal object holding metadata for one cached chunk: key, refcount, per-tier slots, policy bookkeeping. Exactly one entry per cached chunk, regardless of how many tiers hold its payload.
- **Cache key** — a content-derived hash identifying a specific set of bytes independent of pipeline or query. The lookup key.
- **Canonicalization** — the process of reducing a structure to a unique normal form so that two logically equal structures become byte-identical. Used for projection (sort column ids) and filter (normalize expression tree) inputs to fingerprinting.
- **Chunk** — the unit of caching at the scan tier: `(table_id, partition_id, row_group_id, column_group_ids, projection_fingerprint, filter_fingerprint, epoch)`.
- **Content key** — synonym for cache key when emphasizing derivation from content, not from identity.
- **Cross-device hit** — a hit where the entry's GPU-tier payload is on a device other than the consumer's. Resolved via the transfer engine.
- **Epoch** — a monotonically increasing version counter on a table; increments when the table's underlying data is refreshed (bulk reload, schema change, etc.). Tables are not mutated row-by-row; an epoch bump is the only kind of change. Invalidation is triggered by epoch bumps, propagated via the `invalidation_source` hook. Epoch is folded into the content key so that entries from older epochs become unreachable by new lookups automatically. In the cluster version, epoch bumps arrive via a broadcast channel (gossip / pub-sub); in single-node v1, the `invalidation_source` is a no-op.
- **Fingerprint** — a fixed-size (typically 64-bit) hash value standing in for a larger structure, such that two logically equal structures produce the same fingerprint and logically different structures produce different fingerprints (with overwhelming probability). Used to compress variable-size things (column-id lists, filter expression trees) into constants that combine into the cache key. Fingerprinting presumes canonicalization of the input first.
- **Handle** — RAII object returned on hit or successful insert. Holds a pin; destructor unpins.
- **Index** — the sharded map from cache key to cache entry. Single logical index; sharded internally for concurrency.
- **Materialize-and-cache / materialize-and-forget** — two options for what happens to a payload transferred cross-device: keep it as an additional populated slot on the entry, or bind it to the handle's lifetime. Open question, §15.
- **NIXL** — NVIDIA's inference transfer library; one possible backend for the transfer engine, not mandated.
- **One entry per chunk** — the principle that the cache holds exactly one `cache_entry` per cached chunk, regardless of how many tiers currently hold the chunk's payload. "Miss" means absence from the index; it never means absence from a particular tier.
- **Pimpl** — pointer-to-implementation idiom; a public class holds a `unique_ptr` to an opaque `impl`.
- **Pin / refcount** — an entry with refcount > 0 cannot be evicted. Acquired on hit and insert; released on handle destruction.
- **Placement policy** — cluster-readiness hook returning `local` or `foreign` for a key. Trivial today.
- **Policy** — pluggable decision-making component (admission, eviction, migration, prefetch) behind a concept and a type-erased wrapper.
- **Projection** — the set of columns the query actually needs from a row group; the scan materializes only these columns.
- **Projection fingerprint** — a `uint64_t` hash of the canonicalized (sorted) projected column-id list. Two queries reading the same set of columns produce the same projection fingerprint even if the columns were specified in different orders. Two queries reading different column sets must produce different fingerprints.
- **Pushdown filter** — a predicate the planner pushes into the scan; rows not satisfying the predicate are dropped during the read rather than in a later operator. Changes the row set emitted by the scan.
- **Pushdown filter fingerprint** — a `uint64_t` hash of the canonicalized pushdown predicate (normalized expression tree). Two queries with semantically identical filters produce the same fingerprint; two queries with different filters must produce different fingerprints — any collision here is a correctness bug because it lets one query's filtered bytes serve a query with a different filter.
- **Representation** — the storage backing of a batch (`gpu_table_representation`, etc.). Defined in `data-batch-caching.md` §2.
- **Reservation** — a cucascade-managed memory claim at a specific tier. The cache holds reservations, not raw allocations.
- **Shard** — an internal partition of the cache index, selected by `hash(key) & (N-1)`. Limits lock contention. Not a separate manager; not tied to a GPU.
- **Tier** — a level of representation at which a payload can reside: `gpu_table`, `host_decoded`, `host_parquet`, `byte_range`.
- **Tier slot** — per-entry storage for one tier's payload plus its cucascade reservation. An entry has one slot per tier; any subset may be populated at any time.
- **Transfer engine** — pluggable, backend-agnostic abstraction for moving payload between devices (and later, nodes). Shapes cluster-readiness.
- **Type erasure** — C++ technique wrapping any implementation of a concept in a single non-template class, enabling runtime polymorphism without a user-visible virtual base.

---

*End of design document.*

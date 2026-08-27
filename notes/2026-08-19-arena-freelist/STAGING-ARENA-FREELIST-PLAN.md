# Staging Arena: Reclaiming Allocator — Execution Plan

**Target repo:** `sirius`, branch `demo-multi-cn` (written against `fa836455` plus the
uncommitted arena-logging change in `src/exec/exchange_staging_arena.cpp`).

**Scope:** replace the exchange staging arena's bump allocator with a coalescing free-list
allocator, and make staging leases release themselves via RAII. Keep the single registered
region, the byte-offset wire contract, and true zero-copy contiguous RDMA writes into a
peer-addressable region — all of that is unchanged.

**Self-contained:** every file, line number, current snippet and replacement is inline. No
context from outside this document is needed.

---

## 0. Why — the defect in one paragraph

`exchange_staging_arena` allocates by bumping `head_` and reclaims by dropping `head_` back to
the end of the *highest* outstanding lease (`src/exec/exchange_staging_arena.cpp:228-251`).
Space below the head that belongs to an already-released lease is **not reachable** — the
capacity check is `free = capacity_ - head_` (`:210`), not live bytes. On the send path, where
exactly one lease is outstanding at a time, this is harmless. On the receive path it is not:
concurrent grants from `handle_staging_lease` (`experimental/starrocks/src/compute_node_service.rs:1106`)
are released out of order (`experimental/starrocks/src/engine.rs:367, :518`), so every release
leaves a gap while every new lease advances the head. `head_` therefore tracks **lifetime
totals, not concurrency**, and a long-running CN drifts toward exhaustion independently of how
much is actually live.

Two observable consequences, both already in the record:

- Arena capacity has to be sized for cumulative drift rather than working set. `SIRIUS_EXCHANGE_STAGING_BYTES`
  ranges from 1280MiB to 32GiB across launchers with no principled derivation.
- The exhaustion message reports `free` bytes that are free-but-unreachable, so it cannot
  distinguish "genuinely concurrent" from "drifted".

A coalescing free list makes all released space reachable. Nothing else about the design needs
to move.

---

## 1. What is explicitly NOT changing

State this up front because it constrains every edit below.

| Property | Stays |
|---|---|
| One device region, allocated once | `cudaMalloc` or the VMM/`CU_MEM_HANDLE_TYPE_FABRIC` path, `:59-160` |
| Region registered once with nixl | `nixl_transport.rs` registers `[base, capacity)` at bring-up |
| Arena is non-movable | copy/move deleted, `exchange_staging_arena.hpp:70-74` |
| Leases are **byte offsets**, `kAlignment`-aligned | `lease()` returns `uint64_t`, `kAlignment = 256` |
| Every lease is one **contiguous** range | required by `cudf::unpack` and by the WRITE target |
| Wire contract `offset == 0 && len == 0` ⇒ no lease | `sirius_ffi.cpp:751` |
| Arena **never blocks** | no condvar; see §6 for why this must not change |
| `require()` / `from_env()` / env var names | unchanged |

Because the API surface is byte-identical, **no Rust signature, no `.proto`, and no wire format
changes in Phases 0-1.** Phase 2 changes one Rust struct.

---

## 2. Phase 0 — Measure before building (do this first)

**This phase decides whether Phases 1-2 are worth doing, and it is ~20 lines.**

The benefit of a free list is bounded above by the ratio

```
peak_head_bytes  /  peak_live_bytes
```

where `peak_head_bytes` is today's `high_water_` and `peak_live_bytes` is the peak sum of
outstanding lease lengths. If that ratio is ~1.0 the allocator is not your problem and you
should stop after Phase 0. If it is 2x or more, Phases 1-2 recover that memory.

Nothing currently measures `peak_live_bytes`. Add it to the **existing** allocator.

### 2.1 `src/include/exec/exchange_staging_arena.hpp`

Add two accessors after `high_water()` (currently line 109):

```cpp
  /// Peak sum of outstanding lease lengths — the working set the workload actually needed,
  /// independent of allocator drift. Compare against `high_water()`: the ratio between them is
  /// the arena capacity currently being spent on unreachable gaps rather than on live bytes.
  [[nodiscard]] std::uint64_t peak_live_bytes() const;

  /// Current sum of outstanding lease lengths.
  [[nodiscard]] std::uint64_t live_bytes() const;
```

And two members, after `high_water_` (currently line 124):

```cpp
  std::uint64_t live_bytes_      = 0;
  std::uint64_t peak_live_bytes_ = 0;
```

### 2.2 `src/exec/exchange_staging_arena.cpp`

In `lease()`, immediately after the existing `leases_.emplace(offset, aligned);` (line 224):

```cpp
  live_bytes_      += aligned;
  peak_live_bytes_  = std::max(peak_live_bytes_, live_bytes_);
```

In `release()`, replace the bare `leases_.erase(it);` (line 238). The length must be read
**before** the erase, since `it` is invalidated by it:

```cpp
  const auto released_len = it->second;   // read BEFORE erase
  leases_.erase(it);
  live_bytes_ -= released_len;
```

Add the two accessors next to `outstanding()` / `high_water()` at the end of the file:

```cpp
std::uint64_t exchange_staging_arena::live_bytes() const
{
  std::lock_guard lock(mutex_);
  return live_bytes_;
}

std::uint64_t exchange_staging_arena::peak_live_bytes() const
{
  std::lock_guard lock(mutex_);
  return peak_live_bytes_;
}
```

### 2.3 Extend the teardown log line

The destructor already logs high water (uncommitted change, `:154-165`). Replace that
`SIRIUS_LOG_INFO` call with:

```cpp
    SIRIUS_LOG_INFO(
      "exchange staging arena: high water {} of {} bytes, peak live {} bytes (drift ratio "
      "{:.2f}x), {} leases outstanding",
      high_water_,
      capacity_,
      peak_live_bytes_,
      peak_live_bytes_ > 0 ? static_cast<double>(high_water_) / static_cast<double>(peak_live_bytes_)
                           : 1.0,
      leases_.size());
```

### 2.4 Measure

```bash
cd <repo>
make release
# Then run the representative workload. Whatever launcher you normally use, e.g.:
#   experimental/starrocks/benchmarks/cluster8.sh
#   experimental/starrocks/configs/gb200-4gpu/cluster4-numa.sh
# and grep the CN logs at shutdown:
grep "exchange staging arena: high water" <cn-log>
```

**Decision gate.** Record `drift ratio` per CN for a full TPC-H sweep at your target SF.

| drift ratio | action |
|---|---|
| < 1.2x | **Stop.** The allocator is not the constraint; your arena pressure is real retention (see §7). Keep Phase 0 — the metric is worth having. |
| 1.2x – 2x | Phase 2 (RAII) alone is likely enough; Phase 1 optional. |
| > 2x | Do Phases 1 and 2. The ratio is the fraction of arena you get back. |

Phase 0 is independently shippable and has no behavioural risk. Commit it separately.

---

## 3. Phase 1 — Coalescing free-list allocator (C++)

### 3.1 Design

Replace `head_` with an address-ordered free list. Keep `leases_` for identity checks.

- `free_` : `std::map<uint64_t /*offset*/, uint64_t /*len*/>` — free blocks, address-ordered,
  **never adjacent** (always coalesced on insert).
- `leases_` : `std::map<uint64_t, uint64_t>` — unchanged, offset → aligned length.
- Constructor seeds `free_ = {{0, capacity_}}`.
- `lease(len)`: first-fit scan of `free_` for a block with `len >= aligned`; split, keeping the
  remainder in `free_`; record in `leases_`.
- `release(offset)`: look up in `leases_` (unchanged error path), erase, insert `(offset, len)`
  into `free_`, then coalesce with the immediately-preceding and immediately-following blocks.

Address-ordered first-fit with coalescing is deliberate: it is the classic well-understood
policy, it keeps low addresses dense, and it has no pathological drift. Best-fit was considered
and rejected — it needs a second index for no measurable gain at the tens-of-blocks scale here.

**Why not fixed-size slabs / size classes?** Lease sizes here span three orders of magnitude —
the 16 MiB canary, the 8 MiB zero-row pack slack, and ~768 MB payloads. Any static class split
either wastes a large fraction to internal fragmentation or needs a class the workload does not
use. Coalescing gives zero internal fragmentation and solves the same problem. Revisit only if
§8's fragmentation soak shows external fragmentation is real.

### 3.2 Header edits — `src/include/exec/exchange_staging_arena.hpp`

**Replace the class doc paragraph** (lines 36-41, the one beginning "Leases are bump-allocated")
with:

```cpp
/// Leases are allocated from an address-ordered free list under a mutex and freed explicitly
/// (`lease` / `release` cross an FFI). Each release returns the block and coalesces it with its
/// free neighbours, so ANY released space is immediately reusable — the allocator has no bump
/// head and cannot drift: capacity bounds concurrent live bytes, not lifetime totals. A lease
/// remains one contiguous, `kAlignment`-aligned range, so it is still a valid RDMA target and a
/// valid `cudf::unpack` source.
```

**Replace** the private members (lines 122-125):

```cpp
  mutable std::mutex mutex_;
  std::uint64_t head_       = 0;  // next free offset; always kAlignment-aligned
  std::uint64_t high_water_ = 0;
  std::map<std::uint64_t, std::uint64_t> leases_;  // offset -> aligned length
```

with:

```cpp
  mutable std::mutex mutex_;
  //! Free blocks, address-ordered and always coalesced: no two entries are adjacent.
  //! Seeded with the whole region at construction.
  std::map<std::uint64_t, std::uint64_t> free_;
  std::map<std::uint64_t, std::uint64_t> leases_;  // offset -> aligned length
  std::uint64_t live_bytes_      = 0;
  std::uint64_t peak_live_bytes_ = 0;

  //! Total free bytes and the largest single contiguous free block. Both are reported on
  //! exhaustion: the gap between them IS the external fragmentation, and it is the number that
  //! tells an operator whether to raise capacity or to fix retention.
  [[nodiscard]] std::uint64_t total_free_locked() const;
  [[nodiscard]] std::uint64_t largest_free_locked() const;
```

**Replace** `high_water()`'s doc + declaration (lines 108-109) with:

```cpp
  /// Peak sum of outstanding lease lengths — the working set the workload actually needed.
  [[nodiscard]] std::uint64_t peak_live_bytes() const;

  /// Current sum of outstanding lease lengths.
  [[nodiscard]] std::uint64_t live_bytes() const;

  /// Largest single contiguous free block. `total_free() - largest_free()` is the external
  /// fragmentation; a lease no larger than this is guaranteed to succeed.
  [[nodiscard]] std::uint64_t largest_free() const;

  /// Sum of all free blocks.
  [[nodiscard]] std::uint64_t total_free() const;
```

> `high_water()` is **removed**, not redefined — a peak bump head has no meaning without a bump
> head, and silently changing what an existing accessor means is worse than breaking the build.
> Its two call sites are the destructor log and `test/cpp/exec/test_exchange_staging_arena.cpp`;
> both are updated below. It is not bridged to Rust, so nothing outside C++ breaks.

**Update** the `release()` doc (lines 90-95):

```cpp
  /// Return the lease at `offset`. The block goes back to the free list and coalesces with its
  /// free neighbours, so the space is immediately reusable regardless of release order.
  /// @throws sirius::invalid_input_exception when `offset` is not an outstanding lease
  ///         (double release, or a corrupted offset).
  void release(std::uint64_t offset);
```

### 3.3 Implementation — `src/exec/exchange_staging_arena.cpp`

**Seed the free list.** At the very end of *both* constructor success paths — i.e. after the
`cudaMalloc` branch sets `capacity_`, and after the fabric branch sets `capacity_ = size` — the
free list must contain exactly the whole region. Because the fabric path *overwrites* `capacity_`
with the granularity-rounded size (`:145`), seed it in one place at the end of the constructor
body rather than per-branch. Add immediately before the closing brace of the constructor, and
convert the `cudaMalloc` branch's early `return;` (line 79) into a fall-through so both paths
reach it:

```cpp
  // Seeded last, after BOTH allocation paths have settled `capacity_` — the fabric path rounds
  // it up to the VMM granularity, and the free list must describe the region actually mapped.
  free_.emplace(0, capacity_);
```

> Mechanically: change the `cudaMalloc` branch's trailing `return;` to a guarded skip of the
> fabric block, or hoist the fabric block into `if (want_fabric_arena()) { ... } else { ... }`.
> The latter is cleaner; either is acceptable as long as `free_.emplace(0, capacity_)` runs
> exactly once, after `capacity_` is final, on every success path — and **not** on a throw path.

**Replace `lease()` in full** (lines 202-226):

```cpp
std::uint64_t exchange_staging_arena::lease(std::uint64_t len)
{
  if (len == 0) {
    // A zero-length lease would alias the next lease's offset and break release-by-offset.
    throw sirius::invalid_input_exception("exchange staging arena: zero-length lease");
  }
  std::lock_guard lock(mutex_);
  const auto aligned = align_up(len);
  // align_up wraps for len within kAlignment-1 of UINT64_MAX; a wrapped 0 would slip past the
  // fit scan and register a zero-length lease aliasing a live one. `length` is wire-supplied
  // (handle_staging_lease), so this guard is load-bearing, not theoretical.
  if (aligned < len || aligned > capacity_) {
    throw sirius::invalid_input_exception(
      "exchange staging arena: lease of {} bytes exceeds the {} byte capacity", len, capacity_);
  }

  // Address-ordered first fit: keeps low addresses dense and needs no second index.
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < aligned) { continue; }
    const auto offset    = it->first;
    const auto block_len = it->second;
    free_.erase(it);
    if (block_len > aligned) { free_.emplace(offset + aligned, block_len - aligned); }
    leases_.emplace(offset, aligned);
    live_bytes_ += aligned;
    peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_);
    return offset;
  }

  // Both numbers, because they mean different things: total free short of the request means
  // raise capacity (or fix retention); total free ample but largest block short means external
  // fragmentation.
  throw sirius::invalid_input_exception(
    "exchange staging arena exhausted: requested {} bytes ({} aligned), {} free of {} capacity "
    "in {} blocks (largest {}), {} leases outstanding holding {} bytes "
    "(raise SIRIUS_EXCHANGE_STAGING_BYTES)",
    len,
    aligned,
    total_free_locked(),
    capacity_,
    free_.size(),
    largest_free_locked(),
    leases_.size(),
    live_bytes_);
}
```

**Replace `release()` in full** (lines 228-251):

```cpp
void exchange_staging_arena::release(std::uint64_t offset)
{
  std::lock_guard lock(mutex_);
  auto it = leases_.find(offset);
  if (it == leases_.end()) {
    throw sirius::invalid_input_exception(
      "exchange staging arena: release of offset {} which is not an outstanding lease "
      "(double release?)",
      offset);
  }
  const auto len = it->second;
  leases_.erase(it);
  live_bytes_ -= len;

  // Insert and coalesce with both neighbours, so the free list never holds two adjacent blocks
  // and released space is reusable regardless of the order releases arrive in.
  auto [ins, ok] = free_.emplace(offset, len);
  (void)ok;  // offset came out of leases_, so it cannot already be free

  auto next = std::next(ins);
  if (next != free_.end() && ins->first + ins->second == next->first) {
    ins->second += next->second;
    free_.erase(next);
  }
  if (ins != free_.begin()) {
    auto prev = std::prev(ins);
    if (prev->first + prev->second == ins->first) {
      prev->second += ins->second;
      free_.erase(ins);
    }
  }
}
```

**Add the locked helpers and the public accessors** (replacing `high_water()`):

```cpp
std::uint64_t exchange_staging_arena::total_free_locked() const
{
  std::uint64_t sum = 0;
  for (const auto& [offset, len] : free_) { sum += len; }
  return sum;
}

std::uint64_t exchange_staging_arena::largest_free_locked() const
{
  std::uint64_t best = 0;
  for (const auto& [offset, len] : free_) { best = std::max(best, len); }
  return best;
}

std::uint64_t exchange_staging_arena::total_free() const
{
  std::lock_guard lock(mutex_);
  return total_free_locked();
}

std::uint64_t exchange_staging_arena::largest_free() const
{
  std::lock_guard lock(mutex_);
  return largest_free_locked();
}

std::uint64_t exchange_staging_arena::live_bytes() const
{
  std::lock_guard lock(mutex_);
  return live_bytes_;
}

std::uint64_t exchange_staging_arena::peak_live_bytes() const
{
  std::lock_guard lock(mutex_);
  return peak_live_bytes_;
}
```

**Update the destructor log** (uncommitted change at `:154-165`) — `high_water_` no longer
exists:

```cpp
    SIRIUS_LOG_INFO(
      "exchange staging arena: peak live {} of {} bytes ({} leases outstanding, {} free blocks, "
      "largest {})",
      peak_live_bytes_,
      capacity_,
      leases_.size(),
      free_.size(),
      largest_free_locked());
```

Add `#include <iterator>` for `std::next` / `std::prev` if not already transitively available.

### 3.4 Invariants the implementation must hold

Assert these in tests (§5), not in production code:

1. `total_free() + live_bytes() == capacity()` at all times. *(Because every lease length is
   `align_up`ped and free blocks are exact remainders, this is exact, not approximate.)*
2. No two entries in `free_` are adjacent (`offset + len != next.offset`).
3. No two entries in `leases_` overlap.
4. Every lease offset is `kAlignment`-aligned.
5. Every lease range lies within `[0, capacity_)`.

---

## 4. Phase 2 — RAII lease guard (Rust)

Phase 1 stops drift. Phase 2 stops leaks. They are independent; ship them separately.

### 4.1 The change

`StagedBatch` (`experimental/starrocks/src/fragment_executor.rs:57-64`) is a plain data struct
carrying a lease it does not own. Every release is manual, so any early return between grant
and push abandons the lease permanently.

Give it a `Drop`. Feasibility is good here:

- `sirius::StagingArena` is already `Send + Sync` over the same `shared_ptr`
  (`rust/crates/sirius/src/lib.rs:388-399`) and is cheap to clone at the C++ level.
- `StagedBatch` is constructed in exactly **two** production sites — `engine.rs:312` (send) and
  `compute_node_service.rs:646` (receive) — plus test builders.
- Despite deriving `Clone, Debug, PartialEq, Eq`, it is **never cloned in production code**
  (verified: no `.clone()` on a `StagedBatch` in `local_exchange.rs` or `nixl_transport.rs`).

### 4.2 Edits

`experimental/starrocks/src/fragment_executor.rs` — replace the struct:

```rust
/// One packed batch sitting in an exchange staging arena as cudf packed bytes.
///
/// Owns its lease: dropping this releases it. That is the only release path, so an early return
/// anywhere between the grant and the push cannot orphan arena space. `arena` is `None` only for
/// a metadata-only batch (`len == 0`), which never held a lease.
#[derive(Debug)]
pub struct StagedBatch {
    pub metadata: Vec<u8>,
    pub offset: u64,
    pub len: u64,
    /// The arena the lease lives in. `None` ⇒ nothing to release.
    arena: Option<sirius::StagingArena>,
}

impl StagedBatch {
    /// A batch owning `offset` in `arena`.
    pub fn owned(metadata: Vec<u8>, offset: u64, len: u64, arena: sirius::StagingArena) -> Self {
        Self { metadata, offset, len, arena: Some(arena) }
    }

    /// A metadata-only batch: no payload, no lease, nothing to release.
    pub fn metadata_only(metadata: Vec<u8>) -> Self {
        Self { metadata, offset: 0, len: 0, arena: None }
    }

    /// Give up ownership of the lease — the caller becomes responsible for releasing it.
    /// Only for the paths that hand the offset to a peer and must not release locally.
    pub fn into_unowned_offset(mut self) -> (Vec<u8>, u64, u64) {
        self.arena = None;
        (std::mem::take(&mut self.metadata), self.offset, self.len)
    }
}

impl Drop for StagedBatch {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let Some(arena) = self.arena.as_ref() else { return };
        if let Err(err) = arena.release(self.offset) {
            // Drop cannot fail; a release error here means the lease is already gone or the
            // offset is corrupt, both of which are bugs worth a loud line but not a panic
            // during unwind.
            tracing::warn!(
                offset = self.offset,
                error = %err,
                "failed to release a staging lease on drop"
            );
        }
    }
}
```

Then, mechanically:

1. **Delete every explicit `staging_release` on a `StagedBatch` offset.** Sites:
   `engine.rs:367` (the sweep), `engine.rs:518` (the push loop), `nixl_transport.rs:743`
   (the send loop). Each becomes a `drop(batch)` or simply falls out of scope.
2. **Delete the `released: HashSet<u64>` bookkeeping** in `engine.rs:349, :364-373, :523` and
   the whole `run_fragment` sweep wrapper (`engine.rs:343-378`) — its entire purpose was
   double-release avoidance, which `Drop` makes structural. `run_fragment_inner` becomes
   `run_fragment`.
3. **Update the two construction sites** to `StagedBatch::owned(...)` / `::metadata_only(...)`,
   threading the arena handle in. Both already have access to one:
   `engine.rs` holds `SiriusEngine::staging`, `compute_node_service.rs` reaches it through
   `self.core.executor`.
4. **Drop the `Clone, PartialEq, Eq` derives.** Fix the test builders
   (`local_exchange.rs:372`, `compute_node_service.rs:2620`) — compare fields explicitly rather
   than `assert_eq!` on the struct.
5. **The send loop** (`nixl_transport.rs:706-750`) currently takes `metadata` out of the batch
   with `std::mem::take` and releases the local lease explicitly. Replace with: keep the batch
   alive across the WRITE, let it drop at the end of the iteration. The existing comment
   "The local lease goes back whether the send succeeded or not" is then enforced by the
   language rather than by the `if let Err` arms.
6. **The receiver's grant** in `handle_staging_lease` (`compute_node_service.rs:1106`) hands a
   raw offset to a peer over RPC. It must **not** get a `Drop` guard on the local side — the
   peer owns it until `transmit_packed` arrives. Leave it raw and note why inline. Closing the
   orphan-on-transport-failure hole there needs a release RPC and is out of scope for this plan
   (see §7).

### 4.3 What this closes

Every abandonment path becomes unreachable rather than audited: the cleared-`sources`
cancellation path, the post-`take_ready` validation refusals in `compute_node_service.rs`
(sink/destination checks and translation refusal, which sit *after* `take_ready` has already
removed the batches from the rendezvous and so are invisible to the `engine.rs` sweep), and the
dropped-oneshot path at `engine.rs:281`.

---

## 5. Tests

Run all C++ tests with:

```bash
make release
./build/release/extension/sirius/test/cpp/sirius_unittest "[staging_arena]"
```

Existing tests live in `test/cpp/exec/test_exchange_staging_arena.cpp` (registered at
`CMakeLists.txt:642`). No new file registration is needed — extend that file.

### 5.1 Existing tests that MUST change (they assert the old behaviour)

These are not incidental breakages; they encode the bump semantics. Update them deliberately.

**ARENA-1** (`:36-69`) asserts a released non-trailing lease leaves an unreachable gap:

```cpp
  arena.release(b);
  auto d = arena.lease(1);
  REQUIRE(d == c + 256);        // <-- OLD: gap unreachable
```

Under the free list, `d` must land **in** the gap. Replace with:

```cpp
  // The gap left by a non-trailing release is immediately reusable — this is the whole point.
  arena.release(b);
  auto d = arena.lease(1);
  REQUIRE(d == b);              // NEW: reuses b's block
  REQUIRE(arena.outstanding() == 3);
```

and replace the final `high_water()` assertion with `peak_live_bytes()`.

**ARENA-3** (`:96-111`) asserts the exhaustion string contains `"3072 free"`. Under the free
list the wording changes. Update to assert on the new fields, and — importantly — make the
"refused lease consumes nothing" check non-vacuous, which it currently is not:

```cpp
TEST_CASE("ARENA-3: exhaustion names requested, free, capacity and fragmentation",
          "[staging_arena]")
{
  exchange_staging_arena arena(4096);
  auto a = arena.lease(1024);

  REQUIRE_THROWS_WITH(arena.lease(8192),
                      Catch::Contains("requested 8192") && Catch::Contains("4096 capacity") &&
                        Catch::Contains("largest") &&
                        Catch::Contains("SIRIUS_EXCHANGE_STAGING_BYTES"));

  // A refused lease must consume nothing — checked against live bytes and the free list, not
  // against a head that a later full release would have reset anyway.
  REQUIRE(arena.outstanding() == 1);
  REQUIRE(arena.live_bytes() == 1024);
  REQUIRE(arena.total_free() == 4096 - 1024);
  arena.release(a);
  REQUIRE(arena.lease(4096) == 0);
}
```

**ARENA-7** (`:170-214`) asserts trailing-reclamation specifics (`REQUIRE(arena.lease(256) == y)`
after releasing `z` then `y`). Under a free list those offsets still hold by address-ordered
first-fit, but the *reason* changes. Keep the test, rewrite the comments, and add the assertion
that actually distinguishes the two allocators — see ARENA-10.

**ARENA-2, ARENA-4, ARENA-5, ARENA-6** are unaffected.

### 5.2 New tests — correctness

Append to the same file.

```cpp
// ============================================================================
// ARENA-8: capacity conservation — the invariant that makes drift impossible
// ============================================================================
TEST_CASE("ARENA-8: free + live == capacity at every step", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);
  auto check = [&] { REQUIRE(arena.total_free() + arena.live_bytes() == arena.capacity()); };

  check();
  std::vector<std::uint64_t> held;
  for (int i = 0; i < 16; ++i) {
    held.push_back(arena.lease(1000 + i * 137));
    check();
  }
  // Release in an order chosen to be neither LIFO nor FIFO.
  for (std::size_t i = 0; i < held.size(); ++i) {
    arena.release(held[(i * 7) % held.size()]);
    check();
  }
  REQUIRE(arena.outstanding() == 0);
  REQUIRE(arena.live_bytes() == 0);
  REQUIRE(arena.total_free() == arena.capacity());
  // Fully coalesced back to one block.
  REQUIRE(arena.largest_free() == arena.capacity());
}

// ============================================================================
// ARENA-9: adjacent frees coalesce
// ============================================================================
TEST_CASE("ARENA-9: neighbouring released blocks merge", "[staging_arena]")
{
  exchange_staging_arena arena(4096);
  auto a = arena.lease(1024);
  auto b = arena.lease(1024);
  auto c = arena.lease(1024);
  REQUIRE(arena.largest_free() == 1024);

  // Releasing the middle block alone cannot yet satisfy a 2048 lease.
  arena.release(b);
  REQUIRE(arena.largest_free() == 1024);

  // Releasing a neighbour merges both — coalescing forward.
  arena.release(c);
  REQUIRE(arena.largest_free() == 1024 + 1024 + 1024);  // b + c + the tail

  // And backward.
  arena.release(a);
  REQUIRE(arena.largest_free() == 4096);
  REQUIRE(arena.lease(4096) == 0);
}
```

### 5.3 New test — the one that proves the improvement

This is the test that fails on the old allocator and passes on the new one. It reproduces the
receive-side pattern: several concurrent grants, released out of order, repeated. On a bump
allocator the head walks and it exhausts; on a free list it runs forever in bounded space.

```cpp
// ============================================================================
// ARENA-10: out-of-order steady state does not drift
// ============================================================================
// The defect this pins: with a bump head, every out-of-order release leaves a gap while every
// new lease advances the head, so the head tracks LIFETIME TOTALS rather than concurrency. A
// CN whose receive path grants concurrently and releases out of order therefore drifts toward
// exhaustion no matter how little is actually live. Here 8 leases of 4 KiB are live at any
// moment — 32 KiB of a 1 MiB arena — and the loop runs 10,000 cycles. A bump allocator
// exhausts within the first few hundred.
TEST_CASE("ARENA-10: out-of-order lease/release never drifts", "[staging_arena]")
{
  constexpr std::uint64_t kCapacity  = kMiB;
  constexpr std::uint64_t kLeaseSize = 4096;
  constexpr int kConcurrent          = 8;
  constexpr int kCycles              = 10000;

  exchange_staging_arena arena(kCapacity);
  std::vector<std::uint64_t> live;
  for (int i = 0; i < kConcurrent; ++i) { live.push_back(arena.lease(kLeaseSize)); }

  // Deterministic pseudo-shuffle: release a rotating non-adjacent slot each cycle and
  // immediately re-lease, so the live set size is constant but the release order is not.
  std::uint64_t idx = 0;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    idx = (idx + 5) % kConcurrent;  // 5 is coprime with 8: visits every slot, never in order
    arena.release(live[idx]);
    live[idx] = arena.lease(kLeaseSize);  // must not throw
    REQUIRE(arena.live_bytes() == kConcurrent * kLeaseSize);
    REQUIRE(arena.total_free() + arena.live_bytes() == kCapacity);
  }

  REQUIRE(arena.outstanding() == kConcurrent);
  REQUIRE(arena.peak_live_bytes() == kConcurrent * kLeaseSize);
  for (auto offset : live) { arena.release(offset); }
  REQUIRE(arena.largest_free() == kCapacity);
}
```

> **Run this against the unmodified allocator first.** It should throw `exchange staging arena
> exhausted` well before 10,000 cycles. That failure is the executable statement of the problem
> and belongs in the commit message.

### 5.4 New tests — gaps in the current suite

The existing suite has no concurrency test, never touches the device memory it hands out, and
never checks that two live leases do not alias. All three matter more once the allocator can
hand back previously-used space.

```cpp
// ============================================================================
// ARENA-11: live leases never alias, verified through the device memory itself
// ============================================================================
TEST_CASE("ARENA-11: concurrent leases address disjoint device memory", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);
  constexpr std::size_t kLen = 4096;
  std::vector<std::uint64_t> offsets;
  for (int i = 0; i < 16; ++i) { offsets.push_back(arena.lease(kLen)); }

  // Stamp each lease with its own byte pattern...
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    auto* p = reinterpret_cast<void*>(arena.base() + offsets[i]);
    REQUIRE(cudaMemset(p, static_cast<int>(i + 1), kLen) == cudaSuccess);
  }
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // ...and read every one back. Any overlap shows up as a wrong pattern.
  std::vector<unsigned char> host(kLen);
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    const auto* p = reinterpret_cast<const void*>(arena.base() + offsets[i]);
    REQUIRE(cudaMemcpy(host.data(), p, kLen, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(host.front() == static_cast<unsigned char>(i + 1));
    REQUIRE(host.back() == static_cast<unsigned char>(i + 1));
  }

  // Release a subset and re-lease: the reused space must still be disjoint from what is live.
  for (std::size_t i = 0; i < offsets.size(); i += 2) { arena.release(offsets[i]); }
  std::vector<std::uint64_t> reused;
  for (std::size_t i = 0; i < offsets.size() / 2; ++i) { reused.push_back(arena.lease(kLen)); }
  for (auto offset : reused) {
    auto* p = reinterpret_cast<void*>(arena.base() + offset);
    REQUIRE(cudaMemset(p, 0xEE, kLen) == cudaSuccess);
  }
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  // The odd-indexed leases were never released and must be untouched.
  for (std::size_t i = 1; i < offsets.size(); i += 2) {
    const auto* p = reinterpret_cast<const void*>(arena.base() + offsets[i]);
    REQUIRE(cudaMemcpy(host.data(), p, kLen, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(host.front() == static_cast<unsigned char>(i + 1));
  }
}

// ============================================================================
// ARENA-12: the allocator is safe under concurrent lease/release
// ============================================================================
// `lease`/`release` are called from transport, RPC and engine threads concurrently; nothing in
// the suite covered that until now.
TEST_CASE("ARENA-12: concurrent lease/release keeps its invariants", "[staging_arena]")
{
  exchange_staging_arena arena(8 * kMiB);
  constexpr int kThreads = 8;
  constexpr int kIters   = 2000;
  std::atomic<int> failures{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&arena, &failures] {
      for (int i = 0; i < kIters; ++i) {
        try {
          auto a = arena.lease(1024);
          auto b = arena.lease(4096);
          if (a % exchange_staging_arena::kAlignment != 0) { ++failures; }
          if (a == b) { ++failures; }
          arena.release(a);
          arena.release(b);
        } catch (const std::exception&) {
          ++failures;  // an 8 MiB arena with 8 threads x 5 KiB must never exhaust
        }
      }
    });
  }
  for (auto& th : threads) { th.join(); }

  REQUIRE(failures.load() == 0);
  REQUIRE(arena.outstanding() == 0);
  REQUIRE(arena.live_bytes() == 0);
  REQUIRE(arena.largest_free() == arena.capacity());
}

// ============================================================================
// ARENA-13: an oversized request is refused, not wrapped
// ============================================================================
TEST_CASE("ARENA-13: align_up overflow cannot alias a live lease", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);
  auto live = arena.lease(1024);

  // align_up wraps to 0 for these; without the guard the request would slip past the fit scan
  // and register a zero-length lease that aliases whatever is allocated next.
  REQUIRE_THROWS_AS(arena.lease(std::numeric_limits<std::uint64_t>::max()),
                    sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(arena.lease(std::numeric_limits<std::uint64_t>::max() - 100),
                    sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(arena.lease(2 * kMiB), sirius::invalid_input_exception);

  REQUIRE(arena.outstanding() == 1);
  REQUIRE(arena.live_bytes() == 1024);
  arena.release(live);
}
```

Add to the includes at the top of the test file:

```cpp
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
```

### 5.5 Rust tests for Phase 2

Run with:

```bash
cd experimental/starrocks
pixi run cn-test-no-engine     # pure Rust, no GPU — what CI runs
pixi run cn-test               # engine-linked, needs a GPU
```

Add to `experimental/starrocks/src/fragment_executor.rs` (or wherever the existing
`StagedBatch` tests live):

```rust
#[cfg(test)]
mod staged_batch_drop {
    use super::*;

    // A batch that goes out of scope on ANY path must return its lease. This is the property
    // that replaces the manual sweep in run_fragment.
    #[test]
    fn drop_releases_the_lease() {
        let arena = test_arena(64 * 1024 * 1024);
        let before = arena_outstanding(&arena);
        {
            let offset = arena.lease(4096).unwrap();
            let _batch = StagedBatch::owned(vec![1, 2, 3], offset, 4096, arena.clone());
            assert_eq!(arena_outstanding(&arena), before + 1);
        }
        assert_eq!(arena_outstanding(&arena), before, "drop must release");
    }

    // The zero-row wire contract: no lease was ever taken, so drop must not attempt a release
    // (which would throw "not an outstanding lease", or worse, free an unrelated lease at 0).
    #[test]
    fn metadata_only_batch_releases_nothing() {
        let arena = test_arena(64 * 1024 * 1024);
        let squatter = arena.lease(4096).unwrap();
        assert_eq!(squatter, 0, "precondition: something else holds offset 0");
        {
            let _batch = StagedBatch::metadata_only(vec![1, 2, 3]);
        }
        // The lease at offset 0 must still be live.
        arena.release(squatter).expect("offset 0 must not have been freed by the drop");
    }

    // An early return between grant and push must not orphan the lease.
    #[test]
    fn early_return_releases_the_lease() {
        let arena = test_arena(64 * 1024 * 1024);
        let before = arena_outstanding(&arena);
        fn fails_after_grant(arena: &sirius::StagingArena) -> Result<(), String> {
            let offset = arena.lease(4096).map_err(|e| e.to_string())?;
            let _batch = StagedBatch::owned(Vec::new(), offset, 4096, arena.clone());
            Err("simulated validation refusal".to_string())
        }
        assert!(fails_after_grant(&arena).is_err());
        assert_eq!(arena_outstanding(&arena), before, "error path must not orphan");
    }

    // into_unowned_offset is the deliberate escape hatch; it must actually disarm the guard.
    #[test]
    fn into_unowned_offset_disarms_drop() {
        let arena = test_arena(64 * 1024 * 1024);
        let offset = arena.lease(4096).unwrap();
        let batch = StagedBatch::owned(vec![9], offset, 4096, arena.clone());
        let (_metadata, off, len) = batch.into_unowned_offset();
        assert_eq!((off, len), (offset, 4096));
        // Still outstanding — the caller owns it now and must release explicitly.
        arena.release(offset).expect("lease must still be outstanding after disarm");
    }
}
```

> `test_arena` / `arena_outstanding` are helpers you will need to add. `outstanding()` is
> **not currently bridged to Rust** — bridge it as part of Phase 0 (§2), it is a two-line
> addition to `rust/crates/sirius-sys/src/lib.rs` (`fn outstanding(self: &StagingArena) -> usize;`)
> plus a forward in `src/sirius_ffi.cpp` and `rust/crates/sirius/src/lib.rs`. Without it these
> tests cannot assert anything and neither can the CN observe a leak.

Also verify the existing end-to-end test still passes:

```bash
cargo test -p sirius --release packed_hop_matches_relay_hop
```

---

## 6. Explicitly out of scope, and why

**Do not make the arena block on exhaustion.** A condvar in `lease()` would reintroduce exactly
the head-of-line problem the `Send + Sync` `StagingArena` handle exists to avoid: a peer's
`request_staging_lease`, served on the caller's thread, would then wait on an arena that only
the local engine thread can drain. Backpressure belongs one level up — bound
`SenderSource::Remote { batches }` in `local_exchange.rs:240` and refuse the grant with a
retryable status. That is a separate change with its own protocol implications; it is listed
here so it is not accidentally folded in.

**Do not touch the fabric/VMM allocation path.** Phases 1-2 sit strictly above it. The one
interaction is that `capacity_` is rounded up on that path, which §3.3 handles by seeding the
free list after `capacity_` is final.

---

## 7. What this does NOT fix

State plainly so the acceptance criteria are honest.

- **Peak live bytes are unchanged.** `take_ready` (`local_exchange.rs:277-279`) withholds a
  receiver until *every* expected sender of *every* exchange node has closed, so a receiver
  holds its entire fan-in from first WRITE until it runs. If 32 GiB is genuinely live at SF500,
  it stays 32 GiB. A free list removes fragmentation waste, not retention.
- **Global serialization is unchanged.** `executor.run` funnels to the single engine thread
  (`engine.rs:139-142, :257`), so receiver B holds its leases across receiver A's whole run.
- **The peer lease orphaned on transport failure** (`nixl_transport.rs:696-708`) still leaks:
  the grant lives in the *peer's* arena and there is no release RPC. Local `Drop` cannot reach
  it. Fixing it needs a `release_staging_lease` RPC, or a lease-expiry sweep keyed on the
  fragment instance.
- **No throughput change.** The fabric path and the wire format are untouched.
- **Nothing here addresses the primary q02 wedge.**

---

## 8. Acceptance criteria

Ship only if all of these hold.

| # | Criterion | How to check |
|---|---|---|
| A1 | ARENA-10 fails on the pre-change allocator and passes after | run it on both, record both outputs |
| A2 | Full `[staging_arena]` suite green | `sirius_unittest "[staging_arena]"` |
| A3 | Full C++ suite green, no new failures | `make test_release` |
| A4 | `packed_hop_matches_relay_hop` still passes | `cargo test -p sirius --release` |
| A5 | CN suites green | `pixi run cn-test-no-engine` and `pixi run cn-test` |
| A6 | A full TPC-H sweep at target SF completes with **the same or smaller** `SIRIUS_EXCHANGE_STAGING_BYTES` | compare against the pre-change baseline |
| A7 | Teardown log reports `0 leases outstanding` after a clean sweep | `grep "exchange staging arena: peak live"` |
| A8 | Soak: 5 consecutive sweeps in one CN process, no exhaustion, `peak live` stable across sweeps | the drift signature is `peak live` climbing sweep over sweep |
| A9 | External fragmentation bounded: at exhaustion (if any), `largest free` within 10% of `total free` | new exhaustion message |

**A8 is the one that matters.** Drift is a function of *uptime*, not of a single query. A
single-sweep pass proves nothing; a stable `peak live` across five sweeps is the evidence that
the allocator no longer accumulates.

**The headline number** to report: pre-change `high water / peak live` ratio from Phase 0
versus post-change `peak live / capacity`. If Phase 0 measured 3x drift, the same workload
should now fit in roughly a third of the arena — and that memory goes back to `GPU_MEM`, which
is the actual win, given the two compete for the same card.

---

## 9. Execution checklist

Commit in this order; each step is independently revertable.

- [ ] **C0** Phase 0 instrumentation (`live_bytes_`, `peak_live_bytes_`, drift ratio in the
      teardown log) + bridge `outstanding()` to Rust. No behaviour change.
- [ ] **C0b** Run the target workload. Record the drift ratio. **Gate: stop here if < 1.2x.**
- [ ] **C1** Free-list allocator (§3), header + impl, `high_water()` removed.
- [ ] **C2** Update ARENA-1 / ARENA-3 / ARENA-7 for the new semantics (§5.1).
- [ ] **C3** New tests ARENA-8..ARENA-13 (§5.2-5.4). Confirm ARENA-10 fails on `C0`.
- [ ] **C4** `StagedBatch` RAII (§4.2 steps 1-4).
- [ ] **C5** Send-loop simplification (§4.2 step 5), sweep + `released` set deleted.
- [ ] **C6** Rust drop tests (§5.5).
- [ ] **C7** Soak run (A8) and the sizing comparison (A6). Update
      `bench/a100x8/TUNING.md` §2 and any launcher whose `SIRIUS_EXCHANGE_STAGING_BYTES` can
      now come down.

### Files touched

```
src/include/exec/exchange_staging_arena.hpp        C0, C1
src/exec/exchange_staging_arena.cpp                C0, C1
test/cpp/exec/test_exchange_staging_arena.cpp      C2, C3
src/sirius_ffi.cpp                                 C0  (outstanding() forward)
src/include/sirius_ffi.hpp                         C0  (declaration)
rust/crates/sirius-sys/src/lib.rs                  C0  (bridge decl)
rust/crates/sirius/src/lib.rs                      C0  (StagingArena::outstanding)
experimental/starrocks/src/fragment_executor.rs    C4, C6
experimental/starrocks/src/engine.rs               C4, C5
experimental/starrocks/src/nixl_transport.rs       C5
experimental/starrocks/src/compute_node_service.rs C4
experimental/starrocks/src/local_exchange.rs       C4  (test builders only)
bench/a100x8/TUNING.md                             C7
```

Nothing in `CMakeLists.txt` changes — the test file is already registered at line 642.

### Build/run reference

```bash
# C++ engine + tests
make release
./build/release/extension/sirius/test/cpp/sirius_unittest "[staging_arena]"
make test_release                       # full suite

# Rust CN
cd experimental/starrocks
pixi run cn-test-no-engine               # no GPU needed
pixi run cn-test                         # engine-linked

# Two-CN smoke over nixl
pixi run cluster2                        # sets SIRIUS_EXCHANGE_STAGING_BYTES=1280MiB
```

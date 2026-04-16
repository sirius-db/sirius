# DuckDB BufferManager::Pin() Overhead Analysis

## Problem

GPU scan throughput is bottlenecked by `BufferManager::Pin()` overhead, not GPU decode.

### Measurement (TPC-H SF100, GH200 480GB, `-readonly` mode)

**`SELECT SUM(l_extendedprice) FROM lineitem`** (600M rows, 1 numeric column):

| Component | Time | % of Wall |
|-----------|------|-----------|
| GPU decode kernels | 7ms | 1.1% |
| H2D data transfer | 29ms | 4.6% |
| **Pin() overhead** | **470ms** | **74.6%** |
| Other CPU overhead | 124ms | 19.7% |
| **Wall clock** | **630ms** | 100% |

DuckDB CPU baseline: **95ms** (6.6x faster than GPU!)

The GPU finishes in 36ms (7ms decode + 29ms H2D). The remaining 594ms is CPU overhead,
dominated by 470ms of `BufferManager::Pin()` calls.

### Root Cause

`StandardBufferManager::Pin()` acquires a **per-block mutex** even for already-loaded
mmap'd blocks:

```
BufferManager::Pin(handle):
  1. block_memory.GetLock()          ← mutex acquisition (~1-5µs)
  2. check state == BLOCK_LOADED     ← always true for mmap'd readonly
  3. handle->Load(context)           ← increment refcount, create BufferHandle
  4. release lock
```

For SF100 lineitem with 1 column: ~29,000 segments × 16µs/Pin = **470ms**.
For Q1 with 7 columns: ~11,000 segments × ~10µs/Pin = **110ms**.

### What We Tried (Exhaustive)

| Approach | Result | Issue |
|----------|--------|-------|
| **Skip Pin (direct GetBuffer)** | Wrong results | Blocks not loaded until first Pin |
| **Conditional skip (if loaded)** | No speedup | Most blocks not pre-loaded |
| **Pre-pin upfront** | Regression (+8%) | Loses overlap with GPU decode |
| **Lock-free Pin fast path (DuckDB mod)** | No speedup | Fast path rarely hit — each block only pinned once |
| **Own mmap (MAP_PRIVATE)** | Correct, cold cache | Separate page table entries — not shared |
| **Own mmap (MAP_SHARED)** | Correct, cold cache | Pages still not shared with DuckDB's mapping |
| **Own mmap (MAP_POPULATE)** | **2.3x faster!** | Works but 16s startup cost for 26GB |
| **Cache warming + own mmap** | 3x regression | DuckDB evicts blocks on handle drop |

### Key Finding

The Pin overhead is NOT from the mutex lock (~1µs). It's from:
1. **Block loading**: `block_manager.Read()` copies 256KB from mmap to buffer pool (~15µs)
2. **BufferHandle creation**: shared_ptr copy + atomic refcount (~2µs)
3. **Segment tree traversal**: linked list walk + virtual calls (~3µs)
Total: ~20µs per segment × 29K segments = 580ms

The lock is only 5-10% of the per-segment cost. Even a lock-free Pin saves <50ms.

### What Would Help — Options for DuckDB

**Option 1: Lock-free Pin for read-only mmap'd databases**

When `DatabaseInstance::IsReadOnly()` and blocks are mmap'd, the buffer is
guaranteed valid for the connection's lifetime. No eviction is possible.
A lock-free fast path could return the pointer directly:

```cpp
BufferHandle StandardBufferManager::Pin(shared_ptr<BlockHandle> &handle) {
    auto &memory = handle->GetMemory();
    // Fast path: read-only mmap'd — no lock needed
    if (is_read_only_ && memory.GetState() == BlockState::BLOCK_LOADED) {
        return BufferHandle(handle, memory.GetBuffer().get());  // no lock, no refcount
    }
    // ... existing slow path with mutex ...
}
```

**Expected impact**: Pin 29K blocks in ~5ms instead of 470ms (lock-free pointer return).

**Option 2: Bulk Pin API**

Single-lock-acquisition for multiple blocks:

```cpp
vector<BufferHandle> StandardBufferManager::PinBatch(
    vector<shared_ptr<BlockHandle>> &handles) {
    // Acquire pool-level lock ONCE
    // Pin all blocks under the same lock
    // Release lock
}
```

**Expected impact**: 29K blocks with 1 lock acquisition instead of 29K (~30ms total).

**Option 3 (RECOMMENDED): Zero-copy Pin for read-only mmap**

For read-only mmap'd databases, `Pin()` should return the mmap'd pointer directly
WITHOUT copying the block into the buffer pool. The block data is already accessible
at `mmap_base + block_offset` — the memcpy into a FileBuffer is unnecessary.

```cpp
BufferHandle StandardBufferManager::Pin(shared_ptr<BlockHandle> &handle) {
    // For persistent blocks on read-only mmap'd storage:
    // return the mmap'd pointer directly, no buffer pool allocation needed.
    if (handle->BlockId() < MAXIMUM_BLOCK && is_read_only_) {
        auto &bm = handle->GetBlockManager();
        // Compute pointer directly from mmap: base + block_start + id * alloc_size + header
        auto ptr = bm.GetMmapPointer(handle->BlockId());
        return BufferHandle::CreateReadOnly(ptr);  // lightweight, no refcount
    }
    // ... existing slow path ...
}
```

This eliminates BOTH the mutex AND the block-loading memcpy.
Expected: 29K × 20µs → 29K × 0.1µs = 3ms (from 470ms).

**Option 4: Pre-load all blocks at database open for read-only mode**

When opening a database read-only, pre-load (mmap) all blocks so that
`block_memory.GetBuffer()` is never null. This enables direct access
without Pin:

```cpp
// At database open:
for (auto& block : all_blocks) {
    Pin(block);  // loads from mmap
    // keep handle alive for session lifetime
}
```

Then client code can use `block->GetMemory().GetBuffer()->buffer` directly.

**Expected impact**: One-time cost at open (few seconds for SF100), then ~0ms per query Pin.

### Why This Matters

At SF100, CPU overhead (primarily Pin) is **10-20x larger than GPU compute**.
The GPU scan is fundamentally limited by how fast we can get block pointers
from DuckDB, not by how fast we can decode on GPU. Eliminating Pin overhead
would make GPU scan **competitive with or faster than CPU** for scan-heavy queries.

### Files Involved

- `src/storage/standard_buffer_manager.cpp:307` — `StandardBufferManager::Pin()`
- `src/include/duckdb/storage/buffer/block_handle.hpp` — `BlockHandle`, `BlockMemory`
- Sirius consumer: `src/op/scan/direct_block_scan.cpp:81` — `buffer_manager.Pin(segment.block)`

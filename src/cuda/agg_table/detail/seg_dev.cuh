// Device code shared by the translation units of the segmented table (seg_table.cu, seg_hot.cu,
// seg_keys.cu): the table as kernels see it, slot addressing, the probe. Ported from
// experimental/agg_bench/src/seg_dev.cuh; the only change to what kernels see is
// seg_layout::seen (the table's reserved-key flag, which the prototype kept in a process-wide
// `__constant__` block, see sql_dev.cuh).
//
// Why two units: CUDA loads a unit's kernels on the first launch of any of them, and the time
// grows with the number of kernels in the unit (round 2's single unit: ~0.6 ms inside the timed
// "alloc" phase, against ~0.1 ms for round 1's table.cu). seg_hot.cu holds only what a pre-sized
// table without growth runs (initialise, fold, gather); everything else stays in seg_table.cu and
// is loaded when first needed.
#pragma once

#include "table_dev.cuh"

#include <cuda_runtime.h>

#include <type_traits>

namespace sirius::agg_table::detail {
namespace segdev {

using namespace tdev;
using ull = unsigned long long;

constexpr uint32_t MAX_SBITS    = 8;
constexpr uint32_t MAX_SEGMENTS = 1u << MAX_SBITS;
// seg_layout::seg_off of a segment that was flushed out of a sealed table
// (seg_table::flush_segment): it holds no key any more, and every lookup of a key that hashes to it
// misses.
constexpr int64_t SEG_EVICTED = INT64_MIN;

// The table as the fold kernels see it. A segment's address is `ref + seg_off[p]` (in words):
// pointer arithmetic on a kernel parameter, so the compiler knows the slots are global memory
// and emits global atomics (ATOMG). A segment pointer loaded from an array has no known address
// space; its atomics become the generic form with a shared-memory fallback, which measured
// 10-20% slower folds for narrow states.
struct seg_layout {
  uint64_t* ref;                  // base of segment 0
  int64_t seg_off[MAX_SEGMENTS];  // (base of segment p - ref) in words (in the kernel's parameters)
  uint64_t cap;                   // slots per segment (multiple of 8)
  uint64_t* count;                // groups in the whole table
  uint64_t limit;                 // groups the table may hold
  uint32_t* full;                 // per segment: set once a probe found no free slot within reach
  uint32_t sbits;
  uint32_t skip_bits;
  uint32_t lazy;  // BUCKET: free slots hold no state (see BUSY_KEY)
  // Sealed BUCKET table with fingerprints (seg_table::build_fingerprints), else null: one word
  // per bucket, byte j = fingerprint of slot j (0: free); segment p's words start at
  // fp + p * (cap / 8).
  uint64_t const* fp;
  uint32_t order;  // direct fold: order of a row's state updates (fold_raw_ordered)
  // Two device words of the table's owner: [0] is set once a kernel skipped a raw row with a
  // reserved key (sql_dev.cuh: reserved_key), [1] is scratch.
  uint32_t* seen;
};

__device__ __forceinline__ uint64_t* seg_base(seg_layout const& L, uint64_t p)
{
  return L.ref + L.seg_off[p];
}

// BUCKET layout with lazy state: the state words of a free slot are never initialised (creating
// or growing a table writes the key lines only, 1/(1 + state words) of the bytes). A thread
// claims a free slot by swapping EMPTY -> BUSY, writes the group's first state with plain
// stores (no identities, no atomics), makes them visible (__threadfence) and only then publishes
// the key; a thread that reads BUSY waits for the key. (Waiting on a lane of the same warp
// relies on independent thread scheduling: Volta and later.) BUSY_KEY, like EMPTY_KEY, cannot
// be a key.
constexpr uint64_t BUSY_KEY = EMPTY_KEY - 1;

__device__ __forceinline__ void publish(uint64_t* claim, uint64_t k)
{
  __threadfence();
  *static_cast<volatile uint64_t*>(claim) = k;
}

// First state of a group from one raw row / from a partial state (plain stores: the slot is
// claimed and not yet published).
template <int NS>
__device__ __forceinline__ void init_raw(
  uint64_t* st, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  st[0] = 1;
  if (NS > 1) st[1] = a;
  if (NS > 2) st[2] = b;
  if (NS > 3) st[3] = c;
  if (NS > 4) st[4] = d;
  if (NS > 5) st[5] = a;
  if (NS > 6) st[6] = c;
  if (NS > 7) st[7] = b;
}
template <int NS>
__device__ __forceinline__ void init_partial(uint64_t* st, uint64_t const* v)
{
  for (int j = 0; j < NS; ++j)
    st[j] = v[j];
}

// Where a key lives: its segment, the hash bits that pick the slot, and (BUCKET layout) the
// slot of the bucket its scan starts at.
struct hpos {
  uint64_t p;
  uint64_t hs;
  uint32_t j0;
};

__device__ __forceinline__ hpos locate(uint64_t k, uint32_t sbits, uint32_t skip_bits)
{
  uint64_t f = fmix64(k);
  uint64_t h = f << skip_bits;
  hpos q;
  q.p  = sbits ? h >> (64 - sbits) : 0;
  q.hs = h << sbits;
  q.j0 = static_cast<uint32_t>(f) & 7u;
  return q;
}

// Slot addressing. ROW: slot i is [key, state...]. BUCKET: bucket i/8 is [8 keys][8 state rows].
template <bool BK>
__device__ __forceinline__ uint64_t* key_ptr(uint64_t* base, uint64_t i, uint32_t NS)
{
  if (BK) return base + (i >> 3) * (8ull * (NS + 1)) + (i & 7);
  return base + i * (NS + 1);
}
template <bool BK>
__device__ __forceinline__ uint64_t* state_ptr(uint64_t* base, uint64_t i, uint32_t NS)
{
  if (BK) return base + (i >> 3) * (8ull * (NS + 1)) + 8 + (i & 7) * NS;
  return base + i * (NS + 1) + 1;
}

// A probe that inserts gives up after this many steps without a free slot (ROW: slots, BUCKET:
// buckets) and flags the segment as full; once flagged, after FULL_CHECK steps. Only reachable
// when keys arrive in hash order (one segment fills while the table as a whole is below its
// limit) or far above the supported load factors.
constexpr uint64_t PMAX_ROW = 8192, PMAX_BUCKET = 1024;
constexpr uint64_t FULL_CHECK_ROW = 256, FULL_CHECK_BUCKET = 32;

// Finds or claims the slot of `k`; returns its state words. nullptr: the key is absent and may
// not be inserted (`may_insert` false, table at its group limit, or no free slot within reach;
// in the last case a key that sits further along a full segment is reported absent too).
// BUCKET: the scan of a bucket starts at slot j0 and wraps; a key sits at the first slot that
// was free in that order when it arrived, so an empty slot ends the search.
// `claim` (lazy state): when set, the caller holds the slot unpublished: it must write the
// group's first state to the returned words and then `publish(claim, k)`.
template <bool BK>
__device__ __forceinline__ uint64_t* seg_probe(
  seg_layout const& L, uint32_t NS, uint64_t k, bool may_insert, bool& inserted, uint64_t*& claim)
{
  hpos q = locate(k, L.sbits, L.skip_bits);
  // (Only a sealed table has evicted segments, and nothing is inserted into a sealed table: the
  // kernels that insert pass a constant `true` and do not carry this test.)
  if (!may_insert && L.seg_off[q.p] == SEG_EVICTED) return nullptr;
  uint64_t* base = seg_base(L, q.p);
  if (BK) {
    uint64_t const nb = L.cap >> 3;
    uint64_t const BW = 8ull * (NS + 1);
    uint64_t b        = __umul64hi(q.hs, nb);
    for (uint64_t t = 0; t < nb; ++t) {
      uint64_t* kb = base + b * BW;
      for (uint32_t jj = 0; jj < 8; ++jj) {
        uint32_t j   = (q.j0 + jj) & 7u;
        uint64_t cur = kb[j];
        if (cur == EMPTY_KEY) {
          if (!may_insert || *L.count >= L.limit) return nullptr;
          if (L.lazy) {
            cur = atomicCAS(reinterpret_cast<ull*>(kb + j), EMPTY_KEY, BUSY_KEY);
            if (cur == EMPTY_KEY) {
              inserted = true;
              claim    = kb + j;
              return kb + 8 + j * NS;
            }
          } else {
            cur = atomicCAS(reinterpret_cast<ull*>(kb + j), EMPTY_KEY, k);
            if (cur == EMPTY_KEY) {
              inserted = true;
              return kb + 8 + j * NS;
            }
          }
        }
        while (cur == BUSY_KEY)
          cur = *static_cast<volatile uint64_t*>(kb + j);
        if (cur == k) return kb + 8 + j * NS;
      }
      if (++b == nb) b = 0;
      if (may_insert && t >= FULL_CHECK_BUCKET && (t >= PMAX_BUCKET || L.full[q.p] != 0)) break;
    }
    if (may_insert) L.full[q.p] = 1u;
    return nullptr;
  }
  uint32_t const W = NS + 1;
  uint64_t in      = __umul64hi(q.hs, L.cap);
  for (uint64_t t = 0; t < L.cap; ++t) {
    uint64_t* kp = base + in * W;
    uint64_t cur = *kp;
    if (cur == EMPTY_KEY) {
      if (!may_insert || *L.count >= L.limit) return nullptr;
      cur = atomicCAS(reinterpret_cast<ull*>(kp), EMPTY_KEY, k);
      if (cur == EMPTY_KEY) {
        inserted = true;
        return kp + 1;
      }
    }
    if (cur == k) return kp + 1;
    if (++in == L.cap) in = 0;
    if (may_insert && t >= FULL_CHECK_ROW && (t >= PMAX_ROW || L.full[q.p] != 0)) break;
  }
  if (may_insert) L.full[q.p] = 1u;
  return nullptr;
}

// Fingerprint of a key: a hash byte that neither the segment and bucket (top bits) nor the
// start slot (low 3 bits) depend on; never 0.
__device__ __forceinline__ uint64_t fp_byte(uint64_t f)
{
  uint64_t b = (f >> 8) & 0xffu;
  return b != 0 ? b : 1u;
}
constexpr uint64_t FP_LO = 0x0101010101010101ull, FP_HI = 0x8080808080808080ull;

// Exact lookup, nothing inserted: the state words of `k`, or nullptr. For a table no insert is
// in flight on (a sealed one): the walk ends at the first free slot of the key's probe sequence.
// With fingerprints (BUCKET) a bucket is first judged by its fingerprint word: slots whose byte
// matches are compared with the key, and a bucket with a free slot ends the search - a key is
// stored in the first bucket of its sequence that had a free slot when it arrived, and slots are
// never freed. So a miss usually reads one 8-byte word of an array 8 x (state words + 1) times
// smaller than the table, and never a key line unless a fingerprint matches (1 slot in 255).
template <bool BK>
__device__ __forceinline__ uint64_t* seg_lookup(seg_layout const& L, uint32_t NS, uint64_t k)
{
  if (BK && L.fp != nullptr) {
    hpos q = locate(k, L.sbits, L.skip_bits);
    if (L.seg_off[q.p] == SEG_EVICTED) return nullptr;
    uint64_t const nb  = L.cap >> 3;
    uint64_t const BW  = 8ull * (NS + 1);
    uint64_t const* fw = L.fp + q.p * nb;
    uint64_t const pat = FP_LO * fp_byte(fmix64(k));
    uint64_t b         = __umul64hi(q.hs, nb);
    for (uint64_t t = 0; t < nb; ++t) {
      uint64_t w = fw[b];
      uint64_t x = w ^ pat;
      // Bytes of x that are zero (candidates; a byte above a true match can be flagged falsely,
      // which only costs a key comparison).
      uint64_t hit = (x - FP_LO) & ~x & FP_HI;
      while (hit != 0) {
        uint32_t j   = static_cast<uint32_t>(__ffsll(static_cast<long long>(hit)) - 1) >> 3;
        uint64_t* kb = seg_base(L, q.p) + b * BW;
        if (kb[j] == k) return kb + 8 + j * NS;
        hit &= hit - 1;
      }
      if (((w - FP_LO) & ~w & FP_HI) != 0) return nullptr;  // the bucket has a free slot
      if (++b == nb) b = 0;
    }
    return nullptr;
  }
  bool inserted   = false;
  uint64_t* claim = nullptr;
  return seg_probe<BK>(L, NS, k, false, inserted, claim);
}

// Claims a slot for a key known to be absent (rehash, restore); returns its state words.
template <bool BK>
__device__ __forceinline__ uint64_t* seg_insert_new(
  uint64_t* base, uint64_t cap, uint32_t NS, hpos q, uint64_t k)
{
  if (BK) {
    uint64_t const nb = cap >> 3;
    uint64_t const BW = 8ull * (NS + 1);
    uint64_t b        = __umul64hi(q.hs, nb);
    for (uint64_t t = 0; t < nb; ++t) {
      uint64_t* kb = base + b * BW;
      for (uint32_t jj = 0; jj < 8; ++jj) {
        uint32_t j = (q.j0 + jj) & 7u;
        if (kb[j] == EMPTY_KEY &&
            atomicCAS(reinterpret_cast<ull*>(kb + j), EMPTY_KEY, k) == EMPTY_KEY) {
          return kb + 8 + j * NS;
        }
      }
      if (++b == nb) b = 0;
    }
    return nullptr;
  }
  uint32_t const W = NS + 1;
  uint64_t in      = __umul64hi(q.hs, cap);
  for (uint64_t t = 0; t < cap; ++t) {
    uint64_t* kp = base + in * W;
    if (*kp == EMPTY_KEY && atomicCAS(reinterpret_cast<ull*>(kp), EMPTY_KEY, k) == EMPTY_KEY)
      return kp + 1;
    if (++in == cap) in = 0;
  }
  return nullptr;
}

// Adds the number of threads of the block whose `flag` is set to *c0 (and *c1): one global atomic
// per block. (One per warp on the same word, from every warp of a large launch, is what made
// round 2's counting and rehash kernels slow.) Must be reached by every thread of the block.
__device__ __forceinline__ void block_count(bool flag, ull* c0, ull* c1 = nullptr)
{
  __shared__ unsigned int acc;
  if (threadIdx.x == 0) acc = 0;
  __syncthreads();
  unsigned int m = __ballot_sync(0xffffffffu, flag);
  if ((threadIdx.x & 31) == 0 && m != 0) atomicAdd(&acc, static_cast<unsigned int>(__popc(m)));
  __syncthreads();
  if (threadIdx.x == 0 && acc != 0) {
    atomicAdd(c0, static_cast<ull>(acc));
    if (c1 != nullptr) atomicAdd(c1, static_cast<ull>(acc));
  }
}

__device__ __forceinline__ void mark(uint32_t* bitmap, uint64_t r)
{
  atomicOr(bitmap + (r >> 5), 1u << (r & 31));
}

// Dispatch on the (compile-time) state width and layout.
template <class Fn>
void by_width(int ns, bool bk, Fn&& fn)
{
  // fn is called with two integral-constant tags: NS and BK.
  if (bk) {
    switch (ns) {
      case 1: fn(std::integral_constant<int, 1>{}, std::true_type{}); break;
      case 2: fn(std::integral_constant<int, 2>{}, std::true_type{}); break;
      case 4: fn(std::integral_constant<int, 4>{}, std::true_type{}); break;
      default: fn(std::integral_constant<int, 8>{}, std::true_type{}); break;
    }
  } else {
    switch (ns) {
      case 1: fn(std::integral_constant<int, 1>{}, std::false_type{}); break;
      case 2: fn(std::integral_constant<int, 2>{}, std::false_type{}); break;
      case 4: fn(std::integral_constant<int, 4>{}, std::false_type{}); break;
      default: fn(std::integral_constant<int, 8>{}, std::false_type{}); break;
    }
  }
}

// ---- seg_hot.cu ---------------------------------------------------------------------------
/// Empty slots: keys only (`lazy`, BUCKET) or keys and identities.
void hot_init(uint64_t* base,
              uint64_t slots,
              uint32_t NS,
              bool bk,
              bool lazy,
              identities const& id,
              cudaStream_t stream);
/// Direct fold of the raw rows [first, end_row): the one-row kernel (`strip` <= 1) or the strip
/// kernel. counters: see seg_hot.cu.
void hot_fold(seg_layout const& L,
              raw_cols const& F,
              uint64_t first,
              uint64_t end_row,
              uint32_t NS,
              bool bk,
              uint32_t strip,
              flat_table::fold_plan const& plan,
              uint32_t* bitmap,
              ull* counters,
              cudaStream_t stream);
/// Adds to *counter how many of 1024 evenly spaced rows of `key` (n >= 2048 rows) have the same
/// key as the row after them.
void hot_sample_runs(uint64_t const* key, uint64_t n, ull* counter, cudaStream_t stream);
/// Occupied slots of segments [p0, p0 + nseg) (`slots` slots each) to dense rows, in one launch
/// (at most `capacity` rows are written; *counter gets the number of occupied slots).
void hot_gather(seg_layout const& L,
                uint64_t p0,
                uint64_t nseg,
                uint64_t slots,
                uint32_t NS,
                bool bk,
                uint64_t* out,
                uint64_t capacity,
                ull* counter,
                cudaStream_t stream);

// ---- Hot keys (seg_table::options::hot_keys) ------------------------------------------------
// A few keys that each take a real share of the rows of a high-cardinality batch: every row of
// such a key is an atomic update of the same slot, issued together with updates of other slots.
// The set of a launch's hot keys is found on the device from a sample of its rows (seg_hot.cu:
// seg_hot_sample_kernel) and handed to the fold as `hot`: word 0 = number of hot keys (0: none),
// words 1 .. HOT_SLOTS = an open-addressing table of them (local_hash, linear probing, EMPTY_KEY
// = free; at most HOT_MAX keys, so a probe always ends).
constexpr uint32_t HOT_SLOTS  = 128;
constexpr uint32_t HOT_MAX    = 64;
constexpr uint32_t HOT_SAMPLE = 4096;  // rows sampled per launch
constexpr uint64_t HOT_WORDS  = 1 + HOT_SLOTS;
/// Samples HOT_SAMPLE rows of key[first .. first + n) (n >= HOT_SAMPLE) and writes the hot set to
/// `hot` (HOT_WORDS words): the keys sampled at least `thr` times - none when together they are
/// fewer than `cover` of the sampled rows. Queued on `stream`, nothing is read back. `n_out`,
/// when not null, also gets word 0 (so a caller can read it back with its own counters).
void hot_sample(uint64_t const* key,
                uint64_t first,
                uint64_t n,
                uint32_t thr,
                uint32_t cover,
                uint64_t* hot,
                unsigned long long* n_out,
                cudaStream_t stream);

// ---- seg_keys.cu --------------------------------------------------------------------------
/// Direct fold of the raw rows [first, end_row) with block-local accumulators for the keys of
/// `hot`: `block` threads of `rpt` rows each per block. counters as hot_fold. flags: bit 0 = the
/// accumulators' counts and integer sums as 32-bit shared atomics (options::pre_v2), bit 1 = a
/// thread keeps the first hot key it meets in registers (options::hot_own).
void keys_fold(seg_layout const& L,
               raw_cols const& F,
               uint64_t first,
               uint64_t end_row,
               uint32_t NS,
               bool bk,
               uint32_t block,
               uint32_t rpt,
               uint32_t flags,
               uint64_t const* hot,
               uint32_t* bitmap,
               ull* counters,
               cudaStream_t stream);

// ---- seg_gen.cu ---------------------------------------------------------------------------
/// Fold of the `n` raw rows of the generic source `S` (2 to 8 state words; word 0 counts the rows)
/// with keys that repeat: `rpt` rows per thread; `hot` null: runs of equal keys are combined
/// (`strided`: a thread takes every 256th row of its block instead of consecutive ones), else
/// block-local accumulators for the keys of `hot`. counters: [0] rows marked in `bitmap` (not
/// applied: the table is full), [2] groups inserted.
void gen_fold(seg_layout const& L,
              uint32_t NS,
              bool bk,
              flat_table::source const& S,
              identities const& id,
              uint64_t n,
              uint32_t rpt,
              uint32_t strided,
              uint64_t const* hot,
              uint32_t* bitmap,
              ull* counters,
              cudaStream_t stream);
}  // namespace segdev
}  // namespace sirius::agg_table::detail

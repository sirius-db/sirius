// Ported from experimental/agg_bench/src/seg_keys.cu (the fold kernel unchanged; the sealed pass
// is not part of the library).
// Hot keys (seg_table::options::hot_keys): the fold and the sealed pass for batches in which a
// few keys take a real share of the rows of a high-cardinality input. A translation unit of its
// own: a run without such batches never loads these kernels (see seg_dev.cuh).
//
// What they are for (GB10, 1 state word, a table of 2^14 keys that stays in the caches, fold only,
// M rows/s): the probe alone runs at 20,000 to 23,000 whatever the keys are; with the count's
// atomic add, uniform keys run at 11,800, and one key on 10 / 25 / 50 / 90% of the rows at 6,400 /
// 4,100 / 3,500 / 4,800 (4 keys on 50%: 6,600; 16: 9,600; 64 and more: 10,400). A plain load /
// add / store of the same word in place of the atomic is as fast as no update at all when the key
// has 99% of the rows, and when EVERY row has the same key the atomic is as fast too (24,900: long
// runs of equal keys were never the problem with narrow states). So it is neither the probe nor
// the layout nor the traffic on the slot's line: it is the atomic update of a slot that many, but
// not all, of the threads in flight address at once. With 8 state words the compare-and-swap loops
// of the f64 sum and the min / max words retry against each other on top of that (one key on 50%
// of the rows: 240 M rows/s against 790 for 1,024 keys on 50%).
//
// Here every block keeps an accumulator per hot key in shared memory (HOT_SLOTS x (state words +
// 2) words: 10 KiB with 8 state words). At block start each hot key is looked up in the table once
// (the fold inserts it: it is in the rows of this launch, it was sampled from them); a row of a
// hot key is then folded into the block's accumulator and the accumulators are added to the table
// once per block; every other row goes to the table directly, as in the direct kernel (one row at
// a time: no run combining, such input has no runs). A hot key the table cannot take (it is at its
// limit; sealed: it does not hold the key) is simply not hot in that block: its rows take the
// direct path, which marks / tags them. So nothing here can fail that the direct path cannot.
#include "seg_dev.cuh"

#include <stdexcept>

namespace sirius::agg_table::detail {
namespace segdev {

namespace {

constexpr int64_t NO_SLOT = INT64_MIN;

// Slot of `k` in the block's copy of the hot table, or HOT_SLOTS.
__device__ __forceinline__ uint32_t hot_find(uint64_t const* hkey, uint64_t k)
{
  uint32_t idx = local_hash(k) & (HOT_SLOTS - 1);
  for (;;) {
    uint64_t cur = hkey[idx];
    if (cur == k) return idx;
    if (cur == EMPTY_KEY) return HOT_SLOTS;
    idx = (idx + 1) & (HOT_SLOTS - 1);
  }
}

// One raw row into a block accumulator (shared memory): the v2 forms with 1 and 2 state words
// (native 32-bit atomics), the generic ones with more (see table_dev.cuh: sh_fold_raw).
template <int NS>
__device__ __forceinline__ void acc_fold(
  uint64_t* st, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint32_t v2)
{
  if (v2 != 0) {
    sh_fold_raw<NS>(st, a, b, c, d);
  } else {
    fold_raw<NS>(st, a, b, c, d);
  }
}
template <int NS>
__device__ __forceinline__ void acc_merge(uint64_t* st, uint64_t const* v, uint32_t v2)
{
  if (v2 != 0) {
    sh_fold_partial<NS>(st, v);
  } else {
    fold_partial<NS>(st, v);
  }
}

// A thread's rows of the first hot key it meets stay in registers (`own`: no atomics, not even
// shared-memory ones - with one key on a third of the rows, a third of a block's threads would
// otherwise meet on one accumulator at every step); its rows of other hot keys go to the block's
// accumulators.
template <int NS>
struct own_acc {
  uint32_t slot = HOT_SLOTS;
  uint64_t v[NS];
};
template <int NS>
__device__ __forceinline__ void hot_row(own_acc<NS>& own,
                                        uint64_t* hacc,
                                        uint32_t h,
                                        uint64_t a,
                                        uint64_t b,
                                        uint64_t c,
                                        uint64_t d,
                                        uint32_t v2,
                                        uint32_t use_own)
{
  if (use_own != 0) {
    if (own.slot == HOT_SLOTS) {
      own.slot = h;
      set_identity<NS>(own.v);
    }
    if (own.slot == h) {
      acc_raw<NS>(own.v, a, b, c, d);
      return;
    }
  }
  acc_fold<NS>(hacc + static_cast<uint64_t>(h) * NS, a, b, c, d, v2);
}

// SM (both kernels of this file): the instance that measures the int64 values it sums
// (AGG_SQL_SUM=fold, sql_dev.cuh), launched only when that mode is on. The instance without is
// the kernel as built with -DAGG_SQL_SUMFOLD_OFF=1, instruction for instruction. With the
// measurement in the one kernel (an accumulator across the row loop, whatever the mode) the
// fold took 4 to 6 registers more on sm_100, and with blocks of 256 threads that is a resident
// block per SM less: 48 -> 52 registers with 2 state words and 64 -> 70 with 8 in the bucket
// layout, 5 -> 4 and 4 -> 3 blocks; GB200 job 24426, mode off, a few hot keys on most rows: 0.81
// to 0.91 of the speed without it at 2 and 8 aggregates (README, section 3b).
//
// The measuring instance of the fold reads its rows' summed columns in a loop of its own - all
// the thread's rows, as the accumulator in the row loop did (a row the table cannot take is
// measured again by the kernel that applies it later) - so that nothing of it is live across the
// row loop. Where that loop stands moves the register count (sm_100, <2, row> <2, bucket>
// <4, row> <4, bucket> <8, row> <8, bucket>; without: 40 48 48 56 64 64; with the accumulator in
// the row loop: 44 52 52 62 64 70; a loop before the row loop: 40 48 48 60 62 69; after it:
// 40 52 54 56 56 64). Hence before the row loop up to 4 state words and after it with 8, which
// leaves all six with the resident blocks of the instance without (registers are granted in
// eights: 6 blocks up to 40, 5 up to 48, 4 up to 64, 3 up to 72). The sealed pass keeps its
// accumulator (it loads the values of the rows it applies only): its measuring instance has the
// registers of the other in all but <8, row> (48 -> 56).
template <int NS>
__device__ __forceinline__ void sum_thread_rows(
  raw_cols const& F, uint64_t base, uint64_t end_row, uint32_t rpt, uint32_t strips)
{
  sum_acc sacc;
  for (uint32_t r = 0; r < rpt; ++r) {
    uint64_t i = strips != 0 ? base + static_cast<uint64_t>(threadIdx.x) * rpt + r
                             : base + static_cast<uint64_t>(r) * blockDim.x + threadIdx.x;
    if (i >= end_row) break;
    sum_row<NS>(sacc, F.col[0][i], NS > 7 ? F.col[1][i] : 0);
  }
  sum_rows_done<NS>(sacc);
}
template <int NS>
constexpr bool SUM_ROWS_FIRST = NS < 8;

template <int NS, bool BK, bool SM>
__global__ void seg_fold_keys_kernel(seg_layout L,
                                     raw_cols F,
                                     uint64_t first,
                                     uint64_t end_row,
                                     uint32_t rpt,
                                     uint32_t v2,  // bit 0: v2 accumulator updates; bit 1: own_acc
                                     uint64_t const* hot,
                                     uint32_t* bitmap,
                                     ull* counters)
{
  __shared__ uint64_t hkey[HOT_SLOTS];
  __shared__ int64_t
    hoff[HOT_SLOTS];  // the key's state words, relative to L.ref (NO_SLOT: not hot here)
  __shared__ uint64_t hacc[HOT_SLOTS * NS];
  __shared__ unsigned int blk_new;
  uint32_t const T = blockDim.x;
  if (threadIdx.x == 0) blk_new = 0;
  unsigned int ins = 0;
  for (uint32_t s = threadIdx.x; s < HOT_SLOTS; s += T) {
    uint64_t k  = hot[1 + s];
    int64_t off = NO_SLOT;
    if (k != EMPTY_KEY) {
      bool inserted   = false;
      uint64_t* claim = nullptr;
      uint64_t* st    = seg_probe<BK>(L, NS, k, true, inserted, claim);
      ins += inserted ? 1u : 0u;
      if (st != nullptr) {
        if (claim != nullptr) {  // lazy state: the group starts at the identities
          set_identity<NS>(st);
          publish(claim, k);
        }
        off = st - L.ref;
      }
    }
    hkey[s] = k;
    hoff[s] = off;
    set_identity<NS>(hacc + static_cast<uint64_t>(s) * NS);
  }
  __syncthreads();

  own_acc<NS> own;
  uint64_t base = first + static_cast<uint64_t>(blockIdx.x) * T * rpt;
  reserved_acc rsv;
  if (SM && SUM_ROWS_FIRST<NS>) sum_thread_rows<NS>(F, base, end_row, rpt, v2 & 4u);
  for (uint32_t r = 0; r < rpt; ++r) {
    uint64_t i = (v2 & 4u) != 0 ? base + static_cast<uint64_t>(threadIdx.x) * rpt + r
                                : base + static_cast<uint64_t>(r) * T + threadIdx.x;
    if (i >= end_row) break;
    uint64_t k = F.key[i];
    uint64_t a = NS > 1 ? F.col[0][i] : 0;
    uint64_t b = NS > 2 ? F.col[1][i] : 0;
    uint64_t c = NS > 3 ? F.col[2][i] : 0;
    uint64_t d = NS > 4 ? F.col[3][i] : 0;
    if (reserved_row(k, a, b, c, d, rsv))
      continue;  // not a table key: the side groups (sql_dev.cuh)
    uint32_t h = hot_find(hkey, k);
    if (h != HOT_SLOTS && hoff[h] != NO_SLOT) {
      hot_row<NS>(own, hacc, h, a, b, c, d, v2 & 1u, v2 & 2u);
      continue;
    }
    bool inserted   = false;
    uint64_t* claim = nullptr;
    uint64_t* st    = seg_probe<BK>(L, NS, k, true, inserted, claim);
    ins += inserted ? 1u : 0u;
    if (st == nullptr) {
      atomicAdd(counters, 1ull);
      mark(bitmap, i);
    } else if (claim != nullptr) {
      init_raw<NS>(st, a, b, c, d);
      publish(claim, k);
    } else {
      fold_raw_ordered<NS>(st, a, b, c, d, L.order);
    }
  }
  reserved_rows_done(L.seen, rsv);
  if (SM && !SUM_ROWS_FIRST<NS>) sum_thread_rows<NS>(F, base, end_row, rpt, v2 & 4u);
  if (own.slot != HOT_SLOTS)
    acc_merge<NS>(hacc + static_cast<uint64_t>(own.slot) * NS, own.v, v2 & 1u);
  if (ins != 0) atomicAdd(&blk_new, ins);
  __syncthreads();

  for (uint32_t s = threadIdx.x; s < HOT_SLOTS; s += T) {
    int64_t off = hoff[s];
    // (The count is word 0 of every canonical set: 0 = no row of the key in this block.)
    if (off == NO_SLOT || hacc[static_cast<uint64_t>(s) * NS] == 0) continue;
    fold_partial<NS>(L.ref + off, hacc + static_cast<uint64_t>(s) * NS);
  }
  if (threadIdx.x == 0 && blk_new != 0) {
    atomicAdd(reinterpret_cast<ull*>(L.count), static_cast<ull>(blk_new));
    atomicAdd(counters + 2, static_cast<ull>(blk_new));
  }
}

uint32_t blocks_for(uint64_t rows, uint32_t block, uint32_t rpt)
{
  uint64_t per = static_cast<uint64_t>(block) * rpt;
  uint64_t g   = (rows + per - 1) / per;
  if (g > 0x7fffffffull) throw std::runtime_error("launch too large");
  return static_cast<uint32_t>(g);
}

}  // namespace

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
               cudaStream_t stream)
{
  if (end_row <= first) return;
  by_width(static_cast<int>(NS), bk, [&](auto ns, auto bkt) {
    constexpr int NSV  = decltype(ns)::value;
    constexpr bool BKV = decltype(bkt)::value;
    uint32_t const g   = blocks_for(end_row - first, block, rpt);
    // (The measuring instance SM = true of the prototype's AGG_SQL_SUM=fold is not instantiated.)
    seg_fold_keys_kernel<NSV, BKV, false>
      <<<g, block, 0, stream>>>(L, F, first, end_row, rpt, flags, hot, bitmap, counters);
  });
  check(cudaGetLastError(), "seg_fold_keys_kernel");
}

}  // namespace segdev
}  // namespace sirius::agg_table::detail

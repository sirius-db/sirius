// The segmented aggregation table, host side and the kernels that are not on the path of a
// pre-sized table. Ported from experimental/agg_bench/src/seg_table.cu; seg_table.hpp lists what
// changed. The kernels in this file (pre-aggregation, generic upsert, rehash) are the prototype's.
#include "seg_dev.cuh"
#include "seg_table.hpp"
#include "table_dev.cuh"

#include <cuda/stream>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sirius::agg_table::detail {

namespace {

using namespace tdev;
using namespace segdev;

// Low-cardinality fold (see agg_table's fold_preagg_kernel): every block aggregates its rows in
// a table in shared memory and merges that into the table. A row whose key finds no local slot
// is marked in the bitmap for the host; a block-local group that finds the table full is written
// as a partial state to `partials` (capacity: one row per local and private slot of every block).
constexpr uint32_t SEG_LOCAL_PROBES = 16;

// A partial state into the table; one that finds it full is appended to `partials`.
template <int NS, bool BK>
__device__ __forceinline__ void merge_to_table(seg_layout const& L,
                                               uint64_t k,
                                               uint64_t const* ls,
                                               uint64_t* partials,
                                               ull* counters,
                                               unsigned int* blk_new)
{
  bool inserted   = false;
  uint64_t* claim = nullptr;
  uint64_t* st    = seg_probe<BK>(L, NS, k, true, inserted, claim);
  if (inserted) atomicAdd(blk_new, 1u);
  if (st == nullptr) {
    uint64_t* o = partials + atomicAdd(counters + 1, 1ull) * (NS + 1);
    o[0]        = k;
    for (int j = 0; j < NS; ++j)
      o[1 + j] = ls[j];
  } else if (claim != nullptr) {
    init_partial<NS>(st, ls);
    publish(claim, k);
  } else {
    fold_partial<NS>(st, ls);
  }
}

// PV > 0 (a handful of groups): every thread first aggregates its rows in PV private slots held
// in registers (no memory traffic, no atomics) and adds them to the block's table once, at the
// end of its rows. Without it, all threads of a block update the same few shared words for every
// row and stall each other.
template <int NS, bool BK, int PV>
__global__ void seg_fold_preagg_kernel(seg_layout L,
                                       raw_cols F,
                                       uint64_t first,
                                       uint64_t end_row,
                                       uint32_t local_slots,
                                       uint32_t rows_per_thread,
                                       uint32_t v2,
                                       uint32_t* bitmap,
                                       uint64_t* partials,
                                       ull* counters)
{
  // v2 (options::pre_v2): the block table's counts and integer sums as native 32-bit shared
  // atomics (table_dev.cuh: sh_fold_raw) instead of 64-bit compare-and-swap loops.
  extern __shared__ uint64_t sh[];
  __shared__ unsigned int blk_new;
  __shared__ unsigned int blk_fail;
  uint64_t* lkey      = sh;
  uint64_t* lstate    = sh + local_slots;
  uint32_t const mask = local_slots - 1;
  uint32_t const T    = blockDim.x;
  if (threadIdx.x == 0) {
    blk_new  = 0;
    blk_fail = 0;
  }
  for (uint32_t s = threadIdx.x; s < local_slots; s += T) {
    lkey[s] = EMPTY_KEY;
    set_identity<NS>(lstate + static_cast<uint64_t>(s) * NS);
  }
  __syncthreads();

  // Private slots: only ever indexed by constants (unrolled loops), so they live in registers.
  uint64_t pkey[PV > 0 ? PV : 1];
  uint64_t pstate[PV > 0 ? PV : 1][NS];
#pragma unroll
  for (int j = 0; j < PV; ++j)
    pkey[j] = EMPTY_KEY;

  uint64_t base = first + static_cast<uint64_t>(blockIdx.x) * T * rows_per_thread;
  reserved_acc rsv;
  sum_acc sacc;
  for (uint32_t r = 0; r < rows_per_thread; ++r) {
    uint64_t i = base + static_cast<uint64_t>(r) * T + threadIdx.x;
    if (i >= end_row) break;
    uint64_t k = F.key[i];
    uint64_t a = NS > 1 ? F.col[0][i] : 0;
    uint64_t b = NS > 2 ? F.col[1][i] : 0;
    uint64_t c = NS > 3 ? F.col[2][i] : 0;
    uint64_t d = NS > 4 ? F.col[3][i] : 0;
    sum_row<NS>(sacc, a, b);
    if (reserved_row(k, a, b, c, d, rsv))
      continue;  // not a table key: the side groups (sql_dev.cuh)
    if (PV > 0) {
      // The first slot that holds the key or is free takes the row (slots fill in order).
      bool done = false;
#pragma unroll
      for (int j = 0; j < PV; ++j) {
        if (!done && (pkey[j] == k || pkey[j] == EMPTY_KEY)) {
          if (pkey[j] == EMPTY_KEY) {
            pkey[j] = k;
            set_identity<NS>(pstate[j]);
          }
          acc_raw<NS>(pstate[j], a, b, c, d);
          done = true;
        }
      }
      if (done) continue;
    }
    uint32_t idx = static_cast<uint32_t>(fmix64(k)) & mask;
    uint64_t* st = nullptr;
    for (uint32_t p = 0; p < SEG_LOCAL_PROBES; ++p) {
      uint64_t cur = lkey[idx];
      if (cur == EMPTY_KEY) {
        cur = atomicCAS(reinterpret_cast<ull*>(lkey + idx), EMPTY_KEY, k);
        if (cur == EMPTY_KEY) cur = k;
      }
      if (cur == k) {
        st = lstate + static_cast<uint64_t>(idx) * NS;
        break;
      }
      idx = (idx + 1) & mask;
    }
    if (st == nullptr) {
      // No local slot (more groups than expected): the row is left to the host, which folds the
      // marked rows afterwards. Keeping the table out of this loop keeps every update in it a
      // shared-memory one.
      mark(bitmap, i);
      atomicAdd(&blk_fail, 1u);
      continue;
    }
    if (v2 != 0) {
      sh_fold_raw<NS>(st, a, b, c, d);
    } else {
      fold_raw<NS>(st, a, b, c, d);
    }
  }
  reserved_rows_done(L.seen, rsv);
  sum_rows_done<NS>(sacc);
#pragma unroll
  for (int j = 0; j < PV; ++j) {  // private slots -> block table (or the table, if it has no room)
    uint64_t k = pkey[j];
    if (k == EMPTY_KEY) continue;
    uint32_t idx = static_cast<uint32_t>(fmix64(k)) & mask;
    uint64_t* st = nullptr;
    for (uint32_t p = 0; p < SEG_LOCAL_PROBES; ++p) {
      uint64_t cur = lkey[idx];
      if (cur == EMPTY_KEY) {
        cur = atomicCAS(reinterpret_cast<ull*>(lkey + idx), EMPTY_KEY, k);
        if (cur == EMPTY_KEY) cur = k;
      }
      if (cur == k) {
        st = lstate + static_cast<uint64_t>(idx) * NS;
        break;
      }
      idx = (idx + 1) & mask;
    }
    if (st != nullptr) {
      if (v2 != 0) {
        sh_fold_partial<NS>(st, pstate[j]);
      } else {
        fold_partial<NS>(st, pstate[j]);
      }
    } else {
      merge_to_table<NS, BK>(L, k, pstate[j], partials, counters, &blk_new);
    }
  }
  __syncthreads();

  for (uint32_t s = threadIdx.x; s < local_slots; s += T) {
    uint64_t k = lkey[s];
    if (k == EMPTY_KEY) continue;
    merge_to_table<NS, BK>(
      L, k, lstate + static_cast<uint64_t>(s) * NS, partials, counters, &blk_new);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (blk_new != 0) {
      atomicAdd(reinterpret_cast<ull*>(L.count), static_cast<ull>(blk_new));
      atomicAdd(counters + 2, static_cast<ull>(blk_new));
    }
    if (blk_fail != 0) atomicAdd(counters + 3, static_cast<ull>(blk_fail));
  }
}

// Generic fold of any source (raw rows or partial states) with a row bitmap; see seg_table::upsert.
template <bool BK>
__global__ void seg_upsert_kernel(seg_layout L,
                                  uint32_t NS,
                                  flat_table::source S,
                                  uint64_t n,
                                  uint32_t* bitmap,
                                  bool only_marked,
                                  bool may_insert,
                                  ull* counters)
{
  uint64_t i    = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool inserted = false;
  if (i < n) {
    uint32_t* w  = bitmap + (i >> 5);
    uint32_t bit = 1u << (i & 31);
    // (A raw row with a reserved key is left alone: the operator's side groups, sql.cu.)
    if ((!only_marked || (*w & bit) != 0) && !reserved_key(L.seen, S.key.p[i * S.key.stride])) {
      uint64_t k      = S.key.p[i * S.key.stride];
      uint64_t* claim = nullptr;
      uint64_t* st    = seg_probe<BK>(L, NS, k, may_insert, inserted, claim);
      if (st == nullptr) {
        atomicAdd(counters, 1ull);
        if (!only_marked) atomicOr(w, bit);
      } else {
        sum_acc sacc;
        if (claim != nullptr) {  // a new group: its first state is this row
          for (int s = 0; s < S.num_state; ++s) {
            uint64_t v = S.op[s] == kop::ADD_ONE ? 1 : S.val[s].p[i * S.val[s].stride];
            sum_src(sacc, S, s, v);
            st[s] = v;
          }
          publish(claim, k);
        } else {
          for (int s = 0; s < S.num_state; ++s) {
            uint64_t v = S.op[s] == kop::ADD_ONE ? 0 : S.val[s].p[i * S.val[s].stride];
            sum_src(sacc, S, s, v);
            apply(st + s, S.op[s], v);
          }
        }
        sum_done(sacc);
        if (only_marked) atomicAnd(w, ~bit);
      }
    }
  }
  block_count(inserted, reinterpret_cast<ull*>(L.count), counters + 2);
}

// Growth: every group of the old segment (layout BO) into the initialised, larger new one (BN).
// Slot order is hash order in both, so reads and writes are both nearly sequential.
template <bool BO, bool BN>
__global__ void seg_rehash_kernel(uint64_t* old_base,
                                  uint64_t old_cap,
                                  uint64_t* new_base,
                                  uint64_t* const* new_segs,
                                  uint64_t new_cap,
                                  uint32_t NS,
                                  uint32_t sbits,
                                  uint32_t skip_bits,
                                  ull* moved)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool ok    = false;
  if (i < old_cap) {
    uint64_t k = *key_ptr<BO>(old_base, i, NS);
    if (k != EMPTY_KEY) {
      hpos q = locate(k, sbits, skip_bits);
      uint64_t* dst =
        seg_insert_new<BN>(new_base != nullptr ? new_base : new_segs[q.p], new_cap, NS, q, k);
      if (dst != nullptr) {
        uint64_t const* src = state_ptr<BO>(old_base, i, NS);
        for (uint32_t j = 0; j < NS; ++j)
          dst[j] = src[j];
        ok = true;
      }
    }
  }
  block_count(ok, moved);
}

constexpr uint64_t PROBE_ROWS = 1ull << 16;  // first fold: rows folded before the plan is chosen
// options::gen_combine (see seg_table::fold): batches below GEN_MIN_ROWS are not sampled; RUNS
// when GEN_RUN_MIN of the 1,024 sampled rows are followed by a row with the same key (the rule of
// options::run_preagg).
constexpr uint64_t GEN_MIN_ROWS    = 1ull << 16;
constexpr uint64_t GEN_RUN_MIN     = 1022;
constexpr uint64_t MIN_TOTAL_SLOTS = 1ull << 18;
// A table up to this size is PACKED: all segments in one allocation, initialised, grown and
// emitted with one launch each (a small table must not pay per-segment launches).
constexpr uint64_t PACK_BYTES = 256ull << 20;

uint64_t round8(uint64_t v) { return (v + 7) & ~7ull; }

double wall()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void zero(void* p, std::size_t bytes, ::cuda::stream_ref stream)
{
  if (bytes != 0) check(cudaMemsetAsync(p, 0, bytes, stream.get()), "memset");
}

constexpr int PRIVATE_SLOTS = 4;  // per-thread slots of the pre-aggregating kernel (few groups)

template <int NS, bool BK>
void launch_preagg(seg_layout const& L,
                   raw_cols const& F,
                   uint64_t first,
                   uint64_t end_row,
                   uint32_t blocks,
                   uint32_t block,
                   uint32_t local_slots,
                   uint32_t rpt,
                   bool private_slots,
                   uint32_t v2,
                   uint32_t* bitmap,
                   uint64_t* partials,
                   ull* counters,
                   cudaStream_t stream,
                   uint64_t local_budget)
{
  std::size_t shmem = static_cast<std::size_t>(local_slots) * (1 + NS) * sizeof(uint64_t);
  if (private_slots) {  // only chosen with a small local table (default shared-memory limit)
    seg_fold_preagg_kernel<NS, BK, PRIVATE_SLOTS><<<blocks, block, shmem, stream>>>(
      L, F, first, end_row, local_slots, rpt, v2, bitmap, partials, counters);
    return;
  }
  if (shmem + 1024 > 48 * 1024) {
    // The attribute belongs to the kernel, not to the launch: it is always set to the device's
    // budget, the same value from every thread. (The prototype set it to this launch's size; two
    // folds in flight with block tables of different sizes - the plan follows the group count,
    // which moves while a table grows - could then lower it between another thread's set and its
    // launch, which fails with "too many resources requested for launch".)
    check(cudaFuncSetAttribute(seg_fold_preagg_kernel<NS, BK, 0>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(local_budget > shmem ? local_budget : shmem)),
          "shared memory size");
  }
  seg_fold_preagg_kernel<NS, BK, 0><<<blocks, block, shmem, stream>>>(
    L, F, first, end_row, local_slots, rpt, v2, bitmap, partials, counters);
}

uint64_t local_budget_uncached()
{
  int dev = 0, option = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&option, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
        cudaSuccess) {
    option = 48 * 1024;
  }
  uint64_t most = option > 2048 ? static_cast<uint64_t>(option) - 1024 : 0;
  return most < 200 * 1024 ? most : 200 * 1024;
}

// A growth step could not allocate its new segments (the table is as it was).
struct grow_failed {};

}  // namespace

seg_table::scratch::scratch(::cuda::stream_ref stream, rmm::device_async_resource_ref mr_)
  : mr(mr_),
    bitmap(0, stream, mr_),
    partials(0, stream, mr_),
    partial_bitmap(0, stream, mr_),
    counters(5, stream, mr_),  // (4 of a fold launch, and the hot-key count of its sample)
    hot(HOT_WORDS, stream, mr_)
{
}

seg_table::seg_table(state_spec const& spec,
                     options const& opt,
                     ::cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
  : _spec(spec),
    _opt(opt),
    _mr(mr),
    _W(spec.row_words()),
    _NS(static_cast<uint32_t>(spec.num_state)),
    _local_budget(local_budget_uncached()),
    _ptrs(1ull << (opt.sbits <= MAX_SBITS ? opt.sbits : 0), stream, mr),
    _count(2, stream, mr),
    _full(1ull << (opt.sbits <= MAX_SBITS ? opt.sbits : 0), stream, mr)
{
  if (opt.seen == nullptr) throw std::runtime_error("seg_table: options::seen is not set");
  init_shape();
  uint64_t minc = min_cap();
  uint64_t cap  = opt.initial_groups != 0 ? cap_for_groups(opt.initial_groups) : minc;
  if (cap < minc) cap = minc;
  uint64_t mc = max_cap();
  if (mc < minc) throw std::runtime_error("seg_table: the budget is below the minimum table");
  if (cap > mc) cap = mc;
  _cap = cap;
  allocate(stream);
  zero(_count.data(), 2 * sizeof(uint64_t), stream);
  upload_pointers(stream);
}

seg_table::seg_table(state_spec const& spec, options const& opt, shape_only)
  : _spec(spec),
    _opt(opt),
    _mr(rmm::mr::get_current_device_resource_ref()),
    _W(spec.row_words()),
    _NS(static_cast<uint32_t>(spec.num_state)),
    _ptrs(0, ::cuda::stream_ref{cudaStream_t{nullptr}}, _mr),
    _count(0, ::cuda::stream_ref{cudaStream_t{nullptr}}, _mr),
    _full(0, ::cuda::stream_ref{cudaStream_t{nullptr}}, _mr)
{
  init_shape();
}

// Option checks and what follows from the options alone: the load factor, the kernel family.
void seg_table::init_shape()
{
  auto const& opt = _opt;
  if (opt.sbits > MAX_SBITS || opt.sbits + opt.skip_bits > 40) {
    throw std::runtime_error("seg_table: too many hash bits");
  }
  if (!(opt.growth > 1.0)) throw std::runtime_error("seg_table: growth must be > 1");
  if (opt.run_cap == 0 || (opt.run_cap & (opt.run_cap - 1)) != 0) {
    throw std::runtime_error("seg_table: run_cap must be a power of two");
  }
  bool big_bk = opt.layout != 0;  // layout of a table past `small_bytes`
  // Wider rows tolerate denser tables (measured on a GB200, 2^26..2^30 groups): 8 state words
  // lose 2% from load 0.5 to 0.85, one or two lose 2% up to 0.75 and 7-10% at 0.85.
  _load = opt.load > 0 ? opt.load : (!big_bk ? 0.75 : _NS >= 8 ? 0.85 : _NS >= 4 ? 0.8 : 0.75);
  if (_load > (big_bk ? 0.97 : 0.85)) _load = big_bk ? 0.97 : 0.85;  // see PMAX_*
  _canon = _W == 2 || _W == 3 || _W == 5 || _W == 9;
  for (int j = 0; _canon && j < _spec.num_state; ++j) {
    _canon = raw_op(_spec.state[j].op) == CANON_OP[j] && _spec.state[j].col == CANON_COL[j];
  }
}

// Slots per segment of the smallest table.
uint64_t seg_table::min_cap() const
{
  uint64_t S = num_segments();
  return round8(MIN_TOTAL_SLOTS / S < 64 ? 64 : MIN_TOTAL_SLOTS / S);
}

seg_table::sizing seg_table::presize(state_spec const& spec, options const& opt)
{
  seg_table t(spec, opt, shape_only{});
  uint64_t S    = t.num_segments();
  uint64_t per  = (S + 1) * t._W * sizeof(uint64_t);  // budget bytes per slot of segment capacity
  uint64_t minc = t.min_cap();
  uint64_t want = opt.initial_groups != 0 ? t.cap_for_groups(opt.initial_groups) : minc;
  if (want < minc) want = minc;
  uint64_t mc  = t.max_cap();
  uint64_t cap = want < mc ? want : mc;
  sizing s{};
  s.need_bytes = want * per;
  s.min_bytes  = minc * per;
  s.slots      = cap * S;
  s.limit      = t.limit_for_cap(cap);
  s.max_limit  = opt.budget_bytes == 0 ? ~0ull : t.limit_for_cap(mc);
  s.bucket     = t.layout_for(cap);
  return s;
}

// Layout by size (layout < 0): ROW while the table is small (cache-resident: one line per group
// beats two), BUCKET once it is not (dense, aligned state, lazy state).
bool seg_table::layout_for(uint64_t cap) const
{
  if (_opt.layout >= 0) return _opt.layout != 0;
  return cap * num_segments() * _W * sizeof(uint64_t) > _opt.small_bytes;
}

bool seg_table::can_pack(uint64_t cap) const
{
  uint64_t total = cap * num_segments() * _W * sizeof(uint64_t);
  return total <= PACK_BYTES && (_opt.budget_bytes == 0 || _opt.budget_bytes >= 4 * PACK_BYTES);
}

// Allocates and initialises empty segments of `_cap` slots.
void seg_table::allocate(::cuda::stream_ref stream)
{
  uint64_t S = num_segments();
  auto id    = make_identities(_spec);
  _packed    = can_pack(_cap);
  _bk        = layout_for(_cap);
  _segs.clear();
  _emitted.assign(S, false);
  zero(_full.data(), _full.size() * sizeof(uint32_t), stream);
  auto init = [&](uint64_t* base, uint64_t slots) {
    hot_init(base, slots, _NS, _bk, lazy(), id, stream.get());
  };
  double t0 = _opt.profile ? wall() : 0.0;
  if (_packed) {
    _segs.emplace_back(S * _cap * _W, stream, _mr);
    init(_segs[0].data(), S * _cap);
  } else {
    for (uint64_t p = 0; p < S; ++p) {
      _segs.emplace_back(_cap * _W, stream, _mr);
      init(_segs[p].data(), _cap);
    }
  }
  if (_opt.profile) check(cudaStreamSynchronize(stream.get()), "sync");
  std::lock_guard<std::mutex> lock(_m);
  if (_stats.first_slots == 0) _stats.first_slots = S * _cap;
  _stats.init_bytes += S * _cap * _W * sizeof(uint64_t);
  if (_opt.profile) _stats.init_seconds += wall() - t0;
}

// Segment capacity that holds `per_seg` groups per segment under the load policy.
uint64_t seg_table::cap_for(double per_seg) const
{
  uint64_t seg_bytes = num_segments() * _W * sizeof(uint64_t);  // table bytes per slot of capacity
  uint64_t prev_edge = 0;  // largest capacity of the previous tier
  for (int t = 0; t < TIERS; ++t) {
    uint64_t top = tier_bytes(t);
    if (top == 0) break;
    double load = _load < TIER_LOAD[t] ? _load : TIER_LOAD[t];
    uint64_t c  = round8(static_cast<uint64_t>(std::ceil(per_seg / load)));
    if (c <= prev_edge) c = prev_edge + 8;  // stay in this tier (load_for_cap goes by size)
    if (c * seg_bytes <= top) return c;
    prev_edge = top / seg_bytes & ~7ull;
  }
  uint64_t c = round8(static_cast<uint64_t>(std::ceil(per_seg / _load)));
  return c > prev_edge ? c : prev_edge + 8;
}

uint64_t seg_table::cap_for_groups(uint64_t groups) const
{
  // 5% headroom, plus four standard deviations of a segment's share of the groups (the limit is
  // on the whole table, so a segment may run slightly above the load factor).
  double g = static_cast<double>(groups) / static_cast<double>(num_segments());
  return cap_for(g * 1.05 + 4.0 * std::sqrt(g) + 8.0);
}

uint64_t seg_table::max_cap() const
{
  if (_opt.budget_bytes == 0) return ~0ull >> 8;
  // S segments plus one more: the new segment during growth, the dense chunk during emit.
  uint64_t mc = (_opt.budget_bytes / ((num_segments() + 1) * _W * sizeof(uint64_t))) & ~7ull;
  if (_packed && !_segs.empty()) {
    // Growing out of a packed table keeps the whole old table until the new one is complete.
    uint64_t old = _segs[0].size() * sizeof(uint64_t);
    uint64_t m2  = ((_opt.budget_bytes - old) / (num_segments() * _W * sizeof(uint64_t))) & ~7ull;
    if (m2 < mc) mc = m2;
  }
  return mc;
}

uint64_t seg_table::max_groups(bool dense) const
{
  if (_opt.budget_bytes == 0) return ~0ull;
  uint64_t mc = max_cap();
  if (dense && _opt.dense_at_budget && _opt.small_bytes != 0 && _opt.layout != 0) {
    return static_cast<uint64_t>(std::floor(static_cast<double>(mc * num_segments()) * _load));
  }
  return limit_for_cap(mc);
}

uint64_t seg_table::bytes() const
{
  std::shared_lock<std::shared_mutex> lk(_rw);
  return bytes_unlocked();
}

uint64_t seg_table::bytes_unlocked() const
{
  uint64_t b = (_ptrs.size() + _count.size()) * sizeof(uint64_t) + _full.size() * sizeof(uint32_t);
  for (auto const& s : _segs)
    b += s.size() * sizeof(uint64_t);
  return b;
}

// Load factor by table size. Tables that fit, or nearly fit, the GPU's caches are bound by probe
// length, not by memory traffic, so they run sparse; measured on a GB200 (2^18 to 2^26 groups,
// 1 to 8 state words, ROW layout at 0.25 / 0.4 / 0.5 / 0.6 / 0.75 against BUCKET at 0.5 / 0.75 /
// 0.85): up to 32 MiB the fastest load is 0.25 to 0.4, up to ~190 MiB 0.4 to 0.5, up to ~450 MiB
// 0.6 (ROW), beyond that BUCKET at the configured load. (Round 2 had a single tier: 0.25 up to
// 256 MiB, which at 2^20 to 2^22 groups made the table 2-4 times larger than the caches hold and
// cost 7-25% against round 1's 0.5.)
double seg_table::load_for_cap(uint64_t cap) const
{
  uint64_t bytes = cap * num_segments() * _W * sizeof(uint64_t);
  for (int t = 0; t < TIERS; ++t) {
    if (bytes <= tier_bytes(t)) return _load < TIER_LOAD[t] ? _load : TIER_LOAD[t];
  }
  return _load;
}

// Largest table (bytes) of size tier t; 0 when the tiers are switched off (small_bytes == 0).
uint64_t seg_table::tier_bytes(int t) const
{
  return t == 0 ? _opt.small_bytes / 10 : t == 1 ? _opt.small_bytes / 10 * 6 : _opt.small_bytes;
}

uint64_t seg_table::limit_for_cap(uint64_t cap) const
{
  return static_cast<uint64_t>(
    std::floor(static_cast<double>(cap * num_segments()) * load_for_cap(cap)));
}

uint64_t seg_table::group_limit() const { return limit_for_cap(_cap); }

seg_table::stats_t seg_table::stats() const
{
  std::lock_guard<std::mutex> lock(_m);
  return _stats;
}

void seg_table::upload_pointers(::cuda::stream_ref stream)
{
  std::vector<uint64_t*> host(num_segments());
  for (std::size_t p = 0; p < host.size(); ++p) {
    host[p] = _packed ? _segs[0].data() + p * _cap * _W : _segs[p].data();
  }
  check(cudaMemcpyAsync(_ptrs.data(),
                        host.data(),
                        host.size() * sizeof(uint64_t*),
                        cudaMemcpyHostToDevice,
                        stream.get()),
        "upload segment pointers");
  check(cudaStreamSynchronize(stream.get()), "sync");
}

void seg_table::note_count(uint64_t count)
{
  uint64_t cur = _total.load();
  while (count > cur && !_total.compare_exchange_weak(cur, count)) {}
}

flat_table::source seg_table::row_source(uint64_t const* rows) const
{
  flat_table::source s{};
  s.key       = {rows, _W};
  s.num_state = _spec.num_state;
  for (int j = 0; j < _spec.num_state; ++j) {
    s.op[j]  = merge_op(_spec.state[j].op);
    s.val[j] = {rows + 1 + j, _W};
  }
  return s;
}

// The table as the fold kernels see it (called with the shared lock held).
void seg_table::fill_layout(void* out) const
{
  auto& L       = *static_cast<seg_layout*>(out);
  uint64_t* ref = const_cast<uint64_t*>(_segs[0].data());
  L.ref         = ref;
  for (uint64_t p = 0; p < num_segments(); ++p) {
    uint64_t const* base = _packed ? ref + p * _cap * _W : _segs[p].data();
    L.seg_off[p]         = (reinterpret_cast<int64_t>(base) - reinterpret_cast<int64_t>(ref)) / 8;
  }
  L.cap        = _cap;
  L.count      = const_cast<uint64_t*>(_count.data());
  L.limit      = group_limit();
  L.full       = const_cast<uint32_t*>(_full.data());
  L.sbits      = _opt.sbits;
  L.skip_bits  = _opt.skip_bits;
  L.lazy       = lazy();
  L.fp         = nullptr;
  L.seen       = _opt.seen;
  uint32_t ord = _opt.order >= 0 ? static_cast<uint32_t>(_opt.order)
                                 : (_runs.load(std::memory_order_relaxed) <= 0.1f ? 1u : 0u);
  L.order      = _NS >= 4 ? ord : 0;
}

// One growth step, or nothing: false when the memory resource could not provide the new
// segments (the table is then exactly as it was).
bool seg_table::try_grow_to(uint64_t new_cap, ::cuda::stream_ref stream)
{
  try {
    grow_to(new_cap, stream);
    return true;
  } catch (grow_failed const&) {
    return false;
  }
}

// One growth step: every segment is rehashed into a segment of `new_cap` slots. Called with the
// exclusive lock held. At any moment a few extra segments exist (a packed table: old and new).
//
// Allocation failures (changed against the prototype, which let them escape with the table half
// moved): every allocation of a step is caught. A packed table allocates all of the new table
// before it touches the old one, so a failure leaves it unchanged. A table of separate segments
// replaces them group by group; a failure after the first group is undone by rehashing the
// segments already grown back into segments of the old capacity (each smaller than the block that
// was just freed for it). Either way grow_failed is thrown with the table as it was. Only if
// that undo cannot allocate either is the table left unusable (`_broken`, a std::runtime_error):
// with a budget the engine has reserved, neither can happen.
void seg_table::grow_to(uint64_t new_cap, ::cuda::stream_ref stream)
{
  double t0   = wall();
  uint64_t S  = num_segments();
  auto id     = make_identities(_spec);
  auto* moved = reinterpret_cast<ull*>(_count.data() + 1);
  zero(moved, sizeof(uint64_t), stream);
  uint64_t old_bytes = bytes_unlocked();
  uint64_t old_cap   = _cap;
  bool const obk     = _bk;  // layout of the old segments
  bool const nbk =
    layout_for(new_cap);  // and of the new ones (a table that stops being small changes)
  bool const old_lazy = lazy();
  _bk                 = nbk;
  bool const new_lazy = lazy();
  _bk                 = obk;
  double t_init = 0, t_rehash = 0;
  uint64_t launches = 0, init_b = 0;
  auto init = [&](uint64_t* base, uint64_t slots, bool bk, bool lz) {
    double ti = _opt.profile ? wall() : 0.0;
    hot_init(base, slots, _NS, bk, lz, id, stream.get());
    launches += 1;
    init_b += slots * _W * sizeof(uint64_t);
    if (_opt.profile) {
      check(cudaStreamSynchronize(stream.get()), "sync");
      t_init += wall() - ti;
    }
  };
  // Rehash `from_slots` slots at `from` (layout fbk) into `to` (segments of `to_cap` slots, layout
  // tbk; null: the segments `_ptrs` names).
  auto rehash = [&](uint64_t* from,
                    uint64_t from_slots,
                    bool fbk,
                    uint64_t* to,
                    uint64_t to_cap,
                    bool tbk,
                    ull* cnt) {
    double tr = _opt.profile ? wall() : 0.0;
    launches += 1;
    auto g  = grid(from_slots);
    auto st = stream.get();
    if (fbk && tbk) {
      seg_rehash_kernel<true, true><<<g, BLOCK, 0, st>>>(
        from, from_slots, to, _ptrs.data(), to_cap, _NS, _opt.sbits, _opt.skip_bits, cnt);
    } else if (!fbk && tbk) {
      seg_rehash_kernel<false, true><<<g, BLOCK, 0, st>>>(
        from, from_slots, to, _ptrs.data(), to_cap, _NS, _opt.sbits, _opt.skip_bits, cnt);
    } else if (fbk && !tbk) {
      seg_rehash_kernel<true, false><<<g, BLOCK, 0, st>>>(
        from, from_slots, to, _ptrs.data(), to_cap, _NS, _opt.sbits, _opt.skip_bits, cnt);
    } else {
      seg_rehash_kernel<false, false><<<g, BLOCK, 0, st>>>(
        from, from_slots, to, _ptrs.data(), to_cap, _NS, _opt.sbits, _opt.skip_bits, cnt);
    }
    check(cudaGetLastError(), "seg_rehash_kernel");
    if (_opt.profile) {
      check(cudaStreamSynchronize(stream.get()), "sync");
      t_rehash += wall() - tr;
    }
  };
  auto const sync = [&] { check(cudaStreamSynchronize(stream.get()), "sync"); };
  auto alloc      = [&](uint64_t words) {
    try {
      return rmm::device_uvector<uint64_t>(words, stream, _mr);
    } catch (std::bad_alloc const&) {
      throw grow_failed{};
    }
  };
  if (can_pack(new_cap)) {  // packed -> packed: one buffer, one launch
    rmm::device_uvector<uint64_t> nw  = alloc(S * new_cap * _W);  // (may fail: nothing changed yet)
    rmm::device_uvector<uint64_t> old = std::move(_segs[0]);
    _segs[0]                          = std::move(nw);
    _cap                              = new_cap;
    _bk                               = nbk;
    zero(_full.data(), _full.size() * sizeof(uint32_t), stream);
    init(_segs[0].data(), S * new_cap, nbk, new_lazy);
    upload_pointers(stream);
    rehash(old.data(), S * old_cap, obk, nullptr, new_cap, nbk, moved);
    sync();              // before the old memory is freed
  } else if (_packed) {  // packed -> segments: the (small) old table is freed at the end
    std::vector<rmm::device_uvector<uint64_t>> nw;
    nw.reserve(S);
    for (uint64_t p = 0; p < S; ++p)
      nw.push_back(alloc(new_cap * _W));  // (may fail: nothing changed yet)
    rmm::device_uvector<uint64_t> old = std::move(_segs[0]);
    _segs                             = std::move(nw);
    zero(_full.data(), _full.size() * sizeof(uint32_t), stream);
    for (uint64_t p = 0; p < S; ++p) {
      init(_segs[p].data(), new_cap, nbk, new_lazy);
      rehash(old.data() + p * old_cap * _W, old_cap, obk, _segs[p].data(), new_cap, nbk, moved);
    }
    sync();
    _packed = false;
    _cap    = new_cap;
    _bk     = nbk;
    upload_pointers(stream);
  } else {
    // `g` old segments at a time (options::grow_segs): their new segments are allocated and
    // filled, the stream is synchronised once, the old ones are freed.
    uint64_t g = _opt.grow_segs > 1 ? _opt.grow_segs : 1;
    if (g > 1) {
      uint64_t by_bytes = 2 * new_cap / old_cap;  // at most two new segments' bytes of old ones
      // options::grow_small_bytes: a small table's old segments may all stay until the step is
      // complete (as a packed table's do), so the step is one synchronisation.
      if (S * old_cap * _W * sizeof(uint64_t) <= _opt.grow_small_bytes) {
        by_bytes = S;
        g        = S;
      }
      if (g > by_bytes) g = by_bytes;
      if (_opt.budget_bytes != 0) {
        uint64_t full = S * new_cap * _W * sizeof(uint64_t);
        uint64_t room = _opt.budget_bytes > full
                          ? (_opt.budget_bytes - full) / (old_cap * _W * sizeof(uint64_t))
                          : 0;
        if (g > room) g = room;
      }
      if (g < 1) g = 1;
    }
    std::vector<rmm::device_uvector<uint64_t>> nw;
    uint64_t done = 0;  // segments [0, done) are at the new capacity
    try {
      for (uint64_t p = 0; p < S; p += g) {
        uint64_t e = p + g < S ? p + g : S;
        nw.clear();
        for (uint64_t q = p; q < e; ++q)
          nw.push_back(alloc(new_cap * _W));
        for (uint64_t q = p; q < e; ++q) {
          init(nw[q - p].data(), new_cap, nbk, new_lazy);
          rehash(_segs[q].data(), old_cap, obk, nw[q - p].data(), new_cap, nbk, moved);
        }
        sync();
        for (uint64_t q = p; q < e; ++q)
          _segs[q] = std::move(nw[q - p]);  // frees the old segments
        done = e;
      }
    } catch (grow_failed const&) {
      nw.clear();
      // Undo: the segments already grown go back to the old capacity, one at a time.
      for (uint64_t q = 0; q < done; ++q) {
        try {
          rmm::device_uvector<uint64_t> back(old_cap * _W, stream, _mr);
          init(back.data(), old_cap, obk, old_lazy);
          rehash(_segs[q].data(), new_cap, nbk, back.data(), old_cap, obk, moved + 0);
          sync();
          _segs[q] = std::move(back);
        } catch (std::bad_alloc const&) {
          _broken = true;
          throw std::runtime_error(
            "seg_table: a growth step ran out of memory and could not be undone");
        }
      }
      upload_pointers(stream);
      throw;
    }
    zero(_full.data(), _full.size() * sizeof(uint32_t), stream);
    _cap = new_cap;
    _bk  = nbk;
    upload_pointers(stream);
  }
  uint64_t got[2] = {0, 0};
  check(
    cudaMemcpyAsync(got, _count.data(), 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.get()),
    "read count");
  check(cudaStreamSynchronize(stream.get()), "sync");
  if (got[1] != got[0]) {
    _broken = true;
    throw std::runtime_error("seg_table: growth lost groups");
  }
  std::lock_guard<std::mutex> lock(_m);
  _stats.growths += 1;
  _stats.grow_seconds += wall() - t0;
  _stats.rehashed_bytes += old_bytes;
  _stats.rehashed_groups += got[1];
  _stats.grow_launches += launches;
  _stats.init_bytes += init_b;
  _stats.init_seconds += t_init;
  _stats.rehash_seconds += t_rehash;
}

// Makes room for `need_groups`, growing by at least one growth step. `seen_cap` is the capacity
// the caller saw: if another thread grew the table since, nothing is done. False: the table is
// at the size its budget allows.
bool seg_table::grow_for(uint64_t need_groups,
                         uint64_t seen_cap,
                         ::cuda::stream_ref stream,
                         uint64_t model_groups,
                         bool by_curve,
                         bool* declined)
{
  std::unique_lock<std::shared_mutex> lk(_rw);
  if (_emitting) throw std::runtime_error("seg_table: used after emit");
  if (_broken)
    throw std::runtime_error("seg_table: unusable after a growth step that could not be undone");
  if (_cap != seen_cap) return true;
  if (_gave_up || _at_max.load(std::memory_order_relaxed)) return false;
  uint64_t need_cap =
    cap_for(static_cast<double>(need_groups) / static_cast<double>(num_segments()) + 1.0);
  uint64_t new_cap =
    round8(static_cast<uint64_t>(std::ceil(static_cast<double>(_cap) * _opt.growth)));
  double rung = 0;
  if (_opt.growth_by_groups) {
    double lim        = static_cast<double>(group_limit());
    rung              = _ladder > 0
                          ? _ladder
                          : (_opt.initial_groups != 0 ? static_cast<double>(_opt.initial_groups) * 1.05 : lim);
    double const from = rung;
    do {
      rung *= _opt.growth;
    } while (rung < lim * 1.125 || rung < static_cast<double>(need_groups));
    new_cap = cap_for(rung / static_cast<double>(num_segments()) + 1.0);
    if (_opt.growth > 2.0 && _opt.big_step_bytes != 0 &&
        new_cap * num_segments() * _W * sizeof(uint64_t) > _opt.big_step_bytes) {
      // Past the bound for large steps: factor 2 (from the rung below, or from the table).
      rung = from > lim ? from : lim;
      do {
        rung *= 2.0;
      } while (rung < lim * 1.125 || rung < static_cast<double>(need_groups));
      new_cap = cap_for(rung / static_cast<double>(num_segments()) + 1.0);
    }
    if (new_cap <= _cap) new_cap = _cap + 8;
  }
  uint64_t const step_cap = new_cap;
  bool const by_model     = model_groups != 0 && (_opt.hint_groups == 0 || _opt.hint_soft);
  if (by_model && _opt.give_up_share > 0 && _opt.budget_bytes != 0 &&
      static_cast<double>(max_groups(true)) <
        _opt.give_up_share * static_cast<double>(model_groups)) {
    _at_max  = true;
    _gave_up = true;
    std::lock_guard<std::mutex> lock(_m);
    _stats.gave_up += 1;
    return false;
  }
  if (by_model) {
    // The counts seen so far say how many groups there will be (see estimate_groups): size the
    // step for that instead of the fixed factor. It may make the step smaller (the table then
    // ends at the size a pre-sized one would have), never smaller than x1.25 (a low estimate must
    // not turn into many rehashes), and larger only up to `growth_trust` (default: the fixed
    // factor, so a wrong estimate cannot cost more memory than plain doubling does).
    uint64_t mg = model_groups;
    // A hint the model does not contradict (within a factor 1.5 either way: keys that are not
    // quite uniform move the model's count by that much) is the better figure.
    if (_opt.hint_soft && _opt.hint_groups != 0 &&
        static_cast<double>(mg) <= 1.5 * static_cast<double>(_opt.hint_groups) &&
        static_cast<double>(_opt.hint_groups) <= 1.5 * static_cast<double>(mg) &&
        _total.load() <= limit_for_cap(cap_for_groups(_opt.hint_groups))) {
      mg = _opt.hint_groups;
    }
    uint64_t mc = cap_for_groups(mg);
    auto lo     = round8(static_cast<uint64_t>(std::ceil(static_cast<double>(_cap) * 1.25)));
    double top  = _opt.growth_trust > _opt.growth ? _opt.growth_trust : _opt.growth;
    auto hi     = round8(static_cast<uint64_t>(std::ceil(static_cast<double>(_cap) * top)));
    if (_opt.jump_cap > 0) {  // ... and to a bounded multiple of the groups held
      double held = static_cast<double>(std::max<uint64_t>(_total.load(), 1024));
      uint64_t jc = cap_for_groups(static_cast<uint64_t>(_opt.jump_cap * held));
      if (jc > step_cap && jc < hi) hi = jc;
    }
    new_cap = mc < lo ? lo : (mc > hi ? hi : mc);
  }
  // options::curve_x: no answer from the model - the groups the measured curve gives at the next
  // rung of the row ladder (first batch x curve_x^j; a table that fills within 6% of a rung goes
  // to the next one, one that fills earlier - the curve was low - is sized for the same rung
  // again).
  bool curve_step = false, curve_fit = false;
  if (by_curve && !by_model && _opt.curve_x > 1.0 && !_curve_off.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(_m);
    double R = _rows_done, D = _ins_done;
    if (_batch0 > 0 && R > 0 && D > 0) {
      auto const rung_above = [&](double r) {
        double h = _batch0;
        while (h <= r)
          h *= _opt.curve_x;
        if (_opt.curve_rows > 0 && h > _opt.curve_rows && r < _opt.curve_rows) h = _opt.curve_rows;
        return h;
      };
      // A caller that grows ahead of its launch (the groups it expects, with margin, for every
      // fold in flight) while the table has room: when the curve says the table holds what the
      // rung it is in will bring, it is left alone - the launch goes ahead, and rows it leaves out
      // after all come back through the retry. (Several streams ask for room for all of them at
      // once: at the end of the input that is several batches' groups in a table sized for it.)
      if (declined != nullptr && _total.load() + 1024 < group_limit()) {
        bool f    = false;
        double P1 = curve_groups(R, D, rung_above(R), &f);
        if (P1 > 0 && cap_for_groups(static_cast<uint64_t>(P1 / 1.04)) <= _cap) {
          *declined = true;
          return true;
        }
      }
      double H = rung_above(R / 0.94);
      double P = curve_groups(R, D, H, &curve_fit);
      if (P > 0) {
        uint64_t cc = cap_for_groups(static_cast<uint64_t>(P));
        if (_opt.jump_cap > 0) {
          uint64_t jc = cap_for_groups(static_cast<uint64_t>(_opt.jump_cap * std::max(D, 1024.0)));
          if (cc > jc) cc = jc;
        }
        // options::curve_cap: with no row count told, the rung is a guess at where the input
        // ends - a step then holds at most curve_cap times the groups held (job 24281: up to 4
        // times the exact table at 128 batches without it).
        if (_opt.curve_cap > 0 && _opt.curve_rows <= 0) {
          uint64_t kc = cap_for_groups(static_cast<uint64_t>(_opt.curve_cap * std::max(D, 1024.0)));
          if (cc > kc) cc = kc;
        }
        auto lo    = round8(static_cast<uint64_t>(std::ceil(static_cast<double>(_cap) * 1.25)));
        new_cap    = cc < lo ? lo : cc;
        curve_step = true;
      }
    }
  }
  if (new_cap < need_cap) new_cap = need_cap;
  if (_opt.hint_groups != 0 && !(by_model && _opt.hint_soft)) {
    uint64_t hc = cap_for_groups(_opt.hint_groups);
    // The hint still holds (the table is below its size and the need fits it): stop there - or
    // (hint_soft, no answer from the model) go there in one step.
    if (hc > _cap && hc >= need_cap && (new_cap > hc || _opt.hint_soft)) new_cap = hc;
    if (_opt.hint_soft && _opt.hint_share < 1.0) {
      uint64_t sc = cap_for_groups(
        static_cast<uint64_t>(static_cast<double>(_opt.hint_groups) * _opt.hint_share));
      if (sc > _cap + _cap / 8 && sc >= need_cap && sc < new_cap) new_cap = sc;
    }
  }
  bool const jump = new_cap > step_cap;
  uint64_t mc     = max_cap();
  if (new_cap > mc && by_model && _opt.dense_at_budget && _opt.small_bytes != 0 &&
      _opt.layout != 0 && model_groups > limit_for_cap(mc)) {
    // The model says the largest table of the budget cannot hold the groups in its sparse size
    // tier: go to the dense layout now (what `densify` would do once that table is full), in this
    // one step.
    uint64_t old_limit = group_limit();
    uint64_t old_small = _opt.small_bytes;
    _opt.small_bytes   = 0;
    uint64_t dc        = max_cap();
    if (dc >= _cap + _cap / 16 + 8 && limit_for_cap(dc) > old_limit + old_limit / 16) {
      mc = dc;
    } else {
      _opt.small_bytes = old_small;
    }
  }
  if (new_cap > mc) {
    new_cap = mc;
    if (new_cap < _cap + _cap / 16 + 8) {
      if (densify(stream)) return true;
      _at_max = true;
      return false;
    }
  }
  if (new_cap <= _cap) return false;
  if (!try_grow_to(new_cap, stream)) {
    _at_max = true;  // the memory resource has no room for the step: the table stays as it is
    return false;
  }
  if (curve_step) {
    std::lock_guard<std::mutex> lock(_m);
    _stats.curve_steps += 1;
    if (curve_fit) _stats.curve_fits += 1;
  }
  // The ladder continues from this step's rung, or (a step sized otherwise) from what the table
  // holds now.
  if (_opt.growth_by_groups) {
    if (new_cap == step_cap && rung > 0) {
      _ladder = rung;
    } else if (!_opt.ladder_anchor) {
      _ladder = static_cast<double>(group_limit());
    } else if (_ladder == 0) {
      _ladder = _opt.initial_groups != 0 ? static_cast<double>(_opt.initial_groups) * 1.05
                                         : static_cast<double>(group_limit());
    }
  }
  if (jump) {
    std::lock_guard<std::mutex> lock(_m);
    _stats.jumps += 1;
  }
  return true;
}

// options::dense_at_budget: the table is at the size its budget allows; if it runs sparser than
// a BUCKET table at the load for its state width would, it is rehashed into that (the size tiers
// are dropped for good). Called with the exclusive lock held. False: nothing to gain.
bool seg_table::densify(::cuda::stream_ref stream)
{
  if (!_opt.dense_at_budget || _opt.small_bytes == 0 || _opt.layout == 0) return false;
  uint64_t old_limit = group_limit();
  uint64_t old_small = _opt.small_bytes;
  _opt.small_bytes   = 0;
  uint64_t mc        = max_cap();
  if (mc < _cap || limit_for_cap(mc) <= old_limit + old_limit / 16) {
    _opt.small_bytes = old_small;
    return false;
  }
  if (!try_grow_to(mc, stream)) {
    _opt.small_bytes = old_small;
    return false;
  }
  return true;
}

uint64_t seg_table::upsert_once(flat_table::source const& src,
                                uint64_t n,
                                uint32_t* bitmap,
                                bool only_marked,
                                bool allow_insert,
                                rmm::device_uvector<uint64_t>& counters,
                                uint64_t& seen_cap,
                                ::cuda::stream_ref stream,
                                uint64_t* inserted,
                                gen_plan const* gp)
{
  auto* c = reinterpret_cast<ull*>(counters.data());
  zero(c, 4 * sizeof(uint64_t), stream);
  uint64_t hc[4] = {0, 0, 0, 0};
  uint64_t count = 0;
  {
    std::shared_lock<std::shared_mutex> lk(_rw);
    if (_emitting) throw std::runtime_error("seg_table: used after emit");
    if (_broken)
      throw std::runtime_error("seg_table: unusable after a growth step that could not be undone");
    seen_cap = _cap;
    seg_layout L;
    fill_layout(&L);
    if (gp != nullptr && gp->rpt != 0 && !only_marked && allow_insert) {
      // Keys that repeat (options::gen_combine; seg_gen.cu).
      gen_fold(L,
               _NS,
               _bk,
               src,
               make_identities(_spec),
               n,
               gp->rpt,
               gp->strided,
               gp->hot,
               bitmap,
               c,
               stream.get());
    } else if (_bk) {
      seg_upsert_kernel<true>
        <<<grid(n), BLOCK, 0, stream.get()>>>(L, _NS, src, n, bitmap, only_marked, allow_insert, c);
    } else {
      seg_upsert_kernel<false>
        <<<grid(n), BLOCK, 0, stream.get()>>>(L, _NS, src, n, bitmap, only_marked, allow_insert, c);
    }
    check(cudaGetLastError(), "seg_upsert_kernel");
    check(cudaMemcpyAsync(hc, c, sizeof hc, cudaMemcpyDeviceToHost, stream.get()), "read counters");
    check(
      cudaMemcpyAsync(&count, _count.data(), sizeof count, cudaMemcpyDeviceToHost, stream.get()),
      "read count");
    check(cudaStreamSynchronize(stream.get()), "sync");
  }
  note_count(count);
  if (inserted != nullptr) *inserted += hc[2];
  return hc[0];
}

// Groups the table must hold after `fails` rows could not be applied. When each row is a group
// of its own (partial states with unique keys) that is the count plus the rows. Raw rows say
// little about how many groups they hold (one new key can be thousands of rows), so the table
// then grows one step at a time: a step is never much more than needed, and a badly undersized
// table takes a few rounds.
uint64_t seg_table::need_after(uint64_t fails, bool rows_are_groups) const
{
  return _total.load() + (rows_are_groups ? fails : 1);
}

uint64_t seg_table::upsert(flat_table::source const& src,
                           uint64_t n,
                           uint32_t* bitmap,
                           bool only_marked,
                           rmm::device_uvector<uint64_t>& counters,
                           ::cuda::stream_ref stream,
                           bool allow_insert,
                           gen_plan const* gp)
{
  if (n == 0) return 0;
  uint64_t seen   = 0;
  uint64_t before = _total.load();
  uint64_t fails =
    upsert_once(src, n, bitmap, only_marked, allow_insert, counters, seen, stream, nullptr, gp);
  if (allow_insert) {
    // Groups that arrive this way are not rows of the stream: the growth model does not apply
    // to a table that got a real share of its groups from here.
    std::lock_guard<std::mutex> lock(_m);
    uint64_t now = _total.load();
    if (now > before && (now - before) * 20 > now && !_opt.keep_model_on_upsert) _model_off = true;
    if (now > before)
      _ins_done += static_cast<double>(now - before);  // (the curve's count, see `_hist`)
  }
  while (fails != 0 && allow_insert) {
    if (!grow_for(need_after(fails, src.op[0] != kop::ADD_ONE), seen, stream)) break;
    fails = upsert_once(src, n, bitmap, true, true, counters, seen, stream);
  }
  return fails;
}

// Rows per thread of the direct fold (see options::strip). Measured on a GB200 (2^31 rows):
// the strip kernel with 4 rows per thread is 16-22% faster than the one-row kernel on clustered
// input at every state width (44% with 8 words), 1-12% faster on uniform input with up to 4 state
// words in the BUCKET layout and with up to 2 in the ROW layout, and 2-8% slower on uniform input
// with 8 words. With 8 words it is therefore only used when sampled rows show runs of equal keys.
uint32_t seg_table::strip_for() const
{
  if (_opt.strip >= 0) return static_cast<uint32_t>(_opt.strip);
  if (!_bk) return _NS <= 2 ? 4u : 0u;
  if (_NS <= 4) return 4u;
  return _runs.load(std::memory_order_relaxed) > 0.1f ? 4u : 0u;
}

// The final number of groups, from the counts seen so far, when they pin it down; 0 otherwise.
// Model: rows draw their key uniformly from G keys, so after R rows the table holds
// D = G (1 - exp(-R / G)) groups. G is solved from the latest (R, D) and accepted only if the same
// G also reproduces an earlier point (at most ~2/3 of the rows) within 3%: input that brings new
// keys at a steady rate (clustered, all-distinct) or with heavy skew fails that test, and growth
// then falls back to one step at a time. Called with `_m` held.
uint64_t seg_table::estimate_groups() const
{
  if (_model_off) return 0;
  double R = _obs_rows, D = static_cast<double>(_total.load());
  if (R < static_cast<double>(1ull << 18) || D <= 0) return 0;
  double q              = D / R;
  bool const by_repeats = _opt.model_min_repeats != 0;
  if (by_repeats) {
    if (D >= R || R - D < static_cast<double>(_opt.model_min_repeats)) return 0;
  } else if (q > 0.98) {
    return 0;  // too few repeated keys to tell
  }
  double cr = _obs_mid_rows <= R / 1.5 ? _obs_mid_rows : _obs_old_rows;
  double cd = _obs_mid_rows <= R / 1.5 ? _obs_mid_groups : _obs_old_groups;
  if (cr <= 0 || cd <= 0) return 0;
  double G = D;
  if (q > 0.02) {  // solve (1 - exp(-x)) / x = q for x = R / G (decreasing in x)
    double lo = by_repeats ? 1e-12 : 1e-9, hi = 60.0;
    for (int it = 0; it < 60; ++it) {
      double x = 0.5 * (lo + hi);
      if ((1.0 - std::exp(-x)) / x > q) {
        lo = x;
      } else {
        hi = x;
      }
    }
    G = R / (0.5 * (lo + hi));
  }
  if (by_repeats) {
    // Repeats among the first `cr` rows: the model's against the observed ones. With few repeats
    // the group counts agree whatever G is (both are nearly the rows), so the test is on the
    // repeats: uniform keys bring them with the square of the rows, keys that arrive at a steady
    // rate (runs, clustered) or with skew in proportion to the rows.
    double pc  = cr + G * std::expm1(-cr / G);
    double oc  = cr - cd;
    double tol = _opt.model_tol * pc + 4.0 * std::sqrt(pc > 0 ? pc : 0) + 16.0;
    if (std::fabs(pc - oc) > tol) return 0;
    // ... and on the groups, as the round-3 rule: long runs of equal keys (16 rows per key) put
    // nearly all rows among the repeats at both points, which the test above lets through.
    double pd = -G * std::expm1(-cr / G);
    if (std::fabs(pd - cd) > 0.03 * cd) return 0;
    return static_cast<uint64_t>(G);
  }
  double pred = G * (1.0 - std::exp(-cr / G));
  if (std::fabs(pred - cd) > 0.03 * cd) return 0;
  return static_cast<uint64_t>(G);
}

// options::curve_x: the groups the table held after `r` rows, interpolated (log-log) between the
// points kept; 0 when no point lies within a factor 1.5 of `r`. Called with `_m` held.
double seg_table::curve_at(double r) const
{
  if (_hist.empty() || r <= 0) return 0;
  std::size_t hi = 0;
  while (hi < _hist.size() && _hist[hi].first < r)
    ++hi;
  if (hi == 0) return _hist[0].first <= r * 1.5 ? _hist[0].second * r / _hist[0].first : 0;
  auto const& a = _hist[hi - 1];
  if (hi == _hist.size()) return a.first * 1.5 >= r ? a.second : 0;
  auto const& b = _hist[hi];
  if (a.first * 1.5 < r && b.first > r * 1.5) return 0;
  if (b.second <= a.second || b.first <= a.first) return a.second;
  double t = std::log(r / a.first) / std::log(b.first / a.first);
  return a.second * std::pow(b.second / a.second, t);
}

// options::curve_x: the groups expected after `H` rows, from `D` groups after `R` rows and the
// points kept (see the option); 0: no statement. `fitted`: by the fitted model. `_m` held.
double seg_table::curve_groups(double R, double D, double H, bool* fitted) const
{
  *fitted = false;
  if (!(H > R)) return 0;
  double d1 = curve_at(R / 2), d0 = curve_at(R / 4), dm = curve_at(R / 8);
  if (d1 <= 0 || d0 <= 0 || dm <= 0) return 0;
  double I0 = D - d1, I1 = d1 - d0, I2 = d0 - dm;
  if (I0 <= 0 || I1 <= 0 || I2 < 256) return 0;
  // Never more than the latest octave's rate brings.
  double const linear = D + I0 / (R / 2) * (H - R);
  double P            = 0;
  // Known keys plus a uniform tail: D = G (1 - exp(-a R)) through (R/2, d1) and (R, D), i.e.
  // D / d1 = 1 + exp(-x) with x = a R / 2, when the curve bends by more than its noise.
  double const bend = 2 * d1 - D;
  if (bend > 6.0 * std::sqrt(D) + 64.0 && D > d1) {
    double x = -std::log(D / d1 - 1.0);
    double G = d1 / -std::expm1(-x);
    double s = G * x / (R / 2);  // new groups per row at the start
    double pc =
      s * (R / 4) + G * std::expm1(-x / 2);  // repeats among the first R/4 rows: predicted
    double oc = s * (R / 4) - d0;            // ... and observed
    double tol =
      _opt.model_tol * pc + 4.0 * std::sqrt(pc > 0 ? pc : 0) + 3.0 * std::sqrt(d0) + 16.0;
    if (std::fabs(pc - oc) <= tol) {
      // (s >= 0.9: nine rows in ten were new keys at the start - uniform keys and nothing else,
      // for which G is the answer the uniform-keys model gives, whatever the rung.)
      double e = -G * std::expm1(-x * H / (R / 2));
      P        = G <= 1.125 * e || s >= 0.9 ? G : e;
      *fitted  = true;
    }
  }
  if (P == 0) {
    // The increments per octave, continued: q = I0 / I1 falls short of 2 by a deficit that keeps
    // its own ratio from octave to octave.
    double q0 = I0 / I1, q1 = I1 / I2;
    if (q0 > 2.0) q0 = 2.0;
    if (q1 > 2.0) q1 = 2.0;
    double def = 2.0 - q0, defp = 2.0 - q1;
    double rho = defp > 1e-9 ? def / defp : 1.0;
    rho        = rho < 0.5 ? 0.5 : (rho > 2.0 ? 2.0 : rho);
    double I = I0, r = R;
    P = D;
    while (r < H) {
      def = def * rho > 2.0 ? 2.0 : def * rho;
      I *= 2.0 - def;
      double part = 2 * r <= H ? 1.0 : (H - r) / r;  // the last octave may be a part of one
      P += I * part;
      r *= 2;
    }
    P *= _opt.curve_margin;
  }
  if (P > linear) P = linear;
  return P > D ? P : D;
}

void seg_table::add_observed(double rows)
{
  std::lock_guard<std::mutex> lock(_m);
  _obs_rows += rows;
  _rows_done += rows;
  if (_obs_mid_rows == 0 || _obs_rows >= 2 * _obs_mid_rows) {
    _obs_old_rows   = _obs_mid_rows;
    _obs_old_groups = _obs_mid_groups;
    _obs_mid_rows   = _obs_rows;
    _obs_mid_groups = static_cast<double>(_total.load());
  }
}

uint64_t seg_table::model_groups() const
{
  std::lock_guard<std::mutex> lock(_m);
  uint64_t e = estimate_groups();
  if (e != 0) _model_last = e;
  return e != 0 ? e : _model_last;
}

seg_table::fold_result seg_table::fold(raw_batch const& b,
                                       flat_table::source const& rsrc,
                                       scratch& sc,
                                       ::cuda::stream_ref stream,
                                       bool allow_preagg)
{
  fold_result res;
  uint64_t n = b.rows;
  if (n == 0) return res;
  if (n > 0xffffffffull) throw std::runtime_error("seg_table::fold: batch too large");
  uint64_t words = bitmap_words(n);
  if (sc.bitmap.size() < words)
    throw std::runtime_error("seg_table::fold: the scratch was not reserved");
  sc.partial_count = 0;
  if (!_canon) {  // no specialised kernel for this aggregate set
    uint64_t const before = _total.load();
    // options::gen_combine: do the keys of this batch repeat - in runs, or a few hot keys? Two
    // samples (1,024 and 4,096 rows, read back with one synchronisation) decide the kernel of the
    // batch's first pass; input without either runs seg_upsert_kernel as before.
    gen_plan gp{};
    if (_opt.gen_combine != 0 && rsrc.op[0] == kop::ADD_ONE && n >= GEN_MIN_ROWS) {
      auto* cnt = reinterpret_cast<ull*>(sc.counters.data());
      zero(cnt, 5 * sizeof(uint64_t), stream);
      hot_sample_runs(rsrc.key.p, n, cnt, stream.get());
      auto cover = static_cast<uint32_t>(std::ceil(_opt.hot_cover * HOT_SAMPLE));
      hot_sample(rsrc.key.p,
                 0,
                 n,
                 _opt.hot_thr,
                 cover < 1 ? 1 : cover,
                 sc.hot.data(),
                 cnt + 4,
                 stream.get());
      uint64_t sm[5] = {0, 0, 0, 0, 0};
      check(cudaMemcpyAsync(sm, cnt, sizeof sm, cudaMemcpyDeviceToHost, stream.get()),
            "read sample");
      check(cudaStreamSynchronize(stream.get()), "sync");
      uint64_t const eq = sm[0];  // of 1,024 sampled rows: followed by a row with the same key
      if (eq >= GEN_RUN_MIN) {
        gp.rpt     = eq >= 1024 ? _opt.gen_run_rows : _opt.gen_run_rows_min;
        gp.strided = _opt.gen_run_strided;
      } else if (sm[4] != 0) {
        gp.rpt = _opt.hot_rows;
        gp.hot = sc.hot.data();
      }
    }
    res.rows = upsert(
      rsrc, n, sc.bitmap.data(), false, sc.counters, stream, true, gp.rpt != 0 ? &gp : nullptr);
    uint64_t const now = _total.load();
    res.inserted       = now > before ? now - before : 0;
    {
      // (The generic kernel has no chunks: the growth model sees the batch as one point.)
      std::lock_guard<std::mutex> lock(_m);
      _stats.chunks += 1;
      if (gp.rpt != 0 && gp.hot == nullptr) _stats.run_chunks += 1;
      if (gp.rpt != 0 && gp.hot != nullptr) _stats.hot_chunks += 1;
      _obs_rows += static_cast<double>(n);
      _rows_done += static_cast<double>(n - res.rows);
      if (_obs_mid_rows == 0 || _obs_rows >= 2 * _obs_mid_rows) {
        _obs_old_rows   = _obs_mid_rows;
        _obs_old_groups = _obs_mid_groups;
        _obs_mid_rows   = _obs_rows;
        _obs_mid_groups = static_cast<double>(now);
      }
    }
    return res;
  }
  raw_cols F{};
  F.key = b.key;
  for (int c = 0; c < NUM_VALUE_COLUMNS; ++c)
    F.col[c] = b.col[c];
  auto* counters = reinterpret_cast<ull*>(sc.counters.data());
  raw_cols Fuse  = F;

  struct active_guard {
    std::atomic<int>& a;
    explicit active_guard(std::atomic<int>& x) : a(x) { a.fetch_add(1); }
    ~active_guard() { a.fetch_sub(1); }
  } guard(_active);
  if (_opt.curve_x > 1.0 && !_opt.curve_shared && _active.load() > 1)
    _curve_off.store(true, std::memory_order_relaxed);

  // Wide states: is the input clustered (runs of equal keys)? Sampled on the first batch and every
  // 16th after it; decides the kernel (see strip_for).
  if (_opt.strip < 0 && _NS > 4 && n >= 2048 && (sc.folds++ % 16) == 0) {
    zero(counters, sizeof(uint64_t), stream);
    hot_sample_runs(F.key, n, counters, stream.get());
    uint64_t eq = 0;
    check(cudaMemcpyAsync(&eq, counters, sizeof eq, cudaMemcpyDeviceToHost, stream.get()),
          "read sample");
    check(cudaStreamSynchronize(stream.get()), "sync");
    _runs.store(static_cast<float>(eq) / 1024.0f, std::memory_order_relaxed);
  }

  // options::run_preagg: does this batch come in long runs of equal keys? (Sampled on every
  // batch: a batch that stops doing so must not go through the launch planned for a few keys.)
  bool long_runs = false;
  bool very_long = false;  // no sampled row ends a run
  if (_opt.run_preagg != 0 && _NS >= 4 && n >= (1ull << 16)) {
    zero(counters, sizeof(uint64_t), stream);
    hot_sample_runs(F.key, n, counters, stream.get());
    uint64_t eq = 0;
    check(cudaMemcpyAsync(&eq, counters, sizeof eq, cudaMemcpyDeviceToHost, stream.get()),
          "read sample");
    check(cudaStreamSynchronize(stream.get()), "sync");
    long_runs = eq >= 1022;
    very_long = eq >= 1024;
  }

  // options::curve_x: the first batch folded is the first rung of the row ladder, and is folded
  // in pieces that end at 1/8, 1/4 and 1/2 of it (its end then has the points the curve needs).
  bool split_first = false;
  if (_opt.curve_x > 1.0) {
    std::lock_guard<std::mutex> lock(_m);
    if (_batch0 == 0) {
      _batch0     = static_cast<double>(n);
      split_first = n >= 8 * PROBE_ROWS;
    }
  }

  bool no_preagg  = !allow_preagg;
  uint64_t marked = 0;
  uint64_t r0     = 0;
  while (r0 < n) {
    uint64_t m                   = n - r0;
    uint64_t const marked_before = marked;
    double rate;
    {
      std::lock_guard<std::mutex> lock(_m);
      rate = _rate;
    }
    uint64_t total  = _total.load();
    uint64_t hint_c = _opt.hint_groups != 0 ? cap_for_groups(_opt.hint_groups) : 0;
    uint64_t hint_l = limit_for_cap(hint_c);
    uint64_t cap, lim;
    auto look = [&] {
      std::shared_lock<std::shared_mutex> lk(_rw);
      cap = _cap;
      lim = group_limit();
    };
    look();
    // Settled: no growth is wanted (at the budget, or at the hinted size while the hint holds).
    bool settled = _at_max.load(std::memory_order_relaxed) ||
                   (_opt.hint_groups != 0 && total <= hint_l && cap >= hint_c);
    if (!settled) {
      // The table follows the group count: a chunk of rows is folded only when the new groups it
      // is expected to bring fit; otherwise the chunk shrinks, and when even a minimum chunk does
      // not fit the table grows one step. (Rows that hit the limit anyway are retried below, so
      // a wrong expectation costs time, not correctness.) The first rows ever folded are a probe
      // that measures how many new groups a row brings.
      if (rate < 0 && m > PROBE_ROWS) m = PROBE_ROWS;
      // Expected new groups per row, with margin.
      double per_row    = (rate < 0 ? 1.0 : rate) * 1.5 * static_cast<double>(_active.load());
      uint64_t minchunk = m / 64 > PROBE_ROWS ? m / 64 : PROBE_ROWS;
      if (minchunk > m) minchunk = m;
      auto room = [&] {
        return lim > total + 1024 ? static_cast<double>(lim - total - 1024) : 0.0;
      };
      if (per_row * static_cast<double>(m) > room()) {
        auto fit = static_cast<uint64_t>(room() / per_row);
        if (fit < minchunk) {
          auto need = total + 1024 + static_cast<uint64_t>(per_row * static_cast<double>(minchunk));
          uint64_t model = 0;
          if (_opt.grow_ahead) {
            // Grow ahead: room for what the rest of this batch is expected to bring (no further
            // ahead than twice the rows seen so far: the rate of new groups is only known to hold
            // over a stretch like the one behind us), and a step sized by the final group count
            // when the counts seen so far determine it (see estimate_groups and grow_for).
            double seen_rows;
            {
              std::lock_guard<std::mutex> lock(_m);
              model = estimate_groups();
              if (model != 0) _model_last = model;
              seen_rows = _obs_rows;
            }
            if (rate >= 0 && model == 0) {
              double ahead = static_cast<double>(n - r0) < 2 * seen_rows
                               ? static_cast<double>(n - r0)
                               : 2 * seen_rows;
              auto rest    = total + 1024 + static_cast<uint64_t>(rate * 1.1 * ahead);
              if (rest > need) need = rest;
            }
          }
          if (_opt.hint_groups != 0 && !(_opt.hint_soft && model != 0) && total <= hint_l &&
              need > hint_l) {
            need = hint_l;
          }
          bool declined = false;
          if (need > lim) grow_for(need, cap, stream, model, true, &declined);
          look();
          fit = static_cast<uint64_t>(room() / per_row);
          if (declined) {
            fit = m;  // options::curve_x: the table is the size for its rung, the launch goes ahead
          } else if (fit < minchunk) {
            fit = minchunk;  // could not grow enough: fold anyway
          }
        }
        if (fit < m) m = fit;
      }
    }

    if (split_first) {
      for (uint64_t bnd : {n / 8, n / 4, n / 2}) {
        if (r0 < bnd && r0 + m > bnd) {
          m = bnd - r0;
          break;
        }
      }
    }

    // Unknown rate (the probe chunk): pre-aggregate with a mid-sized local table. With few groups
    // that avoids contended global atomics; with many, rows just fall through to the table.
    double exp_new  = rate < 0
                        ? (_opt.hint_groups != 0 ? static_cast<double>(_opt.hint_groups) : 128.0)
                        : rate * static_cast<double>(m);
    auto est        = total + static_cast<uint64_t>(exp_new);
    auto plan       = flat_table::plan_for(est, _spec, _local_budget);
    plan.count_mode = rate < 0 || rate >= 1.0 / 64 ? 2 : 1;
    plan.run_cap    = _opt.run_cap;
    if (rate < 0 && _opt.hint_groups == 0) {
      // Few threads per local table: with a handful of groups, hundreds of threads on the same
      // shared-memory words stall each other.
      plan.block           = 64;
      plan.rows_per_thread = 128;
    }
    if (no_preagg ||
        static_cast<uint64_t>(plan.local_slots) * _W * sizeof(uint64_t) > _local_budget) {
      plan.local_slots = 0;
    }
    // Long runs (options::run_preagg): a small block table and private slots, whatever the count.
    // (Also instead of a block table planned for all the groups: without private slots its
    // threads meet on the same shared-memory words, with the same compare-and-swap loops.)
    bool const run_pre = long_runs && !no_preagg && m >= (1ull << 16) && _opt.private_slots;
    if (run_pre) {
      // A block must see only a few keys (its threads hold 4 each in registers): 32,768 rows per
      // block when no sampled row ended a run, 2,048 when one or two did (runs of 500 to 2,000).
      plan.local_slots     = 64;
      plan.block           = very_long ? 128 : 64;
      plan.rows_per_thread = very_long ? 256 : 32;
    }
    uint32_t blocks = 0, rpt = 0;
    // A handful of groups: threads aggregate privately first (see the kernel).
    bool priv = plan.local_slots != 0 && plan.local_slots <= 64 &&
                (est <= PRIVATE_SLOTS || run_pre) && _opt.private_slots;
    if (plan.local_slots != 0) {  // experiments: override the shape of the pre-aggregating launch
      if (_opt.pre_rows != 0) plan.rows_per_thread = _opt.pre_rows;
      if (_opt.pre_block != 0) plan.block = _opt.pre_block;
      if (_opt.pre_slots != 0 && _opt.pre_slots >= plan.local_slots)
        plan.local_slots = _opt.pre_slots;
    }
    if (plan.local_slots != 0) {
      uint64_t r   = plan.rows_per_thread;
      uint64_t fit = (m + 128ull * plan.block - 1) / (128ull * plan.block);
      if (r > fit) r = fit < 8 ? 8 : fit;
      uint64_t per_block = static_cast<uint64_t>(plan.block) * r;
      blocks             = static_cast<uint32_t>((m + per_block - 1) / per_block);
      rpt                = static_cast<uint32_t>(r);
      // Room for every group a block can fail to merge: its local table and, with private
      // slots, what each thread could not place there.
      uint64_t pw = static_cast<uint64_t>(blocks) *
                    (plan.local_slots + (priv ? plan.block * PRIVATE_SLOTS : 0)) * _W;
      if (sc.partials.size() < pw) {
        if (sc.fixed) {
          // The scratch was reserved before the batch and may not grow now (see reserve_scratch):
          // this chunk is folded directly; the next reservation covers a launch like this one.
          if (pw > sc.want_partials) sc.want_partials = pw;
          plan.local_slots = 0;
          priv             = false;
          blocks           = 0;
          rpt              = 0;
        } else {
          sc.partials = rmm::device_uvector<uint64_t>(pw, stream, sc.mr);
        }
      }
    }

    // options::hot_keys: sample this launch's rows for hot keys (on the device), and take the
    // kernel with block-local accumulators when the previous sample of this scratch had any.
    bool hot_sampled = false, hot_run = false;
    if (_opt.hot_keys != 0 && plan.local_slots == 0 && m >= _opt.hot_min_rows &&
        m >= 2 * HOT_SAMPLE) {
      if (sc.hot_state > 0 || sc.hot_skip + 1 >= _opt.hot_every) {
        auto cover = static_cast<uint32_t>(std::ceil(_opt.hot_cover * HOT_SAMPLE));
        hot_sample(F.key,
                   r0,
                   m,
                   _opt.hot_thr,
                   cover < 1 ? 1 : cover,
                   sc.hot.data(),
                   counters + 4,
                   stream.get());
        hot_sampled = true;
        sc.hot_skip = 0;
        if (sc.hot_state < 0) {  // the first sample of this scratch is read back before the fold
          uint64_t hn = 0;
          check(
            cudaMemcpyAsync(&hn, sc.hot.data(), sizeof hn, cudaMemcpyDeviceToHost, stream.get()),
            "read hot");
          check(cudaStreamSynchronize(stream.get()), "sync");
          sc.hot_state = hn != 0 ? 1 : 0;
        }
        hot_run = sc.hot_state > 0;
      } else {
        ++sc.hot_skip;
      }
    }

    zero(counters, 4 * sizeof(uint64_t), stream);
    uint64_t hc[5] = {
      0, 0, 0, 0, 0};  // (hc[4]: the hot keys of this launch's sample, read with the rest)
    uint64_t count = 0;
    uint64_t seen  = 0;
    {
      std::shared_lock<std::shared_mutex> lk(_rw);
      if (_emitting) throw std::runtime_error("seg_table: used after emit");
      if (_broken)
        throw std::runtime_error(
          "seg_table: unusable after a growth step that could not be undone");
      seen = _cap;
      seg_layout L;
      fill_layout(&L);
      if (hot_run) {
        keys_fold(L,
                  F,
                  r0,
                  r0 + m,
                  _NS,
                  _bk,
                  _opt.hot_block,
                  _opt.hot_rows,
                  hot_flags(),
                  sc.hot.data(),
                  sc.bitmap.data(),
                  counters,
                  stream.get());
      } else if (plan.local_slots != 0) {
        by_width(static_cast<int>(_NS), _bk, [&](auto ns, auto bk) {
          constexpr int NS  = decltype(ns)::value;
          constexpr bool BK = decltype(bk)::value;
          launch_preagg<NS, BK>(L,
                                Fuse,
                                r0,
                                r0 + m,
                                blocks,
                                plan.block,
                                plan.local_slots,
                                rpt,
                                priv,
                                pre_v2_on() ? 1u : 0u,
                                sc.bitmap.data(),
                                sc.partials.data(),
                                counters,
                                stream.get(),
                                _local_budget);
        });
      } else {
        hot_fold(
          L, F, r0, r0 + m, _NS, _bk, strip_for(), plan, sc.bitmap.data(), counters, stream.get());
      }
      check(cudaGetLastError(), "seg fold kernel");
      check(cudaMemcpyAsync(hc,
                            counters,
                            (hot_sampled ? 5 : 4) * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost,
                            stream.get()),
            "read counters");
      check(
        cudaMemcpyAsync(&count, _count.data(), sizeof count, cudaMemcpyDeviceToHost, stream.get()),
        "read count");
      check(cudaStreamSynchronize(stream.get()), "sync");
    }
    note_count(count);
    uint64_t const hot_n = hot_sampled ? hc[4] : 0;
    if (hot_sampled) sc.hot_state = hot_n != 0 ? 1 : 0;

    // marked: rows of this batch whose bit is set (not applied yet), across all its chunks.
    marked += hc[0] + hc[3];
    uint64_t new_groups = hc[2];
    if (hc[3] != 0) {  // rows the pre-aggregating kernel left to the host
      marked = upsert_once(
        rsrc, n, sc.bitmap.data(), true, true, sc.counters, seen, stream, &res.inserted);
      uint64_t now = _total.load();
      if (now > count) new_groups += now - count;  // what those rows inserted
    }
    res.inserted += hc[2];
    {
      std::lock_guard<std::mutex> lock(_m);
      _rate = static_cast<double>(new_groups < m ? new_groups : m) / static_cast<double>(m);
      if (hc[0] != 0 || hc[1] != 0 || hc[3] != 0) _stats.retries += 1;
      _stats.chunks += 1;
      if (run_pre && plan.local_slots != 0) _stats.run_chunks += 1;
      if (hot_sampled) {
        _stats.hot_samples += 1;
        _stats.hot_found += hot_n;
      }
      if (hot_run) _stats.hot_chunks += 1;
      _stats.retry_rows += hc[0] + hc[3];
      // (rows folded, groups held) history for estimate_groups.
      _ins_done += static_cast<double>(new_groups);
      _rows_done +=
        static_cast<double>(m) + static_cast<double>(marked_before) - static_cast<double>(marked);
      _obs_rows += static_cast<double>(m);
      double d = static_cast<double>(_total.load());
      if (_obs_mid_rows == 0 || _obs_rows >= 2 * _obs_mid_rows) {
        _obs_old_rows   = _obs_mid_rows;
        _obs_old_groups = _obs_mid_groups;
        _obs_mid_rows   = _obs_rows;
        _obs_mid_groups = d;
      }
    }
    uint64_t fp = hc[1];
    if ((hc[0] != 0 || hc[3] != 0 || fp != 0) && (marked != 0 || fp != 0)) {
      // The table hit its limit: grow, then apply what was left out. Partial states are always
      // offered once more (their key may have been inserted meanwhile).
      bool pmarked = false;
      uint64_t pc  = fp;
      if (fp != 0) {
        uint64_t pwords = bitmap_words(pc);
        if (sc.partial_bitmap.size() < pwords) {
          if (sc.fixed) throw std::logic_error("seg_table::fold: reserved scratch too small");
          sc.partial_bitmap = rmm::device_uvector<uint32_t>(pwords, stream, sc.mr);
        }
        zero(sc.partial_bitmap.data(), pwords * sizeof(uint32_t), stream);
      }
      uint64_t const ins_before = res.inserted, marked_retry = marked;
      // options::retry_ahead: room for what the rows left out are expected to bring.
      uint64_t ahead = 0;
      if (_opt.retry_ahead && marked != 0) {
        uint64_t out   = hc[0] + hc[3];
        double applied = static_cast<double>(m > out ? m - out : 1);
        double rr      = static_cast<double>(new_groups) / applied;
        if (rr > 1.0) rr = 1.0;
        ahead =
          _total.load() + 1024 + static_cast<uint64_t>(rr * 1.1 * static_cast<double>(marked));
      }
      for (;;) {
        uint64_t need = need_after(marked + fp, false);
        if (ahead > need) need = ahead;
        ahead     = 0;
        bool grew = grow_for(need, seen, stream, 0, fp == 0);
        if (fp != 0) {
          fp      = upsert_once(row_source(sc.partials.data()),
                           pc,
                           sc.partial_bitmap.data(),
                           pmarked,
                           true,
                           sc.counters,
                           seen,
                           stream,
                           &res.inserted);
          pmarked = true;
        }
        if (marked != 0 && grew) {
          marked = upsert_once(
            rsrc, n, sc.bitmap.data(), true, true, sc.counters, seen, stream, &res.inserted);
        }
        if (!grew || (marked == 0 && fp == 0)) break;
      }
      if (fp != 0) {
        sc.partial_count = pc;
        res.partials     = fp;
        no_preagg        = true;  // the partial buffer now belongs to the caller
      }
      if (res.inserted > ins_before ||
          marked != marked_retry) {  // (the curve's counts, see `_hist`)
        std::lock_guard<std::mutex> lock(_m);
        _ins_done += static_cast<double>(res.inserted - ins_before);
        _rows_done += static_cast<double>(marked_retry) - static_cast<double>(marked);
      }
    }
    if (_opt.curve_x > 1.0 && marked == 0) {  // a clean (rows, groups) point for the curve
      std::lock_guard<std::mutex> lock(_m);
      if (_ins_done > 0 && (_hist.empty() || (_rows_done >= 1.19 * _hist.back().first &&
                                              _ins_done >= _hist.back().second))) {
        _hist.emplace_back(_rows_done, _ins_done);
      }
    }
    r0 += m;
  }
  res.rows = marked;
  return res;
}

// First emit call: the exact group count of the whole table. Called with the exclusive lock held.
void seg_table::begin_emit(::cuda::stream_ref stream)
{
  if (_emitting) return;
  uint64_t count = 0;
  check(cudaMemcpyAsync(&count, _count.data(), sizeof count, cudaMemcpyDeviceToHost, stream.get()),
        "read count");
  check(cudaStreamSynchronize(stream.get()), "sync");
  _emit_total = count;
  _emitting   = true;
}

void seg_table::reserve_scratch(scratch& sc, uint64_t n, ::cuda::stream_ref stream) const
{
  uint32_t const W = _W;
  uint64_t words   = bitmap_words(n);
  if (sc.bitmap.size() < words) sc.bitmap = rmm::device_uvector<uint32_t>(words, stream, sc.mr);
  zero(sc.bitmap.data(), sc.bitmap.size() * sizeof(uint32_t), stream);
  if (!_canon) {  // the generic kernel folds rows only: no partial rows
    sc.fixed = true;
    return;
  }
  // The pre-aggregating launch of a probe chunk (the first rows a table ever folds, when nothing
  // is known about the group count): at most ~130 blocks, each with a local table of at most the
  // shared-memory budget plus the private slots of its (64) threads.
  uint64_t max_slots = 32;
  while (max_slots * 2 * W * sizeof(uint64_t) <= _local_budget)
    max_slots *= 2;
  uint64_t probe = n < PROBE_ROWS ? n : PROBE_ROWS;
  uint64_t want  = (probe / 8192 + 130) * (max_slots + 64 * PRIVATE_SLOTS) * W;
  if (want < sc.want_partials) want = sc.want_partials;
  if (sc.partials.size() < want) sc.partials = rmm::device_uvector<uint64_t>(want, stream, sc.mr);
  uint64_t pwords = bitmap_words(sc.partials.size() / W);
  if (sc.partial_bitmap.size() < pwords)
    sc.partial_bitmap = rmm::device_uvector<uint32_t>(pwords, stream, sc.mr);
  sc.fixed = true;
}

void seg_table::set_budget(uint64_t bytes)
{
  std::unique_lock<std::shared_mutex> lk(_rw);
  // (The prototype refused a budget below the table. Here the table keeps what it holds - the
  // owner flushes it if it must give the memory back - and only stops growing.)
  uint64_t const need = budget_need();
  _opt.budget_bytes   = bytes != 0 && bytes < need ? need : bytes;
  if (bytes != 0 && bytes <= need) _at_max = true;
}

uint64_t seg_table::segments_per_chunk(uint64_t max_bytes) const
{
  uint64_t S = num_segments();
  if (_packed) return S;
  double per_seg =
    static_cast<double>(_total.load()) / static_cast<double>(S) * _W * sizeof(uint64_t);
  if (per_seg < 1) return S;
  auto g = static_cast<uint64_t>(static_cast<double>(max_bytes) / per_seg);
  return g < 1 ? 1 : (g > S ? S : g);
}

seg_table::chunk seg_table::emit_segments(uint64_t p0,
                                          uint64_t count,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref out_mr)
{
  std::unique_lock<std::shared_mutex> lk(_rw);
  if (_broken)
    throw std::runtime_error("seg_table: unusable after a growth step that could not be undone");
  begin_emit(stream);
  uint64_t S = num_segments();
  if (count == 0 || p0 + count > S) throw std::runtime_error("seg_table: bad segment range");
  for (uint64_t p = p0; p < p0 + count; ++p) {
    if (_emitted[p]) throw std::runtime_error("seg_table: segment already emitted");
  }
  auto const mark_emitted = [&] {
    for (uint64_t p = p0; p < p0 + count; ++p)
      _emitted[p] = true;
  };
  if (_packed && p0 != 0) {  // all with segment 0
    mark_emitted();
    return chunk{rmm::device_uvector<uint64_t>(0, stream, out_mr), 0};
  }
  uint64_t nseg  = _packed ? 1 : count;  // launches address a packed table as one segment
  uint64_t slots = _packed ? _cap * S : _cap;
  // The chunk is allocated for the most groups these segments can plausibly hold (a packed table:
  // exactly the table's count; otherwise their share of the count plus six standard deviations, a
  // few KiB more than needed) and the gather counts as it goes. Segments that hold more are
  // gathered again into a chunk of the size the first pass counted.
  uint64_t guess = _emit_total;
  if (!_packed) {
    double mean =
      static_cast<double>(_emit_total) * static_cast<double>(count) / static_cast<double>(S);
    guess = static_cast<uint64_t>(mean + 6.0 * std::sqrt(mean) + 64.0);
  }
  if (guess > slots * nseg) guess = slots * nseg;
  auto* c = reinterpret_cast<ull*>(_count.data() + 1);
  seg_layout L;
  fill_layout(&L);
  if (!_packed) {  // (fill_layout addresses segments relative to segment 0, which may be gone)
    L.ref = _segs[p0].data();
    for (uint64_t p = p0; p < p0 + count; ++p) {
      L.seg_off[p] =
        (reinterpret_cast<int64_t>(_segs[p].data()) - reinterpret_cast<int64_t>(L.ref)) / 8;
    }
  }
  auto gather = [&](uint64_t* out, uint64_t capacity) {
    zero(c, sizeof(uint64_t), stream);
    hot_gather(L, _packed ? 0 : p0, nseg, slots, _NS, _bk, out, capacity, c, stream.get());
    uint64_t got = 0;
    check(cudaMemcpyAsync(&got, c, sizeof got, cudaMemcpyDeviceToHost, stream.get()),
          "read gathered");
    check(cudaStreamSynchronize(stream.get()), "sync");
    return got;
  };
  // (Allocations may throw: the segments are only marked and freed once their chunk is complete.)
  chunk out{rmm::device_uvector<uint64_t>(guess * _W, stream, out_mr), 0};
  uint64_t groups = guess != 0 ? gather(out.rows.data(), guess) : 0;
  if (groups > guess) {  // `groups` is the exact count now
    out.rows = rmm::device_uvector<uint64_t>(0, stream, out_mr);
    out.rows = rmm::device_uvector<uint64_t>(groups * _W, stream, out_mr);
    if (gather(out.rows.data(), groups) != groups) {
      throw std::runtime_error("seg_table: segment count does not match its slots");
    }
  }
  out.groups = groups;
  mark_emitted();
  if (_packed) {
    _segs[0] = rmm::device_uvector<uint64_t>(0, stream, _mr);
  } else {
    for (uint64_t p = p0; p < p0 + count; ++p)
      _segs[p] = rmm::device_uvector<uint64_t>(0, stream, _mr);
  }
  return out;
}

// The shape of a fold launch for an estimate of the group count (the prototype's
// agg_table::plan_for without its front-table kernels: no split fold, no per-thread stripes).
// Measured on a GB200 (2^31 rows, 2^25-row batches).
flat_table::fold_plan flat_table::plan_for(uint64_t est_groups,
                                           state_spec const& spec,
                                           uint64_t local_bytes)
{
  fold_plan p{};
  auto ns = static_cast<uint64_t>(spec.num_state);
  // Block-local pre-aggregation pays off when all of the groups fit in a shared-memory table at
  // load <= 0.5 (a block then folds thousands of rows into it without touching global memory).
  uint64_t max_slots = 32;
  while (max_slots * 2 * (1 + ns) * sizeof(uint64_t) <= local_bytes)
    max_slots *= 2;
  if (est_groups * 2 <= max_slots) {
    uint32_t s = 32;
    while (s < 2 * est_groups)
      s *= 2;
    p.local_slots = s;
    if (est_groups <= 16) {  // a handful of hot slots: fewer threads per table contend less
      p.block           = 64;
      p.rows_per_thread = 128;
    } else if (s >= 2048) {  // big local tables: amortise their init and merge over more rows
      p.block           = 1024;
      p.rows_per_thread = s / 32;
    } else {
      p.block           = ns >= 4 ? 512 : 256;
      p.rows_per_thread = 32;
    }
  }
  // Many new groups per row: one counter update per block. Few: one per inserting warp.
  p.count_mode = est_groups >= (1ull << 20) ? 2 : 1;
  return p;
}

}  // namespace sirius::agg_table::detail

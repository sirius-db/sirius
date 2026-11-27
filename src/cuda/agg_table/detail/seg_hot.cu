// Ported unchanged from experimental/agg_bench/src/seg_hot.cu (namespace, and the reserved-key
// flag through seg_layout::seen).
// The kernels a pre-sized seg_table runs from start to end (see seg_dev.cuh for why they have a
// translation unit of their own): initialise, fold raw rows directly, gather.
#include "seg_dev.cuh"

#include <stdexcept>

namespace sirius::agg_table::detail {
namespace segdev {

namespace {

// counters: [0] rows not applied (table full), [1] partial states not applied, [2] groups inserted
// by this launch, [3] rows the pre-aggregating kernel left to the host.
//
// Direct fold of the raw rows [first, end). As agg_table's fold_direct_kernel (run combining,
// new groups counted per warp or per block), with rows that find the table full marked in a
// bitmap. (The profiling variants of round 2 live on in the frozen seg_table_r2.cu:
// AGG_T1_R2=1 AGG_T1_ABLATE=n.)
template <int NS, bool BK>
__global__ void seg_fold_direct_kernel(seg_layout L,
                                       raw_cols F,
                                       uint64_t first,
                                       uint64_t end_row,
                                       uint32_t run_cap,
                                       uint32_t count_mode,
                                       uint32_t* bitmap,
                                       ull* counters)
{
  __shared__ unsigned int blk_new;
  if (count_mode == 2) {
    if (threadIdx.x == 0) blk_new = 0;
    __syncthreads();
  }
  uint64_t i    = first + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool inserted = false;
  if (i < end_row &&
      !reserved_key(L.seen, F.key[i])) {  // (reserved keys: the operator's side groups, sql.cu)
    uint64_t k = F.key[i];
    {
      uint64_t end = i + 1;
      bool head    = true;
      if (run_cap > 1) {
        uint64_t g0 = i & ~static_cast<uint64_t>(run_cap - 1);
        if (i != g0 && i != first && F.key[i - 1] == k) {
          head = false;
        } else {
          uint64_t gend = g0 + run_cap < end_row ? g0 + run_cap : end_row;
          while (end < gend && F.key[end] == k)
            ++end;
        }
      }
      if (head) {
        uint64_t* claim = nullptr;
        uint64_t* st    = seg_probe<BK>(L, NS, k, true, inserted, claim);
        if (st == nullptr) {
          atomicAdd(counters, static_cast<ull>(end - i));
          for (uint64_t r = i; r < end; ++r)
            mark(bitmap, r);
        } else if (end == i + 1) {
          uint64_t a         = NS > 1 ? F.col[0][i] : 0;
          uint64_t b         = NS > 2 ? F.col[1][i] : 0;
          uint64_t c         = NS > 3 ? F.col[2][i] : 0;
          uint64_t d         = NS > 4 ? F.col[3][i] : 0;
          uint32_t const len = sum_row_len<NS>(a, b);
          if (claim != nullptr) {
            init_raw<NS>(st, a, b, c, d);
            publish(claim, k);
          } else {
            fold_raw_ordered<NS>(st, a, b, c, d, L.order);
          }
          sum_row_done<NS>(len);
        } else {
          uint64_t acc[NS];
          set_identity<NS>(acc);
          for (uint64_t r = i; r < end; ++r) {
            uint64_t a = NS > 1 ? F.col[0][r] : 0;
            uint64_t b = NS > 2 ? F.col[1][r] : 0;
            uint64_t c = NS > 3 ? F.col[2][r] : 0;
            uint64_t d = NS > 4 ? F.col[3][r] : 0;
            acc_raw<NS>(acc, a, b, c, d);
          }
          if (claim != nullptr) {
            init_partial<NS>(st, acc);
            publish(claim, k);
          } else {
            fold_partial<NS>(st, acc);
          }
          // (AGG_SQL_SUM=fold: the run's rows are measured by their own threads, below)
        }
      } else {
        sum_run_row<NS>(F.col[0],
                        F.col[1],
                        i);  // a row inside a run: combined by the run's first thread (sql_dev.cuh)
      }
    }
  }
  // New groups reach the table's counter once per warp, or (count_mode 2: many new groups) once
  // per block.
  unsigned int m = __ballot_sync(0xffffffffu, inserted);
  if (count_mode != 2) {
    if ((threadIdx.x & 31) == 0 && m != 0) {
      atomicAdd(reinterpret_cast<ull*>(L.count), static_cast<ull>(__popc(m)));
      atomicAdd(counters + 2, static_cast<ull>(__popc(m)));
    }
    return;
  }
  if ((threadIdx.x & 31) == 0 && m != 0) atomicAdd(&blk_new, static_cast<unsigned int>(__popc(m)));
  __syncthreads();
  if (threadIdx.x == 0 && blk_new != 0) {
    atomicAdd(reinterpret_cast<ull*>(L.count), static_cast<ull>(blk_new));
    atomicAdd(counters + 2, static_cast<ull>(blk_new));
  }
}

// Direct fold, several rows per thread: thread t owns the `strip` consecutive rows starting at
// first + t * strip and walks them run by run (a run: consecutive rows with the same key, never
// crossing the strip), one probe per run. With clustered input most threads of the one-row kernel
// are not the head of a run and do nothing; here every thread has table work to do, a block is
// `strip` times fewer threads for the same rows, and no row reads its neighbour's key twice.
// (Measured and dropped: loading the first key word of all rows of the strip before probing any of
// them, to overlap their cache misses, is 10-40% slower; so is keeping the strip's keys in
// registers instead of reading each when its turn comes.)
template <int NS, bool BK>
__global__ void seg_fold_strip_kernel(seg_layout L,
                                      raw_cols F,
                                      uint64_t first,
                                      uint64_t end_row,
                                      uint32_t strip,
                                      uint32_t* bitmap,
                                      ull* counters)
{
  __shared__ unsigned int blk_new;
  if (threadIdx.x == 0) blk_new = 0;
  __syncthreads();
  uint64_t i       = first + (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * strip;
  uint64_t lim     = i + strip < end_row ? i + strip : end_row;
  unsigned int ins = 0;
  while (i < lim) {
    uint64_t k   = F.key[i];
    uint64_t end = i + 1;
    while (end < lim && F.key[end] == k)
      ++end;
    if (reserved_key(L.seen, k)) {  // not a table key: the operator's side groups (sql.cu)
      i = end;
      continue;
    }
    bool inserted   = false;
    uint64_t* claim = nullptr;
    uint64_t* st    = seg_probe<BK>(L, NS, k, true, inserted, claim);
    ins += inserted ? 1u : 0u;
    if (st == nullptr) {
      atomicAdd(counters, static_cast<ull>(end - i));
      for (uint64_t r = i; r < end; ++r)
        mark(bitmap, r);
    } else if (end == i + 1) {
      uint64_t a         = NS > 1 ? F.col[0][i] : 0;
      uint64_t b         = NS > 2 ? F.col[1][i] : 0;
      uint64_t c         = NS > 3 ? F.col[2][i] : 0;
      uint64_t d         = NS > 4 ? F.col[3][i] : 0;
      uint32_t const len = sum_row_len<NS>(a, b);
      if (claim != nullptr) {
        init_raw<NS>(st, a, b, c, d);
        publish(claim, k);
      } else {
        fold_raw_ordered<NS>(st, a, b, c, d, L.order);
      }
      sum_row_done<NS>(len);
    } else {
      uint64_t acc[NS];
      set_identity<NS>(acc);
      for (uint64_t r = i; r < end; ++r) {
        uint64_t a = NS > 1 ? F.col[0][r] : 0;
        uint64_t b = NS > 2 ? F.col[1][r] : 0;
        uint64_t c = NS > 3 ? F.col[2][r] : 0;
        uint64_t d = NS > 4 ? F.col[3][r] : 0;
        acc_raw<NS>(acc, a, b, c, d);
      }
      if (claim != nullptr) {
        init_partial<NS>(st, acc);
        publish(claim, k);
      } else {
        fold_partial<NS>(st, acc);
      }
      sum_run<NS>(
        F.col[0], F.col[1], i, end);  // AGG_SQL_SUM=fold: the run just combined (sql_dev.cuh)
    }
    i = end;
  }
  if (ins != 0) atomicAdd(&blk_new, ins);
  __syncthreads();
  if (threadIdx.x == 0 && blk_new != 0) {
    atomicAdd(reinterpret_cast<ull*>(L.count), static_cast<ull>(blk_new));
    atomicAdd(counters + 2, static_cast<ull>(blk_new));
  }
}

// BUCKET with lazy state: only the key line of every bucket is written.
__global__ void seg_init_keys_kernel(uint64_t* base, uint64_t buckets, uint32_t NS)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= buckets) return;
  uint64_t* kb = base + i * (8ull * (NS + 1));
  for (uint32_t j = 0; j < 8; ++j)
    kb[j] = EMPTY_KEY;
}

template <bool BK>
__global__ void seg_init_kernel(uint64_t* base, uint64_t cap, uint32_t NS, identities id)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  *key_ptr<BK>(base, i, NS) = EMPTY_KEY;
  uint64_t* st              = state_ptr<BK>(base, i, NS);
  for (uint32_t j = 0; j < NS; ++j)
    st[j] = id.v[j];
}

// Segments -> dense rows. Block b works on segment p0 + b / bps (bps blocks per segment); blocks
// reserve output ranges in launch order, so the chunk stays close to slot (hash) order.
template <bool BK>
__global__ void seg_gather_kernel(seg_layout L,
                                  uint64_t p0,
                                  uint32_t bps,
                                  uint64_t cap,
                                  uint32_t NS,
                                  uint64_t* out,
                                  uint64_t capacity,
                                  ull* counter)
{
  uint64_t* base = seg_base(L, p0 + blockIdx.x / bps);
  uint64_t i     = static_cast<uint64_t>(blockIdx.x % bps) * blockDim.x + threadIdx.x;
  uint64_t k     = i < cap ? *key_ptr<BK>(base, i, NS) : EMPTY_KEY;
  bool occ       = k != EMPTY_KEY;
  uint64_t pos   = block_position(occ, counter);
  if (occ && pos < capacity) {
    uint64_t* o         = out + pos * (NS + 1);
    uint64_t const* src = state_ptr<BK>(base, i, NS);
    o[0]                = k;
    for (uint32_t j = 0; j < NS; ++j)
      o[1 + j] = src[j];
  }
}

// 1024 evenly spaced rows: how many have the same key as the row after them.
__global__ void seg_sample_runs_kernel(uint64_t const* key, uint64_t n, ull* counter)
{
  uint64_t t = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t i = t * ((n - 1) / 1024);
  block_count(key[i] == key[i + 1], counter);
}

// options::hot_keys: the hot set of the rows [first, first + n) (see seg_dev.cuh). One block of
// 1,024 threads, four sampled rows each (evenly spaced, each moved by a hashed offset inside its
// stride). The sampled keys are counted in a shared-memory table of 2 x HOT_SAMPLE words - bits
// 0-15: a fingerprint of the key, 16-30: its count, 31: taken - and every key counted `thr` times
// or more is entered into the hot table by the first of its threads. Two keys that share a slot
// and a fingerprint are counted as one (1 pair in 2^29; the hot set is a set of keys that get an
// accumulator, any key is correct). More than HOT_MAX qualifying keys: the first HOT_MAX.
// (n >= 2 * HOT_SAMPLE.)
constexpr uint32_t HS_TAB = 2 * HOT_SAMPLE;
__global__ void seg_hot_sample_kernel(uint64_t const* key,
                                      uint64_t first,
                                      uint64_t n,
                                      uint32_t thr,
                                      uint32_t cover,
                                      uint64_t* hot,
                                      ull* n_out)
{
  __shared__ unsigned int tab[HS_TAB];
  __shared__ uint64_t hk[HOT_SLOTS];
  __shared__ unsigned int nh, cov, adj;
  constexpr uint32_t PER = HOT_SAMPLE / 1024;
  for (uint32_t s = threadIdx.x; s < HS_TAB; s += blockDim.x)
    tab[s] = 0;
  if (threadIdx.x < HOT_SLOTS) hk[threadIdx.x] = EMPTY_KEY;
  if (threadIdx.x == 0) {
    nh  = 0;
    cov = 0;
    adj = 0;
  }
  __syncthreads();
  uint64_t const stride = n / HOT_SAMPLE;
  unsigned int eq       = 0;
  uint64_t myk[PER];
  uint32_t mys[PER];
#pragma unroll
  for (uint32_t j = 0; j < PER; ++j) {
    uint64_t t = static_cast<uint64_t>(threadIdx.x) * PER + j;
    uint64_t i = first + t * stride +
                 (static_cast<uint64_t>(static_cast<uint32_t>(t) * 0x9E3779B1u) >> 8) % stride;
    uint64_t k = key[i];
    eq += key[i + 1 < first + n ? i + 1 : i - 1] == k ? 1u : 0u;
    myk[j] = k;
    mys[j] = 0xffffffffu;
    if (reserved_key_quiet(k)) continue;  // never a hot key: the fold skips its rows (sql.cu)
    uint64_t h = fmix64(k);
    auto fp    = static_cast<unsigned int>(h & 0xffffu);
    auto slot  = static_cast<uint32_t>(h >> 40) & (HS_TAB - 1);
    myk[j]     = k;
    mys[j]     = 0xffffffffu;
    for (uint32_t p = 0; p < 32; ++p) {
      unsigned int cur = tab[slot];
      if (cur == 0) {
        cur = atomicCAS(&tab[slot], 0u, fp | 0x10000u);
        if (cur == 0) {
          mys[j] = slot;
          break;
        }
      }
      if ((cur & 0xffffu) == fp) {
        atomicAdd(&tab[slot], 0x10000u);
        mys[j] = slot;
        break;
      }
      slot = (slot + 1) & (HS_TAB - 1);
    }
  }
  if (eq != 0) atomicAdd(&adj, eq);
  __syncthreads();
#pragma unroll
  for (uint32_t j = 0; j < PER; ++j) {
    if (mys[j] == 0xffffffffu) continue;
    unsigned int cnt = (tab[mys[j]] >> 16) & 0x7fffu;
    if (cnt < thr) continue;
    if ((atomicOr(&tab[mys[j]], 0x80000000u) & 0x80000000u) != 0)
      continue;  // another thread has it
    if (atomicAdd(&nh, 1u) >= HOT_MAX) continue;
    atomicAdd(&cov, cnt);
    uint32_t idx = local_hash(myk[j]) & (HOT_SLOTS - 1);
    for (;;) {
      ull old = atomicCAS(
        reinterpret_cast<ull*>(hk + idx), static_cast<ull>(EMPTY_KEY), static_cast<ull>(myk[j]));
      if (old == EMPTY_KEY || old == myk[j]) break;
      idx = (idx + 1) & (HOT_SLOTS - 1);
    }
  }
  __syncthreads();
  // (Long runs of equal keys - all but 8 of the sampled rows are followed by an equal key - are
  // not hot keys: the direct kernels combine runs, and all threads then address one slot.)
  unsigned int const found =
    cov >= cover && adj < HOT_SAMPLE - 8 ? (nh < HOT_MAX ? nh : HOT_MAX) : 0;
  if (threadIdx.x < HOT_SLOTS) hot[1 + threadIdx.x] = found != 0 ? hk[threadIdx.x] : EMPTY_KEY;
  if (threadIdx.x == 0) {
    hot[0] = found;
    if (n_out != nullptr) *n_out = found;
  }
}

template <int NS, bool BK>
void launch_direct(seg_layout const& L,
                   raw_cols const& F,
                   uint64_t first,
                   uint64_t end_row,
                   flat_table::fold_plan const& plan,
                   uint32_t* bitmap,
                   ull* counters,
                   cudaStream_t stream)
{
  uint64_t g = (end_row - first + plan.block - 1) / plan.block;
  seg_fold_direct_kernel<NS, BK><<<static_cast<uint32_t>(g), plan.block, 0, stream>>>(
    L, F, first, end_row, plan.run_cap, plan.count_mode, bitmap, counters);
}

template <int NS, bool BK>
void launch_strip(seg_layout const& L,
                  raw_cols const& F,
                  uint64_t first,
                  uint64_t end_row,
                  uint32_t strip,
                  uint32_t block,
                  uint32_t* bitmap,
                  ull* counters,
                  cudaStream_t stream)
{
  uint64_t threads = (end_row - first + strip - 1) / strip;
  auto blocks      = static_cast<uint32_t>((threads + block - 1) / block);
  seg_fold_strip_kernel<NS, BK>
    <<<blocks, block, 0, stream>>>(L, F, first, end_row, strip, bitmap, counters);
}

}  // namespace

void hot_init(uint64_t* base,
              uint64_t slots,
              uint32_t NS,
              bool bk,
              bool lazy,
              identities const& id,
              cudaStream_t stream)
{
  if (lazy) {
    seg_init_keys_kernel<<<grid(slots / 8), BLOCK, 0, stream>>>(base, slots / 8, NS);
  } else if (bk) {
    seg_init_kernel<true><<<grid(slots), BLOCK, 0, stream>>>(base, slots, NS, id);
  } else {
    seg_init_kernel<false><<<grid(slots), BLOCK, 0, stream>>>(base, slots, NS, id);
  }
  check(cudaGetLastError(), "seg_init_kernel");
}

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
              cudaStream_t stream)
{
  by_width(static_cast<int>(NS), bk, [&](auto ns, auto bkt) {
    constexpr int NSV  = decltype(ns)::value;
    constexpr bool BKV = decltype(bkt)::value;
    if (strip > 1) {
      launch_strip<NSV, BKV>(L, F, first, end_row, strip, plan.block, bitmap, counters, stream);
    } else {
      launch_direct<NSV, BKV>(L, F, first, end_row, plan, bitmap, counters, stream);
    }
  });
}

void hot_sample_runs(uint64_t const* key, uint64_t n, ull* counter, cudaStream_t stream)
{
  seg_sample_runs_kernel<<<4, 256, 0, stream>>>(key, n, counter);
  check(cudaGetLastError(), "seg_sample_runs_kernel");
}

void hot_sample(uint64_t const* key,
                uint64_t first,
                uint64_t n,
                uint32_t thr,
                uint32_t cover,
                uint64_t* hot,
                ull* n_out,
                cudaStream_t stream)
{
  seg_hot_sample_kernel<<<1, 1024, 0, stream>>>(key, first, n, thr, cover, hot, n_out);
  check(cudaGetLastError(), "seg_hot_sample_kernel");
}

void hot_gather(seg_layout const& L,
                uint64_t p0,
                uint64_t nseg,
                uint64_t slots,
                uint32_t NS,
                bool bk,
                uint64_t* out,
                uint64_t capacity,
                ull* counter,
                cudaStream_t stream)
{
  uint64_t bps = (slots + CBLOCK - 1) / CBLOCK;
  if (bps * nseg > 0x7fffffffull) throw std::runtime_error("launch too large");
  auto g = static_cast<uint32_t>(bps * nseg);
  auto b = static_cast<uint32_t>(bps);
  if (bk) {
    seg_gather_kernel<true><<<g, CBLOCK, 0, stream>>>(L, p0, b, slots, NS, out, capacity, counter);
  } else {
    seg_gather_kernel<false><<<g, CBLOCK, 0, stream>>>(L, p0, b, slots, NS, out, capacity, counter);
  }
  check(cudaGetLastError(), "seg_gather_kernel");
}

}  // namespace segdev
}  // namespace sirius::agg_table::detail

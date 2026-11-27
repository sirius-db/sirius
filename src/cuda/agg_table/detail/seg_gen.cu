// Generic fold of raw rows whose keys repeat: long runs of equal keys, or a few hot keys
// (seg_table::options::gen_combine). What the specialised kernels do for the canonical aggregate
// sets - seg_table.cu / seg_hot.cu: runs of equal keys are combined before the table is touched;
// seg_keys.cu: block-local accumulators for the hot keys - for ANY set of state words, the
// nullable layout included: the operations and operands are launch parameters, as in
// seg_upsert_kernel (seg_table.cu), which is the kernel for every other input and sends each row
// to the table on its own. With keys that repeat, the threads in flight then update the same few
// slots, and the f64 sum and the min / max words are compare-and-swap loops that retry against
// each other (GB200 job 24654: 3,000 keys in runs of 716,000 rows, 8 aggregates, 2^31 rows:
// 5.995 s, against 0.30 s for 187,500 uniform keys).
//
// A translation unit of its own: input without such batches never loads these kernels (see
// seg_dev.cuh), and the kernels of the other units are compiled exactly as before.
//
// A row is only ever accumulated outside the table for a key that is ALREADY in the table: the
// thread (or the block) holds the address of the group's state words, and the table cannot move
// while a launch is in flight (growth is exclusive). So nothing can be left over when the table
// is full: a row whose key is absent and cannot be inserted is marked in the bitmap, as in
// seg_upsert_kernel, and every other row is applied by this launch - no partial states, and the
// set of rows a batch applied is still exactly the rows that are not marked (what the wide sums
// and the fold tickets of agg_table.cu rely on).
#include "seg_dev.cuh"

#include <stdexcept>

namespace sirius::agg_table::detail {
namespace segdev {

namespace {

constexpr int64_t NO_SLOT    = INT64_MIN;
constexpr uint32_t GEN_BLOCK = 256;  // threads per block
// A thread adds the groups it inserted to the table's count every GEN_COUNT_STEP of them (the
// rest once per block): the count is what stops inserts at the group limit, and a thread with
// hundreds of rows must not keep its new groups to itself until the block ends.
constexpr unsigned int GEN_COUNT_STEP = 16;

__device__ __forceinline__ uint32_t gen_hot_find(uint64_t const* hkey, uint64_t k)
{
  uint32_t idx = local_hash(k) & (HOT_SLOTS - 1);
  for (;;) {
    uint64_t cur = hkey[idx];
    if (cur == k) return idx;
    if (cur == EMPTY_KEY) return HOT_SLOTS;
    idx = (idx + 1) & (HOT_SLOTS - 1);
  }
}

// One raw-row value into a thread-private partial state word (no atomics).
__device__ __forceinline__ void acc_op(uint64_t& st, kop op, uint64_t v)
{
  switch (op) {
    case kop::ADD_ONE: st += 1; break;
    case kop::ADD_I64: st += v; break;
    case kop::ADD_F64: st = as_bits(as_f64(st) + as_f64(v)); break;
    case kop::MIN_I64:
      if (static_cast<long long>(v) < static_cast<long long>(st)) st = v;
      break;
    case kop::MAX_I64:
      if (static_cast<long long>(v) > static_cast<long long>(st)) st = v;
      break;
    case kop::MIN_F64:
      if (f64_less(v, st)) st = v;
      break;
    case kop::MAX_F64:
      if (f64_greater(v, st)) st = v;
      break;
  }
}
// The operation that merges a partial state word of `op` into a state word.
__device__ __forceinline__ kop merge_of(kop op) { return op == kop::ADD_ONE ? kop::ADD_I64 : op; }

// RUNS (`hot` null). A thread folds `rpt` rows: consecutive ones, or (`strided`) every
//   GEN_BLOCK-th of the block's GEN_BLOCK x rpt rows. Rows with the key of the thread's previous
//   row are accumulated in registers; the accumulated state is merged into the group's state
//   words when the key changes and after the thread's last row: one probe and one state update
//   per piece of a run instead of one per row.
// HOT (`hot`: the hot set of the batch's sample, seg_dev.cuh). A block folds GEN_BLOCK x rpt
//   rows, a thread every GEN_BLOCK-th. The block looks every hot key up once - one that is not in
//   the table yet is not hot in this block, its rows take the path of all other keys, which
//   inserts it - and keeps an accumulator per hot key in shared memory, merged into the table
//   once, at the end of the block. A thread keeps its rows of the first hot key it meets in
//   registers (added to the accumulator after its rows: with one key on a third of the rows a
//   third of the block's threads would otherwise meet on one accumulator at every step) and
//   sends its rows of other hot keys to the accumulators as shared-memory atomics. Rows of the
//   other keys go to the table one by one, as in seg_upsert_kernel.
// The source is RAW rows (word 0 counts them: ADD_ONE; every stride is 1).
// counters: [0] rows marked in the bitmap, [2] groups inserted (also added to the table's count).
template <int NS, bool BK>
__global__ void seg_fold_gen_kernel(seg_layout L,
                                    flat_table::source S,
                                    identities id,
                                    uint64_t n,
                                    uint32_t rpt,
                                    uint32_t strided,
                                    uint64_t const* hot,
                                    uint32_t* bitmap,
                                    ull* counters)
{
  __shared__ uint64_t hkey[HOT_SLOTS];
  __shared__ int64_t hoff[HOT_SLOTS];  // the key's state words, relative to L.ref (NO_SLOT: none)
  __shared__ uint64_t hacc[HOT_SLOTS * NS];
  __shared__ unsigned int blk_new, blk_fail;
  uint32_t const T    = blockDim.x;
  bool const hot_mode = hot != nullptr;
  if (threadIdx.x == 0) {
    blk_new  = 0;
    blk_fail = 0;
  }
  if (hot_mode) {
    for (uint32_t s = threadIdx.x; s < HOT_SLOTS; s += T) {
      uint64_t k  = hot[1 + s];
      int64_t off = NO_SLOT;
      if (k != EMPTY_KEY) {
        bool ins        = false;
        uint64_t* claim = nullptr;
        uint64_t* st    = seg_probe<BK>(L, NS, k, false, ins, claim);
        if (st != nullptr) off = st - L.ref;
      }
      hkey[s] = k;
      hoff[s] = off;
#pragma unroll
      for (int w = 0; w < NS; ++w)
        hacc[s * NS + w] = id.v[w];
    }
  }
  __syncthreads();

  kop op[NS];
  uint64_t const* vp[NS];
#pragma unroll
  for (int w = 0; w < NS; ++w) {
    op[w] = S.op[w];
    vp[w] = S.val[w].p;
  }
  uint64_t const* const key = S.key.p;
  // The accumulators as the row loop addresses them: through a pointer whose address space the
  // compiler cannot tell (it depends on a launch parameter). On a pointer it knows to be shared
  // memory every 64-bit atomic is a load + compare-and-swap loop; on this one the generic atomic
  // comes first and the loop is only its fallback (table_dev.cuh has the prototype's measurement;
  // here, GB10, 8 state words, 4 / 16 / 40 uniform keys: 1,630 / 1,600 / 1,860 M rows/s with
  // `hacc` itself, 2,460 / 2,280 / 2,310 with this).
  uint64_t* const accs = hot_mode ? hacc : L.ref;

  // The thread's private state: `okey` (EMPTY_KEY: none), where it goes - RUNS: the group's state
  // words (relative to L.ref); HOT: the key's accumulator - and the state.
  uint64_t okey = EMPTY_KEY;
  int64_t oref  = 0;
  uint64_t oacc[NS];
  unsigned int newg = 0, fails = 0;
  reserved_acc rsv;
  uint64_t const base = hot_mode || strided != 0
                          ? static_cast<uint64_t>(blockIdx.x) * T * rpt + threadIdx.x
                          : (static_cast<uint64_t>(blockIdx.x) * T + threadIdx.x) * rpt;
  uint64_t const step = hot_mode || strided != 0 ? T : 1;
  for (uint32_t r = 0; r < rpt; ++r) {
    uint64_t i = base + static_cast<uint64_t>(r) * step;
    if (i >= n) break;
    uint64_t k = key[i];
    uint64_t v[NS];
#pragma unroll
    for (int w = 0; w < NS; ++w)
      v[w] = vp[w][i];
    if (reserved_key_quiet(k)) {
      // Not a table key: the operator's side groups. (The values are used on this path too, so
      // that their loads stay in front of the branch: see sql_dev.cuh, reserved_row.)
#pragma unroll
      for (int w = 0; w < NS; ++w)
        rsv.sink += v[w];
      rsv.any = true;
      continue;
    }
    if (k == okey) {
#pragma unroll
      for (int w = 0; w < NS; ++w)
        acc_op(oacc[w], op[w], v[w]);
      continue;
    }
    if (hot_mode) {
      uint32_t h = gen_hot_find(hkey, k);
      if (h != HOT_SLOTS && hoff[h] != NO_SLOT) {
        if (okey == EMPTY_KEY) {
          okey = k;
          oref = static_cast<int64_t>(h);
#pragma unroll
          for (int w = 0; w < NS; ++w) {
            oacc[w] = id.v[w];
            acc_op(oacc[w], op[w], v[w]);
          }
        } else {
          uint64_t* a = accs + static_cast<uint64_t>(h) * NS;
#pragma unroll
          for (int w = 0; w < NS; ++w)
            apply(a + w, op[w], v[w]);
        }
        continue;
      }
    } else if (okey != EMPTY_KEY) {  // the run ended: its state into the table
      if (oacc[0] != 0) {            // (word 0 counts the rows accumulated)
        uint64_t* a = L.ref + oref;
#pragma unroll
        for (int w = 0; w < NS; ++w)
          apply(a + w, merge_of(op[w]), oacc[w]);
      }
      okey = EMPTY_KEY;
    }
    bool inserted   = false;
    uint64_t* claim = nullptr;
    uint64_t* st    = seg_probe<BK>(L, NS, k, true, inserted, claim);
    if (st == nullptr) {
      mark(bitmap, i);
      ++fails;
      continue;
    }
    if (inserted && ++newg == GEN_COUNT_STEP) {
      atomicAdd(reinterpret_cast<ull*>(L.count), static_cast<ull>(newg));
      atomicAdd(counters + 2, static_cast<ull>(newg));
      newg = 0;
    }
    if (claim != nullptr) {  // a new group (lazy state): its first state is this row
#pragma unroll
      for (int w = 0; w < NS; ++w)
        st[w] = op[w] == kop::ADD_ONE ? 1 : v[w];
      publish(claim, k);
    } else if (hot_mode) {
#pragma unroll
      for (int w = 0; w < NS; ++w)
        apply(st + w, op[w], v[w]);
    }
    if (!hot_mode) {  // the next rows of this key stay with the thread
      okey = k;
      oref = st - L.ref;
#pragma unroll
      for (int w = 0; w < NS; ++w) {
        oacc[w] = id.v[w];
        if (claim == nullptr) acc_op(oacc[w], op[w], v[w]);
      }
    }
  }
  reserved_rows_done(L.seen, rsv);
  if (okey != EMPTY_KEY && oacc[0] != 0) {
    if (hot_mode) {
      uint64_t* a = accs + static_cast<uint64_t>(oref) * NS;
#pragma unroll
      for (int w = 0; w < NS; ++w)
        apply(a + w, merge_of(op[w]), oacc[w]);
    } else {
      uint64_t* a = L.ref + oref;
#pragma unroll
      for (int w = 0; w < NS; ++w)
        apply(a + w, merge_of(op[w]), oacc[w]);
    }
  }
  if (newg != 0) atomicAdd(&blk_new, newg);
  if (fails != 0) atomicAdd(&blk_fail, fails);
  __syncthreads();

  if (hot_mode) {
    for (uint32_t s = threadIdx.x; s < HOT_SLOTS; s += T) {
      int64_t off = hoff[s];
      // (Word 0 counts the rows: zero when the block had no row of the key.)
      if (off == NO_SLOT || hacc[s * NS] == 0) continue;
      uint64_t* a = L.ref + off;
#pragma unroll
      for (int w = 0; w < NS; ++w)
        apply(a + w, merge_of(op[w]), hacc[s * NS + w]);
    }
  }
  if (threadIdx.x == 0) {
    if (blk_new != 0) {
      atomicAdd(reinterpret_cast<ull*>(L.count), static_cast<ull>(blk_new));
      atomicAdd(counters + 2, static_cast<ull>(blk_new));
    }
    if (blk_fail != 0) atomicAdd(counters, static_cast<ull>(blk_fail));
  }
}

template <int NS>
void launch_gen(seg_layout const& L,
                bool bk,
                flat_table::source const& S,
                identities const& id,
                uint64_t n,
                uint32_t rpt,
                uint32_t strided,
                uint64_t const* hot,
                uint32_t* bitmap,
                ull* counters,
                cudaStream_t stream)
{
  uint64_t const per = static_cast<uint64_t>(GEN_BLOCK) * rpt;
  uint64_t const g   = (n + per - 1) / per;
  if (g > 0x7fffffffull) throw std::runtime_error("launch too large");
  auto const blocks = static_cast<uint32_t>(g);
  if (bk) {
    seg_fold_gen_kernel<NS, true>
      <<<blocks, GEN_BLOCK, 0, stream>>>(L, S, id, n, rpt, strided, hot, bitmap, counters);
  } else {
    seg_fold_gen_kernel<NS, false>
      <<<blocks, GEN_BLOCK, 0, stream>>>(L, S, id, n, rpt, strided, hot, bitmap, counters);
  }
}

}  // namespace

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
              cudaStream_t stream)
{
  if (n == 0) return;
  switch (NS) {
    case 2: launch_gen<2>(L, bk, S, id, n, rpt, strided, hot, bitmap, counters, stream); break;
    case 3: launch_gen<3>(L, bk, S, id, n, rpt, strided, hot, bitmap, counters, stream); break;
    case 4: launch_gen<4>(L, bk, S, id, n, rpt, strided, hot, bitmap, counters, stream); break;
    case 5: launch_gen<5>(L, bk, S, id, n, rpt, strided, hot, bitmap, counters, stream); break;
    case 6: launch_gen<6>(L, bk, S, id, n, rpt, strided, hot, bitmap, counters, stream); break;
    case 7: launch_gen<7>(L, bk, S, id, n, rpt, strided, hot, bitmap, counters, stream); break;
    case 8: launch_gen<8>(L, bk, S, id, n, rpt, strided, hot, bitmap, counters, stream); break;
    default: throw std::logic_error("gen_fold: state width");
  }
  check(cudaGetLastError(), "seg_fold_gen_kernel");
}

}  // namespace segdev
}  // namespace sirius::agg_table::detail

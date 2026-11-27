// Device and host helpers of the aggregation table (ported unchanged from
// experimental/agg_bench/src/table_dev.cuh, except for the namespace): the hash, the atomic state
// updates, the canonical aggregate set and small launch utilities.
#pragma once

#include "common.hpp"
#include "sql_dev.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::agg_table::detail {
namespace tdev {

using namespace sqldev;

constexpr int BLOCK = 256;

__host__ __device__ inline uint64_t fmix64(uint64_t k)
{
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

struct identities {
  uint64_t v[MAX_STATE];
};

__device__ inline void apply(uint64_t* st, kop op, uint64_t v)
{
  auto* u = reinterpret_cast<unsigned long long*>(st);
  switch (op) {
    case kop::ADD_ONE: atomicAdd(u, 1ull); break;
    case kop::ADD_I64: atomicAdd(u, static_cast<unsigned long long>(v)); break;
    case kop::ADD_F64:
      atomicAdd(reinterpret_cast<double*>(st), __longlong_as_double(static_cast<long long>(v)));
      break;
    // min/max: the state only ever moves one way, so a plain read that already beats the value
    // makes the atomic unnecessary.
    case kop::MIN_I64:
      if (static_cast<long long>(v) < static_cast<long long>(*st)) {
        atomicMin(reinterpret_cast<long long*>(st), static_cast<long long>(v));
      }
      break;
    case kop::MAX_I64:
      if (static_cast<long long>(v) > static_cast<long long>(*st)) {
        atomicMax(reinterpret_cast<long long*>(st), static_cast<long long>(v));
      }
      break;
    case kop::MIN_F64: {
      unsigned long long old = *u;
      while (f64_less(v, old)) {
        unsigned long long prev = atomicCAS(u, old, static_cast<unsigned long long>(v));
        if (prev == old) break;
        old = prev;
      }
      break;
    }
    case kop::MAX_F64: {
      unsigned long long old = *u;
      while (f64_greater(v, old)) {
        unsigned long long prev = atomicCAS(u, old, static_cast<unsigned long long>(v));
        if (prev == old) break;
        old = prev;
      }
      break;
    }
  }
}

/// State word `w` of a generic source (upsert kernels): measured (sql_dev.cuh: sum_acc) when the
/// source is RAW rows - its count is ADD_ONE; a merge of partial states adds its counts - and the
/// word is an int64 sum.
__device__ __forceinline__ void sum_src(sum_acc& s, flat_table::source const& S, int w, uint64_t v)
{
  sum_val(s, S.op[0] == kop::ADD_ONE && S.op[w] == kop::ADD_I64, v);
}

constexpr kop CANON_OP[MAX_STATE]  = {kop::ADD_ONE,
                                      kop::ADD_I64,
                                      kop::MIN_I64,
                                      kop::MAX_F64,
                                      kop::ADD_F64,
                                      kop::MAX_I64,
                                      kop::MIN_F64,
                                      kop::ADD_I64};
constexpr int CANON_COL[MAX_STATE] = {-1, 0, 1, 2, 3, 0, 2, 1};
constexpr uint64_t I64_MAX_BITS    = 0x7fffffffffffffffULL;
constexpr uint64_t I64_MIN_BITS    = 0x8000000000000000ULL;
constexpr uint64_t F64_PINF_BITS   = 0x7ff0000000000000ULL;
constexpr uint64_t F64_NINF_BITS   = 0xfff0000000000000ULL;

struct raw_cols {
  uint64_t const* key;
  uint64_t const* col[NUM_VALUE_COLUMNS];
};

__device__ inline double as_f64(uint64_t v)
{
  return __longlong_as_double(static_cast<long long>(v));
}
__device__ inline uint64_t as_bits(double d)
{
  return static_cast<uint64_t>(__double_as_longlong(d));
}

// Atomic updates of one state word (device or shared memory).
__device__ inline void a_add(uint64_t* st, uint64_t v)
{
  atomicAdd(reinterpret_cast<unsigned long long*>(st), static_cast<unsigned long long>(v));
}
__device__ inline void a_addf(uint64_t* st, uint64_t v)
{
  atomicAdd(reinterpret_cast<double*>(st), as_f64(v));
}
__device__ inline void a_min(uint64_t* st, uint64_t v)
{
  if (static_cast<long long>(v) < static_cast<long long>(*st)) {
    atomicMin(reinterpret_cast<long long*>(st), static_cast<long long>(v));
  }
}
__device__ inline void a_max(uint64_t* st, uint64_t v)
{
  if (static_cast<long long>(v) > static_cast<long long>(*st)) {
    atomicMax(reinterpret_cast<long long*>(st), static_cast<long long>(v));
  }
}
__device__ inline void a_minf(uint64_t* st, uint64_t v)
{
  auto* u                = reinterpret_cast<unsigned long long*>(st);
  unsigned long long old = *u;
  while (f64_less(v, old)) {
    unsigned long long prev = atomicCAS(u, old, static_cast<unsigned long long>(v));
    if (prev == old) break;
    old = prev;
  }
}
__device__ inline void a_maxf(uint64_t* st, uint64_t v)
{
  auto* u                = reinterpret_cast<unsigned long long*>(st);
  unsigned long long old = *u;
  while (f64_greater(v, old)) {
    unsigned long long prev = atomicCAS(u, old, static_cast<unsigned long long>(v));
    if (prev == old) break;
    old = prev;
  }
}

// Compile-time profiling switch (wrong results unless 0): AGG_R1_ABLATE bit j drops state word j
// from fold_raw (the round-1 block-local kernels and the direct kernels in listing order) and
// from fold_partial (the runs a direct kernel combined); bit 8 (256) replaces the count's atomic
// add by a plain load / add / store (same memory traffic, no atomic).
#ifndef AGG_R1_ABLATE
#define AGG_R1_ABLATE 0
#endif

// One raw row (a, b, c, d) into the state words at `st`.
template <int NS>
__device__ inline void fold_raw(uint64_t* st, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  if (AGG_R1_ABLATE & 256) {
    st[0] += 1;  // (profiling: a plain load / add / store in place of the atomic; loses updates)
  } else if (!(AGG_R1_ABLATE & 1)) {
    a_add(st + 0, 1);
  }
  if (NS > 1 && !(AGG_R1_ABLATE & 2)) a_add(st + 1, a);
  if (NS > 2 && !(AGG_R1_ABLATE & 4)) a_min(st + 2, b);
  if (NS > 3 && !(AGG_R1_ABLATE & 8)) a_maxf(st + 3, c);
  if (NS > 4 && !(AGG_R1_ABLATE & 16)) a_addf(st + 4, d);
  if (NS > 5 && !(AGG_R1_ABLATE & 32)) a_max(st + 5, a);
  if (NS > 6 && !(AGG_R1_ABLATE & 64)) a_minf(st + 6, c);
  if (NS > 7 && !(AGG_R1_ABLATE & 128)) a_add(st + 7, b);
}

// The same update with the state words taken in another order (seg_table::options::order).
// 1: the four min / max words first. Each starts with a plain load the thread waits for, so the
//    state line is in the cache before the sums are sent as atomics.
// 2: the four sums first (atomics nothing waits for), then the min / max words.
template <int NS>
__device__ __forceinline__ void fold_raw_ordered(
  uint64_t* st, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint32_t order)
{
  if (order == 1) {
    if (NS > 2) a_min(st + 2, b);
    if (NS > 3) a_maxf(st + 3, c);
    if (NS > 5) a_max(st + 5, a);
    if (NS > 6) a_minf(st + 6, c);
    a_add(st + 0, 1);
    if (NS > 1) a_add(st + 1, a);
    if (NS > 4) a_addf(st + 4, d);
    if (NS > 7) a_add(st + 7, b);
  } else if (order == 2) {
    a_add(st + 0, 1);
    if (NS > 1) a_add(st + 1, a);
    if (NS > 4) a_addf(st + 4, d);
    if (NS > 7) a_add(st + 7, b);
    if (NS > 2) a_min(st + 2, b);
    if (NS > 3) a_maxf(st + 3, c);
    if (NS > 5) a_max(st + 5, a);
    if (NS > 6) a_minf(st + 6, c);
  } else {
    fold_raw<NS>(st, a, b, c, d);
  }
}

// One state word of the canonical set, by its index J (compile time): a raw row into it, a partial
// state into it, its identity, and the value columns its operand comes from (bit c: column c).
template <int J>
__device__ __forceinline__ void fold_word(
  uint64_t* st, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  if (J == 0) a_add(st, 1);
  if (J == 1) a_add(st, a);
  if (J == 2) a_min(st, b);
  if (J == 3) a_maxf(st, c);
  if (J == 4) a_addf(st, d);
  if (J == 5) a_max(st, a);
  if (J == 6) a_minf(st, c);
  if (J == 7) a_add(st, b);
}
template <int J>
__device__ __forceinline__ void fold_partial_word(uint64_t* st, uint64_t v)
{
  if (J == 0 || J == 1 || J == 7) a_add(st, v);
  if (J == 2) a_min(st, v);
  if (J == 3) a_maxf(st, v);
  if (J == 4) a_addf(st, v);
  if (J == 5) a_max(st, v);
  if (J == 6) a_minf(st, v);
}
__device__ __forceinline__ uint64_t canon_identity(int j)
{
  return j == 2   ? I64_MAX_BITS
         : j == 3 ? F64_NINF_BITS
         : j == 5 ? I64_MIN_BITS
         : j == 6 ? f64_min_identity()
                  : 0;
}
// Value columns the state words [first, first + count) read.
__host__ __device__ constexpr uint32_t canon_columns(int first, int count)
{
  uint32_t m = 0;
  for (int j = first; j < first + count; ++j) {
    if (CANON_COL[j] >= 0) m |= 1u << CANON_COL[j];
  }
  return m;
}

// A partial state `v` (NS words) into the state words at `st`.
template <int NS>
__device__ inline void fold_partial(uint64_t* st, uint64_t const* v)
{
  if (AGG_R1_ABLATE & 256) {
    st[0] += v[0];
  } else if (!(AGG_R1_ABLATE & 1)) {
    a_add(st + 0, v[0]);
  }
  if (NS > 1 && !(AGG_R1_ABLATE & 2)) a_add(st + 1, v[1]);
  if (NS > 2 && !(AGG_R1_ABLATE & 4)) a_min(st + 2, v[2]);
  if (NS > 3 && !(AGG_R1_ABLATE & 8)) a_maxf(st + 3, v[3]);
  if (NS > 4 && !(AGG_R1_ABLATE & 16)) a_addf(st + 4, v[4]);
  if (NS > 5 && !(AGG_R1_ABLATE & 32)) a_max(st + 5, v[5]);
  if (NS > 6 && !(AGG_R1_ABLATE & 64)) a_minf(st + 6, v[6]);
  if (NS > 7 && !(AGG_R1_ABLATE & 128)) a_add(st + 7, v[7]);
}

template <int NS>
__device__ inline void set_identity(uint64_t* st)
{
  st[0] = 0;
  if (NS > 1) st[1] = 0;
  if (NS > 2) st[2] = I64_MAX_BITS;
  if (NS > 3) st[3] = F64_NINF_BITS;
  if (NS > 4) st[4] = 0;  // +0.0
  if (NS > 5) st[5] = I64_MIN_BITS;
  if (NS > 6) st[6] = f64_min_identity();
  if (NS > 7) st[7] = 0;
}

// Compile-time profiling switches (wrong results unless 0): AGG_PRE2_ABLATE bit j drops state
// word j from the v2 block-local kernel's row loop; AGG_PRE2_ADDF 2 / 3 replaces its f64 sum by a
// 64-bit / native 32-bit integer add.
#ifndef AGG_PRE2_ABLATE
#define AGG_PRE2_ABLATE 0
#endif
#ifndef AGG_PRE2_ADDF
#define AGG_PRE2_ADDF 0
#endif

// One raw row into a thread-private partial state (no atomics).
template <int NS>
__device__ inline void acc_raw(uint64_t* st, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  if (!(AGG_PRE2_ABLATE & 1)) st[0] += 1;
  if (NS > 1 && !(AGG_PRE2_ABLATE & 2)) st[1] += a;
  if (NS > 2 && !(AGG_PRE2_ABLATE & 4) && static_cast<long long>(b) < static_cast<long long>(st[2]))
    st[2] = b;
  if (NS > 3 && !(AGG_PRE2_ABLATE & 8) && f64_greater(c, st[3])) st[3] = c;
  if (NS > 4 && !(AGG_PRE2_ABLATE & 16)) st[4] = as_bits(as_f64(st[4]) + as_f64(d));
  if (NS > 5 && !(AGG_PRE2_ABLATE & 32) &&
      static_cast<long long>(a) > static_cast<long long>(st[5]))
    st[5] = a;
  if (NS > 6 && !(AGG_PRE2_ABLATE & 64) && f64_less(c, st[6])) st[6] = c;
  if (NS > 7 && !(AGG_PRE2_ABLATE & 128)) st[7] += b;
}

// ---- State words in SHARED memory (block-local pre-aggregation, "v2") -------------------------
// (Later measurement, GB10, kernel only, 8 state words: the 64-bit forms below, on a pointer the
// compiler knows to be shared memory, are 25 to 60% SLOWER than the same updates through a pointer
// that may be shared or global (fold_raw in the round-1 kernels): there the generic atomic
// (ATOM.E.ADD.64 ...) comes first and the compare-and-swap loop is only its fallback. The f64 sum
// is the exception: a loop either way. See fold_split_kernel in table.cu.)
// A 64-bit atomicAdd / atomicMin / atomicMax on a shared-memory word is not one instruction: the
// compiler emits a load + compare-and-swap loop (LDS, op, ATOMS.CAST.SPIN.64, branch), and a
// pointer that may be shared or global costs a failed global atomic and an address-space query
// before that loop. Every lane of a warp that hits the same word repeats the loop once per lane
// in front of it. 32-bit shared atomics are native (ATOMS.ADD, one instruction, no loop), so:
//   count      one 32-bit add (a block folds far fewer than 2^32 rows)
//   int64 sum  32-bit add of the low half; its carry joins the high half, which is only touched
//              when high half + carry is not zero (the halves are independent: the sum is exact
//              modulo 2^64 in any interleaving, and nothing reads the word before the block's
//              barrier)
//   min / max, f64 sum: as before (no native form)
__device__ __forceinline__ void sh_count(uint64_t* st)
{
  atomicAdd(reinterpret_cast<unsigned int*>(st), 1u);
}
__device__ __forceinline__ void sh_add(uint64_t* st, uint64_t v)
{
  auto* w         = reinterpret_cast<unsigned int*>(st);
  unsigned int lo = static_cast<unsigned int>(v);
  unsigned int hi = static_cast<unsigned int>(v >> 32);
  if (lo != 0) {
    unsigned int old = atomicAdd(w, lo);
    hi += static_cast<unsigned int>(old + lo) < lo ? 1u : 0u;
  }
  if (hi != 0) atomicAdd(w + 1, hi);
}
// (Measured and dropped, GB10: explicit CAS loops for the f64 sum and the integer min / max, which
// carry the value a failed CAS returned instead of reloading: 3x slower under contention than the
// compiler's load + CAST.SPIN form.)
__device__ __forceinline__ void sh_addf(uint64_t* st, uint64_t v)
{
#if AGG_PRE2_ADDF == 2  // profiling, wrong results: a 64-bit integer add in its place
  atomicAdd(reinterpret_cast<unsigned long long*>(st), static_cast<unsigned long long>(v));
#elif AGG_PRE2_ADDF == 3  // profiling, wrong results: a native 32-bit add in its place
  atomicAdd(reinterpret_cast<unsigned int*>(st), static_cast<unsigned int>(v >> 32));
#else
  atomicAdd(reinterpret_cast<double*>(st), as_f64(v));
#endif
}
__device__ __forceinline__ void sh_min(uint64_t* st, uint64_t v) { a_min(st, v); }
__device__ __forceinline__ void sh_max(uint64_t* st, uint64_t v) { a_max(st, v); }

// One raw row / one partial state into state words in shared memory.
template <int NS>
__device__ __forceinline__ void sh_fold_raw(
  uint64_t* st, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  // AGG_PRE2_ABLATE (compile-time profiling mask, wrong results): bit j drops state word j.
  if (!(AGG_PRE2_ABLATE & 1)) sh_count(st + 0);
  if (NS > 1 && !(AGG_PRE2_ABLATE & 2)) sh_add(st + 1, a);
  if (NS > 2 && !(AGG_PRE2_ABLATE & 4)) sh_min(st + 2, b);
  if (NS > 3 && !(AGG_PRE2_ABLATE & 8)) a_maxf(st + 3, c);
  if (NS > 4 && !(AGG_PRE2_ABLATE & 16)) sh_addf(st + 4, d);
  if (NS > 5 && !(AGG_PRE2_ABLATE & 32)) sh_max(st + 5, a);
  if (NS > 6 && !(AGG_PRE2_ABLATE & 64)) a_minf(st + 6, c);
  if (NS > 7 && !(AGG_PRE2_ABLATE & 128)) sh_add(st + 7, b);
}
template <int NS>
__device__ __forceinline__ void sh_fold_partial(uint64_t* st, uint64_t const* v)
{
  sh_add(st + 0, v[0]);
  if (NS > 1) sh_add(st + 1, v[1]);
  if (NS > 2) sh_min(st + 2, v[2]);
  if (NS > 3) a_maxf(st + 3, v[3]);
  if (NS > 4) sh_addf(st + 4, v[4]);
  if (NS > 5) sh_max(st + 5, v[5]);
  if (NS > 6) a_minf(st + 6, v[6]);
  if (NS > 7) sh_add(st + 7, v[7]);
}

// Slot index of a key in a block-local table of 2^bits slots. Independent of the table's own hash
// (fmix64: two 64-bit multiplies), so one 32-bit multiply is enough.
__device__ __forceinline__ uint32_t local_hash(uint64_t k)
{
  return (static_cast<uint32_t>(k) ^ static_cast<uint32_t>(k >> 32)) * 0x9E3779B1u;
}

// In-place compaction. `counter` reserves a range of output positions per block; a thread's
// position inside the range is the number of flagged threads before it in the block.
constexpr int CBLOCK = 256;

__device__ inline uint64_t block_position(bool flag, unsigned long long* counter)
{
  __shared__ unsigned int warp_count[CBLOCK / 32];
  __shared__ unsigned long long block_base;
  unsigned int m    = __ballot_sync(0xffffffffu, flag);
  unsigned int lane = threadIdx.x & 31;
  unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) warp_count[warp] = __popc(m);
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int total = 0;
    for (unsigned int w = 0; w < CBLOCK / 32; ++w)
      total += warp_count[w];
    block_base = total != 0 ? atomicAdd(counter, static_cast<unsigned long long>(total)) : 0;
  }
  __syncthreads();
  unsigned int before = __popc(m & ((1u << lane) - 1u));
  for (unsigned int w = 0; w < warp; ++w)
    before += warp_count[w];
  return block_base + before;
}

inline uint32_t grid(uint64_t n)
{
  uint64_t g = (n + BLOCK - 1) / BLOCK;
  if (g > 0x7fffffffull) throw std::runtime_error("launch too large");
  return static_cast<uint32_t>(g);
}

inline void check(cudaError_t e, char const* what)
{
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

inline identities make_identities(state_spec const& spec)
{
  identities id{};
  auto bits = [](double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
  };
  for (int j = 0; j < spec.num_state; ++j) {
    switch (spec.state[j].op) {
      case op_t::COUNT:
      case op_t::SUM_I64: id.v[j] = 0; break;
      case op_t::SUM_F64: id.v[j] = bits(0.0); break;
      case op_t::MIN_I64:
        id.v[j] = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        break;
      case op_t::MAX_I64:
        id.v[j] = static_cast<uint64_t>(std::numeric_limits<int64_t>::min());
        break;
      case op_t::MIN_F64: id.v[j] = f64_min_identity_host(); break;
      case op_t::MAX_F64: id.v[j] = bits(-std::numeric_limits<double>::infinity()); break;
    }
  }
  return id;
}

inline kop raw_op(op_t op)
{
  switch (op) {
    case op_t::COUNT: return kop::ADD_ONE;
    case op_t::SUM_I64: return kop::ADD_I64;
    case op_t::MIN_I64: return kop::MIN_I64;
    case op_t::MAX_I64: return kop::MAX_I64;
    case op_t::SUM_F64: return kop::ADD_F64;
    case op_t::MIN_F64: return kop::MIN_F64;
    case op_t::MAX_F64: return kop::MAX_F64;
  }
  return kop::ADD_ONE;
}

// Merging partial states: a partial count is added, everything else keeps its operation.
inline kop merge_op(op_t op) { return op == op_t::COUNT ? kop::ADD_I64 : raw_op(op); }

}  // namespace tdev
}  // namespace sirius::agg_table::detail

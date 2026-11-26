// sirius::agg_table::table - the engine-independent front of the segmented aggregation table
// (detail/seg_table.*, the kernels of experimental/agg_bench). This file adds what the prototype
// had in its benchmark driver and its process-wide SQL layer (sql.cu), per table:
// types, NULLs, the SQL-correctness layer, fold tickets and partial rows.
//
// STATE LAYOUT. A slot is `[key, word 0 .. word NS-1]`, 8 bytes each, NS <= 8. The key is the
// column's bit pattern zero-extended to 64 bits. Word 0 is always count(*). The other words
// follow from the aggregates, in one of two layouts chosen by the first fold and kept for the
// table's life:
//   compact   one word per distinct (operation, value column) of the SUM / MIN / MAX aggregates.
//             COUNT(col) reads word 0. Used when the first batch has no NULL in an aggregated
//             column. A later row with such a NULL is not folded: it is returned in `unapplied`
//             (partial rows with a NULL state, or whose COUNT(col) differs from their COUNT(*),
//             likewise). When the words can be placed on the prototype's canonical set
//             [count, sum_i64(a), min_i64(b), max_f64(c), sum_f64(d), max_i64(a), min_f64(c),
//             sum_i64(b)] without leaving more than half of it unused, the table runs the
//             specialised kernels (block-local pre-aggregation, strips, hot keys); unused words
//             read the key column and are ignored. Otherwise the generic kernels (operations as
//             launch parameters): one row per thread, or - a batch whose sampled keys come in
//             long runs or have hot keys - runs combined in registers / block-local accumulators
//             for the hot keys (detail/seg_gen.cu; seg_table::options::gen_combine).
//   nullable  as compact plus one word per aggregated value column that counts its non-null
//             rows (COUNT(col) is that word; SUM / MIN / MAX are NULL when it is zero); a NULL
//             value is folded as the operation's identity. Always the generic kernels. Chosen
//             when the first batch has a NULL in an aggregated column and the words fit.
// Values are widened to 8 bytes (integers and decimals sign-extended, FLOAT32 to FLOAT64) in a
// scratch copy; a 64-bit column without NULLs is read in place.
//
// SQL CORRECTNESS (the prototype's sql.{hpp,cu}, made per table).
//   Reserved keys. The table cannot hold the key patterns -1 and -2, and has no NULL key. Rows
//     with such a key are skipped by every kernel (a NULL key is handed over as -1) and folded
//     into three side groups of the table object (-1, -2, NULL), which flush() emits as rows.
//     Narrow keys are zero-extended, so only their NULL group exists.
//   Float MIN / MAX use one total order (NaN greatest, -0.0 below +0.0), in the kernels.
//   Integer SUM. An int64 state word holds the sum modulo 2^64. A running bound per sum advances
//     with every batch by rows x max |value|: from the batch's `column_bound` (no pass), else from
//     a pass over the column. While the bound stays below 2^62 the word is the sum. Beyond, the
//     batch's rounded high halves (v + 2^31) >> 32 are summed per key in sorted runs outside the
//     table and flush() recovers the 128-bit sum H * 2^32 + int64(W - H * 2^32), emitted as
//     DECIMAL128. A SUM whose partial type cannot carry that (DECIMAL32 input: DECIMAL64) makes
//     the fold hand the whole batch back instead.
//
// PARTIAL ROWS (flush / fold_partial; the engine's merge re-aggregates them with SUM for counts
// and sums, MIN, MAX): key, then per aggregate one column - see partial_schema in the header.
//   count_star -> word 0;  count(col) -> word 0 (compact) or the column's non-null word
//   sum -> its word, or the 128-bit sum; NULL when the column's non-null word is zero
//   min / max -> its word narrowed to the input type; NULL likewise
// fold_partial maps the columns back: counts and sums are added, MIN / MAX merged; a non-null
// word without a COUNT(col) aggregate gets 1 for a non-null state (only "is it zero" is read).
//
// CONCURRENCY. `_tmx` (shared: every fold, for its whole duration; exclusive: creating the table
// on the first fold, flush / emit). Inside, the segmented table has its own reader-writer lock:
// fold launches hold it shared from launch to stream synchronisation, a growth step takes it
// exclusively (so growth waits for the launches in flight and blocks new ones; a fold that saw
// the old capacity re-applies its marked rows after growth). `_m` guards the host-side sum
// bounds and the high-half runs. The side groups are device words updated with atomics.
//
// EXACTLY ONCE. fold() runs in stages recorded in the ticket's progress:
//   prepare   normalise columns, reserve all fold scratch, decide the sum bounds. Nothing of the
//             table is changed; an exception leaves the ticket `not_started` (a replay can only
//             make a sum bound more conservative).
//   table     the fold kernels. No allocation: growth that cannot allocate reports "cannot grow"
//             and the rows come back unapplied. The ticket is `mutated` from here on.
//   side      reserved / NULL keys into the side groups (no allocation).
//   wide      the batch's high halves into a run (allocates; each sum is recorded when done).
//   result    the gather map of unapplied rows, from the fold's bitmap (allocates).
// A replay with a `mutated` ticket resumes at the first stage not recorded; with `done` it
// rebuilds the result from the bitmap. A failure inside the table stage (a CUDA error) is not
// recoverable and is reported as std::runtime_error, never as an out-of-memory.
//
// DEVICE BINDING. Everything is allocated from the resource given to create(); a call made with
// another device current throws. No state is shared between tables.
#include "agg_table.hpp"
#include "detail/seg_table.hpp"
#include "detail/table_dev.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

namespace sirius::agg_table {

using namespace detail;
using namespace detail::tdev;

namespace {

using ull  = unsigned long long;
using u128 = unsigned __int128;

// ---- Configuration: the prototype's measured defaults (compile-time; no environment) ----------
constexpr uint32_t SEGMENT_BITS = 6;  // 64 segments (seg_table::options::sbits)
// Growth without a group count: the prototype's blind mode (method_adaptive.cu, AGG_AD_BLIND=1
// with AGG_AD_BLIND_GROW=2 and AGG_AD_BLIND_CURVE=1).
constexpr double BLIND_GROWTH       = 2.0;           // AGG_AD_BLIND_GROWTH
constexpr double BLIND_TRUST        = 1e9;           // AGG_AD_BLIND_TRUST
constexpr uint64_t BLIND_REPEATS    = 16384;         // AGG_AD_BLIND_REPEATS
constexpr uint32_t BLIND_GROW_SEGS  = 8;             // AGG_AD_BLIND_GROW_SEGS
constexpr uint64_t BLIND_GROW_SMALL = 256ull << 20;  // AGG_AD_BLIND_GROW_SMALL
constexpr double BLIND_MODEL_TOL    = 0.05;          // AGG_AD_BLIND_MODEL_TOL
constexpr double BLIND_JUMP_CAP     = 64.0;          // AGG_AD_BLIND_JUMP_CAP
constexpr double BLIND_CURVE_X      = 8.0;           // AGG_AD_BLIND_CURVE_X (the row prior)
constexpr double BLIND_CURVE_MARGIN = 1.125;         // AGG_AD_BLIND_CURVE_MARGIN
constexpr double BLIND_CURVE_CAP    = 0.0;           // AGG_AD_BLIND_CURVE_CAP: off
// The sum bound (sql.cu): whole batches stop at 2^62, batches taken on their statistics at 2^61,
// a widened row adds at most 2^31 to the low part of its group's sum.
constexpr uint64_t NARROW_MAX = (1ull << 62) - 1, STAT_MAX = 1ull << 61, HALF = 1ull << 31;
constexpr int SIDE_SLOTS                 = 3;  // key -1, key -2, NULL
constexpr uint32_t KBLOCK                = 256;
constexpr std::size_t MAX_POOLED_SCRATCH = 8;
constexpr std::size_t MAX_RUNS           = 8;

// ---- Types -----------------------------------------------------------------------------------
enum class wt : uint8_t { I8, I16, I32, I64, F32, F64, I128, NONE };

wt storage_of(cudf::type_id id)
{
  using t = cudf::type_id;
  switch (id) {
    case t::INT8:
    case t::UINT8: return wt::I8;
    case t::INT16:
    case t::UINT16: return wt::I16;
    case t::INT32:
    case t::UINT32:
    case t::DECIMAL32:
    case t::TIMESTAMP_DAYS:
    case t::DURATION_DAYS: return wt::I32;
    case t::INT64:
    case t::UINT64:
    case t::DECIMAL64:
    case t::TIMESTAMP_SECONDS:
    case t::TIMESTAMP_MILLISECONDS:
    case t::TIMESTAMP_MICROSECONDS:
    case t::TIMESTAMP_NANOSECONDS:
    case t::DURATION_SECONDS:
    case t::DURATION_MILLISECONDS:
    case t::DURATION_MICROSECONDS:
    case t::DURATION_NANOSECONDS: return wt::I64;
    case t::FLOAT32: return wt::F32;
    case t::FLOAT64: return wt::F64;
    case t::DECIMAL128: return wt::I128;
    default: return wt::NONE;
  }
}
uint32_t width_of(wt s)
{
  switch (s) {
    case wt::I8: return 1;
    case wt::I16: return 2;
    case wt::I32:
    case wt::F32: return 4;
    case wt::I64:
    case wt::F64: return 8;
    case wt::I128: return 16;
    default: return 0;
  }
}
bool is_signed_int(cudf::type_id id)
{
  using t = cudf::type_id;
  return id == t::INT8 || id == t::INT16 || id == t::INT32 || id == t::INT64;
}
bool is_unsigned_int(cudf::type_id id)
{
  using t = cudf::type_id;
  return id == t::UINT8 || id == t::UINT16 || id == t::UINT32 || id == t::UINT64;
}
bool is_float(cudf::type_id id)
{
  return id == cudf::type_id::FLOAT32 || id == cudf::type_id::FLOAT64;
}
bool is_dec(cudf::type_id id)
{
  return id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64;
}
bool is_time(cudf::type_id id)
{
  auto s = storage_of(id);
  return (s == wt::I32 || s == wt::I64) && !is_signed_int(id) && !is_unsigned_int(id) &&
         !is_dec(id);
}
bool key_type_ok(cudf::type_id id)
{
  return is_signed_int(id) || is_unsigned_int(id) || is_time(id) || is_dec(id);
}
bool sum_type_ok(cudf::type_id id) { return is_signed_int(id) || is_dec(id) || is_float(id); }
bool minmax_type_ok(cudf::type_id id)
{
  return is_signed_int(id) || is_dec(id) || is_float(id) || is_time(id);
}
bool count_type_ok(cudf::type_id id) { return minmax_type_ok(id); }
// A SUM whose partial type can carry a 128-bit sum.
bool sum_can_widen(cudf::type_id id) { return is_signed_int(id) || id == cudf::type_id::DECIMAL64; }

cudf::data_type partial_type(agg_spec const& a, bool wide)
{
  using t = cudf::type_id;
  switch (a.kind) {
    case agg_kind::count_star:
    case agg_kind::count: return cudf::data_type{t::INT64};
    case agg_kind::sum:
      if (is_signed_int(a.input.id()))
        return wide ? cudf::data_type{t::DECIMAL128, 0} : cudf::data_type{t::INT64};
      if (a.input.id() == t::DECIMAL32) return cudf::data_type{t::DECIMAL64, a.input.scale()};
      if (a.input.id() == t::DECIMAL64) return cudf::data_type{t::DECIMAL128, a.input.scale()};
      return a.input;
    default: return a.input;
  }
}

// ---- Layout ----------------------------------------------------------------------------------
enum class wsrc : uint8_t { ONE, VALUE, VALID, DUMMY };
struct word_def {
  op_t op  = op_t::COUNT;
  wsrc src = wsrc::DUMMY;
  int vcol = -1;  // value column (VALUE, VALID)
  int agg  = -1;  // an aggregate whose partial column feeds this word (VALUE: its own;
                  // VALID: a COUNT(col) when there is one, else any aggregate of the column)
  bool agg_is_count = false;  // VALID: `agg` is a COUNT(col)
};
struct layout {
  bool fits     = false;
  bool canon    = false;
  bool nullable = false;
  int ns        = 0;
  word_def w[MAX_STATE];
  std::vector<int> agg_word;   // per aggregate: the word with its value
  std::vector<int> agg_valid;  // per aggregate: the word that is zero when it is NULL (-1: never)
  state_spec spec{};
};

op_t op_of(agg_spec const& a)
{
  bool f = is_float(a.input.id());
  switch (a.kind) {
    case agg_kind::sum: return f ? op_t::SUM_F64 : op_t::SUM_I64;
    case agg_kind::min: return f ? op_t::MIN_F64 : op_t::MIN_I64;
    case agg_kind::max: return f ? op_t::MAX_F64 : op_t::MAX_I64;
    default: return op_t::COUNT;
  }
}

op_t canon_op(int j)
{
  switch (CANON_OP[j]) {
    case kop::ADD_ONE: return op_t::COUNT;
    case kop::ADD_I64: return op_t::SUM_I64;
    case kop::ADD_F64: return op_t::SUM_F64;
    case kop::MIN_I64: return op_t::MIN_I64;
    case kop::MAX_I64: return op_t::MAX_I64;
    case kop::MIN_F64: return op_t::MIN_F64;
    default: return op_t::MAX_F64;
  }
}

// Places the words 1.. of `L` on the canonical set: slot[i] for word i, or false. Minimises the
// highest slot used.
bool place_canonical(layout const& L, int slot[MAX_STATE])
{
  int best_top = MAX_STATE + 1;
  int cur[MAX_STATE];
  int best[MAX_STATE];
  int colv[NUM_VALUE_COLUMNS];
  bool used[MAX_STATE] = {true};
  for (int& c : colv)
    c = -1;
  auto rec = [&](auto&& self, int i, int top) -> void {
    if (top >= best_top) return;
    if (i == L.ns) {
      best_top = top;
      for (int k = 1; k < L.ns; ++k)
        best[k] = cur[k];
      return;
    }
    for (int j = 1; j < MAX_STATE; ++j) {
      if (used[j] || canon_op(j) != L.w[i].op) continue;
      int c = CANON_COL[j];
      if (colv[c] != -1 && colv[c] != L.w[i].vcol) continue;
      int saved = colv[c];
      colv[c]   = L.w[i].vcol;
      used[j]   = true;
      cur[i]    = j;
      self(self, i + 1, top > j ? top : j);
      used[j] = false;
      colv[c] = saved;
    }
  };
  rec(rec, 1, 0);
  if (best_top > MAX_STATE) return false;
  for (int k = 1; k < L.ns; ++k)
    slot[k] = best[k];
  return true;
}

layout build_layout(config const& cfg, bool nullable)
{
  layout L;
  L.nullable = nullable;
  int n      = 1;
  L.w[0]     = word_def{op_t::COUNT, wsrc::ONE, -1, -1, false};
  bool over  = false;
  auto add   = [&](word_def const& d) {
    if (n >= MAX_STATE) {
      over = true;
      return 0;
    }
    L.w[n] = d;
    return n++;
  };
  auto valid_word = [&](int vcol, int agg, bool is_count) {
    for (int i = 1; i < n; ++i) {
      if (L.w[i].src == wsrc::VALID && L.w[i].vcol == vcol) {
        if (is_count && !L.w[i].agg_is_count) {
          L.w[i].agg          = agg;
          L.w[i].agg_is_count = true;
        }
        return i;
      }
    }
    return add(word_def{op_t::SUM_I64, wsrc::VALID, vcol, agg, is_count});
  };
  for (std::size_t a = 0; a < cfg.aggs.size(); ++a) {
    auto const& ag = cfg.aggs[a];
    int word = 0, valid = -1;
    if (ag.kind == agg_kind::count) {
      if (nullable) word = valid_word(ag.value_column, static_cast<int>(a), true);
    } else if (ag.kind != agg_kind::count_star) {
      op_t op = op_of(ag);
      word    = 0;
      for (int i = 1; i < n; ++i) {
        if (L.w[i].src == wsrc::VALUE && L.w[i].op == op && L.w[i].vcol == ag.value_column)
          word = i;
      }
      if (word == 0)
        word = add(word_def{op, wsrc::VALUE, ag.value_column, static_cast<int>(a), false});
      if (nullable) valid = valid_word(ag.value_column, static_cast<int>(a), false);
    }
    L.agg_word.push_back(word);
    L.agg_valid.push_back(valid);
  }
  if (over) return L;
  L.fits = true;
  L.ns   = n;
  int slot[MAX_STATE];
  if (n == 1) {
    L.canon = true;
  } else if (!nullable && place_canonical(L, slot)) {
    int top = 0;
    for (int i = 1; i < n; ++i)
      top = slot[i] > top ? slot[i] : top;
    int cns = top <= 1 ? 2 : top <= 3 ? 4 : 8;
    // (Unmeasured rule: the specialised kernels update every word of their set, so a set that is
    // mostly unused words is left to the generic kernel.)
    if (2 * n >= cns) {
      layout C = L;
      C.canon  = true;
      C.ns     = cns;
      for (int j = 1; j < cns; ++j)
        C.w[j] = word_def{canon_op(j), wsrc::DUMMY, -1, -1, false};
      for (int i = 1; i < n; ++i)
        C.w[slot[i]] = L.w[i];
      for (auto& w : C.agg_word) {
        if (w != 0) w = slot[w];
      }
      L = C;
    }
  }
  L.spec.num_aggs       = L.ns;
  L.spec.num_state      = L.ns;
  L.spec.num_value_cols = L.canon ? (L.ns == 1 ? 0 : L.ns == 2 ? 1 : L.ns == 4 ? 3 : 4) : 0;
  for (int j = 0; j < L.ns; ++j) {
    L.spec.state[j] =
      L.canon ? state_def{canon_op(j), CANON_COL[j]} : state_def{L.w[j].op, 100 + j};
  }
  return L;
}

seg_table::options table_options(config const& cfg,
                                 std::size_t budget,
                                 uint64_t first_groups,
                                 uint32_t* seen)
{
  seg_table::options o;
  o.sbits        = SEGMENT_BITS;
  o.budget_bytes = budget;
  o.seen         = seen;
  if (cfg.groups_hint && *cfg.groups_hint != 0) {  // a trusted count: the table is pre-sized
    o.initial_groups = *cfg.groups_hint;
    o.hint_groups    = *cfg.groups_hint;
    return o;
  }
  o.initial_groups    = first_groups;  // (0: the smallest table; after a flush: what it held)
  o.dense_at_budget   = true;          // AGG_AD_BLIND_DENSE
  o.growth            = BLIND_GROWTH;
  o.growth_trust      = BLIND_TRUST;
  o.model_min_repeats = BLIND_REPEATS;
  o.grow_segs         = BLIND_GROW_SEGS;
  o.growth_by_groups  = true;
  o.ladder_anchor     = true;
  o.grow_small_bytes  = BLIND_GROW_SMALL;
  o.model_tol         = BLIND_MODEL_TOL;
  o.jump_cap          = BLIND_JUMP_CAP;
  o.retry_ahead       = true;  // AGG_AD_BLIND_RETRY
  o.curve_x           = BLIND_CURVE_X;
  o.curve_margin      = BLIND_CURVE_MARGIN;
  o.curve_rows        = cfg.rows_hint ? static_cast<double>(*cfg.rows_hint) : 0.0;
  o.curve_cap         = BLIND_CURVE_CAP;
  return o;
}

// ---- Kernels ---------------------------------------------------------------------------------
__device__ __forceinline__ bool bit_at(uint32_t const* mask, int64_t i)
{
  return (mask[i >> 5] >> (i & 31)) & 1u;
}

// A value widened to a state word: integers sign-extended, FLOAT32 as FLOAT64 bits, a 128-bit
// integer's low word.
__device__ __forceinline__ uint64_t load_value(uint8_t const* p, wt s, uint64_t i)
{
  switch (s) {
    case wt::I8:
      return static_cast<uint64_t>(static_cast<int64_t>(reinterpret_cast<int8_t const*>(p)[i]));
    case wt::I16:
      return static_cast<uint64_t>(static_cast<int64_t>(reinterpret_cast<int16_t const*>(p)[i]));
    case wt::I32:
      return static_cast<uint64_t>(static_cast<int64_t>(reinterpret_cast<int32_t const*>(p)[i]));
    case wt::F32: return as_bits(static_cast<double>(reinterpret_cast<float const*>(p)[i]));
    case wt::I128: return reinterpret_cast<uint64_t const*>(p)[2 * i];
    default: return reinterpret_cast<uint64_t const*>(p)[i];
  }
}
// A key: the bit pattern, zero-extended.
__device__ __forceinline__ uint64_t load_key(uint8_t const* p, wt s, uint64_t i)
{
  switch (s) {
    case wt::I8: return p[i];
    case wt::I16: return reinterpret_cast<uint16_t const*>(p)[i];
    case wt::I32: return reinterpret_cast<uint32_t const*>(p)[i];
    default: return reinterpret_cast<uint64_t const*>(p)[i];
  }
}

// dst[i] = the key of row i, or EMPTY_KEY for a NULL key and for a row that is not folded.
__global__ void norm_key_kernel(uint64_t* dst,
                                uint8_t const* src,
                                wt s,
                                uint64_t n,
                                uint32_t const* mask,
                                int64_t moff,
                                uint32_t const* punt)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bool skip = (mask != nullptr && !bit_at(mask, moff + static_cast<int64_t>(i))) ||
              (punt != nullptr && bit_at(punt, static_cast<int64_t>(i)));
  dst[i] = skip ? EMPTY_KEY : load_key(src, s, i);
}

__global__ void norm_value_kernel(uint64_t* dst,
                                  uint8_t const* src,
                                  wt s,
                                  uint64_t n,
                                  uint32_t const* mask,
                                  int64_t moff,
                                  uint64_t fill)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bool ok = mask == nullptr || bit_at(mask, moff + static_cast<int64_t>(i));
  dst[i]  = ok ? load_value(src, s, i) : fill;
}

__global__ void valid01_kernel(uint64_t* dst, uint64_t n, uint32_t const* mask, int64_t moff)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = bit_at(mask, moff + static_cast<int64_t>(i)) ? 1 : 0;
}

// Rows the compact layout cannot take: a NULL in one of the `masks`, or (partial rows) a count
// column that differs from the reference count. One thread per bitmap word.
struct punt_in {
  uint32_t const* mask[MAX_STATE];
  int64_t moff[MAX_STATE];
  int nmask;
  int64_t const* cnt[MAX_STATE];
  int ncnt;
  int64_t const* ref;
};
__global__ void punt_kernel(punt_in P, uint64_t n, uint32_t* punt, ull* count)
{
  uint64_t w = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (w * 32 >= n) return;
  uint32_t bits = 0;
  for (uint32_t b = 0; b < 32; ++b) {
    uint64_t i = w * 32 + b;
    if (i >= n) break;
    bool bad = false;
    for (int m = 0; m < P.nmask; ++m)
      bad |= !bit_at(P.mask[m], P.moff[m] + static_cast<int64_t>(i));
    for (int c = 0; c < P.ncnt; ++c)
      bad |= P.cnt[c][i] != P.ref[i];
    if (bad) bits |= 1u << b;
  }
  punt[w] = bits;
  if (bits != 0) atomicAdd(count, static_cast<ull>(__popc(bits)));
}

constexpr uint32_t SCAN_RPT = 64;
// out[0] = max |v| over an int64 column (2^63 for INT64_MIN).
__global__ void maxabs_kernel(uint64_t const* col, uint64_t n, ull* out)
{
  uint64_t base = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * SCAN_RPT;
  uint64_t m    = 0;
  for (uint32_t r = 0; r < SCAN_RPT; ++r) {
    uint64_t i = base + r;
    if (i >= n) break;
    uint64_t v = col[i];
    uint64_t a = static_cast<long long>(v) < 0 ? 0ull - v : v;
    m          = a > m ? a : m;
  }
  if (m > *out) atomicMax(out, static_cast<ull>(m));
}
// The same over a 128-bit column (NULLs skipped): out[0] = min(max |v|, 2^63); out[1] is set when
// some |v| >= 2^94 (its rounded high half would not fit the int64 high sum).
__global__ void maxabs128_kernel(
  __int128 const* col, uint64_t n, uint32_t const* mask, int64_t moff, ull* out)
{
  uint64_t base = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * SCAN_RPT;
  uint64_t m    = 0;
  bool huge     = false;
  for (uint32_t r = 0; r < SCAN_RPT; ++r) {
    uint64_t i = base + r;
    if (i >= n) break;
    if (mask != nullptr && !bit_at(mask, moff + static_cast<int64_t>(i))) continue;
    __int128 v = col[i];
    u128 a     = v < 0 ? static_cast<u128>(0) - static_cast<u128>(v) : static_cast<u128>(v);
    uint64_t c = a >= (static_cast<u128>(1) << 63) ? (1ull << 63) : static_cast<uint64_t>(a);
    m          = c > m ? c : m;
    huge |= a >= (static_cast<u128>(1) << 94);
  }
  if (m > *out) atomicMax(out, static_cast<ull>(m));
  if (huge) out[1] = 1;
}

// v = hi * 2^32 + lo with lo in [-2^31, 2^31) (sql.cu: high_half).
__host__ __device__ inline uint64_t high_half(uint64_t v)
{
  return static_cast<uint64_t>(static_cast<long long>(v) >> 32) + ((v >> 31) & 1ull);
}
__global__ void high_kernel(uint64_t* hi, uint64_t const* col, uint64_t n)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) hi[i] = high_half(col[i]);
}
__global__ void high128_kernel(
  uint64_t* hi, __int128 const* col, uint64_t n, uint32_t const* mask, int64_t moff)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bool ok = mask == nullptr || bit_at(mask, moff + static_cast<int64_t>(i));
  hi[i]   = ok ? static_cast<uint64_t>(
                 static_cast<__int128>((col[i] + (static_cast<__int128>(1) << 31)) >> 32))
               : 0;
}

// Which side group a row with a reserved key pattern belongs to (-1: none, the row is not folded).
__device__ __forceinline__ int side_slot(
  uint64_t k, uint64_t i, uint32_t const* kmask, int64_t koff, uint32_t const* punt)
{
  if (punt != nullptr && bit_at(punt, static_cast<int64_t>(i))) return -1;
  if (kmask != nullptr && !bit_at(kmask, koff + static_cast<int64_t>(i))) return 2;
  return k == EMPTY_KEY ? 0 : 1;
}

// After the fold: the high halves of the rows the table did not take are dropped, those of the
// rows with a reserved / NULL key go to their side group (hi_side[slot]).
__global__ void high_mask_kernel(uint64_t* hi,
                                 uint64_t const* key,
                                 uint64_t n,
                                 uint32_t const* unapplied,
                                 uint32_t const* punt,
                                 uint32_t const* kmask,
                                 int64_t koff,
                                 uint64_t* hi_side)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t h = hi[i];
  if (h == 0) return;
  uint64_t k = key[i];
  if (bit_at(unapplied, static_cast<int64_t>(i))) {
    hi[i] = 0;
  } else if (k + 2 < 2) {
    int s = side_slot(k, i, kmask, koff, punt);
    if (s >= 0) atomicAdd(reinterpret_cast<ull*>(hi_side + s), static_cast<ull>(h));
    hi[i] = 0;
  }
}

// One state word into a private partial state (no atomics).
__device__ __forceinline__ void acc_word(uint64_t& st, kop op, uint64_t v)
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
// The operation that merges a partial state of `op` into a state word.
__device__ __forceinline__ kop merge_kop(kop op) { return op == kop::ADD_ONE ? kop::ADD_I64 : op; }

// Side groups (sql.cu: side_fold_kernel, for any aggregate set): every thread folds its rows with
// a reserved / NULL key into private partial states, one per side slot, and adds those to the
// table's side groups once. side: SIDE_SLOTS x MAX_STATE state words, then SIDE_SLOTS "present"
// words.
constexpr uint32_t SIDE_RPT = 16;
__global__ void side_fold_kernel(flat_table::source S,
                                 identities id,
                                 uint64_t n,
                                 uint32_t const* kmask,
                                 int64_t koff,
                                 uint32_t const* punt,
                                 uint64_t* side)
{
  uint64_t acc[SIDE_SLOTS][MAX_STATE];
  bool any[SIDE_SLOTS] = {false, false, false};
  uint64_t base        = static_cast<uint64_t>(blockIdx.x) * blockDim.x * SIDE_RPT;
  for (uint32_t r = 0; r < SIDE_RPT; ++r) {
    uint64_t i = base + static_cast<uint64_t>(r) * blockDim.x + threadIdx.x;
    if (i >= n) break;
    uint64_t k = S.key.p[i * S.key.stride];
    if (k + 2 >= 2) continue;
    int q = side_slot(k, i, kmask, koff, punt);
    if (q < 0) continue;
    if (!any[q]) {
      for (int s = 0; s < S.num_state; ++s)
        acc[q][s] = id.v[s];
      any[q] = true;
    }
    for (int s = 0; s < S.num_state; ++s) {
      uint64_t v = S.op[s] == kop::ADD_ONE ? 0 : S.val[s].p[i * S.val[s].stride];
      acc_word(acc[q][s], S.op[s], v);
    }
  }
  for (int q = 0; q < SIDE_SLOTS; ++q) {
    if (!any[q]) continue;
    for (int s = 0; s < S.num_state; ++s)
      apply(side + q * MAX_STATE + s, merge_kop(S.op[s]), acc[q][s]);
    side[SIDE_SLOTS * MAX_STATE + q] = 1;
  }
}

// The rows of a dense row-layout buffer whose bit is set in `bitmap`, compacted (the prototype's
// select_marked_kernel).
__global__ void select_marked_kernel(
  uint64_t const* rows, uint64_t n, uint32_t W, uint32_t const* bitmap, uint64_t* out, ull* counter)
{
  uint64_t i   = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool flag    = i < n && ((bitmap[i >> 5] >> (i & 31)) & 1u) != 0;
  uint64_t pos = block_position(flag, counter);
  if (flag) {
    for (uint32_t j = 0; j < W; ++j)
      out[pos * W + j] = rows[i * W + j];
  }
}

// ---- Output kernels: dense rows `[key, words]` to typed columns
// ----------------------------------
__global__ void out_key_kernel(uint8_t* dst, wt s, uint64_t const* rows, uint32_t W, uint64_t n)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t k = rows[i * W];
  switch (s) {
    case wt::I8: dst[i] = static_cast<uint8_t>(k); break;
    case wt::I16: reinterpret_cast<uint16_t*>(dst)[i] = static_cast<uint16_t>(k); break;
    case wt::I32: reinterpret_cast<uint32_t*>(dst)[i] = static_cast<uint32_t>(k); break;
    default: reinterpret_cast<uint64_t*>(dst)[i] = k; break;
  }
}
// A state word narrowed to the column's storage; `hi` (I128 only, may be null): the group's high
// sum, for the 128-bit sum H * 2^32 + int64(W - H * 2^32).
__global__ void out_value_kernel(uint8_t* dst,
                                 wt s,
                                 uint64_t const* rows,
                                 uint32_t W,
                                 uint32_t word,
                                 uint64_t const* hi,
                                 uint64_t n)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t v = rows[i * W + 1 + word];
  switch (s) {
    case wt::I8: reinterpret_cast<int8_t*>(dst)[i] = static_cast<int8_t>(v); break;
    case wt::I16: reinterpret_cast<int16_t*>(dst)[i] = static_cast<int16_t>(v); break;
    case wt::I32: reinterpret_cast<int32_t*>(dst)[i] = static_cast<int32_t>(v); break;
    case wt::F32: reinterpret_cast<float*>(dst)[i] = static_cast<float>(as_f64(v)); break;
    case wt::I128: {
      uint64_t h = hi != nullptr ? hi[i] : 0;
      auto low   = static_cast<long long>(v - (h << 32));
      __int128 sum =
        (static_cast<__int128>(static_cast<long long>(h)) << 32) + static_cast<__int128>(low);
      reinterpret_cast<__int128*>(dst)[i] = sum;
      break;
    }
    default: reinterpret_cast<uint64_t*>(dst)[i] = v; break;
  }
}
// Validity words: row i is valid when its state word `word` is not zero (`word` < 0: when it is
// not row `null_row`).
__global__ void out_mask_kernel(
  uint32_t* mask, uint64_t const* rows, uint32_t W, int word, int64_t null_row, uint64_t n)
{
  uint64_t w = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (w * 32 >= n) return;
  uint32_t bits = 0;
  for (uint32_t b = 0; b < 32; ++b) {
    uint64_t i = w * 32 + b;
    if (i >= n) break;
    bool ok = word >= 0 ? rows[i * W + 1 + word] != 0 : static_cast<int64_t>(i) != null_row;
    if (ok) bits |= 1u << b;
  }
  mask[w] = bits;
}
// hi[i] = the high sum of row i's key in a sorted run (0 when it has none).
__global__ void run_lookup_kernel(uint64_t* hi,
                                  uint64_t const* rows,
                                  uint32_t W,
                                  uint64_t n,
                                  uint64_t const* rk,
                                  uint64_t const* rh,
                                  uint64_t rn)
{
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t k  = rows[i * W];
  uint64_t lo = 0, up = rn;
  while (lo < up) {
    uint64_t mid = (lo + up) >> 1;
    if (rk[mid] < k) {
      lo = mid + 1;
    } else {
      up = mid;
    }
  }
  hi[i] = lo < rn && rk[lo] == k ? rh[lo] : 0;
}

struct unapplied_fn {
  uint32_t const* a;
  uint32_t const* b;
  __device__ bool operator()(int32_t i) const
  {
    return (a != nullptr && ((a[i >> 5] >> (i & 31)) & 1u)) ||
           (b != nullptr && ((b[i >> 5] >> (i & 31)) & 1u));
  }
};
struct nonzero_fn {
  __device__ bool operator()(uint64_t h) const { return h != 0; }
};

uint32_t grid_for(uint64_t n, uint64_t per_thread = 1)
{
  uint64_t threads = (n + per_thread - 1) / per_thread;
  uint64_t g       = (threads + KBLOCK - 1) / KBLOCK;
  if (g > 0x7fffffffull) throw std::runtime_error("agg_table: launch too large");
  return static_cast<uint32_t>(g == 0 ? 1 : g);
}

void sync(::cuda::stream_ref stream)
{
  check(cudaStreamSynchronize(stream.get()), "agg_table: sync");
}

// A column of a batch as the kernels read it.
struct colref {
  uint8_t const* data  = nullptr;
  wt store             = wt::NONE;
  uint32_t const* mask = nullptr;  // null: no NULLs
  int64_t moff         = 0;
  uint64_t n           = 0;
  bool direct64() const { return (store == wt::I64 || store == wt::F64) && mask == nullptr; }
  uint64_t const* words() const { return reinterpret_cast<uint64_t const*>(data); }
};
colref ref_of(cudf::column_view const& c)
{
  colref r;
  r.store = storage_of(c.type().id());
  r.n     = static_cast<uint64_t>(c.size());
  r.data  = static_cast<uint8_t const*>(c.head()) +
           static_cast<std::size_t>(c.offset()) * width_of(r.store);
  if (c.has_nulls()) {
    r.mask = c.null_mask();
    r.moff = c.offset();
  }
  return r;
}

// A sorted run of (key, high sum): what the widened batches added, reduced per key (sql.cu).
struct hi_run {
  rmm::device_uvector<uint64_t> key;
  rmm::device_uvector<uint64_t> h;
};

hi_run make_run(rmm::device_uvector<uint64_t>&& k,
                rmm::device_uvector<uint64_t>&& h,
                ::cuda::stream_ref stream,
                rmm::device_async_resource_ref mr)
{
  auto pol = rmm::exec_policy(stream, mr);
  thrust::sort_by_key(pol, k.begin(), k.end(), h.begin());
  hi_run r{rmm::device_uvector<uint64_t>(k.size(), stream, mr),
           rmm::device_uvector<uint64_t>(k.size(), stream, mr)};
  auto ends = thrust::reduce_by_key(pol, k.begin(), k.end(), h.begin(), r.key.begin(), r.h.begin());
  auto n    = static_cast<std::size_t>(ends.first - r.key.begin());
  r.key.resize(n, stream);
  r.h.resize(n, stream);
  return r;
}

hi_run merge_runs(hi_run const& a,
                  hi_run const& b,
                  ::cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
{
  auto pol      = rmm::exec_policy(stream, mr);
  std::size_t n = a.key.size() + b.key.size();
  rmm::device_uvector<uint64_t> k(n, stream, mr), h(n, stream, mr);
  thrust::merge_by_key(pol,
                       a.key.begin(),
                       a.key.end(),
                       b.key.begin(),
                       b.key.end(),
                       a.h.begin(),
                       b.h.begin(),
                       k.begin(),
                       h.begin());
  hi_run r{rmm::device_uvector<uint64_t>(n, stream, mr),
           rmm::device_uvector<uint64_t>(n, stream, mr)};
  auto ends = thrust::reduce_by_key(pol, k.begin(), k.end(), h.begin(), r.key.begin(), r.h.begin());
  auto m    = static_cast<std::size_t>(ends.first - r.key.begin());
  r.key.resize(m, stream);
  r.h.resize(m, stream);
  return r;
}

// One int64 SUM word of the table.
struct sum_state {
  int word       = 0;
  int agg        = 0;  // an aggregate it serves
  bool can_widen = false;
  u128 bound     = 0;
  u128 stat_pos = 0, stat_neg = 0;
  bool widened = false;
  std::vector<hi_run> runs;
};

// Fold scratch that outlives a batch: handed from one ticket to the next.
struct scratch_pool {
  std::mutex m;
  std::vector<std::unique_ptr<seg_table::scratch>> free;
};

}  // namespace

// The record of one batch (fold_ticket::progress).
struct progress {
  void const* owner = nullptr;
  uint64_t n        = 0;
  bool partial_in   = false;
  bool refused      = false;  // nothing of the batch was folded: every row is unapplied
  int stage         = 0;      // 0 prepare done, 1 table, 2 side groups, 3 wide sums
  std::weak_ptr<scratch_pool> pool;
  std::unique_ptr<seg_table::scratch> sc;
  std::map<int, rmm::device_uvector<uint64_t>> cols;  // normalised columns, by role
  std::unique_ptr<rmm::device_uvector<uint32_t>> punt;
  uint64_t punt_count = 0;
  bool allow_preagg   = true;
  std::vector<uint8_t> widen;   // per sum: this batch adds its high halves
  std::vector<uint8_t> masked;  // ... they were masked after the fold
  std::vector<uint8_t> pushed;  // ... and their run was added
  std::vector<std::unique_ptr<rmm::device_uvector<uint64_t>>> hi;
  uint64_t left_rows = 0, left_partials = 0;
  ~progress()
  {
    if (auto p = pool.lock(); p && sc) {
      std::lock_guard<std::mutex> lk(p->m);
      if (p->free.size() < MAX_POOLED_SCRATCH) p->free.push_back(std::move(sc));
    }
  }
};

struct table::impl {
  config cfg;
  rmm::device_async_resource_ref mr;
  int device = 0;
  layout compact, nullable;
  layout const* lay = nullptr;  // decided by the first fold
  identities ident{};
  cudf::data_type key_type{cudf::type_id::EMPTY};
  wt key_store = wt::NONE;

  mutable std::shared_mutex tmx;
  std::unique_ptr<seg_table> t;
  std::atomic<std::size_t> budget{0};
  uint64_t last_groups = 0;
  std::unique_ptr<rmm::device_uvector<uint32_t>> seen;  // [0] reserved-key flag, [1] scratch
  std::atomic<bool> side_seen{false};
  std::atomic<bool> side_dirty{false};
  std::atomic<int> fail_after_mutation{0};
  seg_table::stats_t past{};  // of the tables flushed so far
  uint64_t flushes = 0;
  std::unique_ptr<rmm::device_uvector<uint64_t>>
    side;  // SIDE_SLOTS x MAX_STATE words + present flags
  std::unique_ptr<rmm::device_uvector<uint64_t>> hi_side;  // [sum][slot]
  std::shared_ptr<scratch_pool> pool = std::make_shared<scratch_pool>();

  mutable std::mutex m;
  std::vector<sum_state> sums;

  // A flush in progress (resumable): chunks that left the table and are not yet converted, and
  // the tables already built.
  bool flushing     = false;
  uint64_t next_seg = 0;
  std::unique_ptr<seg_table::chunk> pending;
  std::vector<std::unique_ptr<cudf::table>> ready;

  impl(config const& c, rmm::device_async_resource_ref r) : cfg(c), mr(r) {}

  void check_device() const
  {
    int d = -1;
    check(cudaGetDevice(&d), "agg_table: device");
    if (d != device)
      throw std::logic_error("agg_table: called with another CUDA device current than at create()");
  }

  // ---- lazy state ----
  void choose_layout(bool want_nullable, ::cuda::stream_ref stream)
  {
    if (lay != nullptr) return;
    layout const* L = want_nullable && nullable.fits ? &nullable : &compact;
    ident           = make_identities(L->spec);
    std::vector<sum_state> s;
    for (int j = 0; j < L->ns; ++j) {
      if (L->w[j].src != wsrc::VALUE || L->w[j].op != op_t::SUM_I64) continue;
      sum_state q;
      q.word      = j;
      q.agg       = L->w[j].agg;
      q.can_widen = sum_can_widen(cfg.aggs[static_cast<std::size_t>(q.agg)].input.id());
      s.push_back(std::move(q));
    }
    auto sd = std::make_unique<rmm::device_uvector<uint64_t>>(
      SIDE_SLOTS * MAX_STATE + SIDE_SLOTS, stream, mr);
    auto hs = std::make_unique<rmm::device_uvector<uint64_t>>(
      (s.empty() ? 1 : s.size()) * SIDE_SLOTS, stream, mr);
    auto sn = std::make_unique<rmm::device_uvector<uint32_t>>(2, stream, mr);
    check(cudaMemsetAsync(sn->data(), 0, 2 * sizeof(uint32_t), stream.get()), "agg_table: memset");
    side    = std::move(sd);
    hi_side = std::move(hs);
    seen    = std::move(sn);
    sums    = std::move(s);
    lay     = L;
    reset_side(stream);
  }

  void reset_side(::cuda::stream_ref stream)
  {
    uint64_t host[SIDE_SLOTS * MAX_STATE + SIDE_SLOTS] = {};
    for (int q = 0; q < SIDE_SLOTS; ++q) {
      for (int s = 0; s < lay->ns; ++s)
        host[q * MAX_STATE + s] = ident.v[s];
    }
    check(cudaMemcpyAsync(side->data(), host, sizeof host, cudaMemcpyHostToDevice, stream.get()),
          "agg_table: side");
    check(cudaMemsetAsync(hi_side->data(), 0, hi_side->size() * sizeof(uint64_t), stream.get()),
          "agg_table: memset");
    sync(stream);
    side_dirty = false;
  }

  // False: the budget is below the smallest table.
  bool ensure_table(::cuda::stream_ref stream)
  {
    if (t) return true;
    auto opt = table_options(cfg, budget.load(), last_groups, seen->data());
    if (opt.budget_bytes != 0 && seg_table::presize(lay->spec, opt).min_bytes > opt.budget_bytes)
      return false;
    t = std::make_unique<seg_table>(lay->spec, opt, stream, mr);
    return true;
  }

  // ---- a batch as the table reads it ----
  struct plan {
    raw_batch rb;
    flat_table::source src{};
    uint64_t const* key        = nullptr;
    uint32_t const* kmask      = nullptr;  // the key column's validity (NULL keys), or null
    int64_t koff               = 0;
    bool key_reserved_possible = false;
    // Per sum: the column its high halves are computed from.
    struct sum_in {
      uint64_t const* words = nullptr;  // int64 values (normalised), or
      colref wide;                      // a 128-bit partial column
      column_bound bound;
    };
    std::vector<sum_in> sum;
  };

  // A normalised column of role `role`: allocated and filled when `fill` is set, else looked up.
  template <typename Fn>
  uint64_t const* column(
    progress& tk, int role, uint64_t n, bool materialise, ::cuda::stream_ref stream, Fn&& fill)
  {
    auto it = tk.cols.find(role);
    if (it == tk.cols.end()) {
      if (!materialise)
        throw std::logic_error("agg_table: a replayed batch differs from the one first passed");
      it = tk.cols.emplace(role, rmm::device_uvector<uint64_t>(n, stream, mr)).first;
      fill(it->second.data());
    }
    return it->second.data();
  }

  // Roles of the normalised columns of a batch.
  static constexpr int ROLE_KEY = 0;
  static int role_value(int vcol, int kind)
  {
    return 16 + vcol * 8 + kind;
  }  // kind: fill class / valid

  uint64_t fill_of(int word) const { return lay->nullable ? ident.v[word] : 0ull; }

  // Raw rows. `materialise`: run the normalisation (prepare); else only rebuild the pointers.
  plan plan_raw(progress& tk,
                cudf::column_view const& keys,
                cudf::table_view const& values,
                std::vector<column_bound> const& bounds,
                bool materialise,
                ::cuda::stream_ref stream)
  {
    layout const& L = *lay;
    uint64_t n      = static_cast<uint64_t>(keys.size());
    plan P;
    // Compact layout: rows with a NULL in an aggregated column are not folded.
    std::vector<int> vcols;
    for (auto const& a : cfg.aggs) {
      if (a.kind == agg_kind::count_star) continue;
      bool have = false;
      for (int v : vcols)
        have |= v == a.value_column;
      if (!have) vcols.push_back(a.value_column);
    }
    if (!L.nullable) {
      punt_in pin{};
      for (int v : vcols) {
        colref c = ref_of(values.column(v));
        if (c.mask != nullptr) {
          pin.mask[pin.nmask] = c.mask;
          pin.moff[pin.nmask] = c.moff;
          ++pin.nmask;
        }
      }
      if (pin.nmask != 0) make_punt(tk, pin, n, materialise, stream);
    }
    plan_key(tk, P, keys, n, materialise, stream);
    P.src.key                                    = {P.key, 1};
    P.src.num_state                              = L.ns;
    uint64_t const* canon_col[NUM_VALUE_COLUMNS] = {P.key, P.key, P.key, P.key};
    for (int j = 0; j < L.ns; ++j) {
      word_def const& w = L.w[j];
      P.src.op[j]       = raw_op(L.spec.state[j].op);
      P.src.val[j]      = {P.key, 1};
      if (w.src == wsrc::ONE) {
        P.src.op[j] = kop::ADD_ONE;
      } else if (w.src == wsrc::VALUE) {
        colref c   = ref_of(values.column(w.vcol));
        uint64_t f = fill_of(j);
        uint64_t const* p;
        if (c.direct64()) {
          p = c.words();
        } else {
          // (One copy per column and fill value: the classes are the identities of the operations.)
          int kind = w.op == op_t::SUM_I64 || w.op == op_t::SUM_F64 || !L.nullable ? 0
                     : w.op == op_t::MIN_I64                                       ? 1
                     : w.op == op_t::MAX_I64                                       ? 2
                     : w.op == op_t::MIN_F64                                       ? 3
                                                                                   : 4;
          p = column(tk, role_value(w.vcol, kind), n, materialise, stream, [&](uint64_t* dst) {
            norm_value_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(
              dst, c.data, c.store, n, c.mask, c.moff, f);
            check(cudaGetLastError(), "agg_table: norm_value_kernel");
          });
        }
        P.src.val[j] = {p, 1};
        if (L.canon) canon_col[CANON_COL[j]] = p;
      } else if (w.src == wsrc::VALID) {
        colref c = ref_of(values.column(w.vcol));
        if (c.mask == nullptr) {
          P.src.op[j] = kop::ADD_ONE;
        } else {
          uint64_t const* p =
            column(tk, role_value(w.vcol, 7), n, materialise, stream, [&](uint64_t* dst) {
              valid01_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(dst, n, c.mask, c.moff);
              check(cudaGetLastError(), "agg_table: valid01_kernel");
            });
          P.src.val[j] = {p, 1};
        }
      }
    }
    if (L.canon) {
      for (int j = 1; j < L.ns; ++j)
        P.src.val[j] = {canon_col[CANON_COL[j]], 1};
    }
    P.rb.rows = n;
    P.rb.key  = P.key;
    for (int c = 0; c < NUM_VALUE_COLUMNS; ++c)
      P.rb.col[c] = canon_col[c];
    for (auto const& q : sums) {
      plan::sum_in in;
      in.words = P.src.val[q.word].p;
      auto v   = static_cast<std::size_t>(L.w[q.word].vcol);
      if (v < bounds.size()) in.bound = bounds[v];
      P.sum.push_back(in);
    }
    return P;
  }

  void make_punt(
    progress& tk, punt_in const& pin, uint64_t n, bool materialise, ::cuda::stream_ref stream)
  {
    if (!materialise) {
      if (!tk.punt)
        throw std::logic_error("agg_table: a replayed batch differs from the one first passed");
      return;
    }
    if (!tk.punt)
      tk.punt =
        std::make_unique<rmm::device_uvector<uint32_t>>(seg_table::bitmap_words(n), stream, mr);
    auto* cnt = reinterpret_cast<ull*>(tk.sc->counters.data());
    check(cudaMemsetAsync(cnt, 0, sizeof(ull), stream.get()), "agg_table: memset");
    punt_kernel<<<grid_for(n, 32), KBLOCK, 0, stream.get()>>>(pin, n, tk.punt->data(), cnt);
    check(cudaGetLastError(), "agg_table: punt_kernel");
    uint64_t c = 0;
    check(cudaMemcpyAsync(&c, cnt, sizeof c, cudaMemcpyDeviceToHost, stream.get()),
          "agg_table: read");
    sync(stream);
    tk.punt_count = c;
  }

  void plan_key(progress& tk,
                plan& P,
                cudf::column_view const& keys,
                uint64_t n,
                bool materialise,
                ::cuda::stream_ref stream)
  {
    colref k                = ref_of(keys);
    P.kmask                 = k.mask;
    P.koff                  = k.moff;
    bool const punted       = tk.punt && tk.punt_count != 0;
    P.key_reserved_possible = k.store == wt::I64;
    if (k.store == wt::I64 && k.mask == nullptr && !punted) {
      P.key = k.words();
    } else {
      uint32_t const* pm = punted ? tk.punt->data() : nullptr;
      P.key              = column(tk, ROLE_KEY, n, materialise, stream, [&](uint64_t* dst) {
        norm_key_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(
          dst, k.data, k.store, n, k.mask, k.moff, pm);
        check(cudaGetLastError(), "agg_table: norm_key_kernel");
      });
    }
  }

  // Partial rows (partial_schema): column 0 the key, column 1 + a the state of aggregate a.
  plan plan_partial(progress& tk,
                    cudf::table_view const& partial,
                    bool materialise,
                    ::cuda::stream_ref stream)
  {
    layout const& L = *lay;
    uint64_t n      = static_cast<uint64_t>(partial.num_rows());
    plan P;
    auto agg_col = [&](int a) { return ref_of(partial.column(1 + a)); };
    // The reference count of a compact row (word 0): COUNT(*) if there is one, else a COUNT(col).
    int ref_agg = -1;
    for (std::size_t a = 0; a < cfg.aggs.size() && ref_agg < 0; ++a) {
      if (cfg.aggs[a].kind == agg_kind::count_star) ref_agg = static_cast<int>(a);
    }
    for (std::size_t a = 0; a < cfg.aggs.size() && ref_agg < 0; ++a) {
      if (cfg.aggs[a].kind == agg_kind::count) ref_agg = static_cast<int>(a);
    }
    if (!L.nullable) {
      punt_in pin{};
      for (std::size_t a = 0; a < cfg.aggs.size(); ++a) {
        colref c = agg_col(static_cast<int>(a));
        if (c.mask != nullptr) {
          pin.mask[pin.nmask] = c.mask;
          pin.moff[pin.nmask] = c.moff;
          ++pin.nmask;
        }
        if (cfg.aggs[a].kind == agg_kind::count && static_cast<int>(a) != ref_agg) {
          pin.cnt[pin.ncnt++] = reinterpret_cast<int64_t const*>(c.data);
        }
      }
      if (ref_agg >= 0) pin.ref = reinterpret_cast<int64_t const*>(agg_col(ref_agg).data);
      if (pin.nmask != 0 || pin.ncnt != 0) make_punt(tk, pin, n, materialise, stream);
    }
    plan_key(tk, P, partial.column(0), n, materialise, stream);
    P.src.key       = {P.key, 1};
    P.src.num_state = L.ns;
    for (int j = 0; j < L.ns; ++j) {
      word_def const& w = L.w[j];
      P.src.op[j]       = merge_op(L.spec.state[j].op);
      P.src.val[j]      = {P.key, 1};  // (an unused word of a canonical set: ignored)
      auto take         = [&](int a, uint64_t fill, int kind) {
        colref c = agg_col(a);
        if (c.direct64()) return c.words();
        return column(tk, role_value(a, kind), n, materialise, stream, [&](uint64_t* dst) {
          norm_value_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(
            dst, c.data, c.store, n, c.mask, c.moff, fill);
          check(cudaGetLastError(), "agg_table: norm_value_kernel");
        });
      };
      if (w.src == wsrc::ONE) {
        if (ref_agg >= 0) {
          P.src.op[j]  = kop::ADD_I64;
          P.src.val[j] = {take(ref_agg, 0, 0), 1};
        } else {
          P.src.op[j] = kop::ADD_ONE;
        }
      } else if (w.src == wsrc::VALUE) {
        P.src.val[j] = {take(w.agg, fill_of(j), 0), 1};
      } else if (w.src == wsrc::VALID) {
        colref c    = agg_col(w.agg);
        P.src.op[j] = kop::ADD_I64;
        if (w.agg_is_count) {
          P.src.val[j] = {take(w.agg, 0, 0), 1};
        } else if (c.mask == nullptr) {
          P.src.op[j] = kop::ADD_ONE;
        } else {
          uint64_t const* p =
            column(tk, role_value(w.agg, 7), n, materialise, stream, [&](uint64_t* dst) {
              valid01_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(dst, n, c.mask, c.moff);
              check(cudaGetLastError(), "agg_table: valid01_kernel");
            });
          P.src.val[j] = {p, 1};
        }
      }
    }
    P.rb.rows = n;
    P.rb.key  = P.key;
    for (auto const& q : sums) {
      plan::sum_in in;
      colref c = agg_col(q.agg);
      if (c.store == wt::I128) {
        in.wide = c;
      } else {
        in.words = P.src.val[q.word].p;
      }
      P.sum.push_back(in);
    }
    return P;
  }

  // ---- the sum bound (sql.cu: stat_batch, sum_batch) ----
  // Decides, for every int64 sum, whether this batch stays in the narrow accounting. False: a
  // sum would leave what its partial type can carry - the batch is refused.
  bool decide_sums(progress& tk, plan const& P, uint64_t n, ::cuda::stream_ref stream)
  {
    tk.widen.assign(sums.size(), 0);
    tk.masked.assign(sums.size(), 0);
    tk.pushed.assign(sums.size(), 0);
    tk.hi.resize(sums.size());
    for (std::size_t q = 0; q < sums.size(); ++q) {
      auto const& in = P.sum[q];
      bool decided   = false;
      if (in.bound.known && in.bound.min <= in.bound.max) {
        u128 up = in.bound.max > 0 ? static_cast<u128>(in.bound.max) : 0;
        u128 down =
          in.bound.min < 0 ? static_cast<u128>(0ull - static_cast<uint64_t>(in.bound.min)) : 0;
        std::lock_guard<std::mutex> lk(m);
        auto& s = sums[q];
        if (!(s.widened && (up > HALF || down > HALF))) {
          u128 limit = s.widened ? static_cast<u128>(INT64_MAX) : static_cast<u128>(STAT_MAX);
          u128 pos   = s.stat_pos + static_cast<u128>(n) * up;
          u128 neg   = s.stat_neg + static_cast<u128>(n) * down;
          u128 was   = s.stat_pos > s.stat_neg ? s.stat_pos : s.stat_neg;
          u128 nb    = s.bound - was + (pos > neg ? pos : neg);
          if (nb <= limit) {
            s.bound    = nb;
            s.stat_pos = pos;
            s.stat_neg = neg;
            decided    = true;
          }
        }
      }
      if (decided) continue;
      bool known;
      {
        std::lock_guard<std::mutex> lk(m);
        known = sums[q].widened;
      }
      uint64_t mx[2] = {0, 0};
      if (!known || in.wide.data != nullptr) {  // the batch's largest |value|: one pass
        auto* out = reinterpret_cast<ull*>(tk.sc->counters.data());
        check(cudaMemsetAsync(out, 0, 2 * sizeof(ull), stream.get()), "agg_table: memset");
        if (in.wide.data != nullptr) {
          maxabs128_kernel<<<grid_for(n, SCAN_RPT), KBLOCK, 0, stream.get()>>>(
            reinterpret_cast<__int128 const*>(in.wide.data), n, in.wide.mask, in.wide.moff, out);
        } else {
          maxabs_kernel<<<grid_for(n, SCAN_RPT), KBLOCK, 0, stream.get()>>>(in.words, n, out);
        }
        check(cudaGetLastError(), "agg_table: maxabs_kernel");
        check(cudaMemcpyAsync(mx, out, sizeof mx, cudaMemcpyDeviceToHost, stream.get()),
              "agg_table: read");
        sync(stream);
        if (mx[1] != 0) return false;  // a 128-bit partial sum too large for the high sums
      }
      {
        std::lock_guard<std::mutex> lk(m);
        auto& s = sums[q];
        if (!s.widened) {
          u128 nb = s.bound + static_cast<u128>(n) * mx[0];
          if (nb <= static_cast<u128>(NARROW_MAX)) {
            s.bound = nb;
            continue;  // no group can have left int64
          }
        }
        if (!s.can_widen) return false;
        u128 nb = s.bound + static_cast<u128>(n) * HALF;
        if (nb > static_cast<u128>(INT64_MAX))
          return false;  // (about 2^32 rows after the bound was exhausted)
        s.widened = true;
        s.bound   = nb;
        // Runs are merged as they pile up (functional: a failure leaves them as they were).
        if (s.runs.size() >= MAX_RUNS) {
          while (s.runs.size() >= 2) {
            hi_run mrg = merge_runs(s.runs[s.runs.size() - 2], s.runs.back(), stream, mr);
            sync(stream);
            s.runs.pop_back();
            s.runs.back() = std::move(mrg);
          }
        }
      }
      // The batch is summed in halves: its high halves are computed here, masked after the fold.
      tk.widen[q] = 1;
      if (!tk.hi[q]) tk.hi[q] = std::make_unique<rmm::device_uvector<uint64_t>>(n, stream, mr);
      if (in.wide.data != nullptr) {
        high128_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(
          tk.hi[q]->data(),
          reinterpret_cast<__int128 const*>(in.wide.data),
          n,
          in.wide.mask,
          in.wide.moff);
      } else {
        high_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(tk.hi[q]->data(), in.words, n);
      }
      check(cudaGetLastError(), "agg_table: high_kernel");
      tk.allow_preagg =
        false;  // (a pre-aggregated group that finds no room could not be split again)
    }
    return true;
  }

  // ---- dense rows to a cudf table ----
  // rows: n x W words. hi[q]: per sum, the rows' high sums (or null). null_row: the row whose
  // key is NULL (-1: none). wide[q]: the sum leaves as a 128-bit column.
  std::unique_ptr<cudf::table> rows_to_table(uint64_t const* rows,
                                             uint64_t n,
                                             std::vector<uint64_t const*> const& hi,
                                             std::vector<uint8_t> const& wide,
                                             int64_t null_row,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref out_mr)
  {
    layout const& L  = *lay;
    uint32_t const W = static_cast<uint32_t>(L.ns) + 1;
    auto const size  = static_cast<cudf::size_type>(n);
    auto g           = grid_for(n);
    auto make_mask   = [&](int word, int64_t nrow, cudf::size_type& nulls) {
      rmm::device_buffer mk =
        cudf::create_null_mask(size, cudf::mask_state::UNINITIALIZED, stream, out_mr);
      out_mask_kernel<<<grid_for(n, 32), KBLOCK, 0, stream.get()>>>(
        static_cast<uint32_t*>(mk.data()), rows, W, word, nrow, n);
      check(cudaGetLastError(), "agg_table: out_mask_kernel");
      nulls = cudf::null_count(static_cast<cudf::bitmask_type const*>(mk.data()), 0, size, stream);
      return mk;
    };
    std::vector<std::unique_ptr<cudf::column>> cols;
    {
      rmm::device_buffer data(n * width_of(key_store), stream, out_mr);
      out_key_kernel<<<g, KBLOCK, 0, stream.get()>>>(
        static_cast<uint8_t*>(data.data()), key_store, rows, W, n);
      check(cudaGetLastError(), "agg_table: out_key_kernel");
      rmm::device_buffer mk{0, stream, out_mr};
      cudf::size_type nulls = 0;
      if (null_row >= 0) mk = make_mask(-1, null_row, nulls);
      cols.push_back(
        std::make_unique<cudf::column>(key_type, size, std::move(data), std::move(mk), nulls));
    }
    for (std::size_t a = 0; a < cfg.aggs.size(); ++a) {
      int word = L.agg_word[a];
      int sq   = -1;
      for (std::size_t q = 0; q < sums.size(); ++q) {
        if (cfg.aggs[a].kind == agg_kind::sum && sums[q].word == word) sq = static_cast<int>(q);
      }
      cudf::data_type type =
        partial_type(cfg.aggs[a], sq >= 0 && wide[static_cast<std::size_t>(sq)] != 0);
      wt store = storage_of(type.id());
      rmm::device_buffer data(n * width_of(store), stream, out_mr);
      out_value_kernel<<<g, KBLOCK, 0, stream.get()>>>(
        static_cast<uint8_t*>(data.data()),
        store,
        rows,
        W,
        static_cast<uint32_t>(word),
        sq >= 0 ? hi[static_cast<std::size_t>(sq)] : nullptr,
        n);
      check(cudaGetLastError(), "agg_table: out_value_kernel");
      rmm::device_buffer mk{0, stream, out_mr};
      cudf::size_type nulls = 0;
      if (L.agg_valid[a] >= 0) {
        mk = make_mask(L.agg_valid[a], -1, nulls);
        if (nulls == 0) mk = rmm::device_buffer{0, stream, out_mr};
      }
      cols.push_back(
        std::make_unique<cudf::column>(type, size, std::move(data), std::move(mk), nulls));
    }
    sync(stream);
    return std::make_unique<cudf::table>(std::move(cols));
  }

  // ---- fold ----
  fold_result run(progress& tk, plan const& P, fold_ticket& ticket, ::cuda::stream_ref stream)
  {
    uint64_t const n = tk.n;
    if (!tk.refused) {
      if (tk.stage == 0) {  // the table
        if (ticket.st == fold_ticket::state::mutated) {
          throw std::runtime_error(
            "agg_table: a batch that failed inside the table cannot be resumed");
        }
        ticket.st = fold_ticket::state::mutated;
        seg_table::fold_result r;
        if (tk.partial_in) {
          r.rows = t->upsert(P.src, n, tk.sc->bitmap.data(), false, tk.sc->counters, stream);
        } else {
          r = t->fold(P.rb, P.src, *tk.sc, stream, tk.allow_preagg);
        }
        tk.left_rows     = r.rows;
        tk.left_partials = r.partials;
        tk.stage         = 1;
        int f            = fail_after_mutation.load();
        while (f > 0 && !fail_after_mutation.compare_exchange_weak(f, f - 1)) {}
        if (f > 0)
          throw rmm::out_of_memory("agg_table: injected failure after the table was changed");
      }
      if (tk.stage == 1) {  // side groups
        if (P.key_reserved_possible && !side_seen.load(std::memory_order_relaxed)) {
          uint32_t f = 0;
          check(cudaMemcpyAsync(&f, seen->data(), sizeof f, cudaMemcpyDeviceToHost, stream.get()),
                "agg_table: read");
          sync(stream);
          if (f != 0) side_seen = true;
        }
        if (P.kmask != nullptr ||
            (P.key_reserved_possible && side_seen.load(std::memory_order_relaxed))) {
          uint32_t const* pm = tk.punt && tk.punt_count != 0 ? tk.punt->data() : nullptr;
          side_fold_kernel<<<grid_for(n, SIDE_RPT), KBLOCK, 0, stream.get()>>>(
            P.src, ident, n, P.kmask, P.koff, pm, side->data());
          check(cudaGetLastError(), "agg_table: side_fold_kernel");
          sync(stream);
          side_dirty = true;
        }
        tk.stage = 2;
      }
      if (tk.stage == 2) {  // wide sums: this batch's high halves, of the rows that were applied
        for (std::size_t q = 0; q < sums.size(); ++q) {
          if (!tk.widen[q] || tk.pushed[q]) continue;
          uint64_t* hi = tk.hi[q]->data();
          if (!tk.masked[q]) {
            uint32_t const* pm = tk.punt && tk.punt_count != 0 ? tk.punt->data() : nullptr;
            high_mask_kernel<<<grid_for(n), KBLOCK, 0, stream.get()>>>(
              hi,
              P.key,
              n,
              tk.sc->bitmap.data(),
              pm,
              P.kmask,
              P.koff,
              hi_side->data() + q * SIDE_SLOTS);
            check(cudaGetLastError(), "agg_table: high_mask_kernel");
            sync(stream);
            tk.masked[q] = 1;
          }
          auto pol = rmm::exec_policy(stream, mr);
          auto np  = static_cast<std::size_t>(thrust::count_if(pol, hi, hi + n, nonzero_fn{}));
          if (np != 0) {
            rmm::device_uvector<uint64_t> pk(np, stream, mr), ph(np, stream, mr);
            thrust::copy_if(pol,
                            thrust::make_zip_iterator(thrust::make_tuple(P.key, hi)),
                            thrust::make_zip_iterator(thrust::make_tuple(P.key + n, hi + n)),
                            hi,
                            thrust::make_zip_iterator(thrust::make_tuple(pk.begin(), ph.begin())),
                            nonzero_fn{});
            hi_run r = make_run(std::move(pk), std::move(ph), stream, mr);
            sync(stream);
            std::lock_guard<std::mutex> lk(m);
            sums[q].runs.push_back(std::move(r));
          }
          tk.pushed[q] = 1;
        }
        tk.stage = 3;
      }
    }
    // The result, from what the stages left (repeatable).
    fold_result res;
    uint64_t const left = tk.refused ? n : tk.left_rows + tk.punt_count;
    if (left != 0) {
      rmm::device_uvector<cudf::size_type> map(left, stream, mr);
      auto pol = rmm::exec_policy(stream, mr);
      if (tk.refused) {
        thrust::sequence(pol, map.begin(), map.end(), 0);
      } else {
        unapplied_fn fn{tk.left_rows != 0 ? tk.sc->bitmap.data() : nullptr,
                        tk.punt && tk.punt_count != 0 ? tk.punt->data() : nullptr};
        auto end = thrust::copy_if(
          pol,
          thrust::counting_iterator<cudf::size_type>(0),
          thrust::counting_iterator<cudf::size_type>(static_cast<cudf::size_type>(n)),
          map.begin(),
          fn);
        if (static_cast<uint64_t>(end - map.begin()) != left) {
          throw std::runtime_error("agg_table: the unapplied rows do not match their count");
        }
      }
      sync(stream);
      res.unapplied =
        std::make_unique<cudf::column>(std::move(map), rmm::device_buffer{0, stream, mr}, 0);
    }
    if (!tk.refused && tk.left_partials != 0) {
      uint32_t const W = static_cast<uint32_t>(lay->ns) + 1;
      rmm::device_uvector<uint64_t> rows(tk.left_partials * W, stream, mr);
      auto* cnt = reinterpret_cast<ull*>(tk.sc->counters.data());
      check(cudaMemsetAsync(cnt, 0, sizeof(ull), stream.get()), "agg_table: memset");
      select_marked_kernel<<<grid_for(tk.sc->partial_count), CBLOCK, 0, stream.get()>>>(
        tk.sc->partials.data(),
        tk.sc->partial_count,
        W,
        tk.sc->partial_bitmap.data(),
        rows.data(),
        cnt);
      check(cudaGetLastError(), "agg_table: select_marked_kernel");
      uint64_t got = 0;
      check(cudaMemcpyAsync(&got, cnt, sizeof got, cudaMemcpyDeviceToHost, stream.get()),
            "agg_table: read");
      sync(stream);
      if (got != tk.left_partials)
        throw std::runtime_error("agg_table: the unapplied partial rows do not match");
      std::vector<uint64_t const*> hi(sums.size(), nullptr);
      std::vector<uint8_t> wide(sums.size(), 0);
      for (std::size_t q = 0; q < sums.size(); ++q) {
        wide[q] =
          cfg.aggs[static_cast<std::size_t>(sums[q].agg)].input.id() == cudf::type_id::DECIMAL64;
      }
      res.unapplied_partial =
        rows_to_table(rows.data(), tk.left_partials, hi, wide, -1, stream, mr);
    }
    res.rows_applied = n - left;
    ticket.st        = fold_ticket::state::done;
    // What only the stages needed is released; the bitmap stays for a replay of the result.
    tk.cols.clear();
    tk.hi.clear();
    return res;
  }

  template <typename PlanFn>
  fold_result fold_common(uint64_t n,
                          bool partial_in,
                          bool input_nulls,
                          fold_ticket& ticket,
                          ::cuda::stream_ref stream,
                          PlanFn&& make_plan)
  {
    check_device();
    using st = fold_ticket::state;
    if (n > 0x7fffffffull) throw std::invalid_argument("agg_table: batch too large");
    if (n == 0) {
      ticket.st = st::done;
      return fold_result{};
    }
    auto tk = std::static_pointer_cast<progress>(ticket.progress);
    if (ticket.st != st::not_started) {
      if (!tk || tk->owner != this || tk->n != n || tk->partial_in != partial_in) {
        throw std::logic_error("agg_table: the ticket does not belong to this batch and table");
      }
    } else if (!tk || tk->owner != this) {
      tk              = std::make_shared<progress>();
      tk->owner       = this;
      tk->n           = n;
      tk->partial_in  = partial_in;
      tk->pool        = pool;
      ticket.progress = tk;
    }
    std::shared_lock<std::shared_mutex> sl(tmx);
    if (ticket.st == st::not_started) {
      tk->refused = false;
      if (lay == nullptr || !t) {
        // The table itself: created by the first fold (and by the first after a flush), under
        // the exclusive lock. (Only then: a fold must not wait for the folds in flight.)
        sl.unlock();
        {
          std::unique_lock<std::shared_mutex> xl(tmx);
          if (flushing) throw std::logic_error("agg_table: fold during an unfinished flush");
          choose_layout(input_nulls, stream);
          tk->refused = !ensure_table(stream);
        }
        sl.lock();
      }
    }
    if (ticket.st == st::not_started && !tk->refused) {
      if (!t) throw std::logic_error("agg_table: fold concurrent with flush");
      // Prepare: scratch, normalised columns, sum bounds. Nothing of the table changes.
      if (!tk->sc) {
        std::lock_guard<std::mutex> lk(pool->m);
        if (!pool->free.empty()) {
          tk->sc = std::move(pool->free.back());
          pool->free.pop_back();
        }
      }
      if (!tk->sc) tk->sc = std::make_unique<seg_table::scratch>(stream, mr);
      t->reserve_scratch(*tk->sc, n, stream);
      tk->cols.clear();
      tk->punt_count   = 0;
      tk->allow_preagg = true;
      tk->stage        = 0;
      plan P           = make_plan(*tk, true);
      if (!decide_sums(*tk, P, n, stream)) tk->refused = true;
      sync(stream);
      return run(*tk, P, ticket, stream);
    }
    if (tk->refused) return run(*tk, plan{}, ticket, stream);
    plan P = tk->stage < 3 ? make_plan(*tk, false) : plan{};
    return run(*tk, P, ticket, stream);
  }

  // ---- flush ----
  // One step of a flush: appends to `ready`. False: the flush is complete.
  bool flush_step(std::size_t max_rows,
                  ::cuda::stream_ref stream,
                  rmm::device_async_resource_ref out_mr)
  {
    if (lay == nullptr) return false;
    uint32_t const W = static_cast<uint32_t>(lay->ns) + 1;
    std::vector<uint8_t> wide(sums.size(), 0);
    {
      // All runs of a sum into one (functional: a failure leaves them as they were).
      std::lock_guard<std::mutex> lk(m);
      for (std::size_t q = 0; q < sums.size(); ++q) {
        auto& runs = sums[q].runs;
        while (runs.size() >= 2) {
          hi_run mrg = merge_runs(runs[runs.size() - 2], runs.back(), stream, mr);
          sync(stream);
          runs.pop_back();
          runs.back() = std::move(mrg);
        }
        wide[q] = sums[q].widened || cfg.aggs[static_cast<std::size_t>(sums[q].agg)].input.id() ==
                                       cudf::type_id::DECIMAL64;
      }
    }
    flushing     = true;
    auto convert = [&](
                     uint64_t const* rows, uint64_t n, uint64_t const* side_hi, int64_t null_row) {
      std::vector<std::unique_ptr<cudf::table>> out;
      uint64_t const step = max_rows == 0 ? n : static_cast<uint64_t>(max_rows);
      for (uint64_t a = 0; a < n; a += step) {
        uint64_t c = n - a < step ? n - a : step;
        std::vector<std::unique_ptr<rmm::device_uvector<uint64_t>>> tmp(sums.size());
        std::vector<uint64_t const*> hi(sums.size(), nullptr);
        for (std::size_t q = 0; q < sums.size(); ++q) {
          if (side_hi != nullptr) {
            hi[q] = side_hi + q * SIDE_SLOTS + a;
          } else if (!sums[q].runs.empty() && sums[q].runs[0].key.size() != 0) {
            tmp[q]        = std::make_unique<rmm::device_uvector<uint64_t>>(c, stream, mr);
            auto const& r = sums[q].runs[0];
            run_lookup_kernel<<<grid_for(c), KBLOCK, 0, stream.get()>>>(
              tmp[q]->data(), rows + a * W, W, c, r.key.data(), r.h.data(), r.key.size());
            check(cudaGetLastError(), "agg_table: run_lookup_kernel");
            hi[q] = tmp[q]->data();
          }
        }
        int64_t nr = null_row >= static_cast<int64_t>(a) ? null_row - static_cast<int64_t>(a) : -1;
        out.push_back(rows_to_table(rows + a * W, c, hi, wide, nr, stream, out_mr));
      }
      return out;
    };
    if (t) {
      uint64_t const S = t->num_segments();
      if (pending || next_seg < S) {
        if (!pending) {
          uint64_t g =
            max_rows == 0
              ? S
              : t->segments_per_chunk(static_cast<uint64_t>(max_rows) * W * sizeof(uint64_t));
          if (g > S - next_seg) g = S - next_seg;
          pending = std::make_unique<seg_table::chunk>(t->emit_segments(next_seg, g, stream, mr));
          next_seg += g;
        }
        // (The chunk stays `pending` until all of its tables exist: a failure here loses nothing.)
        if (pending->groups != 0) {
          auto pieces = convert(pending->rows.data(), pending->groups, nullptr, -1);
          for (auto& p : pieces)
            ready.push_back(std::move(p));
        }
        pending.reset();
        return true;
      }
      last_groups = t->groups();
      auto st     = t->stats();
      past.growths += st.growths;
      past.chunks += st.chunks;
      past.hot_chunks += st.hot_chunks;
      past.run_chunks += st.run_chunks;
      past.retries += st.retries;
      past.curve_steps += st.curve_steps;
      past.jumps += st.jumps;
      t.reset();
      next_seg = 0;
    }
    if (side_dirty.load()) {
      // The side groups: at most three rows, built on the host.
      uint64_t host[SIDE_SLOTS * MAX_STATE + SIDE_SLOTS];
      std::vector<uint64_t> hs(hi_side->size());
      check(cudaMemcpyAsync(host, side->data(), sizeof host, cudaMemcpyDeviceToHost, stream.get()),
            "agg_table: read");
      check(cudaMemcpyAsync(hs.data(),
                            hi_side->data(),
                            hs.size() * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost,
                            stream.get()),
            "agg_table: read");
      sync(stream);
      std::vector<uint64_t> rows;
      std::vector<uint64_t> his(hs.size(), 0);  // [sum][row]
      uint64_t n       = 0;
      int64_t null_row = -1;
      for (int q = 0; q < SIDE_SLOTS; ++q) {
        if (host[SIDE_SLOTS * MAX_STATE + q] == 0) continue;
        if (q == 2) null_row = static_cast<int64_t>(n);
        rows.push_back(q == 0 ? EMPTY_KEY : q == 1 ? EMPTY_KEY - 1 : 0);
        for (int s = 0; s < lay->ns; ++s)
          rows.push_back(host[q * MAX_STATE + s]);
        for (std::size_t j = 0; j < sums.size(); ++j) {
          his[j * SIDE_SLOTS + n] = hs[j * SIDE_SLOTS + static_cast<std::size_t>(q)];
        }
        ++n;
      }
      if (n != 0) {
        rmm::device_uvector<uint64_t> drows(rows.size(), stream, mr);
        rmm::device_uvector<uint64_t> dhi(his.size(), stream, mr);
        check(cudaMemcpyAsync(drows.data(),
                              rows.data(),
                              rows.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice,
                              stream.get()),
              "agg_table: write");
        check(cudaMemcpyAsync(dhi.data(),
                              his.data(),
                              his.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice,
                              stream.get()),
              "agg_table: write");
        auto pieces = convert(drows.data(), n, dhi.data(), null_row);
        for (auto& p : pieces)
          ready.push_back(std::move(p));
      }
      reset_side(stream);
    }
    {
      std::lock_guard<std::mutex> lk(m);
      for (auto& s : sums) {  // the sums start again with the empty table
        s.runs.clear();
        s.bound = s.stat_pos = s.stat_neg = 0;
        s.widened                         = false;
      }
    }
    flushing = false;
    ++flushes;
    return false;
  }

  std::vector<std::unique_ptr<cudf::table>> flush(std::size_t max_rows,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref out_mr)
  {
    check_device();
    std::unique_lock<std::shared_mutex> xl(tmx);
    while (flush_step(max_rows, stream, out_mr)) {}
    return std::move(ready);
  }

  std::unique_ptr<cudf::table> flush_next(std::size_t max_rows,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref out_mr)
  {
    check_device();
    std::unique_lock<std::shared_mutex> xl(tmx);
    while (ready.empty()) {
      bool more = flush_step(max_rows, stream, out_mr);
      if (!more && ready.empty()) return nullptr;
    }
    auto out = std::move(ready.front());
    ready.erase(ready.begin());
    return out;
  }

  std::string describe() const
  {
    std::shared_lock<std::shared_mutex> sl(tmx);
    if (lay == nullptr) return "layout=undecided";
    seg_table::stats_t st = past;
    if (t) {
      auto s = t->stats();
      st.growths += s.growths;
      st.chunks += s.chunks;
      st.hot_chunks += s.hot_chunks;
      st.run_chunks += s.run_chunks;
      st.retries += s.retries;
      st.curve_steps += s.curve_steps;
      st.jumps += s.jumps;
    }
    std::string wide;
    {
      std::lock_guard<std::mutex> lk(m);
      for (auto const& s : sums)
        wide += s.widened ? "1" : "0";
    }
    return std::string("layout=") + (lay->nullable ? "nullable" : "compact") +
           " words=" + std::to_string(lay->ns) +
           " kernels=" + (lay->canon ? "canonical" : "generic") +
           " groups=" + std::to_string(t ? t->groups() : 0) +
           " bytes=" + std::to_string(t ? t->bytes() : 0) +
           " growths=" + std::to_string(st.growths) +
           " curve_steps=" + std::to_string(st.curve_steps) + " jumps=" + std::to_string(st.jumps) +
           " launches=" + std::to_string(st.chunks) +
           (st.hot_chunks != 0 ? " hot_chunks=" + std::to_string(st.hot_chunks) : "") +
           (st.run_chunks != 0 ? " run_chunks=" + std::to_string(st.run_chunks) : "") +
           " retries=" + std::to_string(st.retries) + " side=" + (side_dirty.load() ? "1" : "0") +
           " wide=" + wide + " flushes=" + std::to_string(flushes);
  }
};

// ---- public ------------------------------------------------------------------------------------
bool eligible(const config& cfg, std::string* why_not)
{
  auto no = [&](char const* why) {
    if (why_not != nullptr) *why_not = why;
    return false;
  };
  if (!key_type_ok(cfg.key_type.id()))
    return no("key type: not a fixed-width integer, date or timestamp of at most 64 bits");
  int max_col = -1;
  for (auto const& a : cfg.aggs) {
    auto id = a.input.id();
    switch (a.kind) {
      case agg_kind::count_star:
        if (a.value_column != -1) return no("count_star takes no value column");
        continue;
      case agg_kind::count:
        if (!count_type_ok(id)) return no("COUNT: value type not supported");
        break;
      case agg_kind::sum:
        if (!sum_type_ok(id))
          return no("SUM: value type not a signed integer, DECIMAL32/64 or float");
        break;
      case agg_kind::min:
      case agg_kind::max:
        if (!minmax_type_ok(id)) return no("MIN/MAX: value type not supported");
        break;
      default: return no("aggregate kind not supported");
    }
    if (a.value_column < 0) return no("aggregate without a value column");
    max_col = a.value_column > max_col ? a.value_column : max_col;
  }
  // (One value column must have one type.)
  for (auto const& a : cfg.aggs) {
    for (auto const& b : cfg.aggs) {
      if (a.kind != agg_kind::count_star && b.kind != agg_kind::count_star &&
          a.value_column == b.value_column && a.input != b.input) {
        return no("aggregates disagree on the type of a value column");
      }
    }
  }
  layout L = build_layout(cfg, false);
  if (!L.fits) return no("more than 8 state words");
  if (cfg.budget_bytes != 0) {
    uint32_t dummy[2];
    auto opt = table_options(cfg, cfg.budget_bytes, 0, dummy);
    if (seg_table::presize(L.spec, opt).min_bytes > cfg.budget_bytes)
      return no("budget below the smallest table");
  }
  return true;
}

std::vector<cudf::data_type> partial_schema(const config& cfg)
{
  std::vector<cudf::data_type> out;
  out.push_back(cfg.key_type);
  for (auto const& a : cfg.aggs)
    out.push_back(partial_type(a, false));
  return out;
}

table::table()  = default;
table::~table() = default;

std::unique_ptr<table> table::create(const config& cfg, rmm::device_async_resource_ref mr)
{
  std::string why;
  if (!eligible(cfg, &why)) throw std::invalid_argument("agg_table: not eligible: " + why);
  std::unique_ptr<table> t(new table());
  t->_impl = std::make_unique<impl>(cfg, mr);
  auto& I  = *t->_impl;
  check(cudaGetDevice(&I.device), "agg_table: device");
  I.compact   = build_layout(cfg, false);
  I.nullable  = build_layout(cfg, true);
  I.key_type  = cfg.key_type;
  I.key_store = storage_of(cfg.key_type.id());
  I.budget    = cfg.budget_bytes;
  return t;
}

fold_result table::fold(const cudf::column_view& keys,
                        const cudf::table_view& values,
                        const std::vector<column_bound>& bounds,
                        fold_ticket& ticket,
                        ::cuda::stream_ref stream)
{
  auto& I = *_impl;
  if (keys.type() != I.cfg.key_type) throw std::invalid_argument("agg_table: key column type");
  bool nulls = false;
  for (auto const& a : I.cfg.aggs) {
    if (a.kind == agg_kind::count_star) continue;
    if (a.value_column >= values.num_columns() || values.column(a.value_column).type() != a.input ||
        values.column(a.value_column).size() != keys.size()) {
      throw std::invalid_argument("agg_table: value column does not match the configuration");
    }
    nulls |= values.column(a.value_column).has_nulls();
  }
  return I.fold_common(static_cast<uint64_t>(keys.size()),
                       false,
                       nulls,
                       ticket,
                       stream,
                       [&](progress& tk, bool materialise) {
                         return I.plan_raw(tk, keys, values, bounds, materialise, stream);
                       });
}

fold_result table::fold_partial(const cudf::table_view& partial,
                                fold_ticket& ticket,
                                ::cuda::stream_ref stream)
{
  auto& I = *_impl;
  if (partial.num_columns() != static_cast<cudf::size_type>(1 + I.cfg.aggs.size()) ||
      partial.column(0).type() != I.cfg.key_type) {
    throw std::invalid_argument("agg_table: partial rows do not match the partial schema");
  }
  bool nulls = false;
  for (std::size_t a = 0; a < I.cfg.aggs.size(); ++a) {
    auto const& c = partial.column(static_cast<cudf::size_type>(1 + a));
    bool ok       = c.type() == partial_type(I.cfg.aggs[a], false) ||
              c.type() == partial_type(I.cfg.aggs[a], true);
    if (!ok) throw std::invalid_argument("agg_table: partial rows do not match the partial schema");
    if ((I.cfg.aggs[a].kind == agg_kind::count_star || I.cfg.aggs[a].kind == agg_kind::count) &&
        c.has_nulls()) {
      throw std::invalid_argument("agg_table: a partial count is NULL");
    }
    nulls |= c.has_nulls();
  }
  return I.fold_common(static_cast<uint64_t>(partial.num_rows()),
                       true,
                       nulls,
                       ticket,
                       stream,
                       [&](progress& tk, bool materialise) {
                         return I.plan_partial(tk, partial, materialise, stream);
                       });
}

std::vector<std::unique_ptr<cudf::table>> table::flush(std::size_t max_rows,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref out_mr)
{
  return _impl->flush(max_rows, stream, out_mr);
}

std::vector<std::unique_ptr<cudf::table>> table::emit(std::size_t max_rows,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref out_mr)
{
  return _impl->flush(max_rows, stream, out_mr);
}

std::unique_ptr<cudf::table> table::flush_next(std::size_t max_rows,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref out_mr)
{
  return _impl->flush_next(max_rows, stream, out_mr);
}

std::string table::describe() const { return _impl->describe(); }

void table::test_fail_after_mutation(int times) { _impl->fail_after_mutation = times; }

std::size_t table::device_bytes() const
{
  auto& I = *_impl;
  std::shared_lock<std::shared_mutex> sl(I.tmx);
  std::size_t b = I.t ? I.t->bytes() : 0;
  if (I.side)
    b += I.side->size() * sizeof(uint64_t) + I.hi_side->size() * sizeof(uint64_t) +
         2 * sizeof(uint32_t);
  if (I.pending) b += I.pending->rows.size() * sizeof(uint64_t);
  std::lock_guard<std::mutex> lk(I.m);
  for (auto const& s : I.sums) {
    for (auto const& r : s.runs)
      b += (r.key.size() + r.h.size()) * sizeof(uint64_t);
  }
  return b;
}

uint64_t table::groups() const
{
  auto& I = *_impl;
  std::shared_lock<std::shared_mutex> sl(I.tmx);
  return I.t ? I.t->groups() : 0;
}

void table::shrink_budget(std::size_t budget_bytes)
{
  auto& I = *_impl;
  std::shared_lock<std::shared_mutex> sl(I.tmx);
  I.budget = budget_bytes;
  if (I.t) I.t->set_budget(budget_bytes);
}

}  // namespace sirius::agg_table

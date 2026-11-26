// Device side of the SQL-correctness layer (ported from experimental/agg_bench/src/sql_dev.cuh):
// what every table kernel needs to know about reserved key values and the ordering of float64
// MIN / MAX.
//
// Changes against the prototype, all of them because a table must not share state with another
// table (two operators on one GPU, two tables on two GPUs):
//  * The prototype kept its settings in `__constant__` memory, one copy per translation unit
//    (`g_sql`), written by sql::enable(). Here the two behaviours are always on (reserved keys:
//    the prototype's AGG_SQL_KEYS=1; NaN order: AGG_SQL_NAN=1) and are compile-time constants, and
//    the flag a kernel raises when it skips a reserved key is a device word of the TABLE, reached
//    through the kernel's own parameters (seg_layout::seen) instead of through `g_sql.seen`. A
//    kernel therefore loads one launch parameter where it loaded three settings words, and stores
//    one word where it stored two (the prototype's zero-copy host page is gone: the wrapper reads
//    the device word back with the fold's counters).
//  * The magnitude measurement of AGG_SQL_SUM=fold is compiled out (the prototype's
//    -DAGG_SQL_SUMFOLD_OFF=1 build, which it showed to be instruction-identical for the hot-key
//    kernels): the library bounds int64 sums from statistics or with a pass (agg_table.cu).
#pragma once

#include "common.hpp"

#include <cuda_runtime.h>

#define AGG_SQL_SUMFOLD_OFF 1

namespace sirius::agg_table::detail {

namespace sqldev {

/// Number of reserved key values at the top of the key space: EMPTY_KEY = -1 and BUSY_KEY = -2.
constexpr uint64_t RESERVED_KEYS = 2;

/// A key the tables cannot hold (their free / claimed slot markers). Every kernel that reads the
/// key of a RAW input row skips such a row as if it were not in the batch - not applied, not
/// marked, not tagged, not parked - and raises the flag; the operator then folds the batch's
/// reserved rows into its side groups (sql.cu). Partial rows and parked rows never carry such a
/// key. `seen`: the table's flag word (seg_layout::seen), followed by a scratch word.
__device__ __forceinline__ bool reserved_key(uint32_t* seen, uint64_t k)
{
  if (k + 2 >= RESERVED_KEYS) return false;  // (k + 2 wraps to 0 / 1 for -2 / -1)
  if (*seen == 0) *seen = 1u;
  return true;
}
/// The same test without the flag (samples, histograms).
__device__ __forceinline__ bool reserved_key_quiet(uint64_t k) { return k + 2 < RESERVED_KEYS; }

/// The row loops that read a row's VALUES together with its key (the block-local and hot-key
/// kernels): `if (reserved_row(k, a, b, c, d, rsv)) continue;` after the loads, and
/// `reserved_rows_done(L.seen, rsv)` once after the loop.
/// A plain `if (reserved_key(L.seen, k)) continue;` - wherever it is written - lets the compiler
/// move the value loads behind that branch, since nothing on the skipping path uses them: every
/// value load then waits for the row's key instead of being issued with it (SASS of
/// fold_preagg2_kernel<2>: LDG key, LDG value before; LDG key, compare, branch, LDG value with the
/// plain test). GB200 job 24294: long runs of equal keys (3,000 keys, 2^31 rows) at 0.81 / 0.81 /
/// 0.71 of the speed before with 2 / 4 / 8 aggregates, 1 aggregate (no values) unchanged. Here the
/// skipping path uses the values too (they go into `rsv.sink`, which the thread stores once if it
/// met a reserved key), so the loads stay where they were, in front of the branch.
struct reserved_acc {
  uint64_t sink = 0;
  bool any      = false;
};
__device__ __forceinline__ bool reserved_row(
  uint64_t k, uint64_t a, uint64_t b, uint64_t c, uint64_t d, reserved_acc& rsv)
{
  if (!reserved_key_quiet(k)) return false;
  rsv.sink += a + b + c + d;
  rsv.any = true;
  return true;
}
__device__ __forceinline__ void reserved_rows_done(uint32_t* seen, reserved_acc const& rsv)
{
  if (!rsv.any) return;
  seen[1] = static_cast<uint32_t>(rsv.sink);  // (a scratch word: only so that the values are used)
  if (*seen == 0) *seen = 1u;
}

// ---- The magnitude of the int64 values a kernel sums (AGG_SQL_SUM=fold; README "SUM over int64:
// the bound measured inside the fold") ------------------------------------------------------------
// Every kernel that loads the value of a RAW input row to add it to a sum - or to park the row -
// ORs `v ^ (v >> 63)` (v for v >= 0, -v - 1 for v < 0) of the row's int64 SUM operands into one
// 64-bit register (sum_row), and once, when it is done with its rows, publishes the bit length p
// of that OR if it is above what was published before (sum_rows_done): every value the thread
// summed then satisfies |v| <= 2^p. The host reads the largest p after the batch (sql.cu), so the
// bound of a batch costs no pass over it. Partial states and parked rows are not measured again.
// One length for all the sums of the aggregate set (columns a and b with 8 state words): a second
// accumulator is two more registers in every kernel, and the kernels with 8 state words are the
// ones nearest to losing a resident block to their register count.
//   per row:    a shift, an exclusive-or and an or per summed column, on a value the kernel has
//               loaded anyway (no load, no store, no branch);
//   per thread: one load of the settings and of the device word, a compare; an atomic max and a
//               store only for a thread that raises the published length (at most 63 times per
//               run, plus races).
// Registers: the block-local and hot-key loops carry the 64-bit accumulator. The kernels with
// one row per thread carry one 32-bit length across the row's state update (sum_row_len /
// sum_row_done); a run that a direct kernel combines is measured by the threads of its other rows
// (sum_run_row), the strip kernel's by a loop after the run's state update (sum_run); the
// partition kernels measure in a loop of their own (sum_parked). The last three are skipped when
// the mode is off.
// (Tried and dropped: keeping the bits in reserved_acc::sink, the register pair the block-local
// loops already have, with the skipped rows' values masked out. On sm_100 the loads of the
// columns that are not summed then moved behind the branch on the key again - the very thing
// reserved_row exists to prevent - because the two paths no longer start with the same uses.)
// -DAGG_SQL_SUMFOLD_OFF=1 (and -DAGG_SQL_OFF=1) removes all of it at compile time.
constexpr uint32_t SUM_BITS = 64;  // published lengths: 0 (only 0 and -1) to 63 (|v| up to 2^63)

struct sum_acc {
  uint64_t m = 0;
};
__device__ __forceinline__ uint64_t sum_mag(uint64_t v)
{
  return v ^ static_cast<uint64_t>(static_cast<long long>(v) >> 63);
}
/// The int64 SUM operands among a raw row's values `a`, `b` in a kernel specialised on NS state
/// words of the canonical set: sum(a) is state word 1, sum(b) state word 7.
template <int NS>
__device__ __forceinline__ uint64_t sum_mag_row(uint64_t a, uint64_t b)
{
  return NS > 7 ? sum_mag(a) | sum_mag(b) : NS > 1 ? sum_mag(a) : 0;
}
template <int NS>
__device__ __forceinline__ void sum_row(sum_acc& s, uint64_t a, uint64_t b)
{
#ifndef AGG_SQL_SUMFOLD_OFF
  if (NS > 1) s.m |= sum_mag_row<NS>(a, b);
#endif
}

/// The same for the launches of the split fold: only the columns whose SUM word is among the
/// state words [W0, W0 + H) of this launch (each sum is measured by the launch that adds it).
template <int W0, int H>
__device__ __forceinline__ void sum_row_words(sum_acc& s, uint64_t a, uint64_t b)
{
#ifndef AGG_SQL_SUMFOLD_OFF
  if (W0 <= 1 && 1 < W0 + H) s.m |= sum_mag(a);
  if (W0 <= 7 && 7 < W0 + H) s.m |= sum_mag(b);
#endif
}
/// One value of the kernels whose aggregate set is a launch parameter: `on` when it is the
/// operand of an int64 SUM. No branch: a select.
__device__ __forceinline__ void sum_val(sum_acc& s, bool on, uint64_t v)
{
#ifndef AGG_SQL_SUMFOLD_OFF
  s.m |= on ? sum_mag(v) : 0;
#endif
}
/// The raw rows the partition kernels park (as they are, packed, or collapsed with their run
/// into a partial row): measured by a loop of their own over the thread's rows, which is skipped
/// when the mode is off (sum_on), so that the kernels' own loops - 16 rows per thread, unrolled -
/// stay as they were. `a` / `b`: the row's columns when they are int64 SUM operands of the
/// aggregate set (a with any value column; b with all four, i.e. 8 state words), else null. The
/// loads are unconditional - a column that is not summed is replaced by the key column and
/// masked out - so that the loop is not compiled once per combination.
struct sum_cols {
  uint64_t const* a;
  uint64_t const* b;
  uint64_t ma, mb;
};
__device__ __forceinline__ sum_cols sum_parked_cols(uint64_t const* key,
                                                    uint64_t const* a,
                                                    uint64_t const* b)
{
  return {a != nullptr ? a : key,
          b != nullptr ? b : key,
          a != nullptr ? ~0ull : 0ull,
          b != nullptr ? ~0ull : 0ull};
}
__device__ __forceinline__ void sum_parked(sum_acc& s, sum_cols const& P, uint64_t row)
{
#ifndef AGG_SQL_SUMFOLD_OFF
  s.m |= (sum_mag(P.a[row]) & P.ma) | (sum_mag(P.b[row]) & P.mb);
#endif
}
/// Whether a kernel specialised on NS state words has a measuring instance of its own, which a
/// launch takes when sql::sum_fold_mode() (the kernels that have one: seg_keys.cu). False where
/// nothing would be measured (1 state word; -DAGG_SQL_SUMFOLD_OFF=1): no second instance then.
template <int NS>
constexpr bool SUM_INSTANCE = false;
/// Once per thread, after its rows.
__device__ __forceinline__ void sum_done(sum_acc const& s) { (void)s; }
template <int NS>
__device__ __forceinline__ void sum_rows_done(sum_acc const& s)
{
  if (NS > 1) sum_done(s);
}
template <int W0, int H>
__device__ __forceinline__ void sum_rows_done_words(sum_acc const& s)
{
  if ((W0 <= 1 && 1 < W0 + H) || (W0 <= 7 && 7 < W0 + H)) sum_done(s);
}
/// One raw row in the kernels with one row per thread (the direct kernels' single rows, the
/// sealed pass): its length is taken where the values are loaded (one 32-bit register from there
/// on) and published after the row's state update, when the registers of that update are free
/// again. (Publishing next to the loads cost those kernels registers.)
template <int NS>
__device__ __forceinline__ uint32_t sum_row_len(uint64_t a, uint64_t b)
{
#ifndef AGG_SQL_SUMFOLD_OFF
  if (NS > 1)
    return 64u - static_cast<uint32_t>(__clzll(static_cast<long long>(sum_mag_row<NS>(a, b))));
#endif
  return 0;
}
template <int NS>
__device__ __forceinline__ void sum_row_done(uint32_t len)
{
  (void)len;
}
/// The direct kernels with run combining: the thread of a run's first row combines the run in
/// registers; the threads of its other rows have nothing else to do, and each of them measures
/// its own row and the one before it (same run, so the first row is covered by the second's
/// thread). The combining thread and its loop stay exactly as they were - any measurement there
/// cost that kernel registers (sm_100, 4 state words: 36 -> 44, also when it was a loop of its
/// own at the end of the thread) - and nothing runs when the mode is off.
template <int NS>
__device__ __forceinline__ void sum_run_row(uint64_t const* a, uint64_t const* b, uint64_t row)
{
  (void)a;
  (void)b;
  (void)row;
}
/// The strip kernel (a thread walks its strip run by run): a run of consecutive rows
/// [first, end) it has just combined in registers is measured by a second loop over the run's
/// summed columns (a few hundred bytes the thread has just read), after the run's state update.
/// Nothing runs when the mode is off.
template <int NS>
__device__ __forceinline__ void sum_run(uint64_t const* a,
                                        uint64_t const* b,
                                        uint64_t first,
                                        uint64_t end)
{
  (void)a;
  (void)b;
  (void)first;
  (void)end;
}

constexpr uint64_t F64_ABS_MASK  = 0x7fffffffffffffffULL;
constexpr uint64_t F64_INF_BITS  = 0x7ff0000000000000ULL;
constexpr uint64_t F64_QNAN_BITS = 0x7ff8000000000000ULL;  // the canonical NaN

/// Total order of float64 bit patterns as a signed integer: -inf < ... < -0.0 < +0.0 < ... < +inf
/// < NaN, every NaN (either sign, any payload) at the same place.
__host__ __device__ __forceinline__ long long f64_ord(uint64_t b)
{
  if ((b & F64_ABS_MASK) > F64_INF_BITS) return static_cast<long long>(F64_QNAN_BITS);
  return static_cast<long long>(b ^ (static_cast<uint64_t>(static_cast<long long>(b) >> 63) >> 1));
}
__device__ __forceinline__ double f64_of(uint64_t v)
{
  return __longlong_as_double(static_cast<long long>(v));
}

/// "v replaces old" in a MIN / MAX state word that holds float64 bits.
__device__ __forceinline__ bool f64_less(uint64_t v, uint64_t old)
{
  return f64_ord(v) < f64_ord(old);
}
__device__ __forceinline__ bool f64_greater(uint64_t v, uint64_t old)
{
  return f64_ord(v) > f64_ord(old);
}
/// Identity of MIN over float64: a value no input is above. With NaN ordered greatest that is NaN
/// itself (MIN of a group of NaNs is NaN); before, +inf.
__host__ __device__ __forceinline__ uint64_t f64_min_identity() { return F64_QNAN_BITS; }
inline uint64_t f64_min_identity_host() { return F64_QNAN_BITS; }

}  // namespace sqldev
}  // namespace sirius::agg_table::detail

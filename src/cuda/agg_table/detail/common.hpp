// Internal vocabulary of the aggregation table (ported from experimental/agg_bench/src/common.hpp
// and table.hpp): state words, their operations, and the description of what a fold reads.
#pragma once

#include <cstdint>

namespace sirius::agg_table::detail {

inline constexpr int MAX_STATE      = 8;
inline constexpr uint64_t EMPTY_KEY = ~0ull;
inline constexpr int NUM_VALUE_COLUMNS =
  4;  // a (i64), b (i64), c (f64), d (f64) of the canonical kernels

/// Partial-aggregate state kinds. Every state is one 8-byte word.
enum class op_t : uint8_t { COUNT, SUM_I64, MIN_I64, MAX_I64, SUM_F64, MIN_F64, MAX_F64 };

inline bool is_f64(op_t op)
{
  return op == op_t::SUM_F64 || op == op_t::MIN_F64 || op == op_t::MAX_F64;
}

/// One state word: the operation and the input value column of the canonical kernels (0=a, 1=b,
/// 2=c, 3=d; -1 for COUNT). For sets the canonical kernels do not cover, `col` is only a label.
struct state_def {
  op_t op;
  int col;
};

/// The state words of a table. The specialised ("canonical") kernels cover the word prefixes
///   1: count(*)   2: + sum_i64(a)   4: + min_i64(b), max_f64(c)
///   8: + sum_f64(d), max_i64(a), min_f64(c), sum_i64(b)
/// (seg_table::init_shape decides); every other set is folded by the generic kernel.
struct state_spec {
  int num_aggs;
  int num_state;
  state_def state[MAX_STATE];
  int num_value_cols;
  uint32_t row_words() const { return 1u + static_cast<uint32_t>(num_state); }  // key + states
};

/// Strided device array: element i is p[i * stride].
struct src_t {
  uint64_t const* p;
  uint32_t stride;
};

enum class kop : uint8_t { ADD_ONE, ADD_I64, ADD_F64, MIN_I64, MAX_I64, MIN_F64, MAX_F64 };

/// The parts of the prototype's round-1 `agg_table` the segmented table uses: the generic source
/// of a fold and the shape of a pre-aggregating launch.
struct flat_table {
  /// What to fold: where each row's key is, and per state word the operation and its operand.
  struct source {
    src_t key;
    int num_state;
    kop op[MAX_STATE];
    src_t val[MAX_STATE];
  };

  /// How a fold launch runs (see the prototype's table.hpp for the measurements).
  struct fold_plan {
    /// 0: every row goes straight to the table. Otherwise a power of two: each thread block first
    /// aggregates its rows in a table of this many slots in shared memory.
    uint32_t local_slots     = 0;
    uint32_t rows_per_thread = 32;
    uint32_t block           = 256;
    /// Direct path: consecutive rows with the same key are combined in registers, up to this many.
    uint32_t run_cap = 8;
    /// Direct path: 1 = new groups counted once per inserting warp, 2 = once per block.
    uint32_t count_mode = 1;
  };
  /// Whether the v2 block-local updates (native 32-bit shared atomics) are the default for a state
  /// of `num_state` words: with 1 and 2 words, not with 4 and more (prototype: AGG_T1_PRE2; GB200
  /// job 23550).
  static bool pre2_default(uint32_t num_state) { return num_state <= 2; }
  /// The shape of a launch for an estimate of the group count. `local_bytes`: shared memory a
  /// block-local table may use on the table's device.
  static fold_plan plan_for(uint64_t est_groups, state_spec const& spec, uint64_t local_bytes);
};

}  // namespace sirius::agg_table::detail

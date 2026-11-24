/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Host-only description of what `sirius_generate` produces. No CUDA and no DuckDB includes: the
// bind, the planner, the operator, the kernels and the tests all share this header.
//
// Value semantics
// ---------------
// Let `g = batch_index * rows_per_batch + row` be the global row. All randomness comes from
// `detail::counter_hash(seed, batch_index, row, column_index, salt)` (generator_detail.cuh), so a
// batch depends only on (spec, batch_index) and can be regenerated in any order, on any GPU.
// INTEGER, BIGINT and VARCHAR columns draw an int64; DOUBLE columns draw a double.
//
//   dist             p1       p2             int64 draw                       double draw
//   uniform          lo       hi             uniform in [lo, hi] inclusive    uniform in [lo, hi)
//   normal           mean     stddev         llround of Box-Muller            Box-Muller
//                                            (salts VALUE_A, VALUE_B)
//   zipf             s (> 0)  -              rank in [1, N], N = cardinality  same rank as double
//                                            (required > 0), rank 1 most
//                                            frequent; rejection-inversion
//                                            (Hormann & Derflinger); s may be
//                                            either side of 1
//   few_distinct     D (>= 1) -              hash % D                         same as double
//   sequential       start    step (0 => 1)  start + step * g                 same
//   sequential_desc  start    step (0 => 1)  start - step * g                 same
//
// - cardinality >= 1 for uniform / normal / sequential*: int64 draws become
//   `lo' + ((v - lo') mod C + C) mod C` with `lo' = p1` for uniform and sequential*, `lo' = 0` for
//   normal; ignored for DOUBLE. For few_distinct it is ignored (D governs); for zipf it is N.
//   0 means no clamp; a value in (0, 1) is rejected, since it would round to "no clamp".
// - INTEGER columns: the int64 draw is narrowed with wrap-around (`static_cast<int32_t>`).
// - Nulls: a row is null iff `to_unit(counter_hash(..., NULL_FLAG)) < null_fraction`.
//   `null_fraction == 0` means no null mask at all.
// - VARCHAR: the draw is clamped to >= 0, then formatted by `detail::format_varchar`: the decimal
//   digits of the key zero-padded on the left to `p1` characters (more digits if the key needs
//   them), then lowercase letters `'a' + (counter_hash(..., STR_CHAR_BASE + i) % 26)` up to a
//   total length `L = p1 + counter_hash(..., STR_LEN) % (p2 - p1 + 1)` (no suffix if `L` is
//   shorter than the prefix). Requires `1 <= p1 <= p2`. So `ORDER BY s` orders by key whenever
//   every key fits in `p1` digits.
//
// `validation_error` holds every rule a column must satisfy; the bind and the kernels both call
// it, so a spec that binds is a spec the kernels accept.

#include "memory/size_arithmetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::op::scan::generator {

/// How a column's values are drawn; see "Value semantics" above.
enum class distribution : uint8_t {
  uniform,
  normal,
  zipf,
  sequential,
  sequential_desc,
  few_distinct
};

/// Output type of a column. SQL names: INTEGER, BIGINT, DOUBLE, VARCHAR.
enum class value_type : uint8_t { int32, int64, float64, varchar };

/// Name of a distribution as written in SQL (exact, lowercase).
[[nodiscard]] inline std::string_view to_string(distribution dist) noexcept
{
  switch (dist) {
    case distribution::uniform: return "uniform";
    case distribution::normal: return "normal";
    case distribution::zipf: return "zipf";
    case distribution::sequential: return "sequential";
    case distribution::sequential_desc: return "sequential_desc";
    case distribution::few_distinct: return "few_distinct";
  }
  return "unknown";
}

/// Inverse of to_string. Matches the exact lowercase names only; anything else is nullopt.
[[nodiscard]] inline std::optional<distribution> parse_distribution(std::string_view name) noexcept
{
  for (auto dist : {distribution::uniform,
                    distribution::normal,
                    distribution::zipf,
                    distribution::sequential,
                    distribution::sequential_desc,
                    distribution::few_distinct}) {
    if (name == to_string(dist)) { return dist; }
  }
  return std::nullopt;
}

/// One generated column. The meaning of p1 / p2 / cardinality depends on `dist` and `type`.
struct column_spec {
  std::string name;
  value_type type{value_type::int64};
  distribution dist{distribution::uniform};
  double p1{0};
  double p2{0};
  double cardinality{0};    // 0 = no clamp, otherwise >= 1
  double null_fraction{0};  // [0, 1]
  friend bool operator==(const column_spec&, const column_spec&) = default;
};

/// Everything that determines the generated data: `num_batches` batches of `rows_per_batch` rows.
struct generator_spec {
  std::vector<column_spec> columns;
  std::uint64_t num_batches{0};
  std::uint64_t rows_per_batch{0};
  std::uint64_t seed{0};
  friend bool operator==(const generator_spec&, const generator_spec&) = default;
};

/// The reason `column` cannot be generated, or nullopt when it is valid. The name is not part of
/// the text: callers prefix it with `sirius_generate: column '<name>': `.
[[nodiscard]] inline std::optional<std::string> validation_error(const column_spec& column)
{
  // Checked first, so that every comparison below is free of NaN.
  if (!std::isfinite(column.p1)) { return "'p1' must be finite"; }
  if (!std::isfinite(column.p2)) { return "'p2' must be finite"; }
  if (!std::isfinite(column.cardinality)) { return "'cardinality' must be finite"; }
  if (!std::isfinite(column.null_fraction)) { return "'null_fraction' must be finite"; }
  if (column.null_fraction < 0 || column.null_fraction > 1) {
    return "null_fraction must be in [0, 1]";
  }
  if (column.cardinality < 0) { return "cardinality must not be negative"; }
  // The kernels round the cardinality to an integer, and 0 means no clamp.
  if (column.cardinality > 0 && column.cardinality < 1) {
    return "cardinality must be 0 (no clamp) or >= 1";
  }
  switch (column.dist) {
    case distribution::uniform:
      if (column.p1 > column.p2) { return "uniform requires p1 (low) <= p2 (high)"; }
      break;
    case distribution::normal:
      if (column.p2 < 0) { return "normal requires p2 (stddev) >= 0"; }
      break;
    case distribution::zipf:
      if (column.p1 <= 0) { return "zipf requires p1 (exponent) > 0"; }
      if (column.cardinality < 1) { return "zipf requires cardinality >= 1"; }
      break;
    case distribution::few_distinct:
      if (column.p1 < 1) { return "few_distinct requires p1 (distinct values) >= 1"; }
      break;
    case distribution::sequential:
    case distribution::sequential_desc: break;
    default: return "unknown distribution";
  }
  switch (column.type) {
    case value_type::int32:
    case value_type::int64:
    case value_type::float64: break;
    case value_type::varchar:
      // A row's length must fit in a cudf::size_type (int32).
      if (column.p1 < 1 || column.p1 > column.p2 ||
          column.p2 > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        return "VARCHAR requires 1 <= p1 (min length) <= p2 (max length) <= 2147483647";
      }
      break;
    default: return "unknown type";
  }
  return std::nullopt;
}

/// Upper bound of the number of decimal digits of any VARCHAR key of `column` (the int64 draw
/// clamped to >= 0) over `total_rows` rows, in [1, 19]. Follows the int64 draws of "Value
/// semantics"; whenever the largest key is not known to stay below 10^18 (a draw that can wrap
/// around int64, an unvalidated spec), the answer is 19, the digits of the largest int64.
[[nodiscard]] inline std::size_t max_key_digits(const column_spec& column,
                                                std::uint64_t total_rows) noexcept
{
  constexpr std::size_t max_digits = 19;
  const double last_row            = total_rows == 0 ? 0 : static_cast<double>(total_rows - 1);
  const bool clamped               = column.cardinality > 0;
  double largest                   = 0;
  switch (column.dist) {
    case distribution::uniform:
      largest = clamped ? std::min(column.p2, column.p1 + column.cardinality) : column.p2;
      break;
    case distribution::normal:
      // Box-Muller over a 53-bit uniform never leaves mean +- 8.6 stddev.
      largest = clamped ? column.cardinality : column.p1 + 9 * std::abs(column.p2);
      break;
    case distribution::zipf: largest = column.cardinality; break;
    case distribution::few_distinct: largest = column.p1; break;
    case distribution::sequential:
    case distribution::sequential_desc: {
      if (clamped) {
        largest = column.p1 + column.cardinality;
        break;
      }
      // The step as the kernels use it: rounded to the nearest integer, 0 => 1.
      const double rounded  = std::round(column.p2);
      const double step     = rounded == 0 ? 1 : rounded;
      const double distance = std::abs(step) * last_row;
      // Far enough to wrap around int64: any key is possible.
      if (!(std::abs(column.p1) + distance < 0x1.0p62)) { return max_digits; }
      const bool rising = (step > 0) == (column.dist == distribution::sequential);
      largest           = column.p1 + (rising ? distance : 0);
      break;
    }
    default: return max_digits;
  }
  // The kernels round parameters to the nearest int64; + 1 covers that.
  largest += 1;
  if (!(largest < 1e18)) { return max_digits; }  // NaN too
  std::size_t digits = 1;
  for (double limit = 10; largest >= limit; limit *= 10) {
    ++digits;
  }
  return digits;
}

/// Bytes per row of the generated batch when only `output_columns` (indices into `spec.columns`,
/// see generate_batch) are produced: the sum of the fixed widths (4 / 8 / 8), plus the expected
/// length and one 4-byte offset for each VARCHAR column, plus `ceil(columns_with_nulls / 8)` for
/// the null masks. Indices outside the spec count as nothing. Never returns 0 for a non-empty
/// selection.
///
/// A VARCHAR row is `max(L, digits(key))` characters with `L` uniform in [p1, p2]. The estimate
/// is the mean of that for a key of `d = max_key_digits` digits: `(p1 + p2) / 2` when `d <= p1`,
/// `d` when `d >= p2`, and `(p1 + p2) / 2 + k * (k + 1) / (2 * (p2 - p1 + 1))` with `k = d - p1`
/// in between. It is an upper bound of the mean row length, since no key has more digits.
[[nodiscard]] inline std::size_t estimated_row_bytes(
  const generator_spec& spec, const std::vector<std::size_t>& output_columns) noexcept
{
  using sirius::memory::saturating_add;
  // Saturates; bind rejects a product that overflows.
  constexpr auto max_rows = std::numeric_limits<std::uint64_t>::max();
  const bool overflows =
    spec.rows_per_batch != 0 && spec.num_batches > max_rows / spec.rows_per_batch;
  const auto total_rows          = overflows ? max_rows : spec.num_batches * spec.rows_per_batch;
  std::size_t bytes              = 0;
  std::size_t columns_with_nulls = 0;
  for (const auto index : output_columns) {
    if (index >= spec.columns.size()) { continue; }
    const auto& column = spec.columns[index];
    switch (column.type) {
      case value_type::int32: bytes = saturating_add(bytes, 4); break;
      case value_type::int64:
      case value_type::float64: bytes = saturating_add(bytes, 8); break;
      case value_type::varchar: {
        // An unvalidated spec may carry NaN, negative or huge lengths; keep the estimate finite.
        constexpr auto max_length = static_cast<double>(std::numeric_limits<std::int32_t>::max());
        const auto digits         = static_cast<double>(max_key_digits(column, total_rows));
        double mean               = (column.p1 + column.p2) / 2;
        if (digits >= column.p2) {
          mean = digits;
        } else if (digits > column.p1) {
          const double below = digits - column.p1;
          mean += below * (below + 1) / (2 * (column.p2 - column.p1 + 1));
        }
        std::size_t chars = static_cast<std::size_t>(digits);  // also what a NaN mean gets
        if (mean >= max_length) {
          chars = static_cast<std::size_t>(max_length);
        } else if (mean > digits) {
          chars = static_cast<std::size_t>(std::ceil(mean));
        }
        bytes = saturating_add(bytes, saturating_add(chars, 4));
        break;
      }
    }
    if (column.null_fraction > 0) { ++columns_with_nulls; }
  }
  return saturating_add(bytes, (columns_with_nulls + 7) / 8);
}

/// Every column of the spec, in spec order: the selection that produces the whole batch.
[[nodiscard]] inline std::vector<std::size_t> all_columns(const generator_spec& spec)
{
  std::vector<std::size_t> columns(spec.columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    columns[i] = i;
  }
  return columns;
}

/// Bytes per row of the whole spec. Never returns 0 for a non-empty spec.
[[nodiscard]] inline std::size_t estimated_row_bytes(const generator_spec& spec)
{
  return estimated_row_bytes(spec, all_columns(spec));
}

/// `rows_per_batch * estimated_row_bytes(spec, output_columns)`, saturating at the largest
/// std::size_t.
[[nodiscard]] inline std::size_t estimated_batch_bytes(
  const generator_spec& spec, const std::vector<std::size_t>& output_columns) noexcept
{
  return sirius::memory::saturating_mul(static_cast<std::size_t>(spec.rows_per_batch),
                                        estimated_row_bytes(spec, output_columns));
}

/// `rows_per_batch * estimated_row_bytes(spec)`, saturating at the largest std::size_t.
[[nodiscard]] inline std::size_t estimated_batch_bytes(const generator_spec& spec)
{
  return estimated_batch_bytes(spec, all_columns(spec));
}

/// Bytes per row that generate_batch holds while it formats a VARCHAR column and that are not
/// part of the batch: the 8-byte int64 key and the 4-byte size of each row. Columns are formatted
/// one at a time, so this is 12 when any of `output_columns` is VARCHAR and 0 otherwise.
[[nodiscard]] inline std::size_t varchar_scratch_row_bytes(
  const generator_spec& spec, const std::vector<std::size_t>& output_columns) noexcept
{
  for (const auto index : output_columns) {
    if (index < spec.columns.size() && spec.columns[index].type == value_type::varchar) {
      return 8 + 4;
    }
  }
  return 0;
}

/// Peak device bytes of one generate_batch call, for reservations: twice the batch, plus the
/// VARCHAR scratch. Columns are generated one at a time, so the peak is the finished columns
/// plus one column's temporaries; 2x the batch covers the draw buffers and a column's offsets
/// and chars, but not the key and sizes of a short VARCHAR column, which can exceed the column
/// itself. Saturates at the largest std::size_t.
[[nodiscard]] inline std::size_t estimated_peak_batch_bytes(
  const generator_spec& spec, const std::vector<std::size_t>& output_columns) noexcept
{
  using sirius::memory::saturating_add;
  using sirius::memory::saturating_mul;
  return saturating_add(saturating_mul(estimated_batch_bytes(spec, output_columns), 2),
                        saturating_mul(static_cast<std::size_t>(spec.rows_per_batch),
                                       varchar_scratch_row_bytes(spec, output_columns)));
}

}  // namespace sirius::op::scan::generator

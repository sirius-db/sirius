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
// - cardinality > 0 for uniform / normal / sequential*: int64 draws become
//   `lo' + ((v - lo') mod C + C) mod C` with `lo' = p1` for uniform and sequential*, `lo' = 0` for
//   normal; ignored for DOUBLE. For few_distinct it is ignored (D governs); for zipf it is N.
// - INTEGER columns: the int64 draw is narrowed with wrap-around (`static_cast<int32_t>`).
// - Nulls: a row is null iff `to_unit(counter_hash(..., NULL_FLAG)) < null_fraction`.
//   `null_fraction == 0` means no null mask at all.
// - VARCHAR: the draw is clamped to >= 0, then formatted by `detail::format_varchar`: the decimal
//   digits of the key zero-padded on the left to `p1` characters (more digits if the key needs
//   them), then lowercase letters `'a' + (counter_hash(..., STR_CHAR_BASE + i) % 26)` up to a
//   total length `L = p1 + counter_hash(..., STR_LEN) % (p2 - p1 + 1)` (no suffix if `L` is
//   shorter than the prefix). Requires `1 <= p1 <= p2`. So `ORDER BY s` orders by key whenever
//   every key fits in `p1` digits.

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
  double cardinality{0};    // 0 = no clamp
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

/// Bytes per row used for reservations: the sum of the fixed widths (4 / 8 / 8), plus
/// `(p1 + p2) / 2 + 4` for each VARCHAR column (mean length plus one offset), plus
/// `ceil(columns_with_nulls / 8)` for the null masks. Never returns 0 for a non-empty spec.
[[nodiscard]] inline std::size_t estimated_row_bytes(const generator_spec& spec) noexcept
{
  constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();
  auto add                 = [](std::size_t a, std::size_t b) noexcept {
    return b > max_bytes - a ? max_bytes : a + b;
  };
  std::size_t bytes              = 0;
  std::size_t columns_with_nulls = 0;
  for (const auto& column : spec.columns) {
    switch (column.type) {
      case value_type::int32: bytes = add(bytes, 4); break;
      case value_type::int64:
      case value_type::float64: bytes = add(bytes, 8); break;
      case value_type::varchar: {
        // An unvalidated spec may carry NaN, negative or huge lengths; keep the estimate finite.
        const double mean = (column.p1 + column.p2) / 2;
        std::size_t chars = 0;
        if (mean >= static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
          chars = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
        } else if (mean > 0) {
          chars = static_cast<std::size_t>(std::ceil(mean));
        }
        bytes = add(bytes, add(chars, 4));
        break;
      }
    }
    if (column.null_fraction > 0) { ++columns_with_nulls; }
  }
  return add(bytes, (columns_with_nulls + 7) / 8);
}

/// `rows_per_batch * estimated_row_bytes`, saturating at the largest std::size_t.
[[nodiscard]] inline std::size_t estimated_batch_bytes(const generator_spec& spec) noexcept
{
  constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();
  const auto row_bytes     = estimated_row_bytes(spec);
  if (row_bytes == 0 || spec.rows_per_batch == 0) { return 0; }
  if (spec.rows_per_batch > max_bytes / row_bytes) { return max_bytes; }
  return static_cast<std::size_t>(spec.rows_per_batch) * row_bytes;
}

}  // namespace sirius::op::scan::generator

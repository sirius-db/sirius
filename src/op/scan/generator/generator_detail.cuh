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

// Internal to the generator: included by generator_batch.cu, generator_strings.cu and their
// tests. The hash helpers define the generator's determinism; changing them changes every
// generated value. The public entry point, generate_batch, is declared in generator_batch.hpp.

#include "op/scan/generator/generator_batch.hpp"
#include "op/scan/generator/generator_spec.hpp"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace sirius::op::scan::generator::detail {

/// Separates the independent random streams of one (seed, batch, row, column) cell.
/// String character `i` uses `STR_CHAR_BASE + i`.
enum salt : uint32_t { VALUE_A = 0, VALUE_B = 1, NULL_FLAG = 2, STR_LEN = 3, STR_CHAR_BASE = 16 };

/// splitmix64 finalizer.
CUDF_HOST_DEVICE inline std::uint64_t mix64(std::uint64_t x) noexcept
{
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

/// The splitmix64 increment, added before each link of the counter_hash chain.
inline constexpr std::uint64_t hash_increment = 0x9e3779b97f4a7c15ULL;

/// The (seed, batch) links of counter_hash: the same for every value of one batch, so it is
/// computed once on the host and passed to the kernels.
CUDF_HOST_DEVICE inline std::uint64_t batch_key(std::uint64_t seed, std::uint64_t batch) noexcept
{
  return mix64((mix64(seed + hash_increment) ^ batch) + hash_increment);
}

/// The row link of counter_hash: the same for every (column, salt) stream of one row.
CUDF_HOST_DEVICE inline std::uint64_t row_key(std::uint64_t batch_key, std::uint64_t row) noexcept
{
  return mix64((batch_key ^ row) + hash_increment);
}

/// The last link of counter_hash, from `row_key(batch_key(seed, batch), row)`.
CUDF_HOST_DEVICE inline std::uint64_t counter_hash_from_row_key(std::uint64_t row_key,
                                                                std::uint32_t column,
                                                                std::uint32_t salt) noexcept
{
  return mix64((row_key ^ ((static_cast<std::uint64_t>(column) << 32) | salt)) + hash_increment);
}

/// counter_hash from `batch_key(seed, batch)`.
CUDF_HOST_DEVICE inline std::uint64_t counter_hash_from_batch_key(std::uint64_t batch_key,
                                                                  std::uint64_t row,
                                                                  std::uint32_t column,
                                                                  std::uint32_t salt) noexcept
{
  return counter_hash_from_row_key(row_key(batch_key, row), column, salt);
}

/// mix64 chain over (seed, batch, row, (uint64(column) << 32) | salt). Each link adds the
/// splitmix64 increment, so the all-zero input does not hash to 0.
CUDF_HOST_DEVICE inline std::uint64_t counter_hash(std::uint64_t seed,
                                                   std::uint64_t batch,
                                                   std::uint64_t row,
                                                   std::uint32_t column,
                                                   std::uint32_t salt) noexcept
{
  return counter_hash_from_batch_key(batch_key(seed, batch), row, column, salt);
}

/// Top 53 bits of `h` as a double in [0, 1).
CUDF_HOST_DEVICE inline double to_unit(std::uint64_t h) noexcept
{
  return static_cast<double>(h >> 11) * 0x1.0p-53;
}

/// Throws std::invalid_argument with the text of validation_error when `column` is not valid.
inline void check_column(const column_spec& column)
{
  if (const auto problem = validation_error(column)) {
    throw std::invalid_argument("sirius_generate: column '" + column.name + "': " + *problem);
  }
}

/// One column's draw, in the vector of its type; the other two vectors are empty. An INTEGER
/// column is drawn already narrowed, so it never holds 8 bytes per row.
struct drawn_column {
  rmm::device_uvector<std::int64_t> ints;    // int64 / varchar (the key before formatting)
  rmm::device_uvector<std::int32_t> ints32;  // int32
  rmm::device_uvector<double> doubles;       // float64
  rmm::device_buffer null_mask;              // empty when null_fraction == 0
  cudf::size_type null_count{0};
};

/// Draws column `column_index` of batch `batch_index` following the value semantics in
/// generator_spec.hpp. Defined in generator_batch.cu.
drawn_column draw_column(const generator_spec& spec,
                         std::size_t column_index,
                         std::uint64_t batch_index,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr);

/// Formats `keys` into a STRING column of `keys.size()` rows (see "VARCHAR" in
/// generator_spec.hpp), taking ownership of the null mask. Negative keys are clamped to 0 here,
/// so the caller passes the raw draw. Defined in generator_strings.cu.
std::unique_ptr<cudf::column> format_varchar(cudf::device_span<const std::int64_t> keys,
                                             rmm::device_buffer null_mask,
                                             cudf::size_type null_count,
                                             const column_spec& column,
                                             std::uint64_t seed,
                                             std::uint64_t batch_index,
                                             std::uint32_t column_index,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace sirius::op::scan::generator::detail

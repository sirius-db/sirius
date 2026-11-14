/*
 * Copyright 2025, Sirius Contributors.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::op::order {

/// When encoded_sorted_order() takes the encoded path.
struct encoded_sort_options {
  /// Tables with fewer rows are left to cudf::sorted_order (the encoded path costs a fixed number
  /// of kernel launches that a small sort does not repay).
  cudf::size_type min_rows = 4096;
  /// A string key column whose longest value exceeds this many bytes is left to
  /// cudf::sorted_order: every 8 bytes of a string are one refinement step.
  cudf::size_type max_string_bytes = 256;
  /// Tables whose sort would need more working memory than this (encoded_sort_working_bytes())
  /// are left to cudf::sorted_order, whose comparison sort needs about a quarter of it. 0: no
  /// limit.
  std::size_t max_working_bytes = 0;
};

/// Device memory encoded_sorted_order() allocates per row at its peak, the result included: the
/// result (4), a 64-bit key and the row index with the radix sort's second buffer of each (24)
/// and, after the first step, the position of each row still tied (4).
inline constexpr std::size_t kEncodedSortWorkingBytesPerRow = 32;

/// Upper bound of the device memory encoded_sorted_order() holds at any time for @p rows rows,
/// apart from the radix sort's histograms (a few MiB at most).
[[nodiscard]] constexpr std::size_t encoded_sort_working_bytes(cudf::size_type rows) noexcept
{
  return static_cast<std::size_t>(rows) * kEncodedSortWorkingBytesPerRow;
}

/// What one encoded_sorted_order() call did; for tests and logs.
struct encoded_sort_stats {
  int words{0};                 ///< 64-bit key words the key columns were encoded into
  int words_skipped{0};         ///< words that separated no tied rows (no sort ran for them)
  int radix_sorts{0};           ///< radix sort calls
  std::int64_t rows_sorted{0};  ///< rows passed to radix sorts, summed over the calls
};

/**
 * @brief Row order of @p keys under (@p column_order, @p null_precedence), computed by radix
 * sorts over an order-preserving fixed-width encoding of the key columns.
 *
 * The result orders the rows exactly as cudf::stable_sorted_order does: by the cuDF row
 * comparator (NaN equal to NaN and above every other value, -0.0 equal to 0.0, strings by
 * unsigned bytes, a null below every value for null_order::BEFORE and above for AFTER, and
 * order::DESCENDING reversing the column, nulls included), with rows of equal keys in their input
 * order. It is therefore a valid result of cudf::sorted_order for the same arguments.
 *
 * How: every key column becomes one or more bit fields whose unsigned order is the column's
 * order (a null flag, the value minus the column minimum, 8-byte big-endian chunks of a string
 * followed by its length), and the fields are packed, as one bit string, into 64-bit words. The
 * words are applied most-significant first: each step radix-sorts the rows that are still tied
 * on all earlier words by (tie group, word), over only the bits in which the word differs inside
 * a group, and drops the rows that became unique. A word that is constant inside every group (a
 * long shared string prefix) costs one pass over the tied rows and no sort.
 *
 * Memory: the result and every temporary are allocated from @p mr, at most
 * encoded_sort_working_bytes(rows) at a time; all temporaries are freed on return. An allocation
 * failure propagates to the caller (nothing is caught), so that a task that runs this inside a
 * memory reservation is retried with a larger one.
 *
 * @return The row order as a non-nullable INT32 column, or nullptr when the keys are not taken
 * by this path and the caller must use cudf::sorted_order: fewer rows than
 * `options.min_rows`, a single fixed-width key of at most 64 bits (cuDF already radix-sorts it),
 * a key type other than fixed-width or STRING, a string longer than
 * `options.max_string_bytes`, or more working memory than `options.max_working_bytes`.
 */
std::unique_ptr<cudf::column> encoded_sorted_order(
  cudf::table_view const& keys,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  encoded_sort_options const& options = {},
  encoded_sort_stats* stats           = nullptr);

}  // namespace sirius::op::order

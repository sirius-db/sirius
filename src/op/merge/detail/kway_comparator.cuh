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

// Device view of kway_key_comparator. CUDA sources only.
//
// Design: no copy of the key data. The comparator owns one device buffer with the
// cudf::column_device_view of every key column of every window (and of their children), built in
// one host buffer and copied in one asynchronous transfer, plus two small device arrays: the
// column orders and the null orders. Its device memory is O(k * key columns) and does not grow
// with the number of rows; creating it never waits on the stream. less() reads the two rows
// directly from their windows and compares them column by column with a type-dispatched element
// comparison.
//
// Nulls: a row is null when its bit in the column's null mask is clear. The null count of a
// window's column is never read, so it may be an upper bound.
//
// Supported key types: fixed-width (integers, floating point, BOOL8, timestamps, durations,
// decimals) and STRING. Ordering follows cuDF's row comparator, so a merge through less() orders
// rows as cudf::merge / cudf::sorted_order do for the same (column_order, null_precedence):
// - floating point: NaN is equal to NaN and greater than every other value;
// - null_order::BEFORE makes a null less than every value and null_order::AFTER greater, two
//   nulls are equal; order::DESCENDING then reverses the whole column comparison, nulls included
//   (DESCENDING with null_order::BEFORE puts nulls last).

#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/cmath>

#include <cstdint>
#include <type_traits>

namespace sirius::op::merge::detail {

/// Three-way comparison of two non-null elements of the same type: negative, zero or positive.
struct kway_element_compare_fn {
  template <typename T>
  __device__ int operator()(cudf::column_device_view const& lhs,
                            cudf::size_type lhs_row,
                            cudf::column_device_view const& rhs,
                            cudf::size_type rhs_row) const noexcept
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      return lhs.element<cudf::string_view>(lhs_row).compare(
        rhs.element<cudf::string_view>(rhs_row));
    } else if constexpr (std::is_floating_point_v<T>) {
      const T a        = lhs.element<T>(lhs_row);
      const T b        = rhs.element<T>(rhs_row);
      const bool a_nan = cuda::std::isnan(a);
      const bool b_nan = cuda::std::isnan(b);
      if (a_nan || b_nan) { return static_cast<int>(a_nan) - static_cast<int>(b_nan); }
      return static_cast<int>(b < a) - static_cast<int>(a < b);
    } else if constexpr (cudf::is_fixed_width<T>()) {
      const T a = lhs.element<T>(lhs_row);
      const T b = rhs.element<T>(rhs_row);
      return static_cast<int>(b < a) - static_cast<int>(a < b);
    } else {
      // kway_key_comparator::create rejects every other type.
      return 0;
    }
  }
};

/// Trivially copyable device view. less() is a strict weak order over (input, row) pairs:
/// key(a) < key(b), or keys equal and ia < ib, or equal keys, ia == ib and ra < rb.
struct kway_key_device_view {
  /// Column c of key window i is columns[i * num_columns + c].
  const cudf::column_device_view* columns{nullptr};
  const cudf::order* column_order{nullptr};          ///< one per key column
  const cudf::null_order* null_precedence{nullptr};  ///< one per key column
  int32_t num_inputs{0};
  cudf::size_type num_columns{0};

  /// Key-only three-way comparison of row @p ra of window @p ia with row @p rb of window @p ib:
  /// negative when a orders first, positive when b orders first, zero when the keys are equal.
  __device__ int compare_keys(int32_t ia,
                              cudf::size_type ra,
                              int32_t ib,
                              cudf::size_type rb) const noexcept
  {
    const cudf::column_device_view* const ta =
      columns + static_cast<std::int64_t>(ia) * num_columns;
    const cudf::column_device_view* const tb =
      columns + static_cast<std::int64_t>(ib) * num_columns;
    for (cudf::size_type c = 0; c < num_columns; ++c) {
      const cudf::column_device_view& ca = ta[c];
      const cudf::column_device_view& cb = tb[c];
      const bool a_null                  = ca.is_null(ra);
      const bool b_null                  = cb.is_null(rb);
      int cmp;
      if (a_null || b_null) {
        if (a_null && b_null) { continue; }
        const bool null_is_less = null_precedence[c] == cudf::null_order::BEFORE;
        cmp                     = (a_null == null_is_less) ? -1 : 1;
      } else {
        cmp = cudf::type_dispatcher(ca.type(), kway_element_compare_fn{}, ca, ra, cb, rb);
        if (cmp == 0) { continue; }
      }
      return column_order[c] == cudf::order::ASCENDING ? cmp : -cmp;
    }
    return 0;
  }

  __device__ bool less(int32_t ia,
                       cudf::size_type ra,
                       int32_t ib,
                       cudf::size_type rb) const noexcept
  {
    const int cmp = compare_keys(ia, ra, ib, rb);
    if (cmp != 0) { return cmp < 0; }
    if (ia != ib) { return ia < ib; }
    return ra < rb;
  }
};

/// The device view of @p cmp; valid while @p cmp is alive. The device state it points at is
/// filled by a copy queued on the stream @p cmp was created with: use the view on that stream.
kway_key_device_view device_view_of(const kway_key_comparator& cmp);

}  // namespace sirius::op::merge::detail

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
// Specialised first key column: when the key is one fixed-width column (and, with
// kway_tuning::specialized_keys = 2, when a fixed-width column is the first of several), less() of
// kway_fixed_key_device_view<T> compares it without the column views and the type dispatch: one
// small entry per window (data pointer, null mask, bit offset), a load of the storage type T and
// one unsigned compare of the normalised value. It orders rows exactly as the generic view does
// (same null rule, NaN rule, -0.0 == 0.0, same tie-break); with more key columns a tie on the
// first one continues in the generic comparison from the second column on. Every other first key
// type (STRING) uses the generic view only. with_key_view() picks the view of a comparator.
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

#include <cudf/utilities/bit.hpp>

#include <cuda/std/cmath>

#include <cstdint>
#include <type_traits>
#include <utility>

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
    return compare_keys_from(0, ia, ra, ib, rb);
  }

  /// compare_keys over the key columns @p first_column, first_column + 1, ...
  __device__ int compare_keys_from(cudf::size_type first_column,
                                   int32_t ia,
                                   cudf::size_type ra,
                                   int32_t ib,
                                   cudf::size_type rb) const noexcept
  {
    const cudf::column_device_view* const ta =
      columns + static_cast<std::int64_t>(ia) * num_columns;
    const cudf::column_device_view* const tb =
      columns + static_cast<std::int64_t>(ib) * num_columns;
    for (cudf::size_type c = first_column; c < num_columns; ++c) {
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

/// How the values of a specialised first key column are stored.
enum class key_storage : std::int8_t {
  generic = 0,  ///< not specialised: the generic view compares every column
  i8,
  i16,
  i32,
  i64,
  i128,
  u8,
  u16,
  u32,
  u64,
  f32,
  f64
};

/// The first key column of one window.
struct fixed_key_input {
  const void* data;                ///< value of row 0 (the column's offset is applied)
  const cudf::bitmask_type* mask;  ///< nullptr when the column has no null mask
  cudf::size_type offset;          ///< bit of row 0 in `mask`
};

/// A value of storage type T as an unsigned integer that orders as the generic comparison of two
/// non-null elements does: integers by value; floating point with -0.0 equal to 0.0, every NaN
/// equal to every NaN and greater than every other value.
template <typename T>
__device__ inline auto normalized_key(T value) noexcept
{
  if constexpr (std::is_floating_point_v<T>) {
    // float -> double is exact and keeps the order.
    const double wide = static_cast<double>(value);
    if (cuda::std::isnan(wide)) { return ~std::uint64_t{0}; }
    // -0.0 == 0.0 is true, so both zeros become +0.0.
    const double canonical   = wide == 0.0 ? 0.0 : wide;
    const std::uint64_t bits = static_cast<std::uint64_t>(__double_as_longlong(canonical));
    return (bits >> 63) != 0 ? ~bits : bits ^ (std::uint64_t{1} << 63);
  } else if constexpr (sizeof(T) == 16) {
    return static_cast<unsigned __int128>(value) ^ (static_cast<unsigned __int128>(1) << 127);
  } else if constexpr (std::is_signed_v<T>) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(value)) ^ (std::uint64_t{1} << 63);
  } else {
    return static_cast<std::uint64_t>(value);
  }
}

/// Trivially copyable device view with a specialised first key column of storage type T. less()
/// is the same strict weak order as kway_key_device_view::less.
template <typename T>
struct kway_fixed_key_device_view {
  const fixed_key_input* inputs{nullptr};  ///< one per window
  kway_key_device_view rest{};             ///< every key column; compared from column 1 on
  int32_t num_inputs{0};
  bool descending{false};    ///< order of the first key column
  bool nulls_first{false};   ///< a null first key orders before a value, the order applied
  bool any_nullable{false};  ///< some window's first key column has a null mask
  bool has_rest{false};      ///< there is more than one key column

  __device__ bool less(int32_t ia,
                       cudf::size_type ra,
                       int32_t ib,
                       cudf::size_type rb) const noexcept
  {
    const fixed_key_input a = inputs[ia];
    const fixed_key_input b = inputs[ib];
    bool first_ties         = false;
    if (any_nullable) {
      const bool a_null = a.mask != nullptr && !cudf::bit_is_set(a.mask, a.offset + ra);
      const bool b_null = b.mask != nullptr && !cudf::bit_is_set(b.mask, b.offset + rb);
      if (a_null || b_null) {
        if (a_null != b_null) { return a_null == nulls_first; }
        first_ties = true;
      }
    }
    if (!first_ties) {
      const auto ka = normalized_key(static_cast<const T*>(a.data)[ra]);
      const auto kb = normalized_key(static_cast<const T*>(b.data)[rb]);
      if (ka != kb) { return (ka < kb) != descending; }
    }
    if (has_rest) {
      const int cmp = rest.compare_keys_from(1, ia, ra, ib, rb);
      if (cmp != 0) { return cmp < 0; }
    }
    if (ia != ib) { return ia < ib; }
    return ra < rb;
  }
};

/// What a kway_fixed_key_device_view is built from; `storage` is key_storage::generic when the
/// comparator is not specialised (the other members are then unset).
struct fixed_key_parts {
  key_storage storage{key_storage::generic};
  const fixed_key_input* inputs{nullptr};
  bool descending{false};
  bool nulls_first{false};
  bool any_nullable{false};
};

/// The device view of @p cmp; valid while @p cmp is alive. The device state it points at is
/// filled by a copy queued on the stream @p cmp was created with: use the view on that stream.
kway_key_device_view device_view_of(const kway_key_comparator& cmp);

/// The parts of the specialised view of @p cmp; same lifetime and stream rule as device_view_of.
fixed_key_parts fixed_key_parts_of(const kway_key_comparator& cmp);

template <typename T>
kway_fixed_key_device_view<T> make_fixed_view(const fixed_key_parts& parts,
                                              const kway_key_device_view& rest)
{
  return kway_fixed_key_device_view<T>{parts.inputs,
                                       rest,
                                       rest.num_inputs,
                                       parts.descending,
                                       parts.nulls_first,
                                       parts.any_nullable,
                                       rest.num_columns > 1};
}

/// Calls @p fn with the device view to compare the rows of @p cmp with: a
/// kway_fixed_key_device_view<T> when the comparator is specialised, else the generic view.
/// @p fn must accept every view type (a generic callable).
template <typename Fn>
void with_key_view(const kway_key_comparator& cmp, Fn&& fn)
{
  const auto rest  = device_view_of(cmp);
  const auto parts = fixed_key_parts_of(cmp);
  switch (parts.storage) {
    case key_storage::i8: fn(make_fixed_view<std::int8_t>(parts, rest)); break;
    case key_storage::i16: fn(make_fixed_view<std::int16_t>(parts, rest)); break;
    case key_storage::i32: fn(make_fixed_view<std::int32_t>(parts, rest)); break;
    case key_storage::i64: fn(make_fixed_view<std::int64_t>(parts, rest)); break;
    case key_storage::i128: fn(make_fixed_view<__int128_t>(parts, rest)); break;
    case key_storage::u8: fn(make_fixed_view<std::uint8_t>(parts, rest)); break;
    case key_storage::u16: fn(make_fixed_view<std::uint16_t>(parts, rest)); break;
    case key_storage::u32: fn(make_fixed_view<std::uint32_t>(parts, rest)); break;
    case key_storage::u64: fn(make_fixed_view<std::uint64_t>(parts, rest)); break;
    case key_storage::f32: fn(make_fixed_view<float>(parts, rest)); break;
    case key_storage::f64: fn(make_fixed_view<double>(parts, rest)); break;
    case key_storage::generic: fn(rest); break;
  }
}

}  // namespace sirius::op::merge::detail

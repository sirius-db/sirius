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

#include "catch.hpp"
#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/detail/kway_merge_detail.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op::merge;
namespace kway_detail = sirius::op::merge::detail;
using namespace sirius::test;

namespace {

// Columns of every window made by make_case.
constexpr cudf::size_type kIntKey        = 0;  // INT32, nullable
constexpr cudf::size_type kStringKey     = 1;  // STRING, nullable
constexpr cudf::size_type kDoubleKey     = 2;  // FLOAT64 with NaN and +inf, nullable
constexpr cudf::size_type kStringPayload = 3;  // STRING, nullable
constexpr cudf::size_type kWindowColumns = 5;  // the last one is a unique INT64 row id

/// One (order, null order) pair per key column of make_case.
struct key_orders {
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
};

/// The four (order, null order) combinations, indexed 0..3.
std::pair<cudf::order, cudf::null_order> order_variant(std::size_t variant)
{
  return {(variant & 1U) != 0 ? cudf::order::DESCENDING : cudf::order::ASCENDING,
          (variant & 2U) != 0 ? cudf::null_order::AFTER : cudf::null_order::BEFORE};
}

/// Key i gets order_variant(first + i * step): step 0 gives every key the same combination.
key_orders make_orders(std::size_t first, std::size_t step)
{
  key_orders out;
  for (std::size_t key = 0; key < 3; ++key) {
    const auto [order, null_order] = order_variant((first + key * step) % 4);
    out.column_order.push_back(order);
    out.null_precedence.push_back(null_order);
  }
  return out;
}

merge_key_spec make_keys(const key_orders& orders)
{
  return merge_key_spec{.key_cols        = {kIntKey, kStringKey, kDoubleKey},
                        .column_order    = orders.column_order,
                        .null_precedence = orders.null_precedence};
}

/// Makes the rows with valids[i] == 0 NULL. One host-built mask, so it stays cheap for millions
/// of rows.
void set_nulls(cudf::column& column, const std::vector<std::uint8_t>& valids)
{
  const auto rows = static_cast<cudf::size_type>(valids.size());
  if (rows == 0) { return; }
  std::vector<cudf::bitmask_type> words(
    cudf::bitmask_allocation_size_bytes(rows) / sizeof(cudf::bitmask_type), 0);
  cudf::size_type null_count = 0;
  for (std::size_t i = 0; i < valids.size(); ++i) {
    if (valids[i] != 0) {
      words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++null_count;
    }
  }
  rmm::device_buffer mask(words.data(),
                          words.size() * sizeof(cudf::bitmask_type),
                          cudf::get_default_stream(),
                          cudf::get_current_device_resource_ref());
  column.set_null_mask(std::move(mask), null_count);
}

template <typename T>
std::unique_ptr<cudf::column> fixed_column(const std::vector<T>& values, cudf::type_id type_id)
{
  if (values.empty()) { return cudf::make_empty_column(type_id); }
  return merge_utils::make_column(values, type_id);
}

std::unique_ptr<cudf::column> string_column(const std::vector<std::string>& values)
{
  if (values.empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }
  return merge_utils::make_string_column(values);
}

/// k sorted windows that are slices of one table, so most of them have a non-zero offset and a
/// null mask that does not start on a word boundary.
struct merge_case {
  /// Owns the rows. Column 0 is the window index; the others are the window columns. Sorted by
  /// (window, keys): the windows are consecutive slices of it.
  std::unique_ptr<cudf::table> all;
  std::vector<cudf::table_view> windows;
  std::vector<cudf::size_type> lengths;
  merge_key_spec keys;

  [[nodiscard]] std::int64_t total_rows() const { return all->num_rows(); }
};

/// Random windows of the given lengths. The keys come from small domains, so equal keys are
/// frequent inside a window and across windows.
merge_case make_case(const std::vector<cudf::size_type>& lengths,
                     const key_orders& orders,
                     std::uint32_t seed)
{
  std::mt19937 rng(seed);
  const std::vector<std::string> strings{"", "a", "ab", "b", "ba"};
  const std::vector<double> doubles{-1.5,
                                    0.0,
                                    2.25,
                                    std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::infinity()};
  const auto total = std::accumulate(lengths.begin(), lengths.end(), std::size_t{0});

  std::vector<std::int32_t> window_ids;
  std::vector<std::int32_t> int_keys;
  std::vector<std::string> string_keys;
  std::vector<double> double_keys;
  std::vector<std::string> string_payloads;
  std::vector<std::int64_t> row_ids;
  std::vector<std::uint8_t> int_valids;
  std::vector<std::uint8_t> string_valids;
  std::vector<std::uint8_t> double_valids;
  std::vector<std::uint8_t> payload_valids;
  window_ids.reserve(total);
  int_keys.reserve(total);
  string_keys.reserve(total);
  double_keys.reserve(total);
  string_payloads.reserve(total);
  row_ids.reserve(total);
  for (std::size_t w = 0; w < lengths.size(); ++w) {
    for (cudf::size_type r = 0; r < lengths[w]; ++r) {
      const auto id = static_cast<std::int64_t>(row_ids.size());
      window_ids.push_back(static_cast<std::int32_t>(w));
      int_valids.push_back(rng() % 10 != 0);
      string_valids.push_back(rng() % 10 != 0);
      double_valids.push_back(rng() % 10 != 0);
      payload_valids.push_back(rng() % 5 != 0);
      int_keys.push_back(static_cast<std::int32_t>(rng() % 4));
      string_keys.push_back(string_valids.back() != 0 ? strings[rng() % strings.size()] : "");
      double_keys.push_back(doubles[rng() % doubles.size()]);
      string_payloads.push_back(
        payload_valids.back() != 0 ? "w" + std::to_string(w) + "-" + std::to_string(id % 977) : "");
      row_ids.push_back(id);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(fixed_column(window_ids, cudf::type_id::INT32));
  columns.push_back(fixed_column(int_keys, cudf::type_id::INT32));
  set_nulls(*columns.back(), int_valids);
  columns.push_back(string_column(string_keys));
  set_nulls(*columns.back(), string_valids);
  columns.push_back(fixed_column(double_keys, cudf::type_id::FLOAT64));
  set_nulls(*columns.back(), double_valids);
  columns.push_back(string_column(string_payloads));
  set_nulls(*columns.back(), payload_valids);
  columns.push_back(fixed_column(row_ids, cudf::type_id::INT64));
  cudf::table unsorted(std::move(columns));

  merge_case out;
  out.lengths = lengths;
  out.keys    = make_keys(orders);

  // Sort every window by the keys: (window, keys...) over the whole table.
  std::vector<cudf::size_type> sort_cols{0};
  std::vector<cudf::order> sort_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> sort_nulls{cudf::null_order::BEFORE};
  for (std::size_t key = 0; key < out.keys.key_cols.size(); ++key) {
    sort_cols.push_back(out.keys.key_cols[key] + 1);
    sort_order.push_back(out.keys.column_order[key]);
    sort_nulls.push_back(out.keys.null_precedence[key]);
  }
  out.all = cudf::stable_sort_by_key(
    unsorted.view(), unsorted.view().select(sort_cols), sort_order, sort_nulls);

  std::vector<cudf::size_type> window_cols(kWindowColumns);
  std::iota(window_cols.begin(), window_cols.end(), 1);
  std::vector<cudf::size_type> bounds;
  cudf::size_type begin = 0;
  for (const auto length : lengths) {
    bounds.push_back(begin);
    bounds.push_back(begin + length);
    begin += length;
  }
  out.windows = cudf::slice(out.all->view().select(window_cols), bounds);
  return out;
}

/// Window lengths in [n / 2, n]; with more than two windows the second one is empty, and the last
/// window always has n rows.
std::vector<cudf::size_type> random_lengths(std::size_t k, cudf::size_type n, std::mt19937& rng)
{
  std::vector<cudf::size_type> lengths;
  for (std::size_t i = 0; i < k; ++i) {
    const auto span = static_cast<std::uint32_t>(n - n / 2 + 1);
    lengths.push_back(n / 2 + static_cast<cudf::size_type>(rng() % span));
  }
  if (k > 2) { lengths[1] = 0; }
  lengths.back() = n;
  return lengths;
}

/// The oracle: merge_utils::reference_bounded_merge (cudf::merge of all rows with ties broken by
/// (window, row), then the safe-target rule on the host) on the default stream.
bounded_merge_result run_oracle(const merge_case& input,
                                const std::vector<std::uint8_t>& truncated,
                                cudf::size_type max_rows)
{
  return merge_utils::reference_bounded_merge(input.windows,
                                              input.keys,
                                              truncated,
                                              max_rows,
                                              cudf::get_default_stream(),
                                              cudf::get_current_device_resource_ref());
}

bounded_merge_result run_merge(const merge_case& input,
                               const std::vector<std::uint8_t>& truncated,
                               cudf::size_type max_rows)
{
  return bounded_kway_merge(input.windows,
                            input.keys,
                            truncated,
                            max_rows,
                            cudf::get_default_stream(),
                            cudf::get_current_device_resource_ref());
}

/// Runs the kernel and the oracle on @p input and requires equal output and consumed counts.
/// Returns the number of rows emitted.
cudf::size_type check_against_oracle(const merge_case& input,
                                     const std::vector<std::uint8_t>& truncated,
                                     cudf::size_type max_rows)
{
  const auto expected = run_oracle(input, truncated, max_rows);
  const auto actual   = run_merge(input, truncated, max_rows);
  REQUIRE(actual.output != nullptr);
  REQUIRE(actual.output->num_rows() == expected.output->num_rows());
  REQUIRE(actual.output->num_columns() == kWindowColumns);
  REQUIRE(actual.consumed == expected.consumed);
  for (std::size_t i = 0; i < input.lengths.size(); ++i) {
    REQUIRE(actual.consumed[i] <= input.lengths[i]);
  }
  REQUIRE(std::accumulate(actual.consumed.begin(), actual.consumed.end(), std::int64_t{0}) ==
          actual.output->num_rows());
  REQUIRE(merge_utils::tables_equal(actual.output->view(), expected.output->view()));
  return actual.output->num_rows();
}

/// True when the key of row @p a_row of @p a orders before or equal to the key of row @p b_row
/// of @p b.
bool key_not_after(const cudf::table_view& a,
                   cudf::size_type a_row,
                   const cudf::table_view& b,
                   cudf::size_type b_row,
                   const merge_key_spec& keys)
{
  const auto first  = cudf::slice(a.select(keys.key_cols), {a_row, a_row + 1}).front();
  const auto second = cudf::slice(b.select(keys.key_cols), {b_row, b_row + 1}).front();
  const auto pair   = cudf::concatenate(std::vector<cudf::table_view>{first, second});
  return cudf::is_sorted(pair->view(), keys.column_order, keys.null_precedence);
}

/// Sets the current device resource for a scope.
class current_resource_guard {
 public:
  explicit current_resource_guard(rmm::device_async_resource_ref resource)
    : _previous(cudf::set_current_device_resource(
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource}))
  {
  }
  current_resource_guard(const current_resource_guard&)            = delete;
  current_resource_guard& operator=(const current_resource_guard&) = delete;
  ~current_resource_guard() { (void)cudf::set_current_device_resource(std::move(_previous)); }

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> _previous;
};

/// Windows of one non-nullable INT32 key column each.
struct int_windows {
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<cudf::table_view> views;
};

int_windows make_int_windows(const std::vector<std::vector<std::int32_t>>& values)
{
  int_windows out;
  for (const auto& window : values) {
    out.columns.push_back(fixed_column(window, cudf::type_id::INT32));
    out.views.emplace_back(std::vector<cudf::column_view>{out.columns.back()->view()});
  }
  return out;
}

const merge_key_spec kSingleIntKey{.key_cols        = {0},
                                   .column_order    = {cudf::order::ASCENDING},
                                   .null_precedence = {cudf::null_order::BEFORE}};

bounded_merge_result merge_ints(const int_windows& windows,
                                const std::vector<std::uint8_t>& truncated,
                                cudf::size_type max_rows)
{
  return bounded_kway_merge(windows.views,
                            kSingleIntKey,
                            truncated,
                            max_rows,
                            cudf::get_default_stream(),
                            cudf::get_current_device_resource_ref());
}

std::vector<std::int32_t> host_ints(const bounded_merge_result& result)
{
  return operator_utils::copy_column_to_host<std::int32_t>(result.output->view().column(0));
}

/// A non-nullable column of @p type whose storage type is T, holding @p values narrowed to T.
template <typename T, typename Source>
std::unique_ptr<cudf::column> storage_column(cudf::data_type type, const std::vector<Source>& values)
{
  std::vector<T> typed;
  typed.reserve(values.size());
  for (const auto v : values) {
    typed.push_back(static_cast<T>(v));
  }
  return std::make_unique<cudf::column>(
    type,
    static_cast<cudf::size_type>(typed.size()),
    rmm::device_buffer(typed.data(),
                       typed.size() * sizeof(T),
                       cudf::get_default_stream(),
                       cudf::get_current_device_resource_ref()),
    rmm::device_buffer{},
    0);
}

/// A key column of @p type: integer-backed types take @p ints, floating-point types @p doubles.
std::unique_ptr<cudf::column> typed_key_column(cudf::data_type type,
                                               const std::vector<std::int64_t>& ints,
                                               const std::vector<double>& doubles)
{
  switch (type.id()) {
    case cudf::type_id::INT8: return storage_column<std::int8_t>(type, ints);
    case cudf::type_id::INT16: return storage_column<std::int16_t>(type, ints);
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS:
    case cudf::type_id::DECIMAL32: return storage_column<std::int32_t>(type, ints);
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
    case cudf::type_id::DECIMAL64: return storage_column<std::int64_t>(type, ints);
    case cudf::type_id::DECIMAL128: return storage_column<__int128_t>(type, ints);
    case cudf::type_id::BOOL8:
    case cudf::type_id::UINT8: return storage_column<std::uint8_t>(type, ints);
    case cudf::type_id::UINT16: return storage_column<std::uint16_t>(type, ints);
    case cudf::type_id::UINT32: return storage_column<std::uint32_t>(type, ints);
    case cudf::type_id::UINT64: return storage_column<std::uint64_t>(type, ints);
    case cudf::type_id::FLOAT32: return storage_column<float>(type, doubles);
    case cudf::type_id::FLOAT64: return storage_column<double>(type, doubles);
    default: throw std::invalid_argument("typed_key_column: unsupported type");
  }
}

/// k sorted windows of (key of @p type, unique INT64 row id), consecutive slices of one table.
/// The keys come from a small domain that includes the extremes of the storage type; windows with
/// an odd index have no nulls.
struct typed_case {
  std::unique_ptr<cudf::table> all;  ///< (window, key, row id), sorted by (window, key)
  std::vector<cudf::table_view> windows;
  merge_key_spec keys;
};

typed_case make_typed_case(cudf::data_type type,
                           const std::vector<cudf::size_type>& lengths,
                           cudf::order order,
                           cudf::null_order null_order,
                           std::uint32_t seed)
{
  std::mt19937 rng(seed);
  const std::vector<std::int64_t> int_domain =
    type.id() == cudf::type_id::BOOL8
      ? std::vector<std::int64_t>{0, 1}
      : std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::min(),
                                  std::numeric_limits<std::int32_t>::min(),
                                  std::numeric_limits<std::int8_t>::min(),
                                  -3,
                                  -1,
                                  0,
                                  1,
                                  2,
                                  std::numeric_limits<std::int8_t>::max(),
                                  std::numeric_limits<std::uint16_t>::max(),
                                  std::numeric_limits<std::int32_t>::max(),
                                  std::numeric_limits<std::int64_t>::max()};
  const std::vector<double> double_domain{-std::numeric_limits<double>::infinity(),
                                          -1.5,
                                          0.0,
                                          2.25,
                                          3.0e38,
                                          std::numeric_limits<double>::infinity(),
                                          std::numeric_limits<double>::quiet_NaN()};
  std::vector<std::int32_t> window_ids;
  std::vector<std::int64_t> ints;
  std::vector<double> doubles;
  std::vector<std::uint8_t> valids;
  std::vector<std::int64_t> row_ids;
  for (std::size_t w = 0; w < lengths.size(); ++w) {
    for (cudf::size_type r = 0; r < lengths[w]; ++r) {
      window_ids.push_back(static_cast<std::int32_t>(w));
      ints.push_back(int_domain[rng() % int_domain.size()]);
      doubles.push_back(double_domain[rng() % double_domain.size()]);
      valids.push_back(w % 2 == 1 || rng() % 8 != 0 ? 1 : 0);
      row_ids.push_back(static_cast<std::int64_t>(row_ids.size()));
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(fixed_column(window_ids, cudf::type_id::INT32));
  columns.push_back(typed_key_column(type, ints, doubles));
  set_nulls(*columns.back(), valids);
  columns.push_back(fixed_column(row_ids, cudf::type_id::INT64));
  const cudf::table unsorted(std::move(columns));

  typed_case out;
  out.keys = merge_key_spec{.key_cols = {0}, .column_order = {order}, .null_precedence = {null_order}};
  out.all  = cudf::stable_sort_by_key(unsorted.view(),
                                     unsorted.view().select({0, 1}),
                                     {cudf::order::ASCENDING, order},
                                     {cudf::null_order::BEFORE, null_order});
  std::vector<cudf::size_type> bounds;
  cudf::size_type begin = 0;
  for (const auto length : lengths) {
    bounds.push_back(begin);
    bounds.push_back(begin + length);
    begin += length;
  }
  out.windows = cudf::slice(out.all->view().select({1, 2}), bounds);
  return out;
}

/// Replaces the kernel switches for a scope.
class tuning_guard {
 public:
  explicit tuning_guard(const kway_detail::kway_tuning& tuning)
  {
    kway_detail::set_kway_tuning_for_test(tuning);
  }
  tuning_guard(const tuning_guard&)            = delete;
  tuning_guard& operator=(const tuning_guard&) = delete;
  ~tuning_guard() { kway_detail::reset_kway_tuning_for_test(); }
};

/// The switches of the baseline kernels: generic comparator, general co-rank kernel, 2048-row
/// tiles and the baseline launch shape.
kway_detail::kway_tuning baseline_tuning()
{
  kway_detail::kway_tuning tuning;
  tuning.specialized_keys = 0;
  tuning.warp_corank      = 0;
  tuning.tile_rows        = 2048;
  tuning.baseline_shape     = 1;
  return tuning;
}

}  // namespace

TEST_CASE("bounded_kway_merge with a specialised key equals the generic kernels and the oracle",
          "[bounded_kway_merge][kway_specialized]")
{
  const auto type_id = GENERATE(cudf::type_id::INT8,
                                cudf::type_id::INT16,
                                cudf::type_id::INT32,
                                cudf::type_id::INT64,
                                cudf::type_id::UINT8,
                                cudf::type_id::UINT16,
                                cudf::type_id::UINT32,
                                cudf::type_id::UINT64,
                                cudf::type_id::FLOAT32,
                                cudf::type_id::FLOAT64,
                                cudf::type_id::BOOL8,
                                cudf::type_id::TIMESTAMP_DAYS,
                                cudf::type_id::TIMESTAMP_MICROSECONDS,
                                cudf::type_id::DURATION_DAYS,
                                cudf::type_id::DURATION_NANOSECONDS,
                                cudf::type_id::DECIMAL32,
                                cudf::type_id::DECIMAL64,
                                cudf::type_id::DECIMAL128);
  const auto order      = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  const auto null_order = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);
  const bool decimal    = type_id == cudf::type_id::DECIMAL32 ||
                          type_id == cudf::type_id::DECIMAL64 || type_id == cudf::type_id::DECIMAL128;
  const cudf::data_type type = decimal ? cudf::data_type{type_id, -2} : cudf::data_type{type_id};
  CAPTURE(static_cast<int>(type_id), static_cast<int>(order), static_cast<int>(null_order));

  const auto seed = static_cast<std::uint32_t>(9000 + static_cast<int>(type_id) * 4 +
                                               static_cast<int>(order) * 2 +
                                               static_cast<int>(null_order));
  // Fan-ins on both sides of the one-warp co-rank kernel's limit; one window is empty.
  for (const std::vector<cudf::size_type>& lengths :
       {std::vector<cudf::size_type>{2500, 0, 4100, 3000, 1},
        std::vector<cudf::size_type>(40, 300)}) {
    const auto input = make_typed_case(type, lengths, order, null_order, seed);
    const auto k     = lengths.size();
    const auto total = static_cast<cudf::size_type>(input.all->num_rows());
    const auto stream = cudf::get_default_stream();
    const auto mr     = cudf::get_current_device_resource_ref();
    for (const cudf::size_type max_rows : {2048, 4097, total}) {
      for (const std::uint8_t flag : {0, 1}) {
        CAPTURE(k, max_rows, static_cast<int>(flag));
        const std::vector<std::uint8_t> truncated(k, flag);
        const auto expected = merge_utils::reference_bounded_merge(
          input.windows, input.keys, truncated, max_rows, stream, mr);
        const auto actual =
          bounded_kway_merge(input.windows, input.keys, truncated, max_rows, stream, mr);
        bounded_merge_result generic;
        {
          const tuning_guard guard(baseline_tuning());
          generic = bounded_kway_merge(input.windows, input.keys, truncated, max_rows, stream, mr);
        }
        REQUIRE(actual.consumed == expected.consumed);
        REQUIRE(actual.consumed == generic.consumed);
        REQUIRE(merge_utils::tables_equal(actual.output->view(), expected.output->view()));
        REQUIRE(merge_utils::tables_equal(actual.output->view(), generic.output->view()));
      }
    }
  }
}

TEST_CASE("bounded_kway_merge treats -0.0 and 0.0, and every NaN, as equal keys",
          "[bounded_kway_merge][kway_specialized]")
{
  const auto type_id = GENERATE(cudf::type_id::FLOAT32, cudf::type_id::FLOAT64);
  const auto order   = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  // Every window holds only zeros of both signs (or only NaNs of both signs), so it is sorted
  // whatever the order of its rows, and the merge must emit the rows by (window, row).
  const double negative_nan = -std::numeric_limits<double>::quiet_NaN();
  for (const auto& values : {std::vector<double>{0.0, -0.0, -0.0, 0.0, -0.0},
                             std::vector<double>{std::numeric_limits<double>::quiet_NaN(),
                                                 negative_nan,
                                                 negative_nan,
                                                 std::numeric_limits<double>::quiet_NaN()}}) {
    std::vector<std::unique_ptr<cudf::table>> tables;
    std::vector<cudf::table_view> views;
    std::vector<std::int64_t> expected_ids;
    for (std::int64_t w = 0; w < 3; ++w) {
      std::vector<std::int64_t> ids;
      for (std::size_t r = 0; r < values.size(); ++r) {
        ids.push_back(w * 100 + static_cast<std::int64_t>(r));
        expected_ids.push_back(ids.back());
      }
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(typed_key_column(cudf::data_type{type_id}, {}, values));
      columns.push_back(fixed_column(ids, cudf::type_id::INT64));
      tables.push_back(std::make_unique<cudf::table>(std::move(columns)));
      views.push_back(tables.back()->view());
    }
    const merge_key_spec keys{
      .key_cols = {0}, .column_order = {order}, .null_precedence = {cudf::null_order::AFTER}};
    const std::vector<std::uint8_t> none(views.size(), 0);
    const auto run = [&] {
      return bounded_kway_merge(views,
                                keys,
                                none,
                                std::numeric_limits<cudf::size_type>::max(),
                                cudf::get_default_stream(),
                                cudf::get_current_device_resource_ref());
    };
    const auto actual = run();
    REQUIRE(operator_utils::copy_column_to_host<std::int64_t>(actual.output->view().column(1)) ==
            expected_ids);
    const tuning_guard guard(baseline_tuning());
    const auto generic = run();
    REQUIRE(operator_utils::copy_column_to_host<std::int64_t>(generic.output->view().column(1)) ==
            expected_ids);
  }
}

TEST_CASE("bounded_kway_merge matches the stable-sort oracle for random windows",
          "[bounded_kway_merge]")
{
  std::size_t combination = 0;
  for (const std::size_t k : {1, 2, 3, 16, 64}) {
    for (const cudf::size_type n : {1, 5, 2049, 50000}) {
      CAPTURE(k, n, combination);
      std::mt19937 rng(static_cast<std::uint32_t>(1000 + combination));
      // A different (order, null order) combination per key, rotating over the combinations.
      const auto input = make_case(random_lengths(k, n, rng), make_orders(combination, 1), rng());
      const std::vector<std::uint8_t> none(k, 0);

      // The merger's call: at most one window of rows.
      const auto emitted = check_against_oracle(input, none, n);
      REQUIRE(emitted == std::min<std::int64_t>(n, input.total_rows()));
      // Everything, with a last tile that is not full.
      const auto all_rows = static_cast<cudf::size_type>(input.total_rows());
      REQUIRE(check_against_oracle(input, none, all_rows + 7) == all_rows);
      // Exactly one full tile, and one row short of the total.
      if (all_rows > 2048) {
        REQUIRE(check_against_oracle(input, none, 2048) == 2048);
        REQUIRE(check_against_oracle(input, none, all_rows - 1) == all_rows - 1);
      }
      ++combination;
    }
  }
}

TEST_CASE("bounded_kway_merge follows every order and null order", "[bounded_kway_merge]")
{
  for (std::size_t variant = 0; variant < 4; ++variant) {
    for (const std::size_t step : {0, 1, 3}) {
      CAPTURE(variant, step);
      std::mt19937 rng(static_cast<std::uint32_t>(77 + variant * 4 + step));
      const auto input = make_case(random_lengths(5, 3000, rng), make_orders(variant, step), rng());
      const std::vector<std::uint8_t> none(5, 0);
      check_against_oracle(input, none, 3000);
      check_against_oracle(input, none, std::numeric_limits<cudf::size_type>::max());
    }
  }
}

TEST_CASE("bounded_kway_merge stops at the safe rank of truncated windows", "[bounded_kway_merge]")
{
  SECTION("hand-made windows")
  {
    const auto windows = make_int_windows({{1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}});

    // Window 0's last row (3) follows rows 0, 1, 2 of window 1 (the 3 of window 1 loses the tie),
    // so its rank is 5 and six rows are safe.
    auto result = merge_ints(windows, {1, 0}, 100);
    REQUIRE(host_ints(result) == std::vector<std::int32_t>{0, 1, 1, 2, 2, 3});
    REQUIRE(result.consumed == std::vector<cudf::size_type>{3, 3});

    // Window 1's last row is the last row overall: nothing is held back.
    result = merge_ints(windows, {0, 1}, 100);
    REQUIRE(host_ints(result) == std::vector<std::int32_t>{0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7});
    REQUIRE(result.consumed == std::vector<cudf::size_type>{3, 8});

    // max_rows below the safe rank wins.
    result = merge_ints(windows, {1, 1}, 4);
    REQUIRE(host_ints(result) == std::vector<std::int32_t>{0, 1, 1, 2});
    REQUIRE(result.consumed == std::vector<cudf::size_type>{2, 2});
  }

  SECTION("a truncated window of equal keys holds back the later windows' ties")
  {
    const auto windows = make_int_windows({{5, 5}, {5, 5, 5}, {4, 5, 6}});
    // Window 1's last 5 follows both rows of window 0 and the 4 of window 2: rank 5.
    const auto result = merge_ints(windows, {0, 1, 0}, 100);
    REQUIRE(host_ints(result) == std::vector<std::int32_t>{4, 5, 5, 5, 5, 5});
    REQUIRE(result.consumed == std::vector<cudf::size_type>{2, 3, 1});
  }

  SECTION("random windows: output equals the oracle and every emitted row is final")
  {
    std::size_t reduced = 0;
    std::size_t cases   = 0;
    for (const std::size_t k : {2, 3, 16}) {
      for (const cudf::size_type n : {5, 2049}) {
        for (std::uint32_t seed = 0; seed < 4; ++seed) {
          CAPTURE(k, n, seed);
          std::mt19937 rng(static_cast<std::uint32_t>(5000 + cases));
          auto lengths = random_lengths(k, n, rng);
          lengths[1]   = n;  // no empty window here: an empty truncated one gives target 0
          std::vector<std::uint8_t> truncated(k, 0);
          for (std::size_t i = 0; i < k; ++i) {
            if (rng() % 2 == 0) {
              // String-limited windows are short.
              truncated[i] = 1;
              lengths[i]   = 1 + static_cast<cudf::size_type>(rng() % (n / 4 + 1));
            }
          }
          const auto input = make_case(lengths, make_orders(cases, 1), rng());
          const auto bound = std::min<std::int64_t>(n, input.total_rows());

          const auto expected = run_oracle(input, truncated, n);
          const auto actual   = run_merge(input, truncated, n);
          REQUIRE(actual.output->num_rows() == expected.output->num_rows());
          REQUIRE(actual.consumed == expected.consumed);
          REQUIRE(merge_utils::tables_equal(actual.output->view(), expected.output->view()));
          reduced += expected.output->num_rows() < bound ? 1 : 0;

          // Final: no unread row of a truncated run can order before an emitted row, because the
          // last emitted row is not after the last row of any truncated window.
          const auto emitted = actual.output->num_rows();
          for (std::size_t i = 0; i < k && emitted > 0; ++i) {
            if (truncated[i] == 0) { continue; }
            CAPTURE(i);
            REQUIRE(key_not_after(
              actual.output->view(), emitted - 1, input.windows[i], lengths[i] - 1, input.keys));
          }
          ++cases;
        }
      }
    }
    // The truncated windows did cut the target in a fair share of the cases.
    REQUIRE(reduced * 4 >= cases);
  }
}

TEST_CASE("bounded_kway_merge edge cases", "[bounded_kway_merge]")
{
  const auto orders = make_orders(0, 0);

  SECTION("all windows empty")
  {
    const auto input = make_case({0, 0, 0}, orders, 1);
    for (const std::uint8_t flag : {0, 1}) {
      const auto result = run_merge(input, std::vector<std::uint8_t>(3, flag), 100);
      REQUIRE(result.output->num_rows() == 0);
      REQUIRE(result.output->num_columns() == kWindowColumns);
      for (cudf::size_type c = 0; c < kWindowColumns; ++c) {
        REQUIRE(result.output->view().column(c).type() == input.windows.front().column(c).type());
      }
      REQUIRE(result.consumed == std::vector<cudf::size_type>{0, 0, 0});
    }
  }

  SECTION("one empty truncated window gives target 0")
  {
    const auto input  = make_case({40, 0, 25}, orders, 2);
    const auto result = run_merge(input, {0, 1, 0}, 100);
    REQUIRE(result.output->num_rows() == 0);
    REQUIRE(result.output->num_columns() == kWindowColumns);
    REQUIRE(result.output->view().column(kStringPayload).type().id() == cudf::type_id::STRING);
    REQUIRE(result.consumed == std::vector<cudf::size_type>{0, 0, 0});
    // The same windows merge once the empty one is known to be exhausted.
    REQUIRE(check_against_oracle(input, {0, 0, 0}, 100) == 65);
  }

  SECTION("max_rows = 0")
  {
    const auto input  = make_case({40, 12, 25}, orders, 3);
    const auto result = run_merge(input, {0, 0, 0}, 0);
    REQUIRE(result.output->num_rows() == 0);
    REQUIRE(result.output->num_columns() == kWindowColumns);
    REQUIRE(result.consumed == std::vector<cudf::size_type>{0, 0, 0});
  }

  SECTION("max_rows above the total with nothing truncated emits every row")
  {
    const auto input = make_case({40, 12, 0, 25}, orders, 4);
    REQUIRE(check_against_oracle(input, {0, 0, 0, 0}, 1000) == 77);
    const auto result = run_merge(input, {0, 0, 0, 0}, 1000);
    REQUIRE(result.consumed == std::vector<cudf::size_type>{40, 12, 0, 25});
  }

  SECTION("a single window is cut at max_rows")
  {
    const auto input = make_case({100}, orders, 5);
    REQUIRE(check_against_oracle(input, {0}, 30) == 30);
    REQUIRE(check_against_oracle(input, {1}, 1000) == 100);
  }

  SECTION("invalid arguments")
  {
    const auto input = make_case({4, 4}, orders, 6);
    const auto call  = [&](std::span<const cudf::table_view> windows,
                          const merge_key_spec& keys,
                          const std::vector<std::uint8_t>& truncated,
                          cudf::size_type max_rows) {
      return bounded_kway_merge(windows,
                                keys,
                                truncated,
                                max_rows,
                                cudf::get_default_stream(),
                                cudf::get_current_device_resource_ref());
    };
    const std::span<const cudf::table_view> windows(input.windows);

    REQUIRE_THROWS_AS(call({}, input.keys, {}, 10), std::invalid_argument);
    REQUIRE_THROWS_AS(call(windows, input.keys, {0}, 10), std::invalid_argument);
    REQUIRE_THROWS_AS(call(windows, input.keys, {0, 0}, -1), std::invalid_argument);

    auto keys = input.keys;
    keys.key_cols.clear();
    keys.column_order.clear();
    keys.null_precedence.clear();
    REQUIRE_THROWS_AS(call(windows, keys, {0, 0}, 10), std::invalid_argument);
    keys = input.keys;
    keys.column_order.pop_back();
    REQUIRE_THROWS_AS(call(windows, keys, {0, 0}, 10), std::invalid_argument);
    keys = input.keys;
    keys.null_precedence.clear();
    REQUIRE_THROWS_AS(call(windows, keys, {0, 0}, 10), std::invalid_argument);
    keys             = input.keys;
    keys.key_cols[0] = kWindowColumns;
    REQUIRE_THROWS_AS(call(windows, keys, {0, 0}, 10), std::invalid_argument);

    // Windows that differ in column count, then in a column's type.
    std::vector<cudf::table_view> mixed{input.windows[0], input.windows[1].select({0, 1, 2, 3})};
    REQUIRE_THROWS_AS(call(mixed, input.keys, {0, 0}, 10), std::invalid_argument);
    mixed[1] = input.windows[1].select({0, 1, 2, 3, 0});
    REQUIRE_THROWS_AS(call(mixed, input.keys, {0, 0}, 10), std::invalid_argument);

    // One window more than the fan-in limit throws even when the target is 0 (empty windows, or
    // an empty truncated one) and no merge kernel would run; the limit itself is accepted.
    const auto empty = make_case({0}, orders, 7);
    std::vector<cudf::table_view> many(kMaxKwayMergeInputs + 1, empty.windows.front());
    std::vector<std::uint8_t> flags(many.size(), 0);
    REQUIRE_THROWS_AS(call(many, empty.keys, flags, 10), std::invalid_argument);
    flags.front() = 1;
    REQUIRE_THROWS_AS(call(many, empty.keys, flags, 10), std::invalid_argument);
    many.assign(many.size(), input.windows.front());
    REQUIRE_THROWS_AS(call(many, input.keys, flags, 10), std::invalid_argument);
    many.pop_back();
    flags.assign(many.size(), 0);
    REQUIRE(call(many, input.keys, flags, 10).output->num_rows() == 10);
  }
}

TEST_CASE("bounded_kway_merge breaks ties by window, then row, across tile boundaries",
          "[bounded_kway_merge]")
{
  // Every key is equal, so the merged order is window 0's rows, then window 1's, ... and each
  // tile boundary falls inside a run of ties. A second, payload column tells the rows apart.
  constexpr std::size_t k         = 7;
  constexpr cudf::size_type n     = 1500;
  constexpr cudf::size_type total = static_cast<cudf::size_type>(k) * n;
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<cudf::table_view> windows;
  for (std::size_t w = 0; w < k; ++w) {
    std::vector<std::int32_t> payload(n);
    std::iota(payload.begin(), payload.end(), static_cast<std::int32_t>(w) * n);
    columns.push_back(fixed_column(std::vector<std::int32_t>(n, 42), cudf::type_id::INT32));
    columns.push_back(fixed_column(payload, cudf::type_id::INT32));
    windows.emplace_back(
      std::vector<cudf::column_view>{columns[columns.size() - 2]->view(), columns.back()->view()});
  }

  for (const cudf::size_type max_rows : {2047, 2048, 2049, 4096 + 5, total}) {
    CAPTURE(max_rows);
    const std::vector<std::uint8_t> none(k, 0);
    const auto result = bounded_kway_merge(windows,
                                           kSingleIntKey,
                                           none,
                                           max_rows,
                                           cudf::get_default_stream(),
                                           cudf::get_current_device_resource_ref());
    REQUIRE(result.output->num_rows() == max_rows);
    const auto payload =
      operator_utils::copy_column_to_host<std::int32_t>(result.output->view().column(1));
    std::vector<std::int32_t> expected(static_cast<std::size_t>(max_rows));
    std::iota(expected.begin(), expected.end(), 0);
    REQUIRE(payload == expected);
    for (std::size_t w = 0; w < k; ++w) {
      const auto before = static_cast<cudf::size_type>(w) * n;
      REQUIRE(result.consumed[w] == std::clamp(max_rows - before, 0, n));
    }
  }

  // Truncating window 3 holds back windows 4.. (their ties follow it) but not window 3 itself.
  std::vector<std::uint8_t> truncated(k, 0);
  truncated[3]      = 1;
  const auto result = bounded_kway_merge(windows,
                                         kSingleIntKey,
                                         truncated,
                                         total,
                                         cudf::get_default_stream(),
                                         cudf::get_current_device_resource_ref());
  REQUIRE(result.output->num_rows() == 4 * n);
  REQUIRE(result.consumed == std::vector<cudf::size_type>{n, n, n, n, 0, 0, 0});
}

TEST_CASE("bounded_kway_merge reads the null masks, not the windows' null counts",
          "[bounded_kway_merge]")
{
  std::mt19937 rng(606);
  // Small windows and windows of several tiles.
  const auto n = GENERATE(cudf::size_type{40}, cudf::size_type{3000});
  auto input   = make_case(random_lengths(9, n, rng), make_orders(1, 1), rng());

  // Every nullable column of every window now reports all of its rows as NULL: an upper bound of
  // the real count (about one row in ten is NULL), as a window over a larger buffer may carry.
  std::vector<cudf::table_view> inflated;
  for (const auto& window : input.windows) {
    std::vector<cudf::column_view> columns;
    for (const auto& column : window) {
      columns.emplace_back(
        column.type(),
        column.size(),
        column.head(),
        column.null_mask(),
        column.nullable() ? column.size() : 0,
        column.offset(),
        std::vector<cudf::column_view>(column.child_begin(), column.child_end()));
    }
    inflated.emplace_back(columns);
  }
  input.windows = std::move(inflated);
  REQUIRE(input.windows.front().column(kIntKey).null_count() == input.lengths.front());

  std::vector<std::uint8_t> truncated(input.lengths.size(), 0);
  const auto all = static_cast<cudf::size_type>(input.total_rows());
  CHECK(check_against_oracle(input, truncated, all) == all);
  check_against_oracle(input, truncated, all / 3);
  truncated[4] = 1;
  truncated[8] = 1;
  CHECK(check_against_oracle(input, truncated, all) < all);

  // The output's own null counts are exact.
  const auto result = run_merge(input, truncated, all);
  const auto rows   = result.output->num_rows();
  REQUIRE(rows > 0);
  for (cudf::size_type c = 0; c < result.output->num_columns(); ++c) {
    const auto column = result.output->view().column(c);
    INFO("column " << c);
    REQUIRE(column.nullable() == (c != kWindowColumns - 1));
    if (column.nullable()) {
      CHECK(column.null_count() == cudf::null_count(column.null_mask(), 0, rows));
    }
  }
}

TEST_CASE("bounded_kway_merge runs on a non-default stream", "[bounded_kway_merge]")
{
  std::mt19937 rng(4242);
  const auto input = make_case(random_lengths(16, 2049, rng), make_orders(1, 1), rng());
  const std::vector<std::uint8_t> none(16, 0);
  const auto expected = run_oracle(input, none, 3000);

  rmm::cuda_stream stream;
  const auto actual = bounded_kway_merge(
    input.windows, input.keys, none, 3000, stream.view(), cudf::get_current_device_resource_ref());
  // The call returns after its last sync, so the output is ready without waiting on the stream.
  REQUIRE(actual.output->num_rows() == expected.output->num_rows());
  REQUIRE(actual.consumed == expected.consumed);
  REQUIRE(merge_utils::tables_equal(actual.output->view(), expected.output->view()));
}

TEST_CASE("bounded_kway_merge allocates in proportion to its output", "[bounded_kway_merge]")
{
  constexpr std::size_t k     = 64;
  constexpr cudf::size_type n = 50000;
  const auto input            = make_case(std::vector<cudf::size_type>(k, n), make_orders(0, 1), 9);
  const std::vector<std::uint8_t> none(k, 0);

  // Everything the call allocates, scratch and output, goes through the adaptor; the windows
  // were allocated before it.
  rmm::mr::statistics_resource_adaptor stats{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
      cudf::get_current_device_resource_ref()}};
  std::unique_ptr<cudf::table> output;
  {
    const current_resource_guard guard{rmm::device_async_resource_ref{stats}};
    auto result = bounded_kway_merge(input.windows,
                                     input.keys,
                                     none,
                                     n,
                                     cudf::get_default_stream(),
                                     rmm::device_async_resource_ref{stats});
    REQUIRE(result.output->num_rows() == n);
    output = std::move(result.output);
  }
  const auto output_bytes = static_cast<std::int64_t>(output->alloc_size());
  const auto peak         = stats.get_bytes_counter().peak;
  CAPTURE(output_bytes, peak);
  REQUIRE(output_bytes > 0);
  REQUIRE(peak < 3 * output_bytes + (std::int64_t{1} << 20));
  // The output outlives the adaptor's scope of use, but not the adaptor itself.
  output.reset();
}

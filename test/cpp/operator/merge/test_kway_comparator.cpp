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
#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::merge::detail {

// Defined in kway_comparator.cu: out[i] = less(ia[i], ra[i], ib[i], rb[i]).
void compare_pairs_for_test(const kway_key_comparator& cmp,
                            cudf::device_span<const int32_t> ia,
                            cudf::device_span<const cudf::size_type> ra,
                            cudf::device_span<const int32_t> ib,
                            cudf::device_span<const cudf::size_type> rb,
                            cudf::device_span<uint8_t> out,
                            rmm::cuda_stream_view stream);

}  // namespace sirius::op::merge::detail

using namespace sirius::op::merge::detail;

namespace {

enum class kind { int32, int64, float64, string, decimal64, timestamp };

/// One key column of one window, on the host. `ints` holds the values of the integer-backed
/// kinds (int32, int64, decimal64, timestamp).
struct host_column {
  kind type{kind::int64};
  std::vector<int64_t> ints;
  std::vector<double> doubles;
  std::vector<std::string> strings;
  std::vector<bool> valid;

  [[nodiscard]] cudf::size_type size() const { return static_cast<cudf::size_type>(valid.size()); }
};

using host_window = std::vector<host_column>;

cudf::data_type cudf_type(kind type)
{
  switch (type) {
    case kind::int32: return cudf::data_type{cudf::type_id::INT32};
    case kind::int64: return cudf::data_type{cudf::type_id::INT64};
    case kind::float64: return cudf::data_type{cudf::type_id::FLOAT64};
    case kind::string: return cudf::data_type{cudf::type_id::STRING};
    case kind::decimal64: return cudf::data_type{cudf::type_id::DECIMAL64, -2};
    case kind::timestamp: return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  }
  throw std::logic_error("unknown kind");
}

template <typename T>
rmm::device_buffer to_device_buffer(const std::vector<T>& host)
{
  return rmm::device_buffer(host.data(),
                            host.size() * sizeof(T),
                            cudf::get_default_stream(),
                            cudf::get_current_device_resource_ref());
}

template <typename T>
std::vector<T> to_host(const T* device, std::size_t count)
{
  std::vector<T> out(count);
  if (count > 0) {
    cudf::get_default_stream().sync();
    REQUIRE(cudaMemcpy(out.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return out;
}

std::unique_ptr<cudf::column> make_column(const host_column& col)
{
  const auto size = col.size();
  if (size == 0) { return cudf::make_empty_column(cudf_type(col.type)); }

  cudf::size_type null_count = 0;
  std::vector<cudf::bitmask_type> mask_words(cudf::bitmask_allocation_size_bytes(size) /
                                             sizeof(cudf::bitmask_type));
  for (cudf::size_type i = 0; i < size; ++i) {
    if (col.valid[i]) {
      mask_words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++null_count;
    }
  }
  // A column without nulls gets no mask, so windows differ in nullability.
  rmm::device_buffer mask = null_count > 0 ? to_device_buffer(mask_words) : rmm::device_buffer{};

  switch (col.type) {
    case kind::int32: {
      std::vector<int32_t> values(col.ints.begin(), col.ints.end());
      return std::make_unique<cudf::column>(
        cudf_type(col.type), size, to_device_buffer(values), std::move(mask), null_count);
    }
    case kind::int64:
    case kind::decimal64:
    case kind::timestamp:
      return std::make_unique<cudf::column>(
        cudf_type(col.type), size, to_device_buffer(col.ints), std::move(mask), null_count);
    case kind::float64:
      return std::make_unique<cudf::column>(
        cudf_type(col.type), size, to_device_buffer(col.doubles), std::move(mask), null_count);
    case kind::string: {
      std::vector<cudf::size_type> offsets(static_cast<std::size_t>(size) + 1, 0);
      std::vector<char> chars;
      for (cudf::size_type i = 0; i < size; ++i) {
        if (col.valid[i]) {
          chars.insert(chars.end(), col.strings[i].begin(), col.strings[i].end());
        }
        offsets[i + 1] = static_cast<cudf::size_type>(chars.size());
      }
      auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                        size + 1,
                                                        to_device_buffer(offsets),
                                                        rmm::device_buffer{},
                                                        0);
      return cudf::make_strings_column(
        size, std::move(offsets_col), to_device_buffer(chars), null_count, std::move(mask));
    }
  }
  throw std::logic_error("unknown kind");
}

std::unique_ptr<cudf::table> make_table(const host_window& window)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (const auto& col : window) {
    columns.push_back(make_column(col));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/// Random column from a small value domain, so equal keys are frequent.
host_column random_column(kind type,
                          cudf::size_type rows,
                          double null_probability,
                          std::mt19937& rng)
{
  static const std::vector<double> double_domain      = {-std::numeric_limits<double>::infinity(),
                                                         -1.5,
                                                         -0.0,
                                                         0.0,
                                                         2.25,
                                                         1e300,
                                                         std::numeric_limits<double>::infinity(),
                                                         std::numeric_limits<double>::quiet_NaN()};
  static const std::vector<std::string> string_domain = {
    "", "a", "ab", "abc", "b", "ba", "B", "\xC3\xA9", "a longer string value", "a longer string"};
  static const std::vector<int64_t> int_domain = {
    std::numeric_limits<int64_t>::min(), -7, -1, 0, 1, 2, 7, std::numeric_limits<int64_t>::max()};

  host_column col;
  col.type = type;
  std::bernoulli_distribution is_null(null_probability);
  for (cudf::size_type i = 0; i < rows; ++i) {
    col.valid.push_back(!is_null(rng));
    switch (type) {
      case kind::int32: col.ints.push_back(static_cast<int64_t>(rng() % 5) - 2); break;
      case kind::int64:
      case kind::decimal64:
      case kind::timestamp: col.ints.push_back(int_domain[rng() % int_domain.size()]); break;
      case kind::float64: col.doubles.push_back(double_domain[rng() % double_domain.size()]); break;
      case kind::string: col.strings.push_back(string_domain[rng() % string_domain.size()]); break;
    }
  }
  return col;
}

/// Three-way comparison of two non-null values: NaN equals NaN and is greater than every number;
/// strings compare bytewise.
int compare_values(const host_column& a,
                   cudf::size_type ra,
                   const host_column& b,
                   cudf::size_type rb)
{
  switch (a.type) {
    case kind::float64: {
      const double x = a.doubles[ra];
      const double y = b.doubles[rb];
      if (std::isnan(x) || std::isnan(y)) {
        return static_cast<int>(std::isnan(x)) - static_cast<int>(std::isnan(y));
      }
      return static_cast<int>(y < x) - static_cast<int>(x < y);
    }
    case kind::string: {
      const int cmp = a.strings[ra].compare(b.strings[rb]);
      return static_cast<int>(cmp > 0) - static_cast<int>(cmp < 0);
    }
    default:
      return static_cast<int>(b.ints[rb] < a.ints[ra]) - static_cast<int>(a.ints[ra] < b.ints[rb]);
  }
}

/// Host reference of kway_key_device_view::less.
bool reference_less(const std::vector<host_window>& windows,
                    const std::vector<cudf::order>& orders,
                    const std::vector<cudf::null_order>& null_orders,
                    int32_t ia,
                    cudf::size_type ra,
                    int32_t ib,
                    cudf::size_type rb)
{
  for (std::size_t c = 0; c < orders.size(); ++c) {
    const auto& a     = windows[ia][c];
    const auto& b     = windows[ib][c];
    const bool a_null = !a.valid[ra];
    const bool b_null = !b.valid[rb];
    int cmp           = 0;
    if (a_null && b_null) {
      cmp = 0;
    } else if (a_null) {
      cmp = null_orders[c] == cudf::null_order::BEFORE ? -1 : 1;
    } else if (b_null) {
      cmp = null_orders[c] == cudf::null_order::BEFORE ? 1 : -1;
    } else {
      cmp = compare_values(a, ra, b, rb);
    }
    if (cmp != 0) { return orders[c] == cudf::order::ASCENDING ? cmp < 0 : cmp > 0; }
  }
  if (ia != ib) { return ia < ib; }
  return ra < rb;
}

struct pair_list {
  std::vector<int32_t> ia;
  std::vector<cudf::size_type> ra;
  std::vector<int32_t> ib;
  std::vector<cudf::size_type> rb;
};

std::vector<uint8_t> run_less(const kway_key_comparator& cmp, const pair_list& pairs)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto upload = [&](const auto& host) {
    using value_type = typename std::decay_t<decltype(host)>::value_type;
    rmm::device_uvector<value_type> out(host.size(), stream, mr);
    if (!host.empty()) {
      REQUIRE(cudaMemcpy(out.data(),
                         host.data(),
                         host.size() * sizeof(value_type),
                         cudaMemcpyHostToDevice) == cudaSuccess);
    }
    return out;
  };
  auto ia = upload(pairs.ia);
  auto ra = upload(pairs.ra);
  auto ib = upload(pairs.ib);
  auto rb = upload(pairs.rb);
  rmm::device_uvector<uint8_t> out(pairs.ia.size(), stream, mr);
  compare_pairs_for_test(cmp,
                         cudf::device_span<const int32_t>(ia.data(), ia.size()),
                         cudf::device_span<const cudf::size_type>(ra.data(), ra.size()),
                         cudf::device_span<const int32_t>(ib.data(), ib.size()),
                         cudf::device_span<const cudf::size_type>(rb.data(), rb.size()),
                         cudf::device_span<uint8_t>(out.data(), out.size()),
                         stream);
  return to_host(out.data(), out.size());
}

kway_key_comparator make_comparator(const std::vector<cudf::table_view>& views,
                                    const std::vector<cudf::order>& orders,
                                    const std::vector<cudf::null_order>& null_orders)
{
  return kway_key_comparator::create(views,
                                     orders,
                                     null_orders,
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref());
}

/// Compares every ordered pair of rows of the windows through the device comparator and checks
/// the result against the host reference and against the position of the two rows in cuDF's
/// stable sort of the concatenated windows (which orders equal keys by input, then by row).
void check_all_pairs(const std::vector<host_window>& windows,
                     const std::vector<cudf::order>& orders,
                     const std::vector<cudf::null_order>& null_orders)
{
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> views;
  for (const auto& window : windows) {
    tables.push_back(make_table(window));
    views.push_back(tables.back()->view());
  }
  auto cmp = make_comparator(views, orders, null_orders);
  REQUIRE(cmp.num_inputs() == static_cast<int32_t>(windows.size()));

  // (input, row) of every row, in concatenation order.
  std::vector<std::pair<int32_t, cudf::size_type>> rows;
  for (std::size_t i = 0; i < windows.size(); ++i) {
    for (cudf::size_type r = 0; r < windows[i].front().size(); ++r) {
      rows.emplace_back(static_cast<int32_t>(i), r);
    }
  }
  if (rows.empty()) { return; }

  auto concatenated = cudf::concatenate(views);
  auto sorted_order = cudf::stable_sorted_order(concatenated->view(), orders, null_orders);
  const auto order  = to_host(sorted_order->view().data<cudf::size_type>(), rows.size());
  std::vector<std::size_t> position(rows.size());
  for (std::size_t p = 0; p < order.size(); ++p) {
    position[static_cast<std::size_t>(order[p])] = p;
  }

  pair_list pairs;
  for (const auto& a : rows) {
    for (const auto& b : rows) {
      pairs.ia.push_back(a.first);
      pairs.ra.push_back(a.second);
      pairs.ib.push_back(b.first);
      pairs.rb.push_back(b.second);
    }
  }
  const auto result = run_less(cmp, pairs);

  std::size_t reference_mismatches = 0;
  std::size_t cudf_mismatches      = 0;
  for (std::size_t a = 0; a < rows.size(); ++a) {
    for (std::size_t b = 0; b < rows.size(); ++b) {
      const bool got      = result[a * rows.size() + b] != 0;
      const bool expected = reference_less(
        windows, orders, null_orders, rows[a].first, rows[a].second, rows[b].first, rows[b].second);
      if (got != expected) { ++reference_mismatches; }
      if (got != (position[a] < position[b])) { ++cudf_mismatches; }
    }
  }
  CHECK(reference_mismatches == 0);
  CHECK(cudf_mismatches == 0);
}

std::vector<host_window> random_windows(const std::vector<kind>& types,
                                        const std::vector<cudf::size_type>& rows,
                                        std::mt19937& rng)
{
  // The middle window has no nulls.
  const std::vector<double> null_probability = {0.2, 0.0, 0.3};
  std::vector<host_window> windows;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    host_window window;
    for (const auto type : types) {
      window.push_back(random_column(type, rows[i], null_probability[i % 3], rng));
    }
    windows.push_back(std::move(window));
  }
  return windows;
}

host_column int64_column(const std::vector<int64_t>& values)
{
  host_column col;
  col.type  = kind::int64;
  col.ints  = values;
  col.valid = std::vector<bool>(values.size(), true);
  return col;
}

}  // namespace

TEST_CASE("kway comparator matches the host reference for single-column keys", "[kway_comparator]")
{
  const auto type = GENERATE(
    kind::int64, kind::float64, kind::string, kind::decimal64, kind::timestamp, kind::int32);
  const auto order      = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  const auto null_order = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);

  std::mt19937 rng(1234 + static_cast<unsigned>(type) * 4 + static_cast<unsigned>(order) * 2 +
                   static_cast<unsigned>(null_order));
  const auto windows = random_windows({type}, {37, 20, 31}, rng);
  check_all_pairs(windows, {order}, {null_order});
}

TEST_CASE("kway comparator orders multi-column keys lexicographically", "[kway_comparator]")
{
  const auto first_order  = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  const auto second_order = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  const auto null_order   = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);

  // The INT32 column draws from five values, so most pairs tie on it and the STRING column
  // decides.
  std::mt19937 rng(99 + static_cast<unsigned>(first_order) * 4 +
                   static_cast<unsigned>(second_order) * 2 + static_cast<unsigned>(null_order));
  const auto windows = random_windows({kind::int32, kind::string}, {33, 28, 30}, rng);
  check_all_pairs(windows, {first_order, second_order}, {null_order, null_order});
}

TEST_CASE("kway comparator breaks ties by input, then by row", "[kway_comparator]")
{
  // Every key is 5 except row 1 of window 2.
  const std::vector<host_window> windows = {
    {int64_column({5, 5, 5})}, {int64_column({5, 5})}, {int64_column({5, 4})}};
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> views;
  for (const auto& window : windows) {
    tables.push_back(make_table(window));
    views.push_back(tables.back()->view());
  }
  auto cmp = make_comparator(views, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});

  pair_list pairs;
  std::vector<uint8_t> expected;
  auto add = [&](int32_t ia, cudf::size_type ra, int32_t ib, cudf::size_type rb, bool less) {
    pairs.ia.push_back(ia);
    pairs.ra.push_back(ra);
    pairs.ib.push_back(ib);
    pairs.rb.push_back(rb);
    expected.push_back(less ? 1 : 0);
  };
  // Equal keys in different inputs: the lower input first, whatever the rows.
  add(0, 2, 1, 0, true);
  add(1, 0, 0, 2, false);
  add(0, 0, 2, 0, true);
  add(2, 0, 1, 1, false);
  add(1, 1, 2, 0, true);
  // Equal keys in the same input: the lower row first.
  add(0, 0, 0, 1, true);
  add(0, 1, 0, 0, false);
  add(1, 0, 1, 1, true);
  add(1, 1, 1, 0, false);
  // A row is not less than itself.
  add(0, 0, 0, 0, false);
  add(1, 1, 1, 1, false);
  add(2, 1, 2, 1, false);
  // A smaller key wins over the tie-break.
  add(2, 1, 0, 0, true);
  add(0, 0, 2, 1, false);
  add(2, 1, 2, 0, true);
  add(2, 0, 2, 1, false);

  CHECK(run_less(cmp, pairs) == expected);
  check_all_pairs(windows, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
}

TEST_CASE("kway comparator handles empty windows and a single input", "[kway_comparator]")
{
  std::mt19937 rng(7);

  SECTION("windows with zero rows")
  {
    const auto type    = GENERATE(kind::int64, kind::string);
    const auto windows = random_windows({type}, {0, 12, 0, 9}, rng);
    check_all_pairs(windows, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  }

  SECTION("every window is empty")
  {
    const auto windows = random_windows({kind::int64, kind::string}, {0, 0}, rng);
    check_all_pairs(windows,
                    {cudf::order::ASCENDING, cudf::order::ASCENDING},
                    {cudf::null_order::AFTER, cudf::null_order::AFTER});
  }

  SECTION("k = 1")
  {
    const auto windows = random_windows({kind::float64, kind::string}, {40}, rng);
    check_all_pairs(windows,
                    {cudf::order::DESCENDING, cudf::order::ASCENDING},
                    {cudf::null_order::AFTER, cudf::null_order::BEFORE});
  }
}

TEST_CASE("kway comparator rejects keys it cannot compare", "[kway_comparator]")
{
  const std::vector<cudf::order> one_order           = {cudf::order::ASCENDING};
  const std::vector<cudf::null_order> one_null_order = {cudf::null_order::AFTER};

  SECTION("nested key column")
  {
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(make_column(int64_column({1, 2})));
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_structs_column(2, std::move(children), 0, rmm::device_buffer{}));
    cudf::table nested(std::move(columns));
    CHECK_THROWS_AS(make_comparator({nested.view()}, one_order, one_null_order),
                    std::invalid_argument);
  }

  SECTION("windows with different key types")
  {
    std::mt19937 rng(3);
    auto ints    = make_table({random_column(kind::int64, 4, 0.0, rng)});
    auto strings = make_table({random_column(kind::string, 4, 0.0, rng)});
    CHECK_THROWS_AS(make_comparator({ints->view(), strings->view()}, one_order, one_null_order),
                    std::invalid_argument);
  }

  SECTION("order vectors that do not match the key columns")
  {
    auto ints = make_table({int64_column({1, 2})});
    CHECK_THROWS_AS(make_comparator({ints->view()}, {}, one_null_order), std::invalid_argument);
    CHECK_THROWS_AS(make_comparator({ints->view()}, one_order, {}), std::invalid_argument);
    CHECK_THROWS_AS(make_comparator({}, one_order, one_null_order), std::invalid_argument);
  }
}

TEST_CASE("kway comparator handles 1024 small windows", "[kway_comparator]")
{
  // Every window has its own column views (and, for STRING, child views) in the comparator's one
  // device buffer: 0 to 2 rows each, some windows with a null mask and some without.
  std::mt19937 rng(1024);
  std::vector<cudf::size_type> rows;
  for (int i = 0; i < 1024; ++i) {
    rows.push_back(static_cast<cudf::size_type>(rng() % 3));
  }
  const auto windows = random_windows({kind::int32, kind::string}, rows, rng);
  check_all_pairs(windows,
                  {cudf::order::ASCENDING, cudf::order::DESCENDING},
                  {cudf::null_order::AFTER, cudf::null_order::BEFORE});
}

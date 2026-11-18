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
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::op::merge::detail::gather_kway;

rmm::cuda_stream_view test_stream() { return cudf::get_default_stream(); }

rmm::device_async_resource_ref test_mr() { return cudf::get_current_device_resource_ref(); }

template <typename T>
rmm::device_buffer to_device_buffer(std::vector<T> const& host)
{
  return rmm::device_buffer{host.data(), host.size() * sizeof(T), test_stream(), test_mr()};
}

template <typename T>
rmm::device_uvector<T> to_device(std::vector<T> const& host)
{
  rmm::device_uvector<T> device{host.size(), test_stream(), test_mr()};
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(device.data(), host.data(), host.size() * sizeof(T), cudaMemcpyDefault) ==
            cudaSuccess);
  }
  return device;
}

/// Null mask and null count of @p validity (true == valid); an empty mask when it is empty.
std::pair<rmm::device_buffer, cudf::size_type> make_mask(std::vector<bool> const& validity)
{
  std::vector<cudf::bitmask_type> words((validity.size() + 31) / 32, 0);
  cudf::size_type null_count = 0;
  for (std::size_t i = 0; i < validity.size(); ++i) {
    if (validity[i]) {
      words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++null_count;
    }
  }
  return {to_device_buffer(words), null_count};
}

/// A fixed-width column of @p type over @p values (its storage representation); nullable when
/// @p validity is not empty.
template <typename T>
std::unique_ptr<cudf::column> make_fixed(cudf::data_type type,
                                         std::vector<T> const& values,
                                         std::vector<bool> const& validity = {})
{
  REQUIRE(cudf::size_of(type) == sizeof(T));
  REQUIRE((validity.empty() || validity.size() == values.size()));
  auto [mask, null_count] = make_mask(validity);
  return std::make_unique<cudf::column>(type,
                                        static_cast<cudf::size_type>(values.size()),
                                        to_device_buffer(values),
                                        std::move(mask),
                                        null_count);
}

std::unique_ptr<cudf::column> make_strings(std::vector<std::string> const& values,
                                           std::vector<bool> const& validity = {})
{
  REQUIRE((validity.empty() || validity.size() == values.size()));
  std::vector<std::int32_t> offsets(values.size() + 1, 0);
  std::string chars;
  for (std::size_t i = 0; i < values.size(); ++i) {
    chars += values[i];
    offsets[i + 1] = static_cast<std::int32_t>(chars.size());
  }
  auto offsets_column     = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                       static_cast<cudf::size_type>(offsets.size()),
                                                       to_device_buffer(offsets),
                                                       rmm::device_buffer{},
                                                       0);
  auto [mask, null_count] = make_mask(validity);
  return cudf::make_strings_column(
    static_cast<cudf::size_type>(values.size()),
    std::move(offsets_column),
    rmm::device_buffer{chars.data(), chars.size(), test_stream(), test_mr()},
    null_count,
    std::move(mask));
}

/// A STRING column whose offsets child is INT64 (the large-strings layout), whatever its size.
std::unique_ptr<cudf::column> make_large_strings(std::vector<std::string> const& values,
                                                 std::vector<bool> const& validity = {})
{
  REQUIRE((validity.empty() || validity.size() == values.size()));
  std::vector<std::int64_t> offsets(values.size() + 1, 0);
  std::string chars;
  for (std::size_t i = 0; i < values.size(); ++i) {
    chars += values[i];
    offsets[i + 1] = static_cast<std::int64_t>(chars.size());
  }
  auto offsets_column     = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                                       static_cast<cudf::size_type>(offsets.size()),
                                                       to_device_buffer(offsets),
                                                       rmm::device_buffer{},
                                                       0);
  auto [mask, null_count] = make_mask(validity);
  return cudf::make_strings_column(
    static_cast<cudf::size_type>(values.size()),
    std::move(offsets_column),
    rmm::device_buffer{chars.data(), chars.size(), test_stream(), test_mr()},
    null_count,
    std::move(mask));
}

/// Sets an environment variable for the scope, restoring its previous value (or absence) after.
class scoped_env {
 public:
  scoped_env(char const* name, std::string const& value) : name_{name}
  {
    if (auto const* old = std::getenv(name)) { old_ = std::string{old}; }
    REQUIRE(::setenv(name, value.c_str(), 1) == 0);
  }
  ~scoped_env()
  {
    if (old_) {
      ::setenv(name_, old_->c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }
  scoped_env(scoped_env const&)            = delete;
  scoped_env& operator=(scoped_env const&) = delete;

 private:
  char const* name_;
  std::optional<std::string> old_;
};

std::vector<bool> host_validity(cudf::column_view const& col)
{
  std::vector<bool> host(static_cast<std::size_t>(col.size()), true);
  if (col.size() == 0 || !col.nullable()) { return host; }
  std::vector<cudf::bitmask_type> words(
    static_cast<std::size_t>((col.offset() + col.size() + 31) / 32));
  REQUIRE(cudaMemcpy(words.data(),
                     col.null_mask(),
                     words.size() * sizeof(cudf::bitmask_type),
                     cudaMemcpyDefault) == cudaSuccess);
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto const bit                    = static_cast<std::size_t>(col.offset() + i);
    host[static_cast<std::size_t>(i)] = ((words[bit / 32] >> (bit % 32)) & 1u) != 0;
  }
  return host;
}

/// The values of a fixed-width column as raw bytes, size_of(type) per row.
std::vector<std::uint8_t> host_bytes(cudf::column_view const& col)
{
  auto const width = cudf::size_of(col.type());
  std::vector<std::uint8_t> host(static_cast<std::size_t>(col.size()) * width);
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(host.data(),
                       static_cast<std::uint8_t const*>(col.head()) +
                         static_cast<std::size_t>(col.offset()) * width,
                       host.size(),
                       cudaMemcpyDefault) == cudaSuccess);
  }
  return host;
}

std::vector<std::string> host_strings(cudf::column_view const& col)
{
  std::vector<std::string> host;
  if (col.size() == 0) { return host; }
  cudf::strings_column_view const strings{col};
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(col.size()) + 1);
  if (strings.offsets().type().id() == cudf::type_id::INT64) {
    REQUIRE(cudaMemcpy(offsets.data(),
                       strings.offsets().data<std::int64_t>() + col.offset(),
                       offsets.size() * sizeof(std::int64_t),
                       cudaMemcpyDefault) == cudaSuccess);
  } else {
    REQUIRE(strings.offsets().type().id() == cudf::type_id::INT32);
    std::vector<std::int32_t> narrow(offsets.size());
    REQUIRE(cudaMemcpy(narrow.data(),
                       strings.offsets().data<std::int32_t>() + col.offset(),
                       narrow.size() * sizeof(std::int32_t),
                       cudaMemcpyDefault) == cudaSuccess);
    std::copy(narrow.begin(), narrow.end(), offsets.begin());
  }
  std::vector<char> chars(static_cast<std::size_t>(offsets.back()));
  if (!chars.empty()) {
    REQUIRE(cudaMemcpy(
              chars.data(), strings.chars_begin(test_stream()), chars.size(), cudaMemcpyDefault) ==
            cudaSuccess);
  }
  host.reserve(static_cast<std::size_t>(col.size()));
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    host.emplace_back(chars.data() + offsets[i], chars.data() + offsets[i + 1]);
  }
  return host;
}

/// Same types, validity and null counts, and the same value in every valid row.
void require_equivalent(cudf::table_view const& actual, cudf::table_view const& expected)
{
  REQUIRE(actual.num_columns() == expected.num_columns());
  REQUIRE(actual.num_rows() == expected.num_rows());
  for (cudf::size_type c = 0; c < actual.num_columns(); ++c) {
    INFO("column " << c);
    auto const& a = actual.column(c);
    auto const& e = expected.column(c);
    REQUIRE(cudf::have_same_types(a, e));
    REQUIRE(a.offset() == 0);
    auto const valid = host_validity(a);
    REQUIRE(valid == host_validity(e));
    REQUIRE(a.null_count() == e.null_count());
    REQUIRE(a.null_count() == std::count(valid.begin(), valid.end(), false));
    if (a.type().id() == cudf::type_id::STRING) {
      auto const actual_strings   = host_strings(a);
      auto const expected_strings = host_strings(e);
      for (std::size_t i = 0; i < valid.size(); ++i) {
        if (valid[i] && actual_strings[i] != expected_strings[i]) {
          FAIL("row " << i << ": '" << actual_strings[i] << "' != '" << expected_strings[i] << "'");
        }
      }
    } else {
      auto const width          = cudf::size_of(a.type());
      auto const actual_bytes   = host_bytes(a);
      auto const expected_bytes = host_bytes(e);
      for (std::size_t i = 0; i < valid.size(); ++i) {
        if (valid[i] &&
            std::memcmp(
              actual_bytes.data() + i * width, expected_bytes.data() + i * width, width) != 0) {
          FAIL("row " << i << " differs");
        }
      }
    }
  }
}

/// The (input, row) map on the host.
struct host_map {
  std::vector<std::int32_t> input;
  std::vector<cudf::size_type> row;
};

/// @p rows random (input, row) pairs over the non-empty windows; repeats are likely.
host_map random_map(std::vector<cudf::table_view> const& windows,
                    std::size_t rows,
                    std::mt19937& rng)
{
  std::vector<std::int32_t> non_empty;
  for (std::size_t i = 0; i < windows.size(); ++i) {
    if (windows[i].num_rows() > 0) { non_empty.push_back(static_cast<std::int32_t>(i)); }
  }
  REQUIRE_FALSE(non_empty.empty());
  host_map map;
  for (std::size_t j = 0; j < rows; ++j) {
    auto const input = non_empty[rng() % non_empty.size()];
    map.input.push_back(input);
    map.row.push_back(static_cast<cudf::size_type>(
      rng() % static_cast<std::uint32_t>(windows[static_cast<std::size_t>(input)].num_rows())));
  }
  return map;
}

std::unique_ptr<cudf::table> run_gather(std::vector<cudf::table_view> const& windows,
                                        host_map const& map)
{
  auto const input = to_device(map.input);
  auto const row   = to_device(map.row);
  auto result      = gather_kway(windows, input, row, test_stream(), test_mr());
  test_stream().synchronize();
  return result;
}

/// cudf::gather over the concatenated windows with the map flattened to row indices.
std::unique_ptr<cudf::table> reference_gather(std::vector<cudf::table_view> const& windows,
                                              host_map const& map)
{
  std::vector<cudf::size_type> starts(windows.size(), 0);
  for (std::size_t i = 1; i < windows.size(); ++i) {
    starts[i] = starts[i - 1] + windows[i - 1].num_rows();
  }
  std::vector<cudf::size_type> flat(map.input.size());
  for (std::size_t j = 0; j < flat.size(); ++j) {
    flat[j] = starts[static_cast<std::size_t>(map.input[j])] + map.row[j];
  }
  auto const indices = to_device(flat);
  auto const all     = cudf::concatenate(windows, test_stream(), test_mr());
  cudf::column_view const indices_view{cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(flat.size()),
                                       indices.data(),
                                       nullptr,
                                       0};
  return cudf::gather(
    all->view(), indices_view, cudf::out_of_bounds_policy::DONT_CHECK, test_stream(), test_mr());
}

/// Runs gather_kway and checks it against the reference and the nullability rule.
void check_gather(std::vector<cudf::table_view> const& windows, host_map const& map)
{
  auto const actual   = run_gather(windows, map);
  auto const expected = reference_gather(windows, map);
  require_equivalent(actual->view(), expected->view());
  if (map.input.empty()) { return; }
  for (cudf::size_type c = 0; c < actual->num_columns(); ++c) {
    INFO("column " << c);
    auto const any_nullable =
      std::any_of(windows.begin(), windows.end(), [c](cudf::table_view const& window) {
        return window.column(c).nullable();
      });
    REQUIRE(actual->view().column(c).nullable() == any_nullable);
  }
}

std::vector<bool> random_validity(std::size_t rows, std::mt19937& rng)
{
  std::vector<bool> validity(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    validity[i] = rng() % 4 != 0;
  }
  return validity;
}

template <typename T>
std::vector<T> random_values(std::size_t rows, std::mt19937& rng)
{
  std::vector<T> values(rows);
  for (auto& value : values) {
    std::uint64_t const low  = (static_cast<std::uint64_t>(rng()) << 32) | rng();
    std::uint64_t const high = (static_cast<std::uint64_t>(rng()) << 32) | rng();
    if constexpr (sizeof(T) == 16) {
      value = static_cast<T>((static_cast<unsigned __int128>(high) << 64) | low);
    } else if constexpr (std::is_floating_point_v<T>) {
      value = static_cast<T>(static_cast<std::int64_t>(low % 2000001) - 1000000) / T{16};
    } else {
      value = static_cast<T>(low);
    }
  }
  return values;
}

std::vector<std::string> random_strings(std::size_t rows, std::mt19937& rng)
{
  std::vector<std::string> values(rows);
  for (auto& value : values) {
    // Mostly short, some empty, a few long.
    auto const kind       = rng() % 10;
    std::size_t const len = kind == 0 ? 0 : (kind == 1 ? 100 + rng() % 200 : 1 + rng() % 12);
    value.resize(len);
    for (auto& ch : value) {
      ch = static_cast<char>('a' + rng() % 26);
    }
  }
  return values;
}

constexpr cudf::size_type kMixedColumns = 10;

/// INT8, INT32, INT64, FLOAT64, DECIMAL64, TIMESTAMP_MS, BOOL8, STRING, INT16, DECIMAL128 with
/// random values; column c is nullable when @p nullable[c].
std::unique_ptr<cudf::table> make_mixed_window(std::size_t rows,
                                               std::vector<bool> const& nullable,
                                               std::mt19937& rng)
{
  REQUIRE(nullable.size() == static_cast<std::size_t>(kMixedColumns));
  auto validity = [&](std::size_t c) {
    return nullable[c] ? random_validity(rows, rng) : std::vector<bool>{};
  };
  using cudf::data_type;
  using cudf::type_id;
  auto bools = random_values<std::uint8_t>(rows, rng);
  for (auto& value : bools) {
    value &= 1u;
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    make_fixed(data_type{type_id::INT8}, random_values<std::int8_t>(rows, rng), validity(0)));
  columns.push_back(
    make_fixed(data_type{type_id::INT32}, random_values<std::int32_t>(rows, rng), validity(1)));
  columns.push_back(
    make_fixed(data_type{type_id::INT64}, random_values<std::int64_t>(rows, rng), validity(2)));
  columns.push_back(
    make_fixed(data_type{type_id::FLOAT64}, random_values<double>(rows, rng), validity(3)));
  columns.push_back(make_fixed(
    data_type{type_id::DECIMAL64, -2}, random_values<std::int64_t>(rows, rng), validity(4)));
  columns.push_back(make_fixed(data_type{type_id::TIMESTAMP_MILLISECONDS},
                               random_values<std::int64_t>(rows, rng),
                               validity(5)));
  columns.push_back(make_fixed(data_type{type_id::BOOL8}, bools, validity(6)));
  columns.push_back(make_strings(random_strings(rows, rng), validity(7)));
  columns.push_back(
    make_fixed(data_type{type_id::INT16}, random_values<std::int16_t>(rows, rng), validity(8)));
  columns.push_back(make_fixed(
    data_type{type_id::DECIMAL128, -4}, random_values<__int128>(rows, rng), validity(9)));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::vector<cudf::table_view> views_of(std::vector<std::unique_ptr<cudf::table>> const& tables)
{
  std::vector<cudf::table_view> views;
  for (auto const& table : tables) {
    views.push_back(table->view());
  }
  return views;
}

/// The same windows through column views that report every row of a nullable column as NULL:
/// a null count that is only an upper bound of the real one.
std::vector<cudf::table_view> with_inflated_null_counts(
  std::vector<cudf::table_view> const& windows)
{
  std::vector<cudf::table_view> out;
  for (auto const& window : windows) {
    std::vector<cudf::column_view> columns;
    for (auto const& col : window) {
      columns.emplace_back(col.type(),
                           col.size(),
                           col.head(),
                           col.null_mask(),
                           col.nullable() ? col.size() : 0,
                           col.offset(),
                           std::vector<cudf::column_view>(col.child_begin(), col.child_end()));
    }
    out.emplace_back(columns);
  }
  return out;
}

}  // namespace

TEST_CASE("gather_kway matches concatenate + gather on a mixed schema", "[kway_gather]")
{
  std::mt19937 rng{20251112};
  for (std::size_t k : {3u, 5u, 8u}) {
    for (int mode = 0; mode < 3; ++mode) {
      // mode 0: no column nullable; 1: every column nullable; 2: nullable per (window, column).
      INFO("k=" << k << " mode=" << mode);
      std::vector<std::unique_ptr<cudf::table>> tables;
      for (std::size_t i = 0; i < k; ++i) {
        std::vector<bool> nullable(kMixedColumns, mode == 1);
        if (mode == 2) {
          for (std::size_t c = 0; c < nullable.size(); ++c) {
            // Column 0 stays non-nullable in every window, column 1 nullable in window 0 only.
            nullable[c] = c == 0 ? false : (c == 1 ? i == 0 : rng() % 2 == 0);
          }
        }
        tables.push_back(make_mixed_window(1 + rng() % 400, nullable, rng));
      }
      auto const windows = views_of(tables);
      check_gather(windows, random_map(windows, 1500, rng));
    }
  }
}

TEST_CASE("gather_kway reads sliced window views", "[kway_gather]")
{
  std::mt19937 rng{7};
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> windows;
  // Offsets that are not multiples of 32 leave the null mask unaligned.
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const ranges{
    {37, 200}, {5, 6}, {64, 300}, {0, 90}, {131, 131}};
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    // Window 1 has no null masks; the others have one on every column.
    tables.push_back(make_mixed_window(300, std::vector<bool>(kMixedColumns, i != 1), rng));
    auto sliced =
      cudf::slice(tables.back()->view(), {ranges[i].first, ranges[i].second}, test_stream());
    REQUIRE(sliced.size() == 1);
    REQUIRE(sliced.front().column(7).offset() == ranges[i].first);
    windows.push_back(sliced.front());
  }
  check_gather(windows, random_map(windows, 2000, rng));

  // Every row of every window exactly once, in window order.
  host_map identity;
  for (std::size_t i = 0; i < windows.size(); ++i) {
    for (cudf::size_type r = 0; r < windows[i].num_rows(); ++r) {
      identity.input.push_back(static_cast<std::int32_t>(i));
      identity.row.push_back(r);
    }
  }
  check_gather(windows, identity);
}

TEST_CASE("gather_kway handles an empty map, empty windows and all-null columns", "[kway_gather]")
{
  std::mt19937 rng{11};
  std::vector<std::unique_ptr<cudf::table>> tables;
  tables.push_back(make_mixed_window(50, std::vector<bool>(kMixedColumns, true), rng));
  tables.push_back(make_mixed_window(0, std::vector<bool>(kMixedColumns, false), rng));
  tables.push_back(make_mixed_window(70, std::vector<bool>(kMixedColumns, false), rng));

  SECTION("empty map")
  {
    auto const windows = views_of(tables);
    auto const result  = run_gather(windows, host_map{});
    REQUIRE(result->num_rows() == 0);
    REQUIRE(result->num_columns() == kMixedColumns);
    REQUIRE(cudf::have_same_types(result->view(), windows.front()));
  }

  SECTION("a window with 0 rows")
  {
    auto const windows = views_of(tables);
    check_gather(windows, random_map(windows, 500, rng));
  }

  SECTION("a 0-row STRING window without children")
  {
    std::vector<std::unique_ptr<cudf::column>> a;
    a.push_back(make_strings({"x", "", "yz"}));
    std::vector<std::unique_ptr<cudf::column>> b;
    b.push_back(cudf::make_empty_column(cudf::type_id::STRING));
    REQUIRE(b.front()->num_children() == 0);
    cudf::table const first{std::move(a)};
    cudf::table const second{std::move(b)};
    std::vector<cudf::table_view> const windows{second.view(), first.view(), second.view()};
    check_gather(windows, host_map{{1, 1, 1, 1}, {2, 0, 1, 2}});
  }

  SECTION("all-null columns")
  {
    auto const all_null = [](std::size_t rows) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(make_fixed(cudf::data_type{cudf::type_id::INT32},
                                   std::vector<std::int32_t>(rows, 7),
                                   std::vector<bool>(rows, false)));
      columns.push_back(
        make_strings(std::vector<std::string>(rows, "null"), std::vector<bool>(rows, false)));
      return std::make_unique<cudf::table>(std::move(columns));
    };
    std::vector<std::unique_ptr<cudf::table>> null_tables;
    null_tables.push_back(all_null(40));
    null_tables.push_back(all_null(3));
    null_tables.push_back(all_null(33));
    auto const windows = views_of(null_tables);
    auto const map     = random_map(windows, 300, rng);
    check_gather(windows, map);
    auto const result = run_gather(windows, map);
    REQUIRE(result->view().column(0).null_count() == 300);
    REQUIRE(result->view().column(1).null_count() == 300);
    REQUIRE(cudf::strings_column_view{result->view().column(1)}.chars_size(test_stream()) == 0);
  }
}

TEST_CASE("gather_kway reads and writes large-strings (int64) offsets", "[kway_gather]")
{
  std::mt19937 rng{31};
  // Window 0 has int64 offsets and is sliced; window 1 has int32 offsets; window 2 int64, nullable.
  auto const values0   = random_strings(120, rng);
  auto const values1   = random_strings(60, rng);
  auto const values2   = random_strings(90, rng);
  auto const validity2 = random_validity(90, rng);
  std::vector<std::unique_ptr<cudf::column>> c0;
  c0.push_back(make_large_strings(values0));
  std::vector<std::unique_ptr<cudf::column>> c1;
  c1.push_back(make_strings(values1));
  std::vector<std::unique_ptr<cudf::column>> c2;
  c2.push_back(make_large_strings(values2, validity2));
  cudf::table const t0{std::move(c0)};
  cudf::table const t1{std::move(c1)};
  cudf::table const t2{std::move(c2)};
  REQUIRE(cudf::strings_column_view{t0.view().column(0)}.offsets().type().id() ==
          cudf::type_id::INT64);

  constexpr cudf::size_type kSliceBegin = 13;
  auto const sliced = cudf::slice(t0.view(), {kSliceBegin, 120}, test_stream()).front();
  std::vector<cudf::table_view> const windows{sliced, t1.view(), t2.view()};
  auto const map = random_map(windows, 800, rng);

  // Expected strings and validity from the host values.
  std::vector<std::string> expected(map.input.size());
  std::vector<bool> expected_valid(map.input.size(), true);
  for (std::size_t j = 0; j < map.input.size(); ++j) {
    auto const row = static_cast<std::size_t>(map.row[j]);
    switch (map.input[j]) {
      case 0: expected[j] = values0[row + kSliceBegin]; break;
      case 1: expected[j] = values1[row]; break;
      default:
        expected_valid[j] = validity2[row];
        expected[j]       = values2[row];
        break;
    }
  }
  auto const null_count = std::count(expected_valid.begin(), expected_valid.end(), false);

  auto const check = [&](cudf::column_view const& out) {
    REQUIRE(out.size() == static_cast<cudf::size_type>(map.input.size()));
    REQUIRE(out.nullable());
    REQUIRE(out.null_count() == null_count);
    auto const valid   = host_validity(out);
    auto const strings = host_strings(out);
    REQUIRE(valid == expected_valid);
    for (std::size_t j = 0; j < expected.size(); ++j) {
      if (valid[j] && strings[j] != expected[j]) {
        FAIL("row " << j << ": '" << strings[j] << "' != '" << expected[j] << "'");
      }
    }
  };

  SECTION("default threshold keeps int32 output offsets")
  {
    auto const result = run_gather(windows, map);
    auto const out    = result->view().column(0);
    REQUIRE(cudf::strings_column_view{out}.offsets().type().id() == cudf::type_id::INT32);
    check(out);
  }

  SECTION("a low threshold switches the output to int64 offsets")
  {
    scoped_env const threshold{"LIBCUDF_LARGE_STRINGS_THRESHOLD", "64"};
    auto const result = run_gather(windows, map);
    auto const out    = result->view().column(0);
    REQUIRE(cudf::strings_column_view{out}.offsets().type().id() == cudf::type_id::INT64);
    check(out);
  }
}

TEST_CASE("gather_kway rejects mismatched and unsupported schemas", "[kway_gather]")
{
  auto const table_of = [](std::unique_ptr<cudf::column> column) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    return std::make_unique<cudf::table>(std::move(columns));
  };
  auto const int32 =
    table_of(make_fixed(cudf::data_type{cudf::type_id::INT32}, std::vector<std::int32_t>{1, 2}));
  auto const int64 =
    table_of(make_fixed(cudf::data_type{cudf::type_id::INT64}, std::vector<std::int64_t>{1, 2}));
  auto const strings  = table_of(make_strings({"a", "b"}));
  auto const decimal2 = table_of(
    make_fixed(cudf::data_type{cudf::type_id::DECIMAL64, -2}, std::vector<std::int64_t>{1, 2}));
  auto const decimal3 = table_of(
    make_fixed(cudf::data_type{cudf::type_id::DECIMAL64, -3}, std::vector<std::int64_t>{1, 2}));
  auto const nested = table_of(cudf::make_structs_column(0, {}, 0, rmm::device_buffer{}));

  host_map const map{{0, 1}, {1, 0}};
  auto const gather = [&map](std::vector<cudf::table_view> const& windows) {
    return run_gather(windows, map);
  };

  REQUIRE_THROWS_AS(gather({int32->view(), int64->view()}), std::invalid_argument);
  REQUIRE_THROWS_AS(gather({int32->view(), strings->view()}), std::invalid_argument);
  REQUIRE_THROWS_AS(gather({decimal2->view(), decimal3->view()}), std::invalid_argument);
  REQUIRE_THROWS_AS(gather({int32->view(), cudf::table_view{{int32->view(), int32->view()}}}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(gather({nested->view(), nested->view()}), std::invalid_argument);
  REQUIRE_THROWS_AS(gather({}), std::invalid_argument);
  // The schema is checked before the empty map returns.
  REQUIRE_THROWS_AS(run_gather({int32->view(), int64->view()}, host_map{}), std::invalid_argument);
  // The two halves of the map must have one length.
  REQUIRE_THROWS_AS(run_gather({int32->view(), int32->view()}, host_map{{0, 1}, {1}}),
                    std::invalid_argument);
  REQUIRE_NOTHROW(gather({int32->view(), int32->view()}));
}

TEST_CASE("gather_kway counts nulls from the masks, not from the windows' null counts",
          "[kway_gather]")
{
  std::mt19937 rng{31};
  std::vector<std::unique_ptr<cudf::table>> tables;
  for (std::size_t i = 0; i < 6; ++i) {
    // Window 0 has no null mask at all; in the others each column is nullable at random.
    std::vector<bool> nullable(kMixedColumns, false);
    for (std::size_t c = 0; i > 0 && c < nullable.size(); ++c) {
      nullable[c] = rng() % 3 != 0;
    }
    tables.push_back(make_mixed_window(50 + rng() % 300, nullable, rng));
  }
  auto const windows  = views_of(tables);
  auto const inflated = with_inflated_null_counts(windows);
  auto const map      = random_map(windows, 2000, rng);

  // require_equivalent also compares the output's null counts with its masks.
  auto const actual   = run_gather(inflated, map);
  auto const expected = reference_gather(windows, map);
  require_equivalent(actual->view(), expected->view());
}

TEST_CASE("gather_kway copies the caller's read-back to the host", "[kway_gather]")
{
  std::mt19937 rng{57};
  std::vector<std::unique_ptr<cudf::table>> tables;
  for (std::size_t i = 0; i < 4; ++i) {
    tables.push_back(make_mixed_window(100, std::vector<bool>(kMixedColumns, i % 2 == 0), rng));
  }
  auto const windows = views_of(tables);

  // Packed after the 10 columns' null counts into 64-bit words: 1 and 37 leave the last word half
  // full, 4 fills it.
  auto const entries = GENERATE(std::size_t{1}, std::size_t{4}, std::size_t{37});
  std::vector<cudf::size_type> values(entries);
  for (auto& value : values) {
    value = static_cast<cudf::size_type>(rng());
  }
  auto const source = to_device(values);

  SECTION("with rows to gather")
  {
    auto const map   = random_map(windows, 700, rng);
    auto const input = to_device(map.input);
    auto const row   = to_device(map.row);
    std::vector<cudf::size_type> destination(entries, -1);
    auto const actual =
      gather_kway(windows, input, row, test_stream(), test_mr(), {source, destination});
    // No synchronize here: the copy has completed when gather_kway returns.
    CHECK(destination == values);
    auto const expected = reference_gather(windows, map);
    require_equivalent(actual->view(), expected->view());
  }

  SECTION("with an empty map")
  {
    auto const none_input = to_device(std::vector<std::int32_t>{});
    auto const none_row   = to_device(std::vector<cudf::size_type>{});
    std::vector<cudf::size_type> destination(entries, -1);
    auto const actual =
      gather_kway(windows, none_input, none_row, test_stream(), test_mr(), {source, destination});
    CHECK(destination == values);
    CHECK(actual->num_rows() == 0);
  }

  SECTION("a destination of another size is rejected")
  {
    auto const map   = random_map(windows, 10, rng);
    auto const input = to_device(map.input);
    auto const row   = to_device(map.row);
    std::vector<cudf::size_type> destination(entries + 1, -1);
    CHECK_THROWS_AS(
      gather_kway(windows, input, row, test_stream(), test_mr(), {source, destination}),
      std::invalid_argument);
  }
}

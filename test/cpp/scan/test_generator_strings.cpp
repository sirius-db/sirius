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

#include "catch.hpp"
#include "op/scan/generator/generator_detail.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using sirius::op::scan::generator::column_spec;
using sirius::op::scan::generator::value_type;
using sirius::op::scan::generator::detail::counter_hash;
using sirius::op::scan::generator::detail::format_varchar;
using sirius::op::scan::generator::detail::salt;

using host_strings = std::vector<std::optional<std::string>>;

constexpr std::uint64_t test_seed    = 42;
constexpr std::uint32_t test_column  = 3;
constexpr std::uint64_t test_batch   = 5;
constexpr std::int64_t max_int64_key = std::numeric_limits<std::int64_t>::max();

column_spec varchar_spec(double p1, double p2)
{
  column_spec spec;
  spec.name = "s";
  spec.type = value_type::varchar;
  spec.p1   = p1;
  spec.p2   = p2;
  return spec;
}

template <typename T>
std::vector<T> download(const void* device, std::size_t count)
{
  std::vector<T> host(count);
  if (count > 0) {
    REQUIRE(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return host;
}

rmm::device_buffer make_null_mask(const std::vector<bool>& valid)
{
  const auto rows = static_cast<cudf::size_type>(valid.size());
  std::vector<cudf::bitmask_type> words(cudf::num_bitmask_words(rows), 0);
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (valid[i]) { cudf::set_bit_unsafe(words.data(), i); }
  }
  return rmm::device_buffer{words.data(),
                            words.size() * sizeof(cudf::bitmask_type),
                            cudf::get_default_stream(),
                            cudf::get_current_device_resource_ref()};
}

/// Copies a STRING column to the host; null rows become nullopt.
host_strings to_host(const cudf::column_view& column)
{
  auto stream = cudf::get_default_stream();
  stream.sync();
  REQUIRE(column.type().id() == cudf::type_id::STRING);
  const auto rows = column.size();
  host_strings result(rows);
  if (rows == 0) { return result; }

  cudf::strings_column_view strings{column};
  const auto offsets_view = strings.offsets();
  std::vector<std::int64_t> offsets(rows + 1);
  if (offsets_view.type().id() == cudf::type_id::INT32) {
    const auto narrow = download<std::int32_t>(offsets_view.data<std::int32_t>(), rows + 1);
    std::copy(narrow.begin(), narrow.end(), offsets.begin());
  } else {
    REQUIRE(offsets_view.type().id() == cudf::type_id::INT64);
    offsets = download<std::int64_t>(offsets_view.data<std::int64_t>(), rows + 1);
  }
  const auto chars = download<char>(strings.chars_begin(stream), offsets.back());
  std::vector<cudf::bitmask_type> mask;
  if (column.nullable()) {
    mask = download<cudf::bitmask_type>(column.null_mask(), cudf::num_bitmask_words(rows));
  }
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (!mask.empty() && !cudf::bit_is_set(mask.data(), i)) { continue; }
    result[i] = std::string(chars.data() + offsets[i], chars.data() + offsets[i + 1]);
  }
  return result;
}

struct format_args {
  std::vector<std::int64_t> keys;
  double p1{1};
  double p2{1};
  std::vector<bool> valid{};  // empty = no null mask
  std::uint64_t seed{test_seed};
  std::uint64_t batch{test_batch};
  std::uint32_t column{test_column};
};

std::unique_ptr<cudf::column> format_column(const format_args& args)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  rmm::device_uvector<std::int64_t> keys(args.keys.size(), stream, mr);
  if (!args.keys.empty()) {
    REQUIRE(cudaMemcpy(keys.data(),
                       args.keys.data(),
                       args.keys.size() * sizeof(std::int64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  rmm::device_buffer mask;
  cudf::size_type null_count = 0;
  if (!args.valid.empty()) {
    REQUIRE(args.valid.size() == args.keys.size());
    mask = make_null_mask(args.valid);
    null_count =
      static_cast<cudf::size_type>(std::count(args.valid.begin(), args.valid.end(), false));
  }
  auto column = format_varchar(cudf::device_span<const std::int64_t>{keys.data(), keys.size()},
                               std::move(mask),
                               null_count,
                               varchar_spec(args.p1, args.p2),
                               args.seed,
                               args.batch,
                               args.column,
                               stream,
                               mr);
  stream.sync();  // `keys` must outlive the kernels
  return column;
}

host_strings format(const format_args& args) { return to_host(format_column(args)->view()); }

/// The VARCHAR rule of generator_spec.hpp, on the host.
std::string reference(std::int64_t key, std::uint64_t row, const format_args& args)
{
  const auto p1 = static_cast<std::uint64_t>(args.p1);
  const auto p2 = static_cast<std::uint64_t>(args.p2);
  auto text     = std::to_string(std::max<std::int64_t>(key, 0));
  if (text.size() < p1) { text.insert(0, p1 - text.size(), '0'); }
  const auto length =
    p1 + counter_hash(args.seed, args.batch, row, args.column, salt::STR_LEN) % (p2 - p1 + 1);
  for (std::uint32_t i = 0; text.size() < length; ++i) {
    text.push_back(static_cast<char>(
      'a' + counter_hash(args.seed, args.batch, row, args.column, salt::STR_CHAR_BASE + i) % 26));
  }
  return text;
}

std::size_t digit_count(std::int64_t key) { return std::to_string(key).size(); }

}  // namespace

TEST_CASE("generator strings: fixed length is the zero-padded key", "[generator_strings]")
{
  const auto strings = format({.keys = {0, 7, 42, 123456}, .p1 = 8, .p2 = 8});
  REQUIRE(strings == host_strings{std::string{"00000000"}, "00000007", "00000042", "00123456"});
}

TEST_CASE("generator strings: a key wider than p1 keeps all its digits", "[generator_strings]")
{
  const auto strings = format({.keys = {123456, 999, 1000, max_int64_key, -5}, .p1 = 3, .p2 = 3});
  REQUIRE(strings == host_strings{std::string{"123456"},
                                  "999",
                                  "1000",
                                  "9223372036854775807",
                                  "000"});  // negative keys clamp to 0
}

TEST_CASE("generator strings: variable length has a deterministic lowercase suffix",
          "[generator_strings]")
{
  constexpr std::size_t p1 = 4;
  constexpr std::size_t p2 = 40;
  format_args args{.p1 = static_cast<double>(p1), .p2 = static_cast<double>(p2)};
  for (std::int64_t i = 0; i < 2000; ++i) {
    args.keys.push_back(i % 7 == 0 ? i * 1000003 : i);  // some keys need more than p1 digits
  }
  const auto strings = format(args);
  REQUIRE(strings.size() == args.keys.size());

  std::size_t shortest = p2;
  std::size_t longest  = 0;
  for (std::size_t row = 0; row < strings.size(); ++row) {
    REQUIRE(strings[row].has_value());
    const auto& text  = *strings[row];
    const auto prefix = std::max(p1, digit_count(args.keys[row]));
    REQUIRE(text.size() >= prefix);
    REQUIRE(text.size() <= p2);
    auto expected_prefix = std::to_string(args.keys[row]);
    expected_prefix.insert(0, prefix - expected_prefix.size(), '0');
    REQUIRE(text.substr(0, prefix) == expected_prefix);
    REQUIRE(std::all_of(text.begin() + static_cast<std::ptrdiff_t>(prefix), text.end(), [](char c) {
      return c >= 'a' && c <= 'z';
    }));
    shortest = std::min(shortest, text.size());
    longest  = std::max(longest, text.size());
  }
  // 2000 draws over 37 lengths reach both ends of [p1, p2].
  REQUIRE(shortest == p1);
  REQUIRE(longest == p2);

  SECTION("same inputs give identical output") { REQUIRE(format(args) == strings); }
  SECTION("another batch, seed or column gives different suffixes")
  {
    auto other_batch  = args;
    other_batch.batch = args.batch + 1;
    REQUIRE(format(other_batch) != strings);
    auto other_seed = args;
    other_seed.seed = args.seed + 1;
    REQUIRE(format(other_seed) != strings);
    auto other_column   = args;
    other_column.column = args.column + 1;
    REQUIRE(format(other_column) != strings);
  }
}

TEST_CASE("generator strings: strings order like their keys", "[generator_strings]")
{
  format_args args{.p1 = 6, .p2 = 24};
  for (std::int64_t key = 0; key < 3000; ++key) {
    args.keys.push_back(key * 331);  // ascending, distinct, at most 6 digits
  }
  const auto column = format_column(args);
  const auto order  = cudf::sorted_order(cudf::table_view{{column->view()}});
  cudf::get_default_stream().sync();
  const auto indices =
    download<cudf::size_type>(order->view().data<cudf::size_type>(), order->size());
  REQUIRE(indices.size() == args.keys.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    REQUIRE(indices[i] == static_cast<cudf::size_type>(i));
  }
}

TEST_CASE("generator strings: masked rows are null", "[generator_strings]")
{
  format_args args{.p1 = 5, .p2 = 12};
  for (std::int64_t key = 0; key < 200; ++key) {
    args.keys.push_back(key * 17);
    args.valid.push_back(key % 3 != 1);
  }
  const auto column = format_column(args);
  REQUIRE(column->null_count() == 67);
  const auto strings = to_host(column->view());

  auto without_nulls = args;
  without_nulls.valid.clear();
  const auto all_valid_column = format_column(without_nulls);
  REQUIRE(all_valid_column->null_count() == 0);
  REQUIRE_FALSE(all_valid_column->nullable());
  const auto all_valid = to_host(all_valid_column->view());

  for (std::size_t row = 0; row < strings.size(); ++row) {
    if (args.valid[row]) {
      REQUIRE(strings[row] == all_valid[row]);
    } else {
      REQUIRE_FALSE(strings[row].has_value());
    }
  }

  SECTION("null rows take no bytes")
  {
    cudf::strings_column_view view{column->view()};
    std::int64_t expected = 0;
    for (std::size_t row = 0; row < strings.size(); ++row) {
      if (strings[row]) { expected += static_cast<std::int64_t>(strings[row]->size()); }
    }
    REQUIRE(view.chars_size(cudf::get_default_stream()) == expected);
  }
  SECTION("a mask without nulls leaves a non-nullable column")
  {
    auto all_set = args;
    all_set.valid.assign(args.keys.size(), true);
    const auto result = format_column(all_set);
    REQUIRE(result->null_count() == 0);
    REQUIRE_FALSE(result->nullable());
    REQUIRE(to_host(result->view()) == all_valid);
  }
}

TEST_CASE("generator strings: device output matches the host rule", "[generator_strings]")
{
  format_args args{.p1 = 3, .p2 = 300, .seed = 0xfeedULL, .batch = 9, .column = 2};
  for (std::int64_t row = 0; row < 1000; ++row) {
    // Mixes short keys, keys wider than p1, a negative key and the largest key.
    const auto h = counter_hash(1, 2, static_cast<std::uint64_t>(row), 0, 0);
    args.keys.push_back(static_cast<std::int64_t>(h >> (1 + h % 63)));
    args.valid.push_back(h % 11 != 0);
  }
  args.keys[10]  = -42;
  args.keys[11]  = max_int64_key;
  args.valid[10] = args.valid[11] = true;

  const auto strings = format(args);
  REQUIRE(strings.size() == args.keys.size());
  for (std::size_t row = 0; row < strings.size(); ++row) {
    if (!args.valid[row]) {
      REQUIRE_FALSE(strings[row].has_value());
      continue;
    }
    REQUIRE(strings[row].has_value());
    REQUIRE(*strings[row] == reference(args.keys[row], row, args));
  }
}

TEST_CASE("generator strings: argument validation and empty input", "[generator_strings]")
{
  REQUIRE_THROWS_AS(format_column({.keys = {1}, .p1 = 0, .p2 = 4}), std::invalid_argument);
  REQUIRE_THROWS_AS(format_column({.keys = {1}, .p1 = 5, .p2 = 4}), std::invalid_argument);
  REQUIRE_THROWS_AS(format_column({.keys = {1}, .p1 = -1, .p2 = -1}), std::invalid_argument);
  REQUIRE_THROWS_AS(
    format_column({.keys = {1}, .p1 = std::numeric_limits<double>::quiet_NaN(), .p2 = 4}),
    std::invalid_argument);
  REQUIRE_THROWS_AS(format_column({.keys = {1}, .p1 = 1, .p2 = 1e12}), std::invalid_argument);
  // The largest length is accepted by the check (no rows, so nothing that long is built).
  REQUIRE(format_column({.keys = {}, .p1 = 1, .p2 = 2147483647.0})->size() == 0);
  REQUIRE_THROWS_AS(format_column({.keys = {}, .p1 = 1, .p2 = 2147483648.0}),
                    std::invalid_argument);

  const auto empty = format_column({.keys = {}, .p1 = 2, .p2 = 4});
  REQUIRE(empty->size() == 0);
  REQUIRE(empty->type().id() == cudf::type_id::STRING);
}

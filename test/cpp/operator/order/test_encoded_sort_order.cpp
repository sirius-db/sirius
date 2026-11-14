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

// Property tests of order::encoded_sorted_order against cuDF: for random keys of every supported
// shape and order option, the encoded row order must be exactly cudf::stable_sorted_order's (the
// cuDF row comparator, equal keys in input order), and the keys gathered by it must be sorted
// under cudf::is_sorted for the same (column_order, null_precedence).

#include "catch.hpp"
#include "op/order/encoded_sort_order.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using sirius::op::order::encoded_sort_options;
using sirius::op::order::encoded_sort_stats;
using sirius::op::order::encoded_sort_working_bytes;
using sirius::op::order::encoded_sorted_order;

namespace {

enum class kind {
  int8,
  int16,
  int32,
  int64,
  uint8,
  uint16,
  uint32,
  uint64,
  bool8,
  float32,
  float64,
  decimal32,
  decimal64,
  decimal128,
  timestamp_days,
  timestamp_us,
  duration_ns,
  string
};

/// One key column on the host. `ints` holds the values of the integer-backed kinds, `doubles`
/// those of both floating-point kinds.
struct host_column {
  kind type{kind::int64};
  std::vector<std::int64_t> ints;
  std::vector<double> doubles;
  std::vector<std::string> strings;
  std::vector<bool> valid;

  [[nodiscard]] cudf::size_type size() const { return static_cast<cudf::size_type>(valid.size()); }
};

cudf::data_type cudf_type(kind type)
{
  switch (type) {
    case kind::int8: return cudf::data_type{cudf::type_id::INT8};
    case kind::int16: return cudf::data_type{cudf::type_id::INT16};
    case kind::int32: return cudf::data_type{cudf::type_id::INT32};
    case kind::int64: return cudf::data_type{cudf::type_id::INT64};
    case kind::uint8: return cudf::data_type{cudf::type_id::UINT8};
    case kind::uint16: return cudf::data_type{cudf::type_id::UINT16};
    case kind::uint32: return cudf::data_type{cudf::type_id::UINT32};
    case kind::uint64: return cudf::data_type{cudf::type_id::UINT64};
    case kind::bool8: return cudf::data_type{cudf::type_id::BOOL8};
    case kind::float32: return cudf::data_type{cudf::type_id::FLOAT32};
    case kind::float64: return cudf::data_type{cudf::type_id::FLOAT64};
    case kind::decimal32: return cudf::data_type{cudf::type_id::DECIMAL32, -2};
    case kind::decimal64: return cudf::data_type{cudf::type_id::DECIMAL64, -2};
    case kind::decimal128: return cudf::data_type{cudf::type_id::DECIMAL128, -2};
    case kind::timestamp_days: return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    case kind::timestamp_us: return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
    case kind::duration_ns: return cudf::data_type{cudf::type_id::DURATION_NANOSECONDS};
    case kind::string: return cudf::data_type{cudf::type_id::STRING};
  }
  throw std::logic_error("unknown kind");
}

/// The DECIMAL128 value of host value @p v: v * (2^64 - 3). It grows with v, and neither 64-bit
/// half orders the values on its own.
__int128 widen_decimal128(std::int64_t v)
{
  return static_cast<__int128>(v) * ((static_cast<__int128>(1) << 64) - 3);
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
std::unique_ptr<cudf::column> fixed_column(const host_column& col,
                                           rmm::device_buffer mask,
                                           cudf::size_type null_count)
{
  std::vector<T> values;
  values.reserve(col.valid.size());
  if (col.type == kind::float32 || col.type == kind::float64) {
    for (const auto v : col.doubles) {
      values.push_back(static_cast<T>(v));
    }
  } else if (col.type == kind::decimal128) {
    for (const auto v : col.ints) {
      values.push_back(static_cast<T>(widen_decimal128(v)));
    }
  } else {
    for (const auto v : col.ints) {
      values.push_back(static_cast<T>(v));
    }
  }
  return std::make_unique<cudf::column>(
    cudf_type(col.type), col.size(), to_device_buffer(values), std::move(mask), null_count);
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
  rmm::device_buffer mask = null_count > 0 ? to_device_buffer(mask_words) : rmm::device_buffer{};

  switch (col.type) {
    case kind::int8: return fixed_column<std::int8_t>(col, std::move(mask), null_count);
    case kind::int16: return fixed_column<std::int16_t>(col, std::move(mask), null_count);
    case kind::int32:
    case kind::decimal32:
    case kind::timestamp_days: return fixed_column<std::int32_t>(col, std::move(mask), null_count);
    case kind::int64:
    case kind::decimal64:
    case kind::timestamp_us:
    case kind::duration_ns: return fixed_column<std::int64_t>(col, std::move(mask), null_count);
    case kind::uint8:
    case kind::bool8: return fixed_column<std::uint8_t>(col, std::move(mask), null_count);
    case kind::uint16: return fixed_column<std::uint16_t>(col, std::move(mask), null_count);
    case kind::uint32: return fixed_column<std::uint32_t>(col, std::move(mask), null_count);
    case kind::uint64: return fixed_column<std::uint64_t>(col, std::move(mask), null_count);
    case kind::float32: return fixed_column<float>(col, std::move(mask), null_count);
    case kind::float64: return fixed_column<double>(col, std::move(mask), null_count);
    case kind::decimal128: return fixed_column<__int128>(col, std::move(mask), null_count);
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

std::unique_ptr<cudf::table> make_table(const std::vector<host_column>& columns)
{
  std::vector<std::unique_ptr<cudf::column>> out;
  for (const auto& col : columns) {
    out.push_back(make_column(col));
  }
  return std::make_unique<cudf::table>(std::move(out));
}

std::vector<cudf::size_type> to_host(const cudf::column_view& order)
{
  std::vector<cudf::size_type> out(static_cast<std::size_t>(order.size()));
  if (!out.empty()) {
    cudf::get_default_stream().sync();
    REQUIRE(cudaMemcpy(out.data(),
                       order.data<cudf::size_type>(),
                       out.size() * sizeof(cudf::size_type),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  return out;
}

/// How the string values of a random column are drawn.
enum class string_shape {
  shared_prefix,  ///< 18 values of 18 bytes sharing 16: the few_distinct benchmark shape
  nested,  ///< short strings over {0x00, 'a', 0xFF}: prefixes of each other, embedded NULs, ties
  random,  ///< 8 to 40 lowercase letters: close to unique
  utf8,    ///< multi-byte UTF-8 words with common prefixes
  long_prefix  ///< a 40-byte common prefix, then a few distinct tails of different lengths
};

std::string random_string(string_shape shape, std::mt19937& rng)
{
  switch (shape) {
    case string_shape::shared_prefix: {
      std::string s(16, '0');
      const auto v = rng() % 18;
      s += static_cast<char>('0' + v / 10);
      s += static_cast<char>('0' + v % 10);
      return s;
    }
    case string_shape::nested: {
      static const char alphabet[] = {'\0', 'a', '\xFF'};
      std::string s(rng() % 21, 'a');
      for (auto& c : s) {
        c = alphabet[rng() % 3];
      }
      // Mostly runs of one byte, so long common prefixes are frequent.
      if (rng() % 2 == 0) { s.assign(s.size(), alphabet[rng() % 3]); }
      return s;
    }
    case string_shape::random: {
      std::string s(8 + rng() % 33, 'a');
      for (auto& c : s) {
        c = static_cast<char>('a' + rng() % 26);
      }
      return s;
    }
    case string_shape::utf8: {
      static const std::vector<std::string> words = {
        "",
        "a",
        "\xC3\xA9",
        "\xC3\xA9t\xC3\xA9",
        "\xC3\xA9t\xC3\xA9s longs et chauds",
        "\xE6\x97\xA5\xE6\x9C\xAC",
        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xAE\xE6\x96\x87\xE5\xAD\x97\xE5\x88\x97",
        "\xF0\x9F\x98\x80",
        "\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82\xF0\x9F\x98\x83\xF0\x9F\x98\x84",
        "z",
        "\x7F",
        "\xC2\x80"};
      return words[rng() % words.size()] + (rng() % 4 == 0 ? words[rng() % words.size()] : "");
    }
    case string_shape::long_prefix: {
      std::string s(40, 'p');
      const auto tail = rng() % 6;
      for (std::uint32_t i = 0; i < tail * 5; ++i) {
        s += static_cast<char>('a' + (tail + i) % 3);
      }
      return s;
    }
  }
  throw std::logic_error("unknown string shape");
}

/// Random column. `distinct` > 0 limits the integer-backed kinds to that many values.
host_column random_column(kind type,
                          cudf::size_type rows,
                          double null_probability,
                          std::mt19937& rng,
                          std::int64_t distinct = 0,
                          string_shape shape    = string_shape::nested)
{
  static const std::vector<double> double_domain = {-std::numeric_limits<double>::infinity(),
                                                    -1.5,
                                                    -0.0,
                                                    0.0,
                                                    2.25,
                                                    1e300,
                                                    -1e-300,
                                                    std::numeric_limits<double>::infinity(),
                                                    std::numeric_limits<double>::quiet_NaN(),
                                                    -std::numeric_limits<double>::quiet_NaN()};
  static const std::vector<double> float_domain  = {-std::numeric_limits<float>::infinity(),
                                                    -1.5,
                                                    -0.0,
                                                    0.0,
                                                    2.25,
                                                    std::numeric_limits<float>::max(),
                                                    std::numeric_limits<float>::infinity(),
                                                    std::numeric_limits<float>::quiet_NaN(),
                                                    -std::numeric_limits<float>::quiet_NaN()};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> wide(-1e6, 1e6);
  host_column col;
  col.type = type;
  for (cudf::size_type i = 0; i < rows; ++i) {
    col.valid.push_back(unit(rng) >= null_probability);
    std::int64_t v = 0;
    double d       = 0;
    std::string s;
    switch (type) {
      case kind::int8: v = static_cast<std::int8_t>(rng()); break;
      case kind::int16: v = static_cast<std::int16_t>(rng()); break;
      case kind::int32:
      case kind::decimal32:
      case kind::timestamp_days: v = static_cast<std::int32_t>(rng()); break;
      case kind::int64:
      case kind::decimal64:
      case kind::timestamp_us:
      case kind::duration_ns:
        v = static_cast<std::int64_t>((static_cast<std::uint64_t>(rng()) << 32) | rng());
        break;
      case kind::decimal128: v = static_cast<std::int32_t>(rng()); break;
      case kind::uint8: v = static_cast<std::uint8_t>(rng()); break;
      case kind::uint16: v = static_cast<std::uint16_t>(rng()); break;
      case kind::uint32: v = static_cast<std::uint32_t>(rng()); break;
      case kind::uint64:
        v = static_cast<std::int64_t>((static_cast<std::uint64_t>(rng()) << 32) | rng());
        break;
      case kind::bool8: v = rng() % 2; break;
      case kind::float32:
        d = rng() % 3 == 0 ? float_domain[rng() % float_domain.size()]
                           : static_cast<double>(static_cast<float>(wide(rng)));
        break;
      case kind::float64:
        d = rng() % 3 == 0 ? double_domain[rng() % double_domain.size()] : wide(rng);
        break;
      case kind::string: s = random_string(shape, rng); break;
    }
    if (distinct > 0 && type != kind::bool8) {
      // A few values around zero and at both ends of the type's range.
      const auto slot = static_cast<std::int64_t>(rng() % distinct);
      switch (type) {
        case kind::int8: v = slot % 2 == 0 ? -128 + slot : 127 - slot; break;
        case kind::int16: v = slot % 2 == 0 ? -32768 + slot : 32767 - slot; break;
        case kind::int32:
        case kind::decimal32:
        case kind::timestamp_days:
          v = slot % 3 == 0 ? std::numeric_limits<std::int32_t>::min() + slot
                            : (slot % 3 == 1 ? std::numeric_limits<std::int32_t>::max() - slot
                                             : slot - distinct / 2);
          break;
        case kind::int64:
        case kind::decimal64:
        case kind::timestamp_us:
        case kind::duration_ns:
          v = slot % 3 == 0 ? std::numeric_limits<std::int64_t>::min() + slot
                            : (slot % 3 == 1 ? std::numeric_limits<std::int64_t>::max() - slot
                                             : slot - distinct / 2);
          break;
        case kind::decimal128: v = slot - distinct / 2; break;
        case kind::uint8: v = slot % 2 == 0 ? slot : 255 - slot; break;
        case kind::uint16: v = slot % 2 == 0 ? slot : 65535 - slot; break;
        case kind::uint32: v = slot % 2 == 0 ? slot : 4294967295LL - slot; break;
        case kind::uint64: v = slot % 2 == 0 ? slot : -1 - slot; break;
        default: break;
      }
    }
    col.ints.push_back(v);
    col.doubles.push_back(d);
    col.strings.push_back(std::move(s));
  }
  return col;
}

encoded_sort_options no_minimum()
{
  encoded_sort_options options;
  options.min_rows = 0;
  return options;
}

/// Requires the encoded order of @p keys to be cudf::stable_sorted_order's.
void require_matches_cudf(const cudf::table_view& keys,
                          const std::vector<cudf::order>& column_order,
                          const std::vector<cudf::null_order>& null_precedence,
                          encoded_sort_stats* stats = nullptr)
{
  const auto stream = cudf::get_default_stream();
  const auto mr     = cudf::get_current_device_resource_ref();
  auto encoded =
    encoded_sorted_order(keys, column_order, null_precedence, stream, mr, no_minimum(), stats);
  REQUIRE(encoded != nullptr);
  REQUIRE(encoded->size() == keys.num_rows());
  REQUIRE(encoded->type().id() == cudf::type_id::INT32);
  REQUIRE(encoded->null_count() == 0);

  auto expected   = cudf::stable_sorted_order(keys, column_order, null_precedence, stream, mr);
  const auto got  = to_host(encoded->view());
  const auto want = to_host(expected->view());
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      FAIL("position " << i << ": encoded order has row " << got[i] << ", cuDF has row "
                       << want[i]);
    }
  }
  // Independent of the reference order: the gathered keys are sorted for cuDF.
  auto gathered = cudf::gather(keys, encoded->view());
  REQUIRE(cudf::is_sorted(gathered->view(), column_order, null_precedence));
}

const std::vector<std::pair<cudf::order, cudf::null_order>> kOrderOptions = {
  {cudf::order::ASCENDING, cudf::null_order::BEFORE},
  {cudf::order::ASCENDING, cudf::null_order::AFTER},
  {cudf::order::DESCENDING, cudf::null_order::BEFORE},
  {cudf::order::DESCENDING, cudf::null_order::AFTER}};

void check_random(const std::vector<kind>& kinds,
                  cudf::size_type rows,
                  double null_probability,
                  std::int64_t distinct,
                  string_shape shape,
                  std::uint32_t seed)
{
  std::mt19937 rng(seed);
  std::vector<host_column> columns;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  for (const auto type : kinds) {
    columns.push_back(random_column(type, rows, null_probability, rng, distinct, shape));
    const auto& option = kOrderOptions[rng() % kOrderOptions.size()];
    column_order.push_back(option.first);
    null_precedence.push_back(option.second);
  }
  INFO("seed " << seed << ", rows " << rows << ", null probability " << null_probability
               << ", distinct " << distinct);
  auto table = make_table(columns);
  require_matches_cudf(table->view(), column_order, null_precedence);
}

}  // namespace

TEST_CASE("encoded sort order: one string key, every order option", "[encoded_sort_order]")
{
  const auto shape            = GENERATE(string_shape::shared_prefix,
                              string_shape::nested,
                              string_shape::random,
                              string_shape::utf8,
                              string_shape::long_prefix);
  const auto null_probability = GENERATE(0.0, 0.2);
  std::mt19937 rng(1234 + static_cast<std::uint32_t>(shape));
  const auto column = random_column(kind::string, 6000, null_probability, rng, 0, shape);
  auto table        = make_table({column});
  for (const auto& option : kOrderOptions) {
    INFO("shape " << static_cast<int>(shape) << ", null probability " << null_probability
                  << ", descending " << (option.first == cudf::order::DESCENDING)
                  << ", nulls after " << (option.second == cudf::null_order::AFTER));
    require_matches_cudf(table->view(), {option.first}, {option.second});
  }
}

TEST_CASE("encoded sort order: a shared prefix costs one sort", "[encoded_sort_order]")
{
  std::mt19937 rng(7);
  const auto column = random_column(kind::string, 50000, 0.0, rng, 0, string_shape::shared_prefix);
  auto table        = make_table({column});
  encoded_sort_stats stats;
  require_matches_cudf(table->view(), {cudf::order::ASCENDING}, {cudf::null_order::BEFORE}, &stats);
  // Three 8-byte chunks, no length field (all strings have 18 bytes): the two chunks of the
  // shared prefix separate nothing and the third orders all rows in one radix sort.
  CHECK(stats.words == 3);
  CHECK(stats.words_skipped == 2);
  CHECK(stats.radix_sorts == 1);
  CHECK(stats.rows_sorted == 50000);
}

TEST_CASE("encoded sort order: degenerate string columns", "[encoded_sort_order]")
{
  const cudf::size_type rows = 3000;
  host_column col;
  col.type = kind::string;
  SECTION("all equal")
  {
    col.strings.assign(rows, "the same value in every row");
    col.valid.assign(rows, true);
  }
  SECTION("all empty")
  {
    col.strings.assign(rows, "");
    col.valid.assign(rows, true);
  }
  SECTION("all null")
  {
    col.strings.assign(rows, "");
    col.valid.assign(rows, false);
  }
  SECTION("empty, null and one value")
  {
    for (cudf::size_type i = 0; i < rows; ++i) {
      col.strings.push_back(i % 3 == 0 ? "" : "x");
      col.valid.push_back(i % 5 != 0);
    }
  }
  SECTION("each row a prefix of the next")
  {
    for (cudf::size_type i = 0; i < rows; ++i) {
      col.strings.push_back(std::string(static_cast<std::size_t>((i * 7919) % 200), 'q'));
      col.valid.push_back(true);
    }
  }
  SECTION("NUL padding against length")
  {
    // "ab", "ab\0", "ab\0\0": equal as zero-padded chunks, ordered by length only.
    for (cudf::size_type i = 0; i < rows; ++i) {
      col.strings.push_back(std::string("ab") + std::string(static_cast<std::size_t>(i % 9), '\0'));
      col.valid.push_back(true);
    }
  }
  col.ints.assign(rows, 0);
  col.doubles.assign(rows, 0);
  auto table = make_table({col});
  for (const auto& option : kOrderOptions) {
    require_matches_cudf(table->view(), {option.first}, {option.second});
  }
}

TEST_CASE("encoded sort order: multi-column keys of every type", "[encoded_sort_order]")
{
  const std::vector<std::vector<kind>> shapes = {
    {kind::int32, kind::int64, kind::string},
    {kind::int32, kind::int64},
    {kind::string, kind::string},
    {kind::string, kind::int32},
    {kind::float64, kind::float32, kind::bool8},
    {kind::float64, kind::int8},
    {kind::decimal128, kind::timestamp_us, kind::decimal64},
    {kind::uint8, kind::int8, kind::int16, kind::uint64},
    {kind::uint16, kind::uint32, kind::decimal32, kind::timestamp_days, kind::duration_ns},
    {kind::bool8,
     kind::bool8,
     kind::int8,
     kind::uint8,
     kind::bool8,
     kind::int16,
     kind::bool8,
     kind::int8,
     kind::bool8,
     kind::uint8,
     kind::bool8,
     kind::string},
    {kind::int64, kind::int64, kind::int64},
    {kind::decimal128, kind::decimal128}};
  const auto shape_index = GENERATE_COPY(range(std::size_t{0}, shapes.size()));
  const auto& kinds      = shapes[shape_index];
  std::uint32_t seed     = 100 * static_cast<std::uint32_t>(shape_index);
  for (const double null_probability : {0.0, 0.15}) {
    // Low cardinality (long tie chains through the columns), then high.
    for (const std::int64_t distinct : {3, 40, 0}) {
      for (const auto strings : {string_shape::nested, string_shape::random}) {
        check_random(kinds, 4000, null_probability, distinct, strings, ++seed);
      }
    }
  }
}

TEST_CASE("encoded sort order: one DECIMAL128 key", "[encoded_sort_order]")
{
  std::uint32_t seed = 900;
  for (const double null_probability : {0.0, 0.3}) {
    for (const std::int64_t distinct : {5, 0}) {
      std::mt19937 rng(++seed);
      auto table =
        make_table({random_column(kind::decimal128, 5000, null_probability, rng, distinct)});
      for (const auto& option : kOrderOptions) {
        require_matches_cudf(table->view(), {option.first}, {option.second});
      }
    }
  }
}

TEST_CASE("encoded sort order: sliced input and row counts around block sizes",
          "[encoded_sort_order]")
{
  std::mt19937 rng(4242);
  const cudf::size_type rows             = GENERATE(2, 3, 17, 255, 257, 70001);
  const std::vector<host_column> columns = {
    random_column(kind::int32, rows + 11, 0.1, rng, 6),
    random_column(kind::string, rows + 11, 0.1, rng, 0, string_shape::nested),
    random_column(kind::decimal128, rows + 11, 0.1, rng, 4),
    random_column(kind::float64, rows + 11, 0.1, rng)};
  auto table       = make_table(columns);
  const auto slice = cudf::slice(table->view(), {7, 7 + rows}).front();
  REQUIRE(slice.num_rows() == rows);
  require_matches_cudf(slice,
                       {cudf::order::ASCENDING,
                        cudf::order::DESCENDING,
                        cudf::order::ASCENDING,
                        cudf::order::DESCENDING},
                       {cudf::null_order::AFTER,
                        cudf::null_order::AFTER,
                        cudf::null_order::BEFORE,
                        cudf::null_order::BEFORE});
}

TEST_CASE("encoded sort order: a large batch of the benchmark shapes", "[encoded_sort_order]")
{
  std::mt19937 rng(99);
  const cudf::size_type rows = 400000;
  SECTION("few distinct strings")
  {
    auto table =
      make_table({random_column(kind::string, rows, 0.0, rng, 0, string_shape::shared_prefix)});
    require_matches_cudf(table->view(), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  }
  SECTION("INTEGER, BIGINT, VARCHAR")
  {
    // The INTEGER key repeats; the BIGINT key is close to unique.
    auto k1 = random_column(kind::int32, rows, 0.0, rng);
    for (auto& v : k1.ints) {
      v %= 50000;
    }
    auto table = make_table({k1,
                             random_column(kind::int64, rows, 0.0, rng),
                             random_column(kind::string, rows, 0.0, rng, 0, string_shape::random)});
    require_matches_cudf(
      table->view(),
      {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING},
      {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER});
  }
  SECTION("close to unique strings")
  {
    auto table =
      make_table({random_column(kind::string, rows, 0.05, rng, 0, string_shape::random)});
    require_matches_cudf(table->view(), {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  }
}

TEST_CASE("encoded sort order: key shapes left to cuDF", "[encoded_sort_order]")
{
  const auto stream = cudf::get_default_stream();
  const auto mr     = cudf::get_current_device_resource_ref();
  std::mt19937 rng(5);
  const std::vector<cudf::order> ascending   = {cudf::order::ASCENDING};
  const std::vector<cudf::null_order> before = {cudf::null_order::BEFORE};

  SECTION("a single fixed-width key")
  {
    for (const auto type : {kind::int64, kind::int32, kind::float64, kind::decimal64}) {
      auto table = make_table({random_column(type, 5000, 0.1, rng)});
      CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, no_minimum()) ==
            nullptr);
    }
  }
  SECTION("fewer rows than the minimum")
  {
    auto table = make_table({random_column(kind::string, 100, 0.0, rng)});
    CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr) == nullptr);
    encoded_sort_options options;
    options.min_rows = 100;
    CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, options) != nullptr);
  }
  SECTION("a string longer than the limit")
  {
    auto column                  = random_column(kind::string, 5000, 0.0, rng);
    auto table                   = make_table({column});
    encoded_sort_options options = no_minimum();
    options.max_string_bytes     = 20;
    CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, options) != nullptr);
    column.strings[77] = std::string(21, 'x');
    table              = make_table({column});
    CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, options) == nullptr);
    // A null row's bytes do not count.
    column.valid[77] = false;
    table            = make_table({column});
    CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, options) != nullptr);
  }
  SECTION("no rows or no columns")
  {
    auto table = make_table({random_column(kind::string, 0, 0.0, rng)});
    CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, no_minimum()) ==
          nullptr);
    CHECK(encoded_sorted_order(cudf::table_view{}, {}, {}, stream, mr, no_minimum()) == nullptr);
  }
}

TEST_CASE("encoded sort order: working memory is bounded and freed", "[encoded_sort_order]")
{
  const auto stream = cudf::get_default_stream();
  // Counts every allocation the sort makes: it must take all its memory from the resource it is
  // given.
  rmm::mr::statistics_resource_adaptor counted{cudf::get_current_device_resource_ref()};
  const rmm::device_async_resource_ref mr{counted};
  std::mt19937 rng(31);
  const cudf::size_type rows = 300000;
  // Histograms and other per-sort scratch of the radix sort, not proportional to the rows.
  const std::int64_t slack = 4 << 20;

  struct shape {
    const char* name;
    std::vector<host_column> columns;
    bool later_steps;  ///< the first word leaves rows tied
  };
  std::vector<shape> shapes;
  shapes.push_back({"shared prefix",
                    {random_column(kind::string, rows, 0.0, rng, 0, string_shape::shared_prefix)},
                    false});
  shapes.push_back({"close to unique strings",
                    {random_column(kind::string, rows, 0.0, rng, 0, string_shape::random)},
                    false});
  shapes.push_back({"nested strings, many ties",
                    {random_column(kind::string, rows, 0.1, rng, 0, string_shape::nested),
                     random_column(kind::int64, rows, 0.1, rng)},
                    true});
  shapes.push_back(
    {"two wide fixed-width keys",
     {random_column(kind::int64, rows, 0.0, rng, 1000), random_column(kind::int64, rows, 0.0, rng)},
     true});
  shapes.push_back({"DECIMAL128 with ties, then a string",
                    {random_column(kind::decimal128, rows, 0.0, rng, 50),
                     random_column(kind::string, rows, 0.0, rng, 0, string_shape::long_prefix)},
                    true});

  for (const auto& s : shapes) {
    INFO(s.name);
    auto table = make_table(s.columns);
    const std::vector<cudf::order> column_order(s.columns.size(), cudf::order::ASCENDING);
    const std::vector<cudf::null_order> null_precedence(s.columns.size(), cudf::null_order::AFTER);
    stream.sync();
    const auto before = counted.get_bytes_counter();
    encoded_sort_stats stats;
    auto order = encoded_sorted_order(
      table->view(), column_order, null_precedence, stream, mr, no_minimum(), &stats);
    REQUIRE(order != nullptr);
    stream.sync();
    const auto after = counted.get_bytes_counter();
    if (s.later_steps) { CHECK(stats.radix_sorts >= 2); }
    // Peak above what was held before the call: within the documented bound.
    CHECK(after.peak - before.value <=
          static_cast<std::int64_t>(encoded_sort_working_bytes(rows)) + slack);
    // On return only the result (4 bytes per row, padded) is still held.
    CHECK(after.value - before.value >= static_cast<std::int64_t>(rows) * 4);
    CHECK(after.value - before.value <= static_cast<std::int64_t>(rows) * 4 + 4096);
    order.reset();
    stream.sync();
    CHECK(counted.get_bytes_counter().value == before.value);
  }
}

TEST_CASE("encoded sort order: inputs over the working memory limit are left to cuDF",
          "[encoded_sort_order]")
{
  const auto stream = cudf::get_default_stream();
  const auto mr     = cudf::get_current_device_resource_ref();
  std::mt19937 rng(77);
  const cudf::size_type rows = 10000;
  auto table                 = make_table({random_column(kind::string, rows, 0.0, rng)});
  const std::vector<cudf::order> ascending   = {cudf::order::ASCENDING};
  const std::vector<cudf::null_order> before = {cudf::null_order::BEFORE};
  encoded_sort_options options               = no_minimum();
  options.max_working_bytes                  = encoded_sort_working_bytes(rows);
  CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, options) != nullptr);
  options.max_working_bytes = encoded_sort_working_bytes(rows) - 1;
  CHECK(encoded_sorted_order(table->view(), ascending, before, stream, mr, options) == nullptr);
  CHECK(encoded_sort_working_bytes(rows) == std::size_t{320000});
}

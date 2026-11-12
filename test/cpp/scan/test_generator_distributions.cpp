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

// Tests of the generator's draw kernels (generator_batch.cu): reproducibility per (seed, batch),
// output schema, the statistics of each distribution, null masks, and a host reference built on
// detail::counter_hash. Numeric columns only; VARCHAR is covered by test_generator_strings.cpp.

#include "catch.hpp"
#include "op/scan/generator/generator_detail.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op::scan::generator;

namespace {

constexpr std::uint64_t million = 1U << 20;

column_spec make_column(std::string name,
                        value_type type,
                        distribution dist,
                        double p1,
                        double p2            = 0,
                        double cardinality   = 0,
                        double null_fraction = 0)
{
  return column_spec{std::move(name), type, dist, p1, p2, cardinality, null_fraction};
}

generator_spec make_spec(std::vector<column_spec> columns,
                         std::uint64_t rows_per_batch,
                         std::uint64_t num_batches = 8,
                         std::uint64_t seed        = 42)
{
  return generator_spec{std::move(columns), num_batches, rows_per_batch, seed};
}

std::unique_ptr<cudf::table> generate(const generator_spec& spec, std::uint64_t batch_index)
{
  return generate_batch(
    spec, batch_index, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

/// Copies the values of a fixed-width column to the host (null rows included).
template <typename T>
std::vector<T> values(cudf::column_view column)
{
  REQUIRE(cudf::size_of(column.type()) == sizeof(T));
  std::vector<T> out(static_cast<std::size_t>(column.size()));
  REQUIRE(::cudaDeviceSynchronize() == cudaSuccess);
  if (!out.empty()) {
    REQUIRE(::cudaMemcpy(
              out.data(), column.data<T>(), out.size() * sizeof(T), ::cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return out;
}

/// Row validity on the host; empty when the column has no null mask.
std::vector<bool> validity(cudf::column_view column)
{
  if (!column.nullable()) { return {}; }
  const auto rows = static_cast<std::size_t>(column.size());
  std::vector<cudf::bitmask_type> words((rows + 31) / 32);
  REQUIRE(::cudaDeviceSynchronize() == cudaSuccess);
  if (!words.empty()) {
    REQUIRE(::cudaMemcpy(words.data(),
                         column.null_mask(),
                         words.size() * sizeof(cudf::bitmask_type),
                         ::cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  std::vector<bool> out(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    out[i] = ((words[i / 32] >> (i % 32)) & 1U) != 0;
  }
  return out;
}

/// One generated int64 / double column of a single-column spec.
template <typename T>
std::vector<T> draw(column_spec column,
                    std::uint64_t rows,
                    std::uint64_t batch_index = 0,
                    std::uint64_t seed        = 42)
{
  const auto spec  = make_spec({std::move(column)}, rows, 8, seed);
  const auto table = generate(spec, batch_index);
  return values<T>(table->view().column(0));
}

template <typename T>
double mean_of(const std::vector<T>& v)
{
  long double sum = 0;
  for (auto x : v) {
    sum += static_cast<long double>(x);
  }
  return static_cast<double>(sum / static_cast<long double>(v.size()));
}

template <typename T>
double stddev_of(const std::vector<T>& v)
{
  const double mean = mean_of(v);
  long double sum   = 0;
  for (auto x : v) {
    const long double d = static_cast<long double>(x) - mean;
    sum += d * d;
  }
  return std::sqrt(static_cast<double>(sum / static_cast<long double>(v.size())));
}

template <typename T>
std::size_t distinct_count(const std::vector<T>& v)
{
  return std::set<T>(v.begin(), v.end()).size();
}

/// Everything observable about a table: per column the raw value bytes and the validity.
struct table_image {
  std::vector<std::vector<std::uint8_t>> data;
  std::vector<std::vector<bool>> valid;
  std::vector<cudf::size_type> null_counts;
  friend bool operator==(const table_image&, const table_image&) = default;
};

table_image image_of(const cudf::table& table)
{
  table_image image;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    const auto column = table.view().column(i);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(column.size()) *
                                    cudf::size_of(column.type()));
    REQUIRE(::cudaDeviceSynchronize() == cudaSuccess);
    if (!bytes.empty()) {
      REQUIRE(::cudaMemcpy(bytes.data(),
                           column.data<std::uint8_t>(),
                           bytes.size(),
                           ::cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    image.data.push_back(std::move(bytes));
    image.valid.push_back(validity(column));
    image.null_counts.push_back(column.null_count());
  }
  return image;
}

/// One column of every distribution and numeric type, some with nulls.
generator_spec mixed_spec(std::uint64_t rows, std::uint64_t seed = 42)
{
  using enum distribution;
  return make_spec({make_column("u64", value_type::int64, uniform, -1000, 1000),
                    make_column("n32", value_type::int32, normal, 50, 10, 0, 0.1),
                    make_column("ud", value_type::float64, uniform, 0, 1, 0, 0.3),
                    make_column("z", value_type::int64, zipf, 1.2, 0, 1000),
                    make_column("fd", value_type::int32, few_distinct, 16),
                    make_column("s", value_type::int64, sequential, 7, 3),
                    make_column("sd", value_type::float64, sequential_desc, 0.5, 0.25),
                    make_column("gauss", value_type::float64, normal, 0, 1)},
                   rows,
                   8,
                   seed);
}

/// Host reference of the uniform int64 draw: `lo + mulhi64(hash, hi - lo + 1)`.
std::int64_t reference_uniform(const generator_spec& spec,
                               std::uint64_t batch,
                               std::uint64_t row,
                               std::uint32_t column,
                               std::int64_t lo,
                               std::int64_t hi)
{
  const std::uint64_t h     = detail::counter_hash(spec.seed, batch, row, column, detail::VALUE_A);
  const std::uint64_t range = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) + 1;
  const auto offset = static_cast<std::uint64_t>((static_cast<unsigned __int128>(h) * range) >> 64);
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(lo) + offset);
}

}  // namespace

TEST_CASE("generator: a batch depends only on (spec, batch_index)", "[generator_distributions]")
{
  const auto spec = mixed_spec(100'000);

  const auto batch3_alone = image_of(*generate(spec, 3));
  SECTION("the same spec and batch give bit-identical data")
  {
    REQUIRE(image_of(*generate(spec, 3)) == batch3_alone);
  }
  SECTION("batch 3 is the same after generating batches 0 to 2")
  {
    std::vector<table_image> earlier;
    for (std::uint64_t b = 0; b < 3; ++b) {
      earlier.push_back(image_of(*generate(spec, b)));
    }
    REQUIRE(image_of(*generate(spec, 3)) == batch3_alone);
    for (const auto& image : earlier) {
      REQUIRE_FALSE(image == batch3_alone);
    }
  }
  SECTION("another batch or another seed gives different data")
  {
    const auto other_batch = image_of(*generate(spec, 4));
    const auto other_seed  = image_of(*generate(mixed_spec(100'000, 43), 3));
    // Every random column differs (the two sequential ones depend on the batch only).
    for (std::size_t c : {0U, 1U, 2U, 3U, 4U, 7U}) {
      REQUIRE(other_batch.data[c] != batch3_alone.data[c]);
      REQUIRE(other_seed.data[c] != batch3_alone.data[c]);
    }
    REQUIRE(other_batch.data[5] != batch3_alone.data[5]);
    REQUIRE(other_seed.data[5] == batch3_alone.data[5]);
    REQUIRE(other_batch.valid[1] != batch3_alone.valid[1]);
    REQUIRE(other_seed.valid[1] != batch3_alone.valid[1]);
  }
  SECTION("two columns with the same spec are independent streams")
  {
    const auto column = make_column("a", value_type::int64, distribution::uniform, 0, 1e9);
    const auto table  = generate(make_spec({column, column}, 10'000), 0);
    REQUIRE(values<std::int64_t>(table->view().column(0)) !=
            values<std::int64_t>(table->view().column(1)));
  }
}

TEST_CASE("generator: schema and row counts follow the spec", "[generator_distributions]")
{
  const auto spec  = mixed_spec(12'345);
  const auto table = generate(spec, 7);
  REQUIRE(table->num_columns() == 8);
  REQUIRE(table->num_rows() == 12'345);
  const std::vector<cudf::type_id> expected{cudf::type_id::INT64,
                                            cudf::type_id::INT32,
                                            cudf::type_id::FLOAT64,
                                            cudf::type_id::INT64,
                                            cudf::type_id::INT32,
                                            cudf::type_id::INT64,
                                            cudf::type_id::FLOAT64,
                                            cudf::type_id::FLOAT64};
  for (cudf::size_type i = 0; i < table->num_columns(); ++i) {
    REQUIRE(table->view().column(i).type().id() == expected[static_cast<std::size_t>(i)]);
    REQUIRE(table->view().column(i).size() == 12'345);
  }

  SECTION("no columns and no rows are valid")
  {
    REQUIRE(generate(make_spec({}, 100), 0)->num_columns() == 0);
    const auto empty = generate(mixed_spec(0), 0);
    REQUIRE(empty->num_columns() == 8);
    REQUIRE(empty->num_rows() == 0);
  }

  SECTION("draw_column fills only the vector of the column's type")
  {
    const auto stream = cudf::get_default_stream();
    const auto mr     = cudf::get_current_device_resource_ref();
    const auto ints   = detail::draw_column(spec, 0, 0, stream, mr);
    REQUIRE(ints.ints.size() == 12'345U);
    REQUIRE(ints.doubles.size() == 0U);
    REQUIRE(ints.null_mask.size() == 0U);
    REQUIRE(ints.null_count == 0);
    const auto doubles = detail::draw_column(spec, 2, 0, stream, mr);
    REQUIRE(doubles.ints.size() == 0U);
    REQUIRE(doubles.doubles.size() == 12'345U);
    REQUIRE(doubles.null_mask.size() > 0U);
    REQUIRE(doubles.null_count > 0);
  }
}

TEST_CASE("generator: invalid requests throw", "[generator_distributions]")
{
  using enum distribution;
  const auto spec = mixed_spec(100);
  REQUIRE_THROWS_AS(generate(spec, 8), std::out_of_range);
  REQUIRE_THROWS_AS(generate(spec, std::numeric_limits<std::uint64_t>::max()), std::out_of_range);
  REQUIRE_THROWS_AS(
    detail::draw_column(
      spec, 8, 0, cudf::get_default_stream(), cudf::get_current_device_resource_ref()),
    std::out_of_range);

  auto one = [](column_spec column) { return make_spec({std::move(column)}, 100); };
  // zipf needs N = cardinality > 0 and s > 0
  REQUIRE_THROWS_AS(generate(one(make_column("z", value_type::int64, zipf, 1.1)), 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(generate(one(make_column("z", value_type::int64, zipf, 1.1, 0, -5)), 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(generate(one(make_column("z", value_type::float64, zipf, 0, 0, 10)), 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(generate(one(make_column("f", value_type::int64, few_distinct, 0)), 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(generate(one(make_column("u", value_type::int64, uniform, 10, 9)), 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
    generate(one(make_column("n", value_type::float64, normal, std::nan(""), 1)), 0),
    std::invalid_argument);

  // rows_per_batch must fit a cudf column
  const auto too_many =
    make_spec({make_column("u", value_type::int64, uniform, 0, 9)}, std::uint64_t{1} << 31);
  REQUIRE_THROWS_AS(generate(too_many, 0), std::invalid_argument);
}

TEST_CASE("generator: uniform", "[generator_distributions]")
{
  using enum distribution;

  SECTION("int64 covers [lo, hi] inclusive with the right mean")
  {
    const auto v =
      draw<std::int64_t>(make_column("u", value_type::int64, uniform, -1000, 1000), million);
    const auto [lo, hi] = std::minmax_element(v.begin(), v.end());
    REQUIRE(*lo == -1000);
    REQUIRE(*hi == 1000);
    REQUIRE(std::abs(mean_of(v)) < 5.0);             // sigma of the mean is ~0.56
    REQUIRE(std::abs(stddev_of(v) - 577.64) < 5.0);  // sqrt((2001^2 - 1) / 12)
    REQUIRE(distinct_count(v) == 2001);
  }
  SECTION("int32 narrows the same draw")
  {
    const auto v32 = draw<std::int32_t>(make_column("u", value_type::int32, uniform, 10, 20), 5000);
    const auto v64 = draw<std::int64_t>(make_column("u", value_type::int64, uniform, 10, 20), 5000);
    REQUIRE(std::equal(v32.begin(), v32.end(), v64.begin(), v64.end()));
    REQUIRE(*std::min_element(v32.begin(), v32.end()) == 10);
    REQUIRE(*std::max_element(v32.begin(), v32.end()) == 20);
  }
  SECTION("a single-value range is constant")
  {
    const auto v = draw<std::int64_t>(make_column("u", value_type::int64, uniform, 5, 5), 1000);
    REQUIRE(std::all_of(v.begin(), v.end(), [](auto x) { return x == 5; }));
  }
  SECTION("ranges at the int64 limits do not overflow")
  {
    constexpr auto max64 = std::numeric_limits<std::int64_t>::max();
    // 2^63 saturates to INT64_MAX; 2^63 - 1024 is the largest double below it.
    const auto top = draw<std::int64_t>(
      make_column("u", value_type::int64, uniform, 0x1.0p63 - 1024, 0x1.0p63), 100'000);
    REQUIRE(*std::min_element(top.begin(), top.end()) >= max64 - 1023);
    REQUIRE(distinct_count(top) == 1024);

    const auto full = draw<std::int64_t>(
      make_column("u", value_type::int64, uniform, -0x1.0p63, 0x1.0p63), million);
    const auto negatives = std::count_if(full.begin(), full.end(), [](auto x) { return x < 0; });
    REQUIRE(std::abs(static_cast<double>(negatives) / million - 0.5) < 0.01);
    // spread over the whole range: |mean| is far below the range's sigma of 2^64 / sqrt(12)
    REQUIRE(std::abs(mean_of(full)) < 0x1.0p63 * 0.01);
    REQUIRE(*std::max_element(full.begin(), full.end()) > max64 / 2);
    REQUIRE(*std::min_element(full.begin(), full.end()) < -(max64 / 2));
  }
  SECTION("double is uniform in [lo, hi)")
  {
    const auto v = draw<double>(make_column("u", value_type::float64, uniform, 2.5, 7.5), million);
    REQUIRE(std::all_of(v.begin(), v.end(), [](double x) { return x >= 2.5 && x < 7.5; }));
    REQUIRE(std::abs(mean_of(v) - 5.0) < 0.02);
    REQUIRE(std::abs(stddev_of(v) - 5.0 / std::sqrt(12.0)) < 0.02);
    REQUIRE(*std::min_element(v.begin(), v.end()) < 2.51);
    REQUIRE(*std::max_element(v.begin(), v.end()) > 7.49);
  }
}

TEST_CASE("generator: normal", "[generator_distributions]")
{
  using enum distribution;

  SECTION("double has mean p1 and stddev p2")
  {
    const auto v = draw<double>(make_column("n", value_type::float64, normal, 10, 3), million);
    REQUIRE(std::abs(mean_of(v) - 10.0) < 0.03);
    REQUIRE(std::abs(stddev_of(v) - 3.0) < 0.03);
    REQUIRE(std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); }));
    // Shape: ~68.27% within one sigma, ~95.45% within two.
    const auto within = [&](double k) {
      return static_cast<double>(std::count_if(
               v.begin(), v.end(), [k](double x) { return std::abs(x - 10.0) < 3.0 * k; })) /
             static_cast<double>(v.size());
    };
    REQUIRE(std::abs(within(1) - 0.6827) < 0.01);
    REQUIRE(std::abs(within(2) - 0.9545) < 0.01);
  }
  SECTION("int64 is the rounded draw")
  {
    const auto column_d = make_column("n", value_type::float64, normal, 100, 15);
    const auto column_i = make_column("n", value_type::int64, normal, 100, 15);
    const auto d        = draw<double>(column_d, million);
    const auto i        = draw<std::int64_t>(column_i, million);
    REQUIRE(std::abs(mean_of(i) - 100.0) < 0.2);
    REQUIRE(std::abs(stddev_of(i) - 15.0) < 0.2);
    std::size_t mismatches = 0;
    for (std::size_t r = 0; r < i.size(); ++r) {
      mismatches += i[r] != std::llround(d[r]) ? 1 : 0;
    }
    REQUIRE(mismatches == 0);
  }
}

TEST_CASE("generator: zipf", "[generator_distributions]")
{
  constexpr std::int64_t n = 1000;
  const double s           = GENERATE(0.7, 1.0, 1.5);
  CAPTURE(s);

  const auto v =
    draw<std::int64_t>(make_column("z", value_type::int64, distribution::zipf, s, 0, n), million);
  REQUIRE(std::all_of(v.begin(), v.end(), [](auto k) { return k >= 1 && k <= n; }));

  double harmonic = 0;
  for (std::int64_t k = 1; k <= n; ++k) {
    harmonic += std::pow(static_cast<double>(k), -s);
  }
  std::vector<double> frequency(n + 1, 0.0);
  for (auto k : v) {
    frequency[static_cast<std::size_t>(k)] += 1.0 / static_cast<double>(v.size());
  }

  // Rank 1 is the most frequent, with probability 1 / H(N, s).
  REQUIRE(std::abs(frequency[1] - 1.0 / harmonic) < 0.01);
  for (std::int64_t k = 1; k <= 8; ++k) {
    CAPTURE(k);
    const double expected = std::pow(static_cast<double>(k), -s) / harmonic;
    REQUIRE(std::abs(frequency[static_cast<std::size_t>(k)] - expected) < 0.1 * expected);
    REQUIRE(frequency[static_cast<std::size_t>(k)] >= frequency[static_cast<std::size_t>(k + 1)]);
  }
  // The tail is populated too.
  double tail          = 0;
  double tail_expected = 0;
  for (std::int64_t k = 901; k <= n; ++k) {
    tail += frequency[static_cast<std::size_t>(k)];
    tail_expected += std::pow(static_cast<double>(k), -s) / harmonic;
  }
  REQUIRE(std::abs(tail - tail_expected) < 0.15 * tail_expected);

  SECTION("DOUBLE and INTEGER columns carry the same rank")
  {
    const auto d =
      draw<double>(make_column("z", value_type::float64, distribution::zipf, s, 0, n), million);
    const auto i32 =
      draw<std::int32_t>(make_column("z", value_type::int32, distribution::zipf, s, 0, n), million);
    std::size_t mismatches = 0;
    for (std::size_t r = 0; r < v.size(); ++r) {
      mismatches += (d[r] != static_cast<double>(v[r]) || i32[r] != v[r]) ? 1 : 0;
    }
    REQUIRE(mismatches == 0);
  }
}

TEST_CASE("generator: zipf edge cardinalities", "[generator_distributions]")
{
  using enum distribution;
  const auto one = draw<std::int64_t>(make_column("z", value_type::int64, zipf, 1.3, 0, 1), 10'000);
  REQUIRE(std::all_of(one.begin(), one.end(), [](auto k) { return k == 1; }));

  const auto two =
    draw<std::int64_t>(make_column("z", value_type::int64, zipf, 1.0, 0, 2), million);
  REQUIRE(std::all_of(two.begin(), two.end(), [](auto k) { return k == 1 || k == 2; }));
  const auto ones = std::count(two.begin(), two.end(), 1);
  REQUIRE(std::abs(static_cast<double>(ones) / million - 2.0 / 3.0) < 0.01);

  // A huge N keeps ranks in range and rank 1 the most frequent.
  const auto big =
    draw<std::int64_t>(make_column("z", value_type::int64, zipf, 2.0, 0, 1e15), million);
  REQUIRE(std::all_of(
    big.begin(), big.end(), [](auto k) { return k >= 1 && k <= 1'000'000'000'000'000; }));
  const auto big_ones = std::count(big.begin(), big.end(), 1);
  REQUIRE(std::abs(static_cast<double>(big_ones) / million - 6.0 / (M_PI * M_PI)) < 0.01);
}

TEST_CASE("generator: few_distinct", "[generator_distributions]")
{
  using enum distribution;
  const auto v = draw<std::int64_t>(make_column("f", value_type::int64, few_distinct, 7), million);
  REQUIRE(std::set<std::int64_t>(v.begin(), v.end()) ==
          std::set<std::int64_t>{0, 1, 2, 3, 4, 5, 6});
  REQUIRE(std::abs(mean_of(v) - 3.0) < 0.02);

  // cardinality is ignored: D governs
  const auto with_cardinality =
    draw<std::int64_t>(make_column("f", value_type::int64, few_distinct, 7, 0, 3), million);
  REQUIRE(with_cardinality == v);

  const auto d = draw<double>(make_column("f", value_type::float64, few_distinct, 7), million);
  std::size_t mismatches = 0;
  for (std::size_t r = 0; r < v.size(); ++r) {
    mismatches += d[r] != static_cast<double>(v[r]) ? 1 : 0;
  }
  REQUIRE(mismatches == 0);

  const auto single =
    draw<std::int32_t>(make_column("f", value_type::int32, few_distinct, 1), 1000);
  REQUIRE(std::all_of(single.begin(), single.end(), [](auto x) { return x == 0; }));
}

TEST_CASE("generator: sequential and sequential_desc", "[generator_distributions]")
{
  using enum distribution;
  constexpr std::uint64_t rows = 1000;

  SECTION("sequential is start + step * g")
  {
    const auto v =
      draw<std::int64_t>(make_column("s", value_type::int64, sequential, 5, 3), rows, 2);
    for (std::uint64_t i = 0; i < rows; ++i) {
      REQUIRE(v[i] == static_cast<std::int64_t>(5 + 3 * (2 * rows + i)));
    }
  }
  SECTION("a step of 0 means 1")
  {
    const auto v =
      draw<std::int64_t>(make_column("s", value_type::int64, sequential, -10), rows, 1);
    for (std::uint64_t i = 0; i < rows; ++i) {
      REQUIRE(v[i] == -10 + static_cast<std::int64_t>(rows + i));
    }
    const auto d = draw<double>(make_column("s", value_type::float64, sequential_desc, 0), rows, 1);
    for (std::uint64_t i = 0; i < rows; ++i) {
      REQUIRE(d[i] == -static_cast<double>(rows + i));
    }
  }
  SECTION("DOUBLE keeps a fractional start and step")
  {
    const auto v =
      draw<double>(make_column("s", value_type::float64, sequential, 0.5, 0.25), rows, 3);
    for (std::uint64_t i = 0; i < rows; ++i) {
      REQUIRE(v[i] == 0.5 + 0.25 * static_cast<double>(3 * rows + i));
    }
  }
  SECTION("sequential_desc is strictly decreasing across batches")
  {
    std::vector<std::int64_t> all;
    std::vector<double> all_d;
    for (std::uint64_t b = 0; b < 4; ++b) {
      const auto v =
        draw<std::int64_t>(make_column("s", value_type::int64, sequential_desc, 100, 2), rows, b);
      all.insert(all.end(), v.begin(), v.end());
      const auto d =
        draw<double>(make_column("s", value_type::float64, sequential_desc, 100, 2), rows, b);
      all_d.insert(all_d.end(), d.begin(), d.end());
    }
    for (std::size_t g = 0; g < all.size(); ++g) {
      REQUIRE(all[g] == 100 - 2 * static_cast<std::int64_t>(g));
      REQUIRE(all_d[g] == static_cast<double>(all[g]));
    }
    REQUIRE(std::adjacent_find(all.begin(), all.end(), std::less_equal<>{}) == all.end());
  }
  SECTION("INTEGER narrows with wrap-around")
  {
    const auto v =
      draw<std::int32_t>(make_column("s", value_type::int32, sequential, 2147483646.0, 1), rows);
    REQUIRE(v[0] == std::numeric_limits<std::int32_t>::max() - 1);
    REQUIRE(v[1] == std::numeric_limits<std::int32_t>::max());
    REQUIRE(v[2] == std::numeric_limits<std::int32_t>::min());
    REQUIRE(v[3] == std::numeric_limits<std::int32_t>::min() + 1);
  }
}

TEST_CASE("generator: cardinality clamps int64 draws", "[generator_distributions]")
{
  using enum distribution;

  SECTION("uniform: lo + (v - lo) mod C")
  {
    const auto raw =
      draw<std::int64_t>(make_column("u", value_type::int64, uniform, -50, 1e6), million);
    const auto v =
      draw<std::int64_t>(make_column("u", value_type::int64, uniform, -50, 1e6, 10), million);
    REQUIRE(std::set<std::int64_t>(v.begin(), v.end()) ==
            std::set<std::int64_t>{-50, -49, -48, -47, -46, -45, -44, -43, -42, -41});
    std::size_t mismatches = 0;
    for (std::size_t r = 0; r < v.size(); ++r) {
      mismatches += v[r] != -50 + (raw[r] + 50) % 10 ? 1 : 0;
    }
    REQUIRE(mismatches == 0);
  }
  SECTION("normal: v mod C, non-negative")
  {
    const auto raw =
      draw<std::int64_t>(make_column("n", value_type::int64, normal, 0, 100), million);
    const auto v =
      draw<std::int64_t>(make_column("n", value_type::int64, normal, 0, 100, 5), million);
    REQUIRE(std::set<std::int64_t>(v.begin(), v.end()) == std::set<std::int64_t>{0, 1, 2, 3, 4});
    std::size_t mismatches = 0;
    for (std::size_t r = 0; r < v.size(); ++r) {
      mismatches += v[r] != ((raw[r] % 5) + 5) % 5 ? 1 : 0;
    }
    REQUIRE(mismatches == 0);
  }
  SECTION("sequential cycles through C values")
  {
    const auto up =
      draw<std::int64_t>(make_column("s", value_type::int64, sequential, 100, 1, 7), 1000, 3);
    const auto down =
      draw<std::int64_t>(make_column("s", value_type::int64, sequential_desc, 100, 1, 7), 1000, 3);
    for (std::int64_t i = 0; i < 1000; ++i) {
      const std::int64_t g = 3000 + i;
      REQUIRE(up[static_cast<std::size_t>(i)] == 100 + g % 7);
      REQUIRE(down[static_cast<std::size_t>(i)] == 100 + ((-g % 7) + 7) % 7);
    }
    REQUIRE(distinct_count(up) == 7);
    REQUIRE(distinct_count(down) == 7);
  }
  SECTION("DOUBLE columns ignore cardinality")
  {
    const auto raw = draw<double>(make_column("u", value_type::float64, uniform, 0, 1e6), 10'000);
    const auto v = draw<double>(make_column("u", value_type::float64, uniform, 0, 1e6, 10), 10'000);
    REQUIRE(raw == v);
  }
  SECTION("a clamp near the int64 limits does not overflow")
  {
    const auto v = draw<std::int64_t>(
      make_column("u", value_type::int64, uniform, -0x1.0p63, 0x1.0p63, 0x1.0p62), million);
    constexpr auto min64 = std::numeric_limits<std::int64_t>::min();
    REQUIRE(std::all_of(v.begin(), v.end(), [&](auto x) { return x >= min64 && x < min64 / 2; }));
  }
}

TEST_CASE("generator: nulls", "[generator_distributions]")
{
  using enum distribution;

  SECTION("null_fraction 0 means no null mask")
  {
    const auto table = generate(make_spec({make_column("u", value_type::int64, uniform, 0, 9),
                                           make_column("d", value_type::float64, normal, 0, 1),
                                           make_column("i", value_type::int32, sequential, 0, 1)},
                                          10'000),
                                0);
    for (cudf::size_type i = 0; i < table->num_columns(); ++i) {
      REQUIRE_FALSE(table->view().column(i).nullable());
      REQUIRE(table->view().column(i).null_count() == 0);
    }
  }

  SECTION("the null fraction is honoured for every type")
  {
    const double fraction = GENERATE(0.01, 0.25, 0.9);
    CAPTURE(fraction);
    const auto spec =
      make_spec({make_column("u", value_type::int64, uniform, 0, 9, 0, fraction),
                 make_column("d", value_type::float64, normal, 0, 1, 0, fraction),
                 make_column("i", value_type::int32, few_distinct, 4, 0, 0, fraction)},
                million + 13);  // not a multiple of the mask word
    const auto table = generate(spec, 1);
    for (cudf::size_type c = 0; c < table->num_columns(); ++c) {
      const auto column = table->view().column(c);
      REQUIRE(column.nullable());
      const auto valid = validity(column);
      const auto nulls = std::count(valid.begin(), valid.end(), false);
      REQUIRE(column.null_count() == nulls);
      REQUIRE(std::abs(static_cast<double>(nulls) / static_cast<double>(valid.size()) - fraction) <
              0.01);
      // The host reference of the null rule reproduces every position.
      std::size_t mismatches = 0;
      for (std::size_t r = 0; r < valid.size(); ++r) {
        const bool is_null =
          detail::to_unit(detail::counter_hash(
            spec.seed, 1, r, static_cast<std::uint32_t>(c), detail::NULL_FLAG)) < fraction;
        mismatches += valid[r] == is_null ? 1 : 0;
      }
      REQUIRE(mismatches == 0);
    }
    // Columns have independent null positions, and they are reproducible.
    REQUIRE(validity(table->view().column(0)) != validity(table->view().column(1)));
    const auto again = generate(spec, 1);
    REQUIRE(validity(table->view().column(0)) == validity(again->view().column(0)));
  }

  SECTION("nulls do not change the values drawn")
  {
    const auto plain =
      draw<std::int64_t>(make_column("u", value_type::int64, uniform, 0, 999), 5000);
    const auto nulls =
      draw<std::int64_t>(make_column("u", value_type::int64, uniform, 0, 999, 0, 0.5), 5000);
    REQUIRE(plain == nulls);
  }

  SECTION("null_fraction 1 makes every row null")
  {
    const auto table =
      generate(make_spec({make_column("u", value_type::int64, uniform, 0, 9, 0, 1.0)}, 1000), 0);
    REQUIRE(table->view().column(0).null_count() == 1000);
  }
}

TEST_CASE("generator: device values match a host counter_hash reference",
          "[generator_distributions]")
{
  using enum distribution;
  constexpr std::uint64_t rows  = 4096;
  constexpr std::uint64_t batch = 5;
  const auto spec  = make_spec({make_column("u", value_type::int64, uniform, -1000, 123456789),
                                make_column("f", value_type::int64, few_distinct, 37),
                                make_column("u32", value_type::int32, uniform, 0, 99),
                                make_column("fd", value_type::float64, few_distinct, 1000),
                                make_column("ud", value_type::float64, uniform, -4, 12)},
                              rows,
                              8,
                              0xdecafbadULL);
  const auto table = generate(spec, batch);
  const auto u     = values<std::int64_t>(table->view().column(0));
  const auto f     = values<std::int64_t>(table->view().column(1));
  const auto u32   = values<std::int32_t>(table->view().column(2));
  const auto fd    = values<double>(table->view().column(3));
  const auto ud    = values<double>(table->view().column(4));

  for (std::uint64_t r = 0; r < rows; ++r) {
    CAPTURE(r);
    REQUIRE(u[r] == reference_uniform(spec, batch, r, 0, -1000, 123456789));
    REQUIRE(f[r] == static_cast<std::int64_t>(
                      detail::counter_hash(spec.seed, batch, r, 1, detail::VALUE_A) % 37));
    REQUIRE(u32[r] == static_cast<std::int32_t>(reference_uniform(spec, batch, r, 2, 0, 99)));
    REQUIRE(fd[r] == static_cast<double>(
                       detail::counter_hash(spec.seed, batch, r, 3, detail::VALUE_A) % 1000));
    REQUIRE(ud[r] ==
            -4.0 + detail::to_unit(detail::counter_hash(spec.seed, batch, r, 4, detail::VALUE_A)) *
                     16.0);
  }
}

namespace {

/// The strings of a STRING column on the host; null rows come back as empty strings.
std::vector<std::string> strings_of(cudf::column_view column)
{
  REQUIRE(column.type().id() == cudf::type_id::STRING);
  const auto rows = static_cast<std::size_t>(column.size());
  std::vector<std::string> out(rows);
  if (rows == 0) { return out; }
  const cudf::strings_column_view strings{column};
  std::vector<std::int64_t> offsets;
  if (strings.offsets().type().id() == cudf::type_id::INT32) {
    const auto narrow = values<std::int32_t>(strings.offsets());
    offsets.assign(narrow.begin(), narrow.end());
  } else {
    offsets = values<std::int64_t>(strings.offsets());
  }
  REQUIRE(offsets.size() == rows + 1);
  std::vector<char> chars(static_cast<std::size_t>(offsets[rows]));
  if (!chars.empty()) {
    REQUIRE(::cudaMemcpy(chars.data(),
                         strings.chars_begin(cudf::get_default_stream()),
                         chars.size(),
                         ::cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  for (std::size_t i = 0; i < rows; ++i) {
    out[i] = std::string(chars.data() + offsets[i], chars.data() + offsets[i + 1]);
  }
  return out;
}

}  // namespace

// For VARCHAR, p1 / p2 are both the distribution's parameters and the min / max string length.
TEST_CASE("generator: VARCHAR columns format the int64 draw", "[generator_distributions]")
{
  using enum distribution;
  constexpr std::uint64_t rows = 1000;

  SECTION("negative draws are clamped to key 0")
  {
    // sequential_desc 3, 3 => key = 3 - 3g; length exactly 3, so no suffix.
    const auto table =
      generate(make_spec({make_column("s", value_type::varchar, sequential_desc, 3, 3)}, rows), 0);
    REQUIRE(table->num_rows() == static_cast<cudf::size_type>(rows));
    REQUIRE(table->view().column(0).type().id() == cudf::type_id::STRING);
    REQUIRE_FALSE(table->view().column(0).nullable());
    const auto s = strings_of(table->view().column(0));
    REQUIRE(s[0] == "003");
    REQUIRE(s[1] == "000");
    REQUIRE(std::all_of(s.begin() + 1, s.end(), [](const auto& x) { return x == "000"; }));
  }
  SECTION("the zero-padded prefix is the BIGINT draw and nulls match it")
  {
    // few_distinct D = 5 => keys 0..4, padded to 5 characters, total length in [5, 9].
    const auto varchar_spec =
      make_spec({make_column("v", value_type::varchar, few_distinct, 5, 9, 0, 0.2)}, 5000);
    const auto bigint_spec =
      make_spec({make_column("v", value_type::int64, few_distinct, 5, 9, 0, 0.2)}, 5000);
    const auto text  = generate(varchar_spec, 1);
    const auto keys  = generate(bigint_spec, 1);
    const auto valid = validity(keys->view().column(0));
    const auto s     = strings_of(text->view().column(0));
    const auto k     = values<std::int64_t>(keys->view().column(0));
    REQUIRE(text->view().column(0).null_count() == keys->view().column(0).null_count());
    REQUIRE(text->view().column(0).null_count() > 0);
    REQUIRE(validity(text->view().column(0)) == valid);
    std::set<std::size_t> lengths;
    for (std::size_t r = 0; r < s.size(); ++r) {
      if (!valid[r]) { continue; }
      CAPTURE(r);
      REQUIRE(s[r].size() >= 5);
      REQUIRE(s[r].size() <= 9);
      REQUIRE(s[r].substr(0, 5) == "0000" + std::to_string(k[r]));
      REQUIRE(
        std::all_of(s[r].begin() + 5, s[r].end(), [](char c) { return c >= 'a' && c <= 'z'; }));
      lengths.insert(s[r].size());
    }
    REQUIRE(lengths == std::set<std::size_t>{5, 6, 7, 8, 9});
    REQUIRE(strings_of(generate(varchar_spec, 1)->view().column(0)) == s);
  }
}

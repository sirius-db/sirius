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

// Tests for gpu_aggregate_impl::hash_encode -- the hash-based drop-in replacement for
// cudf::encode used by the COLLECT_SET (COUNT DISTINCT) label-encode group-by path.
//
// The oracle for most tests is cudf::encode itself: for any input, hash_encode must
// produce the exact same (sorted distinct rows, per-row label) pair. Scenarios:
//
//   1. q16-shaped multi-column string+int keys with heavy duplication.
//   2. NULL key rows: NULL == NULL forms one group, sorted last.
//   3. NaN key rows: NaN == NaN collapses to a single label.
//   4. Empty input, single row, all-rows-identical.
//   5. Large-cardinality randomized differential (exercises hash-table collision
//      handling inside cudf::distinct / distinct_hash_join) including the
//      all-rows-distinct extreme.
//   6. End-to-end: COUNT DISTINCT through local_grouped_aggregate at the label-encode
//      gate scale (>= 2^20 rows, low group cardinality) so the hash_encode path runs
//      inside the real operator and the post-aggregate key recovery is validated.

#include "../operator_test_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <catch.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace sirius;
using namespace sirius::op;
using namespace sirius::test::operator_utils;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

memory_space* get_shared_mem_space()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

template <typename T>
std::unique_ptr<cudf::column> make_numeric_column_from(const std::vector<T>& values,
                                                       cudf::type_id type_id,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto col = cudf::make_numeric_column(cudf::data_type{type_id},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  if (!values.empty()) {
    cudaMemcpyAsync(col->mutable_view().data<T>(),
                    values.data(),
                    values.size() * sizeof(T),
                    cudaMemcpyHostToDevice,
                    stream.value());
  }
  stream.synchronize();
  return col;
}

/// Apply a validity vector (true == valid) as a null mask onto an existing column.
void apply_validity(cudf::column& col,
                    const std::vector<bool>& valid,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  REQUIRE(static_cast<std::size_t>(col.size()) == valid.size());
  auto const mask_bytes =
    cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(valid.size()));
  std::vector<cudf::bitmask_type> words(mask_bytes / sizeof(cudf::bitmask_type), 0);
  cudf::size_type null_count = 0;
  for (std::size_t i = 0; i < valid.size(); ++i) {
    if (valid[i]) {
      words[i / 32] |= (cudf::bitmask_type{1} << (i % 32));
    } else {
      ++null_count;
    }
  }
  rmm::device_buffer mask(mask_bytes, stream, mr);
  cudaMemcpyAsync(mask.data(), words.data(), mask_bytes, cudaMemcpyHostToDevice, stream.value());
  stream.synchronize();
  col.set_null_mask(std::move(mask), null_count);
}

/// Value-level equality of two columns, on host. NaN == NaN. Null rows must line up
/// (values under null rows are not compared).
void expect_columns_equal(const cudf::column_view& actual, const cudf::column_view& expected)
{
  REQUIRE(actual.type().id() == expected.type().id());
  REQUIRE(actual.size() == expected.size());
  auto const actual_valid   = copy_validity_to_host(actual);
  auto const expected_valid = copy_validity_to_host(expected);
  REQUIRE(actual_valid == expected_valid);

  auto compare = [&](auto tag) {
    using T                  = decltype(tag);
    auto const actual_host   = copy_column_to_host<T>(actual);
    auto const expected_host = copy_column_to_host<T>(expected);
    std::size_t mismatches   = 0;
    std::size_t first        = 0;
    for (std::size_t i = 0; i < actual_host.size(); ++i) {
      if (!actual_valid[i]) { continue; }
      if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(actual_host[i]) && std::isnan(expected_host[i])) { continue; }
      }
      if (actual_host[i] != expected_host[i]) {
        if (mismatches == 0) { first = i; }
        ++mismatches;
      }
    }
    INFO("mismatching rows: " << mismatches << ", first at row " << first);
    REQUIRE(mismatches == 0);
  };

  switch (actual.type().id()) {
    case cudf::type_id::INT32: compare(int32_t{}); break;
    case cudf::type_id::INT64: compare(int64_t{}); break;
    case cudf::type_id::FLOAT64: compare(double{}); break;
    case cudf::type_id::STRING: compare(std::string{}); break;
    default: FAIL("expect_columns_equal: unhandled type in test helper");
  }
}

/// Differential oracle: hash_encode(keys) must equal cudf::encode(keys) exactly.
void expect_matches_cudf_encode(const cudf::table_view& keys,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto [actual_keys, actual_labels]     = gpu_aggregate_impl::hash_encode(keys, stream, mr);
  auto [expected_keys, expected_labels] = cudf::encode(keys, stream, mr);
  stream.synchronize();

  REQUIRE(actual_keys->num_columns() == expected_keys->num_columns());
  REQUIRE(actual_keys->num_rows() == expected_keys->num_rows());
  for (cudf::size_type c = 0; c < actual_keys->num_columns(); ++c) {
    INFO("distinct-keys column " << c);
    expect_columns_equal(actual_keys->view().column(c), expected_keys->view().column(c));
  }
  expect_columns_equal(actual_labels->view(), expected_labels->view());
}

}  // namespace

TEST_CASE("hash_encode: q16-shaped string keys with duplicates", "[hash_encode]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  // Unsorted input with heavy duplication, mixed string lengths, and an empty string.
  std::vector<std::string> brand = {"Brand#34",
                                    "Brand#12",
                                    "Brand#34",
                                    "Brand#12",
                                    "Brand#34",
                                    "Brand#12",
                                    "",
                                    "Brand#34",
                                    "Brand#3",
                                    ""};
  std::vector<std::string> type  = {"ECONOMY BRUSHED",
                                    "STANDARD POLISHED",
                                    "ECONOMY BRUSHED",
                                    "STANDARD POLISHED",
                                    "PROMO PLATED",
                                    "STANDARD POLISHED",
                                    "ECONOMY BRUSHED",
                                    "ECONOMY BRUSHED",
                                    "ECONOMY BRUSHED",
                                    "ECONOMY BRUSHED"};
  std::vector<int32_t> size      = {9, 14, 9, 14, 9, 23, 9, 9, 9, 9};

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_string_column(brand, stream, mr));
  cols.push_back(make_string_column(type, stream, mr));
  cols.push_back(make_numeric_column_from(size, cudf::type_id::INT32, stream, mr));
  cudf::table keys(std::move(cols));

  expect_matches_cudf_encode(keys.view(), stream, mr);

  // Hand-checked semantics: 6 distinct tuples; identical tuples share a label; the
  // label dereferences back to the original tuple.
  auto [distinct, labels] = gpu_aggregate_impl::hash_encode(keys.view(), stream, mr);
  stream.synchronize();
  REQUIRE(distinct->num_rows() == 6);
  auto const label_host = copy_column_to_host<int32_t>(labels->view());
  auto const brand_host = copy_column_to_host<std::string>(distinct->view().column(0));
  auto const type_host  = copy_column_to_host<std::string>(distinct->view().column(1));
  auto const size_host  = copy_column_to_host<int32_t>(distinct->view().column(2));
  REQUIRE(label_host.size() == brand.size());
  REQUIRE(label_host[0] == label_host[2]);  // duplicate tuples -> same label
  REQUIRE(label_host[0] != label_host[4]);  // row 4 differs in type only -> new label
  for (std::size_t i = 0; i < label_host.size(); ++i) {
    INFO("row " << i);
    auto const l = label_host[i];
    REQUIRE(l >= 0);
    REQUIRE(l < 6);
    REQUIRE(brand_host[l] == brand[i]);
    REQUIRE(type_host[l] == type[i]);
    REQUIRE(size_host[l] == size[i]);
  }
  // Distinct rows come back lexicographically sorted.
  for (std::size_t i = 1; i < brand_host.size(); ++i) {
    auto const prev = std::tie(brand_host[i - 1], type_host[i - 1], size_host[i - 1]);
    auto const cur  = std::tie(brand_host[i], type_host[i], size_host[i]);
    REQUIRE(prev < cur);
  }
}

TEST_CASE("hash_encode: NULL keys form their own group, sorted last", "[hash_encode]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  std::vector<std::string> str_vals = {"x", "y", "x", "IGNORED", "y", "IGNORED", "x"};
  std::vector<bool> str_valid       = {true, true, true, false, true, false, true};
  std::vector<int32_t> int_vals     = {1, 2, 1, 5, 2, 5, 0 /* IGNORED */};
  std::vector<bool> int_valid       = {true, true, true, true, true, true, false};

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_string_column(str_vals, stream, mr));
  apply_validity(*cols.back(), str_valid, stream, mr);
  cols.push_back(make_numeric_column_from(int_vals, cudf::type_id::INT32, stream, mr));
  apply_validity(*cols.back(), int_valid, stream, mr);
  cudf::table keys(std::move(cols));

  expect_matches_cudf_encode(keys.view(), stream, mr);

  auto [distinct, labels] = gpu_aggregate_impl::hash_encode(keys.view(), stream, mr);
  stream.synchronize();
  // Distinct tuples: ("x",1), ("y",2), (NULL,5), ("x",NULL) -> 4 groups; the two
  // (NULL,5) rows must share one label (NULL == NULL).
  REQUIRE(distinct->num_rows() == 4);
  auto const label_host = copy_column_to_host<int32_t>(labels->view());
  REQUIRE(label_host[3] == label_host[5]);
  REQUIRE(label_host[0] == label_host[2]);
  REQUIRE(label_host[1] == label_host[4]);
  // NULL sorts after valid values within each column (null_order::AFTER): the
  // string-NULL group must come after every valid-string group.
  auto const distinct_str_valid = copy_validity_to_host(distinct->view().column(0));
  REQUIRE_FALSE(distinct_str_valid.back());
}

TEST_CASE("hash_encode: NaN keys collapse to one label", "[hash_encode]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  auto const nan           = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> vals = {1.5, nan, 2.5, nan, 1.5, nan};

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_numeric_column_from(vals, cudf::type_id::FLOAT64, stream, mr));
  cudf::table keys(std::move(cols));

  expect_matches_cudf_encode(keys.view(), stream, mr);

  auto [distinct, labels] = gpu_aggregate_impl::hash_encode(keys.view(), stream, mr);
  stream.synchronize();
  REQUIRE(distinct->num_rows() == 3);  // {1.5, 2.5, NaN}
  auto const label_host = copy_column_to_host<int32_t>(labels->view());
  REQUIRE(label_host[1] == label_host[3]);
  REQUIRE(label_host[1] == label_host[5]);
  REQUIRE(label_host[0] == label_host[4]);
  REQUIRE(label_host[0] != label_host[1]);
}

TEST_CASE("hash_encode: empty input, single row, all rows identical", "[hash_encode]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  SECTION("empty input")
  {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(make_string_column({}, stream, mr));
    cols.push_back(make_numeric_column_from<int32_t>({}, cudf::type_id::INT32, stream, mr));
    cudf::table keys(std::move(cols));

    auto [distinct, labels] = gpu_aggregate_impl::hash_encode(keys.view(), stream, mr);
    stream.synchronize();
    REQUIRE(distinct->num_rows() == 0);
    REQUIRE(labels->size() == 0);
    REQUIRE(labels->type().id() == cudf::type_id::INT32);
  }

  SECTION("single row")
  {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(make_string_column({"only"}, stream, mr));
    cudf::table keys(std::move(cols));
    expect_matches_cudf_encode(keys.view(), stream, mr);
  }

  SECTION("all rows identical")
  {
    std::vector<std::string> vals(1000, "same");
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(make_string_column(vals, stream, mr));
    cudf::table keys(std::move(cols));

    expect_matches_cudf_encode(keys.view(), stream, mr);
    auto [distinct, labels] = gpu_aggregate_impl::hash_encode(keys.view(), stream, mr);
    stream.synchronize();
    REQUIRE(distinct->num_rows() == 1);
    auto const label_host = copy_column_to_host<int32_t>(labels->view());
    REQUIRE(std::all_of(label_host.begin(), label_host.end(), [](int32_t l) { return l == 0; }));
  }

  SECTION("no key columns is rejected")
  {
    cudf::table_view empty_keys{std::vector<cudf::column_view>{}};
    REQUIRE_THROWS_AS(gpu_aggregate_impl::hash_encode(empty_keys, stream, mr), std::runtime_error);
  }
}

TEST_CASE("hash_encode: large-cardinality randomized differential", "[hash_encode]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  std::mt19937_64 rng(0xC0FFEE);

  SECTION("~50% distinct: heavy collision territory")
  {
    constexpr int n = 200000;
    std::uniform_int_distribution<int64_t> dist(0, n / 2);
    std::vector<int64_t> a(n), b(n);
    for (int i = 0; i < n; ++i) {
      a[i] = dist(rng);
      b[i] = dist(rng) % 37;  // second column adds tuple structure
    }
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(make_numeric_column_from(a, cudf::type_id::INT64, stream, mr));
    cols.push_back(make_numeric_column_from(b, cudf::type_id::INT64, stream, mr));
    cudf::table keys(std::move(cols));
    expect_matches_cudf_encode(keys.view(), stream, mr);
  }

  SECTION("all rows distinct: G == N extreme")
  {
    constexpr int n = 100000;
    std::vector<int64_t> a(n);
    for (int i = 0; i < n; ++i) {
      a[i] = i;
    }
    std::shuffle(a.begin(), a.end(), rng);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(make_numeric_column_from(a, cudf::type_id::INT64, stream, mr));
    cudf::table keys(std::move(cols));
    expect_matches_cudf_encode(keys.view(), stream, mr);
  }

  SECTION("random strings with duplicates")
  {
    constexpr int n = 50000;
    std::uniform_int_distribution<int> pick(0, 999);
    std::vector<std::string> s(n);
    for (int i = 0; i < n; ++i) {
      s[i] = "key_" + std::to_string(pick(rng));
    }
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(make_string_column(s, stream, mr));
    cudf::table keys(std::move(cols));
    expect_matches_cudf_encode(keys.view(), stream, mr);
  }
}

// ===========================================================================
// End-to-end: COUNT DISTINCT through local_grouped_aggregate at the label-encode gate
// scale. 1.1M rows (>= 2^20 gate), 100 distinct (brand, type, size) tuples
// (ratio 100/1.1M << 0.01 gate) so the hash_encode label path runs inside the real
// operator. value = i % 7 with group(i) = i % 100 puts all 7 values in every group.
// ===========================================================================
TEST_CASE("hash_encode: COUNT DISTINCT label path end-to-end at gate scale",
          "[hash_encode][physical_grouped_aggregate_count_distinct]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  constexpr int num_rows = 1'100'000;
  static_assert(num_rows >= (1 << 20));

  std::vector<std::string> brand(num_rows), type(num_rows);
  std::vector<int32_t> size_col(num_rows), value(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    brand[i]    = "Brand#" + std::to_string(i % 20);
    type[i]     = "TYPE " + std::to_string(i % 25);
    size_col[i] = i % 20;
    value[i]    = i % 7;
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_string_column(brand, stream, mr));
  cols.push_back(make_string_column(type, stream, mr));
  cols.push_back(make_numeric_column_from(size_col, cudf::type_id::INT32, stream, mr));
  cols.push_back(make_numeric_column_from(value, cudf::type_id::INT32, stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(table), *space, stream);
  auto batch = cucascade::data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));

  auto out_batch = gpu_aggregate_impl::local_grouped_aggregate(
    batch->to_read_only(),
    /*group_idx=*/{0, 1, 2},
    /*aggregates=*/{cudf::aggregation::Kind::COLLECT_SET},
    /*aggregate_idx=*/{3},
    /*aggregate_struct_col_indices=*/{},
    stream,
    *space);
  REQUIRE(out_batch);

  auto ro         = out_batch->to_read_only();
  auto const view = sirius::get_cudf_table_view(ro);
  REQUIRE(view.num_columns() == 4);
  // group(i) = i % 100 (lcm of 20, 25, 20), so exactly 100 groups.
  REQUIRE(view.num_rows() == 100);

  auto const out_brand = copy_column_to_host<std::string>(view.column(0));
  auto const out_type  = copy_column_to_host<std::string>(view.column(1));
  auto const out_size  = copy_column_to_host<int32_t>(view.column(2));

  // Reconstruct each group's key tuple and check it is one of the expected tuples,
  // each appearing exactly once.
  std::vector<std::tuple<std::string, std::string, int32_t>> actual_tuples, expected_tuples;
  for (int g = 0; g < 100; ++g) {
    expected_tuples.emplace_back(
      "Brand#" + std::to_string(g % 20), "TYPE " + std::to_string(g % 25), g % 20);
  }
  for (int r = 0; r < 100; ++r) {
    actual_tuples.emplace_back(out_brand[r], out_type[r], out_size[r]);
  }
  std::sort(expected_tuples.begin(), expected_tuples.end());
  std::sort(actual_tuples.begin(), actual_tuples.end());
  REQUIRE(actual_tuples == expected_tuples);

  // Every group must have collected exactly the 7 distinct values {0..6}.
  cudf::lists_column_view lists(view.column(3));
  auto const offsets = copy_column_to_host<int32_t>(lists.offsets());
  auto const child   = copy_column_to_host<int32_t>(lists.child());
  REQUIRE(offsets.size() == 101);
  for (int g = 0; g < 100; ++g) {
    INFO("group " << g);
    REQUIRE(offsets[g + 1] - offsets[g] == 7);
    std::vector<int32_t> got(child.begin() + offsets[g], child.begin() + offsets[g + 1]);
    std::sort(got.begin(), got.end());
    REQUIRE(got == std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6}));
  }
}

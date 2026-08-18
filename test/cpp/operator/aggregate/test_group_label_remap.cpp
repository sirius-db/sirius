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

// Parity tests for gpu_aggregate_impl::compute_group_labels_via_remap, the
// distinct + cudf::key_remapping replacement for cudf::encode in the grouped-aggregate
// COLLECT_SET label path. The contract is byte-identical output: the same lexicographically
// sorted distinct-key table (nulls last) and the same per-row INT32 labels.

#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace sirius::test::operator_utils;
using sirius::op::gpu_aggregate_impl;
using sirius::test::vector_to_cudf_column;

// Attach a validity mask built from `valid` (true = valid) to `col`.
std::unique_ptr<cudf::column> with_nulls(std::unique_ptr<cudf::column> col,
                                         const std::vector<bool>& valid)
{
  REQUIRE(static_cast<std::size_t>(col->size()) == valid.size());
  auto valid_col          = vector_to_cudf_column<gpu_type_traits<bool>>(valid);
  auto [mask, null_count] = cudf::bools_to_mask(valid_col->view());
  col->set_null_mask(std::move(*mask), null_count);
  return col;
}

// Row-wise equality of two columns, matching values AND validity (null == null).
void require_columns_equal(const cudf::column_view& actual, const cudf::column_view& expected)
{
  REQUIRE(actual.type().id() == expected.type().id());
  REQUIRE(actual.size() == expected.size());
  REQUIRE(actual.null_count() == expected.null_count());
  if (actual.size() == 0) { return; }
  auto eq = cudf::binary_operation(
    actual, expected, cudf::binary_operator::NULL_EQUALS, cudf::data_type{cudf::type_id::BOOL8});
  auto all_agg = cudf::make_all_aggregation<cudf::reduce_aggregation>();
  auto all     = cudf::reduce(eq->view(), *all_agg, cudf::data_type{cudf::type_id::BOOL8});
  REQUIRE(static_cast<cudf::numeric_scalar<bool>&>(*all).value(cudf::get_default_stream()));
}

void require_tables_equal(const cudf::table_view& actual, const cudf::table_view& expected)
{
  REQUIRE(actual.num_columns() == expected.num_columns());
  REQUIRE(actual.num_rows() == expected.num_rows());
  for (cudf::size_type c = 0; c < actual.num_columns(); ++c) {
    require_columns_equal(actual.column(c), expected.column(c));
  }
}

// The parity oracle: the remap path must reproduce cudf::encode byte-for-byte, and its
// labels must round-trip (gathering the distinct-key table by the labels rebuilds the
// original keys -- the invariant the grouped aggregate actually relies on).
void require_encode_parity(const cudf::table_view& keys)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto expected = cudf::encode(keys, stream, mr);
  auto actual   = gpu_aggregate_impl::compute_group_labels_via_remap(keys, stream, mr);

  require_tables_equal(actual.first->view(), expected.first->view());
  require_columns_equal(actual.second->view(), expected.second->view());

  if (keys.num_rows() > 0) {
    auto rebuilt = cudf::gather(actual.first->view(),
                                actual.second->view(),
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                stream,
                                mr);
    require_tables_equal(rebuilt->view(), keys);
  }
}

std::unique_ptr<cudf::table> make_table(std::vector<std::unique_ptr<cudf::column>> cols)
{
  return std::make_unique<cudf::table>(std::move(cols));
}

// Host copy of a fixed-width column's values and validity, for comparisons the
// device-side NULL_EQUALS oracle cannot express (NaN != NaN under NULL_EQUALS).
template <typename T>
std::pair<std::vector<T>, std::vector<uint8_t>> to_host(const cudf::column_view& col)
{
  std::vector<T> vals(col.size());
  std::vector<uint8_t> valid(col.size(), 1);
  REQUIRE(cudaMemcpy(vals.data(), col.data<T>(), col.size() * sizeof(T), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  if (col.null_count() > 0) {
    auto bools = cudf::mask_to_bools(col.null_mask(), col.offset(), col.offset() + col.size());
    REQUIRE(
      cudaMemcpy(valid.data(), bools->view().data<bool>(), col.size(), cudaMemcpyDeviceToHost) ==
      cudaSuccess);
  }
  return {std::move(vals), std::move(valid)};
}

}  // namespace

TEST_CASE("label remap parity: q16-shaped string keys with duplicates", "[aggregate][label_remap]")
{
  // Two string columns + one int32, heavy duplication across a small dictionary --
  // the exact shape of the TPC-H q16 group-by (brand, type, size).
  std::vector<std::string> brands = {"Brand#12", "Brand#34", "Brand#45", "Brand#51"};
  std::vector<std::string> types  = {"STANDARD ANODIZED TIN", "SMALL PLATED COPPER", "PROMO"};
  std::vector<int32_t> sizes      = {3, 9, 14, 23};

  std::vector<std::string> brand_col, type_col;
  std::vector<int32_t> size_col;
  std::minstd_rand rng(7);
  for (int i = 0; i < 4096; ++i) {
    brand_col.push_back(brands[rng() % brands.size()]);
    type_col.push_back(types[rng() % types.size()]);
    size_col.push_back(sizes[rng() % sizes.size()]);
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(brand_col));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(type_col));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(size_col));
  auto keys = make_table(std::move(cols));

  require_encode_parity(keys->view());
}

TEST_CASE("label remap parity: null keys group together and sort last", "[aggregate][label_remap]")
{
  // Nulls in either column, rows with both columns null (duplicated, so null-keys must
  // dedupe together under null_equality::EQUAL), and null vs valid orderings that only
  // match encode if the fixup sort uses null_order::AFTER.
  std::vector<std::string> str_col = {
    "apple", "banana", "apple", "IGNORED", "banana", "IGNORED", "cherry", "IGNORED", "apple"};
  std::vector<bool> str_valid  = {true, true, true, false, true, false, true, false, true};
  std::vector<int32_t> int_col = {1, 2, 1, 5, -7, 5, 0, 5, 0};
  std::vector<bool> int_valid  = {true, true, true, false, true, false, false, false, true};

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(
    with_nulls(vector_to_cudf_column<gpu_type_traits<string_tag>>(str_col), str_valid));
  cols.push_back(with_nulls(vector_to_cudf_column<gpu_type_traits<int32_t>>(int_col), int_valid));
  auto keys = make_table(std::move(cols));

  require_encode_parity(keys->view());
}

TEST_CASE("label remap parity: single distinct key", "[aggregate][label_remap]")
{
  SECTION("one valid key repeated")
  {
    std::vector<std::string> str_col(1000, "only-key");
    std::vector<int32_t> int_col(1000, 42);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(str_col));
    cols.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(int_col));
    auto keys = make_table(std::move(cols));
    require_encode_parity(keys->view());
  }

  SECTION("all rows null")
  {
    std::vector<int32_t> int_col(257, 0);
    std::vector<bool> all_invalid(257, false);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(
      with_nulls(vector_to_cudf_column<gpu_type_traits<int32_t>>(int_col), all_invalid));
    auto keys = make_table(std::move(cols));
    require_encode_parity(keys->view());
  }

  SECTION("single row")
  {
    std::vector<std::string> str_col = {"x"};
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(str_col));
    auto keys = make_table(std::move(cols));
    require_encode_parity(keys->view());
  }
}

TEST_CASE("label remap parity: all rows distinct", "[aggregate][label_remap]")
{
  // ndv == rows: exercises the id-density check at full scale (labels are a permutation).
  std::vector<std::string> str_col;
  std::vector<int32_t> int_col;
  for (int i = 0; i < 4096; ++i) {
    str_col.push_back("key-" + std::to_string(i * 7919 % 100000));
    int_col.push_back(i);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(str_col));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(int_col));
  auto keys = make_table(std::move(cols));

  require_encode_parity(keys->view());
}

TEST_CASE("label remap parity: empty input", "[aggregate][label_remap]")
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>({}));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>({}));
  auto keys = make_table(std::move(cols));

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto actual = gpu_aggregate_impl::compute_group_labels_via_remap(keys->view(), stream, mr);
  REQUIRE(actual.first->num_rows() == 0);
  REQUIRE(actual.first->num_columns() == 2);
  REQUIRE(actual.second->size() == 0);
  REQUIRE(actual.second->type().id() == cudf::type_id::INT32);
}

TEST_CASE("label remap parity: DOUBLE keys with NaN payloads, signed zeros, and nulls",
          "[aggregate][label_remap]")
{
  // Floating-point keys reach this path (the gate only excludes nested keys and a single
  // null-free fixed-width key). NaN-key parity rests on two cudf contracts staying aligned:
  // cudf::distinct's nan_equality::ALL_EQUAL and cudf::key_remapping's "All NaNs are
  // considered equal". A probe-side NaN miss would surface as a negative raw id flowing
  // through the DONT_CHECK gather -- this test pins the current (correct) behavior.
  //
  // The device-side NULL_EQUALS oracle cannot compare NaN-containing key tables (NaN != NaN),
  // so this test asserts (a) label byte-parity with cudf::encode via the INT32 oracle,
  // (b) distinct-count parity, and (c) a host-side label round-trip that treats all NaNs as
  // one key and tolerates the KEEP_ANY representative choice for +/-0.0 and NaN payloads.
  double const qnan = std::numeric_limits<double>::quiet_NaN();
  double const pnan = std::nan("1");  // same key as qnan, different mantissa payload
  {
    // The payload distinction is the point of the test -- fail loudly if it degrades.
    REQUIRE(std::memcmp(&qnan, &pnan, sizeof(double)) != 0);
  }

  std::vector<double> dbl_col = {
    1.5, qnan, -0.0, 2.5, pnan, 0.0, 1.5, 99.0, qnan, 99.0, 3.25, -0.0};
  std::vector<bool> dbl_valid = {
    true, true, true, true, true, true, true, false, true, false, true, true};
  // Distinct keys: {1.5, 2.5, +/-0.0, 3.25, NaN, null} = 6.
  cudf::size_type const expected_distinct = 6;

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(with_nulls(vector_to_cudf_column<gpu_type_traits<double>>(dbl_col), dbl_valid));
  auto keys = make_table(std::move(cols));

  auto stream   = cudf::get_default_stream();
  auto mr       = cudf::get_current_device_resource_ref();
  auto expected = cudf::encode(keys->view(), stream, mr);
  auto actual   = gpu_aggregate_impl::compute_group_labels_via_remap(keys->view(), stream, mr);

  // (a) Label byte-parity (INT32 labels, NULL_EQUALS oracle is exact here).
  require_columns_equal(actual.second->view(), expected.second->view());
  // (b) Distinct-count parity, and the semantically expected group count.
  REQUIRE(actual.first->num_rows() == expected.first->num_rows());
  REQUIRE(actual.first->num_rows() == expected_distinct);

  // (c) Host-side round-trip: dist[label[i]] must reproduce row i's key, where "reproduce"
  // means matching validity, and for valid rows equal values (== admits -0.0 == +0.0) or
  // both NaN (any payload).
  auto [lbl_vals, lbl_valid]   = to_host<int32_t>(actual.second->view());
  auto [dist_vals, dist_valid] = to_host<double>(actual.first->view().column(0));
  for (std::size_t i = 0; i < dbl_col.size(); ++i) {
    REQUIRE(lbl_valid[i] == 1);
    int32_t const label = lbl_vals[i];
    REQUIRE(label >= 0);
    REQUIRE(label < expected_distinct);
    REQUIRE(static_cast<bool>(dist_valid[label]) == static_cast<bool>(dbl_valid[i]));
    if (dbl_valid[i]) {
      double const original = dbl_col[i];
      double const rebuilt  = dist_vals[label];
      bool const both_nan   = std::isnan(original) && std::isnan(rebuilt);
      REQUIRE((both_nan || original == rebuilt));
    }
  }
  // Grouping semantics pinned row-by-row: all NaN payloads one label, +/-0.0 one label,
  // nulls one label, duplicates one label.
  REQUIRE(lbl_vals[1] == lbl_vals[4]);   // qnan == pnan
  REQUIRE(lbl_vals[1] == lbl_vals[8]);   // qnan duplicate
  REQUIRE(lbl_vals[2] == lbl_vals[5]);   // -0.0 == +0.0
  REQUIRE(lbl_vals[2] == lbl_vals[11]);  // -0.0 duplicate
  REQUIRE(lbl_vals[7] == lbl_vals[9]);   // null == null
  REQUIRE(lbl_vals[0] == lbl_vals[6]);   // 1.5 duplicate
  REQUIRE(lbl_vals[0] != lbl_vals[1]);   // value vs NaN separate
  REQUIRE(lbl_vals[1] != lbl_vals[7]);   // NaN vs null separate
}

TEST_CASE("label remap parity: randomized mixed-type keys with nulls", "[aggregate][label_remap]")
{
  // Larger randomized sweep: 50k rows over ~200 distinct (string, string, int32) triples
  // with independent random validity on two of the columns.
  std::vector<std::string> str_a_col, str_b_col;
  std::vector<int32_t> int_col;
  std::vector<bool> str_a_valid, int_valid;
  std::mt19937 rng(20260817);
  for (int i = 0; i < 50000; ++i) {
    str_a_col.push_back("alpha-" + std::to_string(rng() % 10));
    str_b_col.push_back("beta-" + std::to_string(rng() % 5));
    int_col.push_back(static_cast<int32_t>(rng() % 4));
    str_a_valid.push_back(rng() % 17 != 0);
    int_valid.push_back(rng() % 13 != 0);
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(
    with_nulls(vector_to_cudf_column<gpu_type_traits<string_tag>>(str_a_col), str_a_valid));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(str_b_col));
  cols.push_back(with_nulls(vector_to_cudf_column<gpu_type_traits<int32_t>>(int_col), int_valid));
  auto keys = make_table(std::move(cols));

  require_encode_parity(keys->view());
}

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

// sirius::meld::meld_unit over cuDF tables on the GPU: a filter, a product and two aggregates,
// over columns Meld reads where cuDF holds them. The values are dyadic, so every sum is exact.

#include "catch.hpp"
#include "meld/meld_unit.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr char const* filtered_sum_plan = R"(
meld.plan version "0.99.0" {
  meld.extension_urn @comparison
      "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml"
  meld.extension_urn @arithmetic
      "https://github.com/substrait-io/substrait/blob/main/extensions/functions_arithmetic.yaml"
  meld.extension_urn @aggregate
      "https://github.com/substrait-io/substrait/blob/main/extensions/functions_aggregate_generic.yaml"
  meld.function @lt in @comparison "lt:fp64_fp64"
  meld.function @multiply in @arithmetic "multiply:fp64_fp64"
  meld.function @sum in @arithmetic "sum:fp64"
  meld.function @count in @aggregate "count:"

  meld.root names ["total", "rows"] {
    %t = meld.named_table ["t"] columns ["x", "y"]
        : !meld.relation<!meld.nullable<f64>, f64>
    %kept = meld.filter %t {
    ^bb0(%x: !meld.nullable<f64>, %y: f64):
      %limit = meld.literal 4.0 : f64
      %small = meld.call @lt(%x, %limit) : (!meld.nullable<f64>, f64) -> !meld.nullable<i1>
      meld.yield %small : !meld.nullable<i1>
    }
    %products = meld.project %kept {
    ^bb0(%x: !meld.nullable<f64>, %y: f64):
      %product = meld.call @multiply(%x, %y) : (!meld.nullable<f64>, f64) -> !meld.nullable<f64>
      meld.yield %product : !meld.nullable<f64>
    } : !meld.relation<!meld.nullable<f64>>
    %answer = meld.aggregate %products keys {
    ^bb0(%product: !meld.nullable<f64>):
      meld.yield
    } measures {
    ^bb0(%product: !meld.nullable<f64>):
      %total = meld.measure all @sum(%product) : (!meld.nullable<f64>) -> !meld.nullable<f64>
      %rows = meld.measure all @count() : () -> i64
      meld.yield %total, %rows : !meld.nullable<f64>, i64
    } : !meld.relation<!meld.nullable<f64>, i64>
    meld.yield %answer : !meld.relation<!meld.nullable<f64>, i64>
  }
}
)";

std::unique_ptr<cudf::column> doubles(std::vector<double> const& values,
                                      std::vector<bool> const& valid = {})
{
  auto stream = cudf::get_default_stream();
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream);
  cudaMemcpy(column->mutable_view().data<double>(),
             values.data(),
             values.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  if (!valid.empty()) {
    std::vector<cudf::bitmask_type> mask(
      cudf::bitmask_allocation_size_bytes(values.size()) / sizeof(cudf::bitmask_type), 0);
    cudf::size_type nulls = 0;
    for (std::size_t row = 0; row < valid.size(); ++row) {
      if (valid[row]) {
        mask[row / 32] |= cudf::bitmask_type{1} << (row % 32);
      } else {
        ++nulls;
      }
    }
    rmm::device_buffer device_mask(mask.data(), mask.size() * sizeof(cudf::bitmask_type), stream);
    column->set_null_mask(std::move(device_mask), nulls);
  }
  stream.sync();
  return column;
}

template <typename T>
std::optional<T> scalar_at(cudf::column_view const& column)
{
  REQUIRE(column.size() == 1);
  if (column.has_nulls()) { return std::nullopt; }
  T value;
  cudaMemcpy(&value, column.data<T>(), sizeof(T), cudaMemcpyDeviceToHost);
  return value;
}

std::unique_ptr<sirius::meld::meld_unit> checked(std::string plan)
{
  std::string why;
  auto unit = sirius::meld::meld_unit::check(std::move(plan), {"x", "y"}, why);
  INFO(why);
  REQUIRE(unit);
  return unit;
}

}  // namespace

TEST_CASE("meld_unit learns the answer's types when it checks a plan", "[meld_unit]")
{
  auto unit = checked(filtered_sum_plan);
  REQUIRE(unit->answer_types().size() == 2);
  CHECK(unit->answer_types()[0] == cudf::data_type{cudf::type_id::FLOAT64});
  CHECK(unit->answer_types()[1] == cudf::data_type{cudf::type_id::INT64});
}

TEST_CASE("meld_unit runs a plan over a batch on the GPU", "[meld_unit]")
{
  auto unit = checked(filtered_sum_plan);
  auto x    = doubles({0.5, 1.25, 8.0, 3.75, 2.0, 16.0}, {true, true, true, true, false, true});
  auto y    = doubles({2.0, 4.0, 1.0, 0.5, 64.0, 1.0});
  cudf::table_view batch({x->view(), y->view()});

  auto answer =
    unit->run(batch, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  REQUIRE(answer->num_columns() == 2);
  // Rows 0, 1 and 3 pass; row 4 is NULL and does not.
  CHECK(scalar_at<double>(answer->view().column(0)) == 0.5 * 2.0 + 1.25 * 4.0 + 3.75 * 0.5);
  CHECK(scalar_at<std::int64_t>(answer->view().column(1)) == 3);
}

TEST_CASE("meld_unit takes the device memory of a run from the resource it is given", "[meld_unit]")
{
  auto unit = checked(filtered_sum_plan);
  auto x    = doubles({0.5, 1.25, 8.0, 3.75, 2.0, 16.0}, {true, true, true, true, false, true});
  auto y    = doubles({2.0, 4.0, 1.0, 0.5, 64.0, 1.0});
  cudf::table_view batch({x->view(), y->view()});
  rmm::mr::statistics_resource_adaptor allocations{
    cuda::mr::any_resource<cuda::mr::device_accessible>{cudf::get_current_device_resource_ref()}};

  auto answer =
    unit->run(batch, cudf::get_default_stream(), rmm::device_async_resource_ref{allocations});
  CHECK(scalar_at<std::int64_t>(answer->view().column(1)) == 3);
  answer.reset();
  // More than the answer's three buffers, and Meld has given its own back.
  CHECK(allocations.get_allocations_counter().total > 3);
  CHECK(allocations.get_allocations_counter().value == 0);
}

TEST_CASE("meld_unit fails a run the resource it is given cannot hold", "[meld_unit]")
{
  auto unit = checked(filtered_sum_plan);
  auto x    = doubles({0.5, 1.25, 8.0});
  auto y    = doubles({2.0, 4.0, 1.0});
  cudf::table_view batch({x->view(), y->view()});
  rmm::mr::limiting_resource_adaptor tiny{
    cuda::mr::any_resource<cuda::mr::device_accessible>{cudf::get_current_device_resource_ref()},
    1};

  CHECK_THROWS_AS(
    unit->run(batch, cudf::get_default_stream(), rmm::device_async_resource_ref{tiny}),
    sirius::internal_exception);
}

TEST_CASE("meld_unit reads a sliced batch from its offset", "[meld_unit]")
{
  auto unit = checked(filtered_sum_plan);
  auto x    = doubles({0.5, 1.25, 8.0, 3.75, 2.0, 16.0}, {true, true, true, true, false, true});
  auto y    = doubles({2.0, 4.0, 1.0, 0.5, 64.0, 1.0});
  cudf::table_view whole({x->view(), y->view()});
  auto batch = cudf::slice(whole, {3, 6}).front();

  auto answer =
    unit->run(batch, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CHECK(scalar_at<double>(answer->view().column(0)) == 3.75 * 0.5);
  CHECK(scalar_at<std::int64_t>(answer->view().column(1)) == 1);
}

TEST_CASE("meld_unit refuses a plan that reads a column the batches do not have", "[meld_unit]")
{
  std::string why;
  auto unit = sirius::meld::meld_unit::check(filtered_sum_plan, {"x"}, why);
  CHECK_FALSE(unit);
  CHECK(why.find("'y'") != std::string::npos);
}

TEST_CASE("meld_unit says why Meld cannot read a plan", "[meld_unit]")
{
  std::string why;
  auto unit = sirius::meld::meld_unit::check("meld.plan nonsense", {"x", "y"}, why);
  CHECK_FALSE(unit);
  CHECK(why.find("Meld cannot read the plan") != std::string::npos);
}

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

/**
 * @file test_physical_replicate.cpp
 * @brief Tests `gpu_replicate_impl::plan_slices` and `materialize` on plain cuDF tables, then
 * `sirius_physical_replicate::execute` on data batches.
 */

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <catch.hpp>
#include <op/replicate/gpu_replicate_impl.hpp>
#include <op/sirius_physical_replicate.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace sirius::test::operator_utils;
using sirius::op::pipelineable_operator_data;
using sirius::op::sirius_physical_replicate;
namespace replicate = sirius::op::gpu_replicate_impl;

constexpr std::size_t unbounded_bytes = std::size_t{1} << 62;

//! An `INT64` column; row `i` is NULL when `valid` is given and `valid[i]` is false.
std::unique_ptr<cudf::column> int64_column(std::vector<std::int64_t> const& values,
                                           std::optional<std::vector<bool>> const& valid = {})
{
  auto mr     = get_resource_ref(*get_default_gpu_space());
  auto stream = default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());
  auto column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, size, cudf::mask_state::UNALLOCATED, stream, mr);
  if (size > 0) {
    CUDF_CUDA_TRY(cudaMemcpy(column->mutable_view().data<std::int64_t>(),
                             values.data(),
                             values.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice));
  }
  if (valid) {
    auto mask             = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
    cudf::size_type nulls = 0;
    for (cudf::size_type i = 0; i < size; ++i) {
      if (!(*valid)[i]) {
        cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), i, i + 1, false, stream);
        ++nulls;
      }
    }
    column->set_null_mask(std::move(mask), nulls);
  }
  return column;
}

std::unique_ptr<cudf::column> string_column(std::vector<std::string> const& values)
{
  return make_string_column(values, default_stream(), get_resource_ref(*get_default_gpu_space()));
}

std::shared_ptr<cucascade::data_batch> batch_of(std::vector<std::unique_ptr<cudf::column>> columns)
{
  auto& space = *get_default_gpu_space();
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                 space,
                                 default_stream(),
                                 sirius::telemetry::batch_telemetry_info{});
}

replicate::plan plan_of(cudf::table_view const& data,
                        cudf::column_view const& counts,
                        replicate::limits const& caps)
{
  return replicate::plan_slices(
    data, counts, caps, default_stream(), get_resource_ref(*get_default_gpu_space()));
}

//! Every slice of @p expansion, materialized and concatenated in order.
std::unique_ptr<cudf::table> materialize_all(cudf::table_view const& data,
                                             replicate::plan const& expansion)
{
  std::vector<std::unique_ptr<cudf::table>> parts;
  std::vector<cudf::table_view> views;
  for (auto const& part : expansion.slices) {
    parts.push_back(replicate::materialize(
      data, expansion, part, default_stream(), get_resource_ref(*get_default_gpu_space())));
    views.push_back(parts.back()->view());
  }
  return cudf::concatenate(views, default_stream(), get_resource_ref(*get_default_gpu_space()));
}

//! Checks that @p expansion tiles `[0, sum(counts))` in non-empty slices within @p caps, each
//! naming exactly the input rows its output rows copy; `row_bytes[i]` is one copy of row `i`.
void require_exact_tiling(replicate::plan const& expansion,
                          std::vector<std::int64_t> const& counts,
                          std::vector<std::int64_t> const& row_bytes,
                          replicate::limits const& caps)
{
  std::vector<std::int64_t> output_row_source;
  std::vector<std::int64_t> output_row_bytes;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    output_row_source.insert(output_row_source.end(), counts[i], static_cast<std::int64_t>(i));
    output_row_bytes.insert(output_row_bytes.end(), counts[i], row_bytes[i]);
  }
  auto const max_bytes = static_cast<std::int64_t>(
    std::min<std::size_t>(caps.max_bytes, std::numeric_limits<std::int64_t>::max()));

  std::int64_t next = 0;
  for (auto const& part : expansion.slices) {
    REQUIRE(part.output_begin == next);
    REQUIRE(part.output_end > part.output_begin);
    CHECK(part.output_end - part.output_begin <= caps.max_rows);
    std::int64_t bytes = 0;
    for (auto row = part.output_begin; row < part.output_end; ++row) {
      bytes += output_row_bytes[row];
    }
    // Only the slice's last row may cross the byte cap.
    CHECK(bytes - output_row_bytes[part.output_end - 1] < max_bytes);
    CHECK(part.input_begin == output_row_source[part.output_begin]);
    CHECK(part.input_end == output_row_source[part.output_end - 1] + 1);
    next = part.output_end;
  }
  CHECK(next == static_cast<std::int64_t>(output_row_source.size()));
}

//! Bytes of one copy of each row of @p data, as `plan_slices` weighs it.
std::vector<std::int64_t> row_bytes_of(cudf::table_view const& data)
{
  auto const bits =
    cudf::row_bit_count(data, default_stream(), get_resource_ref(*get_default_gpu_space()));
  std::vector<std::int64_t> bytes;
  for (auto const row_bits : copy_column_to_host<std::int32_t>(bits->view())) {
    bytes.push_back(std::max<std::int64_t>(1, (std::int64_t{row_bits} + 7) / 8));
  }
  return bytes;
}

std::vector<std::shared_ptr<cucascade::data_batch>> run(
  sirius_physical_replicate& op, std::vector<std::shared_ptr<cucascade::data_batch>> inputs)
{
  auto output = op.execute(pipelineable_operator_data(inputs), default_stream());
  return dynamic_cast<pipelineable_operator_data const&>(*output).get_data_batches();
}

duckdb::vector<sirius::logical_type> logical_types(std::vector<duckdb::LogicalType> const& types)
{
  return sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>(types.begin(), types.end()));
}

}  // namespace

TEST_CASE("gpu_replicate_impl - slices tile the expansion within both caps",
          "[operator][replicate]")
{
  std::mt19937 rng(20261005);
  std::uniform_int_distribution<std::int64_t> count_of(0, 9);
  std::vector<std::int64_t> counts(300);
  for (auto& count : counts) {
    count = count_of(rng);
  }
  std::vector<std::int64_t> values(counts.size(), 7);
  auto data_column   = int64_column(values);
  auto counts_column = int64_column(counts);
  cudf::table_view const data{{data_column->view()}};
  // A non-nullable INT64 row is 64 bits.
  std::vector<std::int64_t> const row_bytes(counts.size(), 8);

  SECTION("the row cap binds")
  {
    replicate::limits const caps{4, unbounded_bytes};
    require_exact_tiling(plan_of(data, counts_column->view(), caps), counts, row_bytes, caps);
  }
  SECTION("the byte cap binds")
  {
    replicate::limits const caps{1000, 44};
    require_exact_tiling(plan_of(data, counts_column->view(), caps), counts, row_bytes, caps);
  }
  SECTION("both caps bind in turn")
  {
    replicate::limits const caps{7, 50};
    require_exact_tiling(plan_of(data, counts_column->view(), caps), counts, row_bytes, caps);
  }
}

TEST_CASE("gpu_replicate_impl - a total past INT32_MAX is planned in 64 bits",
          "[operator][replicate]")
{
  // Planned only: materializing 2^31 rows is the overflow the plan exists to avoid.
  auto data_column   = int64_column({42});
  auto counts_column = int64_column({(std::int64_t{1} << 31) + 5});
  replicate::limits const caps{1 << 30, unbounded_bytes};
  auto const expansion =
    plan_of(cudf::table_view{{data_column->view()}}, counts_column->view(), caps);

  REQUIRE(expansion.slices.size() == 3);
  CHECK(expansion.slices[0].output_begin == 0);
  CHECK(expansion.slices[1].output_begin == std::int64_t{1} << 30);
  CHECK(expansion.slices[2].output_begin == std::int64_t{1} << 31);
  CHECK(expansion.slices[2].output_end == (std::int64_t{1} << 31) + 5);
  for (auto const& part : expansion.slices) {
    CHECK(part.input_begin == 0);
    CHECK(part.input_end == 1);
  }
}

TEST_CASE("gpu_replicate_impl - a byte cap past INT64_MAX leaves only the row cap",
          "[operator][replicate]")
{
  auto data_column = int64_column({10, 20});
  std::vector<std::int64_t> const counts{3, 5};
  auto counts_column = int64_column(counts);
  cudf::table_view const data{{data_column->view()}};
  replicate::limits const caps{4, std::numeric_limits<std::size_t>::max()};
  auto const expansion = plan_of(data, counts_column->view(), caps);

  REQUIRE(expansion.slices.size() == 2);
  require_exact_tiling(expansion, counts, row_bytes_of(data), caps);
  auto const output = materialize_all(data, expansion);
  CHECK(copy_column_to_host<std::int64_t>(output->get_column(0).view()) ==
        std::vector<std::int64_t>{10, 10, 10, 20, 20, 20, 20, 20});
}

TEST_CASE("gpu_replicate_impl - a sliced input materializes a small slice past INT32_MAX",
          "[operator][replicate]")
{
  auto values = int64_column({99, 10, 20, 30, 40, 50, 99});
  auto counts = int64_column({0, std::int64_t{1} << 31, 0, 3, 0, 2, 0});
  auto const data =
    cudf::slice(cudf::table_view{{values->view()}}, {1, 6}, default_stream()).front();
  auto const count_view = cudf::slice(counts->view(), {1, 6}, default_stream()).front();
  auto const expansion =
    plan_of(data, count_view, {1 << 30, std::numeric_limits<std::size_t>::max()});

  REQUIRE(expansion.slices.size() == 3);
  auto const& part = expansion.slices.back();
  REQUIRE(part.output_begin == std::int64_t{1} << 31);
  REQUIRE(part.output_end - part.output_begin == 5);
  CHECK(part.input_begin == 2);
  CHECK(part.input_end == 5);

  // Only the small final slice is materialized; the preceding rows remain a prefix count.
  auto const output = replicate::materialize(
    data, expansion, part, default_stream(), get_resource_ref(*get_default_gpu_space()));
  CHECK(copy_column_to_host<std::int64_t>(output->get_column(0).view()) ==
        std::vector<std::int64_t>{30, 30, 30, 50, 50});
}

TEST_CASE("gpu_replicate_impl - sliced nullable strings preserve split runs with no byte cap",
          "[operator][replicate]")
{
  auto names = string_column({"prefix", "a", "bb", "ccc", "dddd", "eeeee", "suffix"});
  auto mask  = cudf::create_null_mask(
    7, cudf::mask_state::ALL_VALID, default_stream(), get_resource_ref(*get_default_gpu_space()));
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 3, 4, false, default_stream());
  names->set_null_mask(std::move(mask), 1);
  auto counts = int64_column({0, 0, 2, 3, 0, 1, 0});
  auto const data =
    cudf::slice(cudf::table_view{{names->view()}}, {1, 6}, default_stream()).front();
  auto const count_view = cudf::slice(counts->view(), {1, 6}, default_stream()).front();
  replicate::limits const caps{4, std::numeric_limits<std::size_t>::max()};
  auto const expansion = plan_of(data, count_view, caps);
  require_exact_tiling(expansion, {0, 2, 3, 0, 1}, row_bytes_of(data), caps);
  auto const output  = materialize_all(data, expansion);
  auto const& column = output->get_column(0);

  REQUIRE(column.size() == 6);
  CHECK(copy_validity_to_host(column.view()) ==
        std::vector<bool>{true, true, false, false, false, true});
  auto const strings = copy_column_to_host<std::string>(column.view());
  CHECK(strings[0] == "bb");
  CHECK(strings[1] == "bb");
  CHECK(strings[5] == "eeeee");
}

TEST_CASE("gpu_replicate_impl - one count above the row cap splits that row",
          "[operator][replicate]")
{
  auto data_column   = int64_column({10, 20, 30});
  auto counts_column = int64_column({0, 13, 0});
  cudf::table_view const data{{data_column->view()}};
  auto const expansion = plan_of(data, counts_column->view(), {4, unbounded_bytes});

  REQUIRE(expansion.slices.size() == 4);
  std::vector<std::int64_t> sizes;
  for (auto const& part : expansion.slices) {
    sizes.push_back(part.output_end - part.output_begin);
    CHECK(part.input_begin == 1);
    CHECK(part.input_end == 2);
  }
  CHECK(sizes == std::vector<std::int64_t>{4, 4, 4, 1});
  auto const output = materialize_all(data, expansion);
  CHECK(copy_column_to_host<std::int64_t>(output->get_column(0).view()) ==
        std::vector<std::int64_t>(13, 20));
}

TEST_CASE("gpu_replicate_impl - zero counts produce no slices", "[operator][replicate]")
{
  auto const max_bytes = GENERATE(unbounded_bytes, std::numeric_limits<std::size_t>::max());
  replicate::limits const caps{4, max_bytes};
  SECTION("every count is zero")
  {
    auto data_column   = int64_column({1, 2, 3});
    auto counts_column = int64_column({0, 0, 0});
    auto const expansion =
      plan_of(cudf::table_view{{data_column->view()}}, counts_column->view(), caps);
    CHECK(expansion.slices.empty());
  }
  SECTION("the input is empty")
  {
    auto data_column   = int64_column({});
    auto counts_column = int64_column({});
    auto const expansion =
      plan_of(cudf::table_view{{data_column->view()}}, counts_column->view(), caps);
    CHECK(expansion.slices.empty());
    CHECK(expansion.row_prefix->size() == 0);
  }
}

TEST_CASE("gpu_replicate_impl - a long string row is cut by bytes", "[operator][replicate]")
{
  // One copy is 1 MiB of characters plus a 4-byte offset, so two copies cross 2 MiB.
  auto data_column   = string_column({std::string(std::size_t{1} << 20, 'x')});
  auto counts_column = int64_column({10});
  cudf::table_view const data{{data_column->view()}};
  replicate::limits const caps{std::numeric_limits<cudf::size_type>::max(), std::size_t{2} << 20};
  auto const expansion = plan_of(data, counts_column->view(), caps);

  REQUIRE(expansion.slices.size() == 5);
  for (auto const& part : expansion.slices) {
    CHECK(part.output_end - part.output_begin == 2);
  }
  auto const first = replicate::materialize(data,
                                            expansion,
                                            expansion.slices.front(),
                                            default_stream(),
                                            get_resource_ref(*get_default_gpu_space()));
  REQUIRE(first->num_rows() == 2);
  CHECK(copy_column_to_host<std::string>(first->get_column(0).view()) ==
        std::vector<std::string>(2, std::string(std::size_t{1} << 20, 'x')));
}

TEST_CASE("gpu_replicate_impl - wide rows avoid redundant byte-cut scratch",
          "[operator][replicate]")
{
  auto const max_bytes      = GENERATE(std::size_t{1}, std::size_t{31});
  std::int64_t const copies = max_bytes == 1 ? 1 : 4;
  auto data_column          = string_column({std::string(std::size_t{1} << 20, 'x')});
  auto counts_column        = int64_column({copies});
  cudf::table_view const data{{data_column->view()}};
  replicate::limits const caps{std::numeric_limits<cudf::size_type>::max(), max_bytes};
  auto mr = get_resource_ref(*get_default_gpu_space());
  rmm::mr::statistics_resource_adaptor stats{
    cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};

  auto const expansion = replicate::plan_slices(
    data, counts_column->view(), caps, default_stream(), rmm::device_async_resource_ref{stats});

  REQUIRE(expansion.slices.size() == static_cast<std::size_t>(copies));
  require_exact_tiling(expansion, {copies}, {(std::int64_t{1} << 20) + 4}, caps);
  // Count planner allocations only; the string and counts were allocated before the adaptor.
  CHECK(stats.get_bytes_counter().peak < (std::int64_t{1} << 20));
}

TEST_CASE("gpu_replicate_impl - adaptive byte cuts preserve mixed row widths and zero counts",
          "[operator][replicate]")
{
  std::string const wide(120, 'x');
  auto data_column   = string_column({"unused", "a", wide, "unused", "b"});
  auto counts_column = int64_column({0, 2, 3, 0, 1});
  cudf::table_view const data{{data_column->view()}};
  replicate::limits const caps{3, 31};

  auto const expansion = plan_of(data, counts_column->view(), caps);
  require_exact_tiling(expansion, {0, 2, 3, 0, 1}, {10, 5, 124, 10, 5}, caps);
  // Keep the global byte grid: the first slice contains both narrow copies and one wide copy.
  std::vector<std::int64_t> ends;
  for (auto const& part : expansion.slices) {
    ends.push_back(part.output_end);
  }
  CHECK(ends == std::vector<std::int64_t>{3, 4, 5, 6});
  auto const output = materialize_all(data, expansion);
  CHECK(copy_column_to_host<std::string>(output->get_column(0).view()) ==
        std::vector<std::string>{"a", "a", wide, wide, wide, "b"});
}

TEST_CASE("gpu_replicate_impl - slices concatenate to the whole expansion", "[operator][replicate]")
{
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::int64_t> count_of(0, 6);
  std::vector<std::int64_t> counts(64);
  std::vector<std::int64_t> values(counts.size());
  std::vector<bool> valid(counts.size());
  std::vector<std::string> names(counts.size());
  for (std::size_t i = 0; i < counts.size(); ++i) {
    counts[i] = count_of(rng);
    values[i] = static_cast<std::int64_t>(i) * 3;
    valid[i]  = i % 5 != 0;
    names[i]  = std::string(i % 11, static_cast<char>('a' + i % 26));
  }
  auto value_column  = int64_column(values, valid);
  auto name_column   = string_column(names);
  auto counts_column = int64_column(counts);
  cudf::table_view const data{{value_column->view(), name_column->view()}};

  std::vector<std::int64_t> expected_values;
  std::vector<bool> expected_valid;
  std::vector<std::string> expected_names;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    expected_values.insert(expected_values.end(), counts[i], values[i]);
    expected_valid.insert(expected_valid.end(), counts[i], valid[i]);
    expected_names.insert(expected_names.end(), counts[i], names[i]);
  }

  auto const caps = GENERATE(replicate::limits{4, unbounded_bytes}, replicate::limits{1000, 64});
  auto const expansion = plan_of(data, counts_column->view(), caps);
  require_exact_tiling(expansion, counts, row_bytes_of(data), caps);
  auto const output = materialize_all(data, expansion);
  REQUIRE(output->num_rows() == static_cast<cudf::size_type>(expected_values.size()));
  auto const host_valid = copy_validity_to_host(output->get_column(0).view());
  CHECK(host_valid == expected_valid);
  auto const host_values = copy_column_to_host<std::int64_t>(output->get_column(0).view());
  for (std::size_t row = 0; row < expected_values.size(); ++row) {
    if (expected_valid[row]) { CHECK(host_values[row] == expected_values[row]); }
  }
  CHECK(copy_column_to_host<std::string>(output->get_column(1).view()) == expected_names);
}

TEST_CASE("gpu_replicate_impl - an invalid count column throws", "[operator][replicate]")
{
  auto data_column = int64_column({1, 2});
  cudf::table_view const data{{data_column->view()}};
  replicate::limits const caps{4, unbounded_bytes};
  SECTION("a null count")
  {
    auto counts_column = int64_column({1, 2}, std::vector<bool>{true, false});
    CHECK_THROWS_AS(plan_of(data, counts_column->view(), caps), sirius::internal_exception);
  }
  SECTION("a negative count")
  {
    auto counts_column = int64_column({1, -1});
    CHECK_THROWS_AS(plan_of(data, counts_column->view(), caps), sirius::internal_exception);
  }
  SECTION("a non-integer count")
  {
    CHECK_THROWS_AS(plan_of(data, string_column({"1", "2"})->view(), caps),
                    sirius::internal_exception);
  }
  SECTION("a count column of another length")
  {
    auto counts_column = int64_column({1, 2, 3});
    CHECK_THROWS_AS(plan_of(data, counts_column->view(), caps), sirius::internal_exception);
  }
}

TEST_CASE("gpu_replicate_impl - a zero cap throws", "[operator][replicate]")
{
  auto data_column   = int64_column({1, 2});
  auto counts_column = int64_column({1, 1});
  cudf::table_view const data{{data_column->view()}};
  CHECK_THROWS_AS(plan_of(data, counts_column->view(), {0, unbounded_bytes}),
                  sirius::internal_exception);
  CHECK_THROWS_AS(plan_of(data, counts_column->view(), {4, 0}), sirius::internal_exception);
}

TEST_CASE("gpu_replicate_impl - a narrower integer count column is widened",
          "[operator][replicate]")
{
  auto data_column = int64_column({5, 6});
  auto counts      = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT8},
                                          2,
                                          cudf::mask_state::UNALLOCATED,
                                          default_stream(),
                                          get_resource_ref(*get_default_gpu_space()));
  std::vector<std::int8_t> const host_counts{2, 1};
  CUDF_CUDA_TRY(cudaMemcpy(counts->mutable_view().data<std::int8_t>(),
                           host_counts.data(),
                           host_counts.size(),
                           cudaMemcpyHostToDevice));
  cudf::table_view const data{{data_column->view()}};
  auto const output = materialize_all(data, plan_of(data, counts->view(), {4, unbounded_bytes}));
  CHECK(copy_column_to_host<std::int64_t>(output->get_column(0).view()) ==
        std::vector<std::int64_t>{5, 5, 6});
}

TEMPLATE_TEST_CASE("sirius_physical_replicate repeats every column type",
                   "[operator][replicate]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   bool,
                   decimal64_tag,
                   string_tag,
                   timestamp_us_tag,
                   date32_tag)
{
  using Traits = gpu_type_traits<TestType>;
  auto* space  = get_default_gpu_space();
  auto values  = Traits::sample_values();
  values.resize(3, values.back());
  std::vector<std::int64_t> const counts{0, 1, 3};

  std::shared_ptr<cucascade::data_batch> input;
  if constexpr (Traits::is_decimal) {
    input = make_two_column_batch<std::int64_t, typename Traits::type>(
      *space, counts, values, Traits::cudf_type, Traits::scale, cudf::type_id::INT64);
  } else {
    input = make_two_column_batch<std::int64_t, typename Traits::type>(
      *space, counts, values, Traits::cudf_type, std::nullopt);
  }
  // The helper puts the counts first; REPLICATE takes them last.
  auto swapped = cudf::table{sirius::get_cudf_table_view(*input).select({1, 0})}.release();

  sirius_physical_replicate op(logical_types({Traits::logical_type()}), {4, unbounded_bytes}, 3);
  auto const outputs = run(op, {batch_of(std::move(swapped))});
  REQUIRE(outputs.size() == 1);
  auto const view = sirius::get_cudf_table_view(*outputs[0]);
  REQUIRE(view.num_columns() == 1);
  std::vector<typename Traits::type> expected{values[1], values[2], values[2], values[2]};
  CHECK(copy_column_to_host<typename Traits::type>(view.column(0)) == expected);
}

TEST_CASE("sirius_physical_replicate repeats NULL data values and drops the count column",
          "[operator][replicate]")
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column({1, 2}, std::vector<bool>{true, false}));
  columns.push_back(string_column({"a", "bb"}));
  columns.push_back(int64_column({2, 1}));

  sirius_physical_replicate op(
    logical_types({duckdb::LogicalType::BIGINT, duckdb::LogicalType::VARCHAR}),
    {4, unbounded_bytes},
    2);
  auto const outputs = run(op, {batch_of(std::move(columns))});
  REQUIRE(outputs.size() == 1);
  auto const view = sirius::get_cudf_table_view(*outputs[0]);
  REQUIRE(view.num_columns() == 2);
  CHECK(copy_validity_to_host(view.column(0)) == std::vector<bool>{true, true, false});
  CHECK(copy_column_to_host<std::int64_t>(view.column(0)).front() == 1);
  CHECK(copy_column_to_host<std::string>(view.column(1)) ==
        std::vector<std::string>{"a", "a", "bb"});
}

TEST_CASE("sirius_physical_replicate splits and keeps batches apart", "[operator][replicate]")
{
  sirius_physical_replicate op(
    logical_types({duckdb::LogicalType::BIGINT}), {4, unbounded_bytes}, 0);

  auto make_input = [](std::vector<std::int64_t> values, std::vector<std::int64_t> counts) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column(values));
    columns.push_back(int64_column(counts));
    return batch_of(std::move(columns));
  };
  auto const outputs = run(op, {make_input({1, 2}, {5, 4}), make_input({3}, {2})});

  std::vector<std::int64_t> sizes;
  std::vector<std::int64_t> rows;
  for (auto const& batch : outputs) {
    auto const view = sirius::get_cudf_table_view(*batch);
    sizes.push_back(view.num_rows());
    auto const host = copy_column_to_host<std::int64_t>(view.column(0));
    rows.insert(rows.end(), host.begin(), host.end());
  }
  CHECK(sizes == std::vector<std::int64_t>{4, 4, 1, 2});
  CHECK(rows == std::vector<std::int64_t>{1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3});
}

TEST_CASE("sirius_physical_replicate emits one empty batch for an input with no copies",
          "[operator][replicate]")
{
  sirius_physical_replicate op(
    logical_types({duckdb::LogicalType::VARCHAR}), {4, unbounded_bytes}, 0);
  auto const counts = GENERATE(std::vector<std::int64_t>{}, std::vector<std::int64_t>{0, 0});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(string_column(std::vector<std::string>(counts.size(), "x")));
  columns.push_back(int64_column(counts));
  auto const outputs = run(op, {batch_of(std::move(columns))});
  REQUIRE(outputs.size() == 1);
  auto const view = sirius::get_cudf_table_view(*outputs[0]);
  CHECK(view.num_rows() == 0);
  REQUIRE(view.num_columns() == 1);
  CHECK(view.column(0).type().id() == cudf::type_id::STRING);
}

TEST_CASE("sirius_physical_replicate rejects a bad count", "[operator][replicate]")
{
  sirius_physical_replicate op(
    logical_types({duckdb::LogicalType::BIGINT}), {4, unbounded_bytes}, 0);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column({1, 2}));
  columns.push_back(int64_column({1, -3}));
  CHECK_THROWS_AS(run(op, {batch_of(std::move(columns))}), sirius::internal_exception);

  CHECK_THROWS_AS(sirius_physical_replicate(logical_types({}), {4, 1}, 0),
                  sirius::internal_exception);
}

TEST_CASE("sirius_physical_replicate rejects zero limits and a batch of the wrong width",
          "[operator][replicate]")
{
  auto const types = logical_types({duckdb::LogicalType::BIGINT});
  CHECK_THROWS_AS(sirius_physical_replicate(types, {0, unbounded_bytes}, 0),
                  sirius::internal_exception);
  CHECK_THROWS_AS(sirius_physical_replicate(types, {4, 0}, 0), sirius::internal_exception);

  sirius_physical_replicate op(types, {4, unbounded_bytes}, 0);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column({1, 2}));
  columns.push_back(int64_column({3, 4}));
  columns.push_back(int64_column({1, 1}));
  CHECK_THROWS_AS(run(op, {batch_of(std::move(columns))}), sirius::internal_exception);
}

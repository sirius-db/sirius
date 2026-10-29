/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "io/kvikio/kvikio_context.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/iceberg_gpu_ingestible.hpp"
#include "op/scan/positional_visibility.hpp"
#include "sirius_context.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <sys/resource.h>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <chrono>
#include <ctime>
#include <iostream>
#include <numeric>

using namespace sirius::op::scan;
namespace {
template <typename T>
std::unique_ptr<cudf::column> device_column(std::vector<T> const& values,
                                            cudf::type_id type,
                                            rmm::cuda_stream_view stream)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{type}, values.size(), cudf::mask_state::UNALLOCATED, stream);
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(col->mutable_view().template data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
  }
  return col;
}
template <typename T>
std::vector<T> host_column(cudf::column_view col, rmm::cuda_stream_view stream)
{
  std::vector<T> values(col.size());
  if (!values.empty())
    REQUIRE(cudaMemcpyAsync(values.data(),
                            col.data<T>(),
                            values.size() * sizeof(T),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
  stream.synchronize();
  return values;
}
std::unique_ptr<cudf::table> row_payload(int count, rmm::cuda_stream_view stream)
{
  std::vector<int32_t> data(count);
  std::iota(data.begin(), data.end(), 0);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(device_column(data, cudf::type_id::INT32, stream));
  return std::make_unique<cudf::table>(std::move(cols));
}
}  // namespace

TEST_CASE("Reader positions retain noncontiguous file rows and source context",
          "[scan][row_positions]")
{
  rmm::cuda_stream stream;
  auto const multi    = GENERATE(false, true);
  auto const empty    = GENERATE(false, true);
  auto const deletion = std::string(GENERATE("partial", "all", "no_intersection"));
  CAPTURE(multi, empty, deletion);
  auto positions             = std::make_unique<gpu_row_positions>();
  positions->sources         = {{"a", 1000, {{0, 100}, {300, 400}}}};
  std::vector<uint64_t> rows = empty ? std::vector<uint64_t>{} : std::vector<uint64_t>{11, 301};
  if (multi) {
    positions->sources.push_back({"b", 1000, {{0, 1000}}});
    if (!empty) rows.insert(rows.end(), {11, 301});
    positions->source_indices =
      device_column(empty ? std::vector<int32_t>{} : std::vector<int32_t>{0, 0, 1, 1},
                    cudf::type_id::INT32,
                    stream);
  }
  positions->rows = device_column(rows, cudf::type_id::UINT64, stream);
  std::vector<int64_t> a{301, 301}, b{11};
  if (deletion == "all") a = b = {11, 301};
  if (deletion == "no_intersection") a = b = {999};
  auto result = apply_positional_visibility(
    row_payload(rows.size(), stream),
    std::move(positions),
    [&](std::string const& file) -> std::span<int64_t const> { return file == "a" ? a : b; },
    stream,
    cudf::get_current_device_resource_ref());
  auto const expected = empty || deletion == "all" ? std::vector<int32_t>{}
                        : deletion == "no_intersection"
                          ? (multi ? std::vector<int32_t>{0, 1, 2, 3} : std::vector<int32_t>{0, 1})
                          : (multi ? std::vector<int32_t>{0, 3} : std::vector<int32_t>{0});
  CHECK(host_column<int32_t>(result->view().column(0), stream) == expected);
}

TEST_CASE("Reader positions reject foreign rows and invalid source indices",
          "[scan][row_positions]")
{
  rmm::cuda_stream stream;
  auto const defect = std::string(
    GENERATE("hole", "range", "source", "missing_source", "type", "count", "null", "missing"));
  CAPTURE(defect);
  auto positions     = std::make_unique<gpu_row_positions>();
  positions->sources = {{"a", 1000, {{0, 100}, {300, 400}}}};
  uint64_t row       = defect == "hole" ? 200 : defect == "range" ? 1000 : 11;
  positions->rows    = device_column(std::vector<uint64_t>{row}, cudf::type_id::UINT64, stream);
  if (defect == "source")
    positions->source_indices =
      device_column(std::vector<int32_t>{-1}, cudf::type_id::INT32, stream);
  if (defect == "missing_source") positions->sources.push_back({"b", 1000, {{0, 1000}}});
  if (defect == "type")
    positions->rows = device_column(std::vector<int32_t>{11}, cudf::type_id::INT32, stream);
  if (defect == "count")
    positions->rows = device_column(std::vector<uint64_t>{11, 12}, cudf::type_id::UINT64, stream);
  if (defect == "null")
    positions->rows = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT64}, 1, cudf::mask_state::ALL_NULL, stream);
  if (defect == "missing") positions->rows.reset();
  REQUIRE_THROWS_AS(apply_positional_visibility(
                      row_payload(1, stream),
                      std::move(positions),
                      [](std::string const&) -> std::span<int64_t const> { return {}; },
                      stream,
                      cudf::get_current_device_resource_ref()),
                    std::invalid_argument);
  stream.synchronize();
}

TEST_CASE("Position owners retire safely after a failed payload gather",
          "[scan][row_positions][read_retirement]")
{
  using namespace sirius::op::scan::detail;
  rmm::cuda_stream stream;
  auto const fail_join = GENERATE(false, true);
  auto positions       = std::make_unique<gpu_row_positions>();
  positions->sources   = {{"a", 1000, {{0, 1000}}}};
  positions->rows = device_column(std::vector<uint64_t>{11, 301}, cudf::type_id::UINT64, stream);
  read_retirement_test_hooks fault(read_checkpoint::PAYLOAD_GATHER, fail_join);
  auto apply = [&] {
    return apply_positional_visibility(
      row_payload(2, stream),
      std::move(positions),
      [](std::string const&) -> std::span<int64_t const> { return {}; },
      stream,
      cudf::get_current_device_resource_ref());
  };
  if (fail_join)
    REQUIRE_THROWS_AS(apply(), sirius::unjoined_gpu_work);
  else
    REQUIRE_THROWS_AS(apply(), rmm::out_of_memory);
  CHECK(fault.checkpoint_reached);
  CHECK(fault.retained.has_value() == fail_join);
  stream.synchronize();
}

TEST_CASE("Completed filtering retains answer layout until projection",
          "[scan][execution_record][materialization]")
{
  rmm::cuda_stream stream;
  auto const whole = GENERATE(false, true);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(device_column(std::vector<int32_t>{11}, cudf::type_id::INT32, stream));
  cols.push_back(device_column(std::vector<int8_t>{1}, cudf::type_id::BOOL8, stream));
  materialized_batch batch;
  batch.table                   = owning_table_view(std::make_unique<cudf::table>(std::move(cols)));
  batch.contract                = 16;
  batch.generation              = 2;
  batch.rows                    = 1;
  batch.logical_schema_complete = true;
  batch.whole_static_complete   = whole;
  batch.predicates              = {{0, sirius::op::predicate_stage::answer, 1}};
  materialization_program program{16, 2, {cudf::data_type{cudf::type_id::INT32}}, 1, false};
  program.input_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::STRING}};
  int filters         = 0;
  materialization_steps steps;
  steps.residual = [&](materialized_batch& b) {
    ++filters;
    CHECK(b.predicates.front().answer_slot == 1);
  };
  steps.projection = [](materialized_batch& b) {
    CHECK(b.predicates.front().answer_slot == 1);
    std::vector<std::size_t> output{0};
    b.table.select_columns(output);
  };
  auto result = finish_materialization(std::move(batch), program, steps);
  CHECK(filters == (whole ? 0 : 1));
  CHECK(result.whole_static_complete);
  CHECK_FALSE(result.predicates.front().answer_slot);
  CHECK(result.table.n_columns() == 1);
  stream.synchronize();
}

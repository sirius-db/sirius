/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "io/kvikio/kvikio_context.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/iceberg_dv_preparation.hpp"
#include "op/scan/iceberg_gpu_ingestible.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/scan/positional_visibility.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "transparent/read_view_registry.hpp"
#include "transparent/replay_admission.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda/stream>

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/table_filter_test_utils.hpp>
#include <utils/test_validation_utility.hpp>
#include <utils/tracking_kvikio_context.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <numeric>

using namespace sirius::op::scan;
namespace {
class residual_only_dynamic_filter final : public sirius::op::sirius_dynamic_filter {
 public:
  sirius::op::sirius_dynamic_filter_kind kind() const override
  {
    return sirius::op::sirius_dynamic_filter_kind::BLOOM;
  }
};

class unavailable_reader_filter final : public sirius::op::sirius_dynamic_filter,
                                        public sirius::op::sirius_ast_lowerable {
 public:
  sirius::op::sirius_dynamic_filter_kind kind() const override
  {
    return sirius::op::sirius_dynamic_filter_kind::ZONE_MAP;
  }
  bool is_available_on_device(int) const noexcept override { return false; }
  cudf::ast::expression const& to_ast(cudf::ast::tree&,
                                      cudf::ast::expression const&,
                                      int) const override
  {
    throw std::logic_error("unavailable reader filter must not be lowered");
  }
};

template <typename T>
std::unique_ptr<cudf::column> device_column(std::vector<T> const& values,
                                            cudf::type_id type,
                                            ::cuda::stream_ref stream)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{type}, values.size(), cudf::mask_state::UNALLOCATED, stream);
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(col->mutable_view().template data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
    stream.sync();
  }
  return col;
}
template <typename T>
std::vector<T> host_column(cudf::column_view col, ::cuda::stream_ref stream)
{
  std::vector<T> values(col.size());
  if (!values.empty())
    REQUIRE(cudaMemcpyAsync(values.data(),
                            col.data<T>(),
                            values.size() * sizeof(T),
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
  stream.sync();
  return values;
}
std::unique_ptr<cudf::table> row_payload(int count, ::cuda::stream_ref stream)
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

TEST_CASE("Reader positions preserve interval deletion with negative legacy positions",
          "[scan][row_positions]")
{
  rmm::cuda_stream stream;
  auto const base = GENERATE(int64_t{0}, std::numeric_limits<int64_t>::max() - 23);
  auto const deletion =
    std::string(GENERATE("mixed", "duplicates_and_gaps", "negative_only", "outside_only"));
  CAPTURE(base, deletion);
  std::vector<int64_t> deletes{-1, base + 5};
  std::vector<int32_t> expected{0, 2, 3, 4};
  if (deletion == "duplicates_and_gaps") {
    deletes = {
      -4, -1, base, base + 5, base + 5, base + 10, base + 19, base + 20, base + 22, base + 23};
    expected = {2};
  } else if (deletion == "negative_only") {
    deletes  = {-3, -1};
    expected = {0, 1, 2, 3, 4};
  } else if (deletion == "outside_only") {
    deletes  = {-1, base + 10, base + 19, base + 23};
    expected = {0, 1, 2, 3, 4};
  }
  iceberg_delete_set const delete_set(
    "a", std::make_shared<std::vector<int64_t> const>(std::move(deletes)));
  position_lookup lookup = [&](std::string const&) { return delete_set.positions; };
  auto const first       = static_cast<uint64_t>(base);
  std::vector<uint64_t> const rows{first, first + 5, first + 9, first + 20, first + 22};
  auto positions     = std::make_unique<gpu_row_positions>();
  positions->sources = {{"a", first + 23, {{first, first + 10}, {first + 20, first + 23}}}};
  positions->rows    = device_column(rows, cudf::type_id::UINT64, stream);
  auto result        = apply_positional_visibility(row_payload(rows.size(), stream),
                                            std::move(positions),
                                            lookup,
                                            stream,
                                            cudf::get_current_device_resource_ref());
  std::vector<batch_row_run> runs;
  for (std::size_t i = 0; i < rows.size(); ++i)
    runs.push_back({"a", static_cast<int64_t>(rows[i]), static_cast<int64_t>(i), 1});
  auto legacy = apply_positional_visibility(row_payload(rows.size(), stream),
                                            runs,
                                            lookup,
                                            stream,
                                            cudf::get_current_device_resource_ref());
  CHECK(host_column<int32_t>(legacy->view().column(0), stream) == expected);
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

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Iceberg reader positions and execution facts drive remaining work",
                 "[scan][execution_record][row_positions][iceberg][integration]")
{
  auto const multi            = GENERATE(false, true);
  auto const virtual_row      = GENERATE(false, true);
  auto const wide_payload     = GENERATE(false, true);
  auto const legacy_order     = GENERATE(false, true);
  auto const authority        = GENERATE(false, true);
  auto const offset           = GENERATE(0, 4096);
  bool const reader_positions = virtual_row && !legacy_order;
  CAPTURE(multi, virtual_row, wide_payload, legacy_order, authority, offset);
  sirius::test::scratch_dir directory("iceberg_execution_record");
  std::vector<std::string> paths{directory.file("a.parquet")};
  if (multi) paths.push_back(directory.file("b.parquet"));
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    std::string payload_sql;
    if (wide_payload)
      for (int i = 0; i < 16; ++i)
        payload_sql +=
          ", (i+" + std::to_string((i + 1) * 1000000) + ")::BIGINT payload" + std::to_string(i);
    for (auto const& name : multi ? std::vector<std::string>{"a.parquet", "b.parquet"}
                                  : std::vector<std::string>{"a.parquet"}) {
      auto written =
        con->Query("COPY (SELECT i::INTEGER id, CASE WHEN i-" + std::to_string(offset) +
                   " IN (11,301,701) THEN 200 ELSE 0 END amount, CASE WHEN i = " +
                   std::to_string(offset + 701) + " THEN NULL ELSE 1 END region" + payload_sql +
                   " FROM range(" + std::to_string(offset + 1000) + ") t(i) ORDER BY i) TO " +
                   directory.file_literal(name) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
      REQUIRE(written);
      REQUIRE_FALSE(written->HasError());
    }
  }
  auto info                 = std::make_unique<iceberg_ingestible_table_info>();
  info->resolved_file_paths = paths;
  info->table_path          = "execution-record-table";
  info->names               = {"id", "amount", "region"};
  info->returned_types.assign(3, sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids     = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2)};
  info->projection_ids = {0, 1, 2};
  info->projected_output_count = 1;
  info->approximate_batch_size = std::size_t{1} << 30;
  if (wide_payload) {
    info->projection_ids = {0};
    for (int i = 0; i < 16; ++i) {
      info->names.push_back("payload" + std::to_string(i));
      info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
      info->column_ids.emplace_back(3 + i);
      info->projection_ids.push_back(3 + i);
    }
    info->projection_ids.push_back(1);
    info->projection_ids.push_back(2);
    info->projected_output_count = 17;
  }
  if (virtual_row) {
    auto const id = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
    info->column_ids.emplace_back(id);
    info->projection_ids.insert(info->projection_ids.begin() + 1, wide_payload ? 19 : 3);
    ++info->projected_output_count;
    info->virtual_columns = {{id,
                              "file_row_number",
                              sirius::logical_type::make(sirius::type_id::BIGINT),
                              scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER}};
  }
  info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(1),
    sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                  duckdb::Value::INTEGER(100)));
  info->table_filters->PushFilter(duckdb::ProjectionIndex(2),
                                  sirius::test::null_filter(duckdb::LogicalType::INTEGER, true));
  info->delete_sets.emplace();
  for (std::size_t i = 0; i < paths.size(); ++i) {
    auto deleted =
      std::make_shared<std::vector<int64_t> const>(std::vector<int64_t>{offset + (i ? 11 : 301)});
    info->delete_sets->emplace(paths[i],
                               std::make_shared<iceberg_delete_set const>(paths[i], deleted));
  }
  auto counters            = std::make_shared<physical_check_counters>();
  counters->track_units    = true;
  info->profiles->counters = counters;
  std::vector<std::pair<bool, bool>> prepend;
  counters->parquet_positions_for_testing = [&](bool row, bool source) {
    prepend.emplace_back(row, source);
  };
  int decisions                        = 0;
  counters->scan_execution_for_testing = [&](bool used_producer,
                                             scan_execution_record const& facts) {
    ++decisions;
    CHECK(used_producer == authority);
    CHECK_FALSE(facts.whole_static_complete);
    if (authority) CHECK(facts.visibility_complete);
    CHECK(facts.predicates.size() == (reader_positions && authority ? 1 : 0));
    if (reader_positions && authority)
      CHECK(facts.predicates.front().stage == sirius::op::predicate_stage::enforced);
  };
  auto ingestible = make_ingestible(std::move(info));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto coalescer  = ingestible->create_batch_coalescer();
  std::vector<std::unique_ptr<scan_info>> splits;
  while (auto prepare = ingestible->next_split_provider(
           [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; })) {
    auto prepared = prepare({});
    auto& file    = dynamic_cast<parquet_file_scan_info&>(*prepared);
    CHECK(file.requires_row_positions == virtual_row);
    CHECK(file.disable_filter_pushdown == !virtual_row);
    if (legacy_order) {
      file.requires_row_positions  = false;
      file.disable_filter_pushdown = true;
    }
    auto ready = coalescer->push(std::move(prepared));
    for (auto& split : ready)
      splits.push_back(std::move(split));
  }
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  CHECK(dynamic_cast<parquet_split_info const&>(*splits.front()).requires_row_positions ==
        reader_positions);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto batch =
    ingestible->materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
  REQUIRE(batch.execution);
  CHECK(batch.execution->authoritative);
  CHECK(batch.execution->visibility_applied.size() == paths.size());
  REQUIRE(batch.required_visibility);
  CHECK(batch.required_visibility->size() == paths.size());
  batch.execution->authoritative = authority;
  CHECK_FALSE(batch.row_positions);  // Hidden indices are released at the last consumer.
  CHECK(batch.table.num_rows() == (reader_positions ? 2 : 999) * paths.size());
  if (reader_positions) {
    REQUIRE(prepend.size() == 1);
    CHECK((prepend.front() == std::pair<bool, bool>{true, multi}));
    // The completed, filter-only amount column has no remaining consumer.
    // Poison it: re-evaluating that condition would now discard every survivor.
    if (authority) {
      auto owned = batch.table.release(stream, space->get_default_allocator());
      REQUIRE(cudaMemsetAsync(owned->mutable_view().column(wide_payload ? 17 : 1).data<int32_t>(),
                              0,
                              owned->num_rows() * sizeof(int32_t),
                              stream.value()) == cudaSuccess);
      batch.table = owning_table_view(std::move(owned));
    }
  } else if (virtual_row) {
    REQUIRE(prepend.size() == 1);
    CHECK((prepend.front() == std::pair<bool, bool>{true, false}));
  } else {
    CHECK(prepend.empty());
  }
  auto output = ingestible->post_filter_and_project(std::move(batch), *space, stream);
  CHECK(decisions == 1);
  CHECK(counters->execution_record_shadow_mismatches.load() == 0);
  CHECK(counters->execution_record_shadow_coverage_added.load() ==
        (!authority && reader_positions ? 1 : 0));
  REQUIRE(output->num_columns() == 1 + static_cast<int>(virtual_row) + (wide_payload ? 16 : 0));
  auto ids = host_column<int32_t>(output->view().column(0), stream);
  CHECK(ids == (multi ? std::vector<int32_t>{offset + 11, offset + 301}
                      : std::vector<int32_t>{offset + 11}));
  if (virtual_row) {
    auto rows = host_column<int64_t>(output->view().column(1), stream);
    CHECK(rows == std::vector<int64_t>(ids.begin(), ids.end()));
  }
  if (wide_payload) {
    for (int i = 0; i < 16; ++i) {
      std::vector<int64_t> expected;
      for (auto id : ids)
        expected.push_back(id + (i + 1) * 1000000);
      CHECK(host_column<int64_t>(output->view().column(1 + static_cast<int>(virtual_row) + i),
                                 stream) == expected);
    }
  }
  if (!multi && !virtual_row && !legacy_order && authority && offset == 0) {
    for (int fault : {0, 1, 2, 3}) {
      auto invalid =
        ingestible->materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
      if (fault == 0) invalid.execution->visibility_applied.clear();
      if (fault == 1) invalid.execution->visibility_applied.front().file = "foreign-file";
      if (fault == 2)
        invalid.execution->visibility_applied.front().input = std::make_shared<int const>(1);
      if (fault == 3) invalid.required_visibility.reset();
      REQUIRE_THROWS_AS(ingestible->post_filter_and_project(std::move(invalid), *space, stream),
                        std::invalid_argument);
    }
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Position requests follow actual consumers",
                 "[scan][row_positions][execution_record][integration]")
{
  auto const mode           = std::string(GENERATE("parquet",
                                         "iceberg_empty",
                                         "iceberg_unfiltered",
                                         "iceberg_dynamic",
                                         "iceberg_dynamic_pending",
                                         "iceberg_dynamic_late",
                                         "iceberg_dynamic_before_emission",
                                         "iceberg_dynamic_unlowerable",
                                         "iceberg_dynamic_unavailable",
                                         "iceberg_dynamic_wrong_column",
                                         "iceberg_dynamic_static_pending",
                                         "iceberg_cast",
                                         "iceberg_partial",
                                         "iceberg_all",
                                         "iceberg_unselected"));
  bool const dynamic        = mode.starts_with("iceberg_dynamic");
  bool const static_pending = mode == "iceberg_dynamic_static_pending";
  auto const virtual_kind   = GENERATE(0, 1, 2);  // none, row number, filename
  bool const reader_positions =
    virtual_kind != 0 &&
    (dynamic || mode == "iceberg_partial" || mode == "iceberg_all" || mode == "iceberg_unselected");
  bool const actual_positions =
    reader_positions && (!dynamic || mode == "iceberg_dynamic" ||
                         mode == "iceberg_dynamic_before_emission" || static_pending);
  CAPTURE(mode, virtual_kind);
  sirius::test::scratch_dir directory("position_demand");
  auto const path = directory.file("input.parquet");
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    std::string payload;
    for (int i = 0; i < 32; ++i)
      payload += ", i::BIGINT payload" + std::to_string(i);
    auto written =
      con->Query("COPY (SELECT i::INTEGER id" + payload + " FROM range(1000) t(i)) TO " +
                 directory.file_literal("input.parquet") + " (FORMAT PARQUET)");
    REQUIRE(written);
    REQUIRE_FALSE(written->HasError());
  }
  std::unique_ptr<parquet_ingestible_table_info> info;
  if (mode == "parquet")
    info = std::make_unique<parquet_ingestible_table_info>();
  else {
    auto iceberg        = std::make_unique<iceberg_ingestible_table_info>();
    iceberg->table_path = "position-demand";
    iceberg->delete_sets.emplace();
    std::vector<int64_t> positions;
    if (mode == "iceberg_unfiltered" || dynamic || mode == "iceberg_cast" ||
        mode == "iceberg_partial")
      positions = {301};
    if (mode == "iceberg_unselected") positions = {11};
    if (mode == "iceberg_all") {
      positions.resize(1000);
      std::iota(positions.begin(), positions.end(), 0);
    }
    auto deleted = std::make_shared<std::vector<int64_t> const>(std::move(positions));
    iceberg->delete_sets->emplace(path, std::make_shared<iceberg_delete_set const>(path, deleted));
    info = std::move(iceberg);
  }
  info->resolved_file_paths    = {path};
  info->names                  = {"id"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info->column_ids             = {duckdb::ColumnIndex(0)};
  info->projected_output_count = virtual_kind ? 34 : 33;
  info->approximate_batch_size = std::size_t{1} << 30;
  for (int i = 0; i < 32; ++i) {
    info->names.push_back("payload" + std::to_string(i));
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    info->column_ids.emplace_back(1 + i);
  }
  if (virtual_kind) {
    auto const id = virtual_kind == 1 ? duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER
                                      : duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME;
    info->column_ids.emplace_back(id);
    info->virtual_columns = {
      {id,
       virtual_kind == 1 ? "file_row_number" : "filename",
       sirius::logical_type::make(virtual_kind == 1 ? sirius::type_id::BIGINT
                                                    : sirius::type_id::VARCHAR),
       virtual_kind == 1 ? scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER
                         : scan_plan::parquet_virtual_column_kind::FILENAME}};
  }
  std::optional<sirius::op::sirius_dynamic_filter_set::producer> dynamic_producer;
  auto publish_dynamic = [&] {
    using namespace sirius::op;
    if (mode == "iceberg_dynamic_unlowerable" || mode == "iceberg_dynamic_unavailable") {
      std::shared_ptr<sirius_dynamic_filter const> filter =
        mode == "iceberg_dynamic_unlowerable" ? std::shared_ptr<sirius_dynamic_filter const>(
                                                  std::make_shared<residual_only_dynamic_filter>())
                                              : std::shared_ptr<sirius_dynamic_filter const>(
                                                  std::make_shared<unavailable_reader_filter>());
      REQUIRE(dynamic_producer->push_filter(0, std::move(filter)));
      dynamic_producer->finish(sirius_dynamic_filter_set::completion::PUBLISHED);
      return;
    }
    std::vector<zone_map_entry> zones;
    zones.push_back({std::make_unique<cudf::numeric_scalar<int32_t>>(100, true),
                     std::make_unique<cudf::numeric_scalar<int32_t>>(399, true)});
    REQUIRE(dynamic_producer->push_filter(
      mode == "iceberg_dynamic_wrong_column" ? 33 : 0,
      std::make_shared<sirius_dynamic_zone_map_filter>(std::move(zones))));
    dynamic_producer->finish(sirius_dynamic_filter_set::completion::PUBLISHED);
  };
  if (dynamic) {
    using namespace sirius::op;
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(duckdb::ProjectionIndex(0),
                                    sirius::test::null_filter(duckdb::LogicalType::INTEGER, true));
    info->sirius_dynamic_filters = std::make_shared<sirius_dynamic_filter_set>();
    dynamic_producer.emplace(info->sirius_dynamic_filters->register_producer({0, 33}));
    if (mode == "iceberg_dynamic" || mode == "iceberg_dynamic_unlowerable" ||
        mode == "iceberg_dynamic_unavailable" || mode == "iceberg_dynamic_wrong_column")
      publish_dynamic();
    if (static_pending)
      info->table_filters->PushFilter(
        duckdb::ProjectionIndex(0),
        sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                      duckdb::Value::INTEGER(100)));
  } else if (mode != "iceberg_unfiltered") {
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(0),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                    duckdb::Value::INTEGER(100)));
    if (mode == "iceberg_cast") {
      auto cast = duckdb::BoundCastExpression::AddDefaultCastToType(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
        duckdb::LogicalType::SMALLINT);
      info->table_filters->PushFilter(
        duckdb::ProjectionIndex(0),
        // The NULL test keeps this cast in the residual. Even though the reader
        // subset is a safe comparison, early selection could hide residual errors.
        duckdb::make_uniq<duckdb::ExpressionFilter>(
          duckdb::ExpressionFilter::CreateNullCheckExpression(
            std::move(cast), duckdb::ExpressionType::OPERATOR_IS_NOT_NULL)));
    }
  }
  auto counters            = std::make_shared<physical_check_counters>();
  counters->track_units    = true;
  info->profiles->counters = counters;
  int row_requests = 0, source_requests = 0;
  counters->parquet_positions_for_testing = [&](bool row, bool source) {
    row_requests += row;
    source_requests += source;
  };
  std::shared_ptr<gpu_ingestible> ingestible;
  if (mode == "parquet")
    ingestible = make_ingestible(std::move(info));
  else
    ingestible = make_ingestible(std::unique_ptr<iceberg_ingestible_table_info>(
      static_cast<iceberg_ingestible_table_info*>(info.release())));
  auto ioctx   = std::make_shared<sirius::io::kvikio_context>();
  auto prepare = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  auto prepared                = prepare({});
  auto const preparation_bytes = prepared->preparation_bytes();
  if (!virtual_kind && mode != "parquet") {
    auto const& file = dynamic_cast<parquet_file_scan_info const&>(*prepared);
    CHECK_FALSE(file.position_filter_eligible);
    CHECK_FALSE(file.static_reader_filter_ready);
    for (auto const& group : file.row_groups)
      CHECK_FALSE(group.unfiltered_working_bytes.has_value());
  }
  std::size_t unfiltered_bytes = 0;
  for (auto const& group : dynamic_cast<parquet_file_scan_info const&>(*prepared).row_groups)
    unfiltered_bytes += group.unfiltered_working_bytes.value_or(group.decode_working_bytes);
  CHECK(dynamic_cast<parquet_file_scan_info&>(*prepared).requires_row_positions ==
        (reader_positions && virtual_kind != 0));
  if (mode == "iceberg_cast")
    CHECK(dynamic_cast<parquet_file_scan_info&>(*prepared).disable_filter_pushdown);
  auto coalescer = ingestible->create_batch_coalescer();
  auto splits    = coalescer->push(std::move(prepared));
  if (mode == "iceberg_dynamic_before_emission") publish_dynamic();
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  auto owned        = std::shared_ptr<scan_info>(std::move(splits.front()));
  auto const& split = dynamic_cast<parquet_split_info const&>(*owned);
  if (!virtual_kind) CHECK_FALSE(split.position_filter_snapshot);
  scan_operator_input input(
    owned, nullptr, 0, std::nullopt, scan_operator_input::prefetch_start::deferred);
  auto const device        = rmm::get_current_cuda_device().value();
  auto const working_bytes = input.get_estimated_working_set_size_in_bytes(device);
  if (mode == "iceberg_dynamic_late" || static_pending) publish_dynamic();
  CHECK(input.get_estimated_working_set_size_in_bytes(device) == working_bytes);
  CHECK(split.preparation_bytes() == preparation_bytes);
  if (dynamic) {
    CHECK(split.position_reader_filter_available(device) == actual_positions);
    CHECK(split.reader_filter_disabled(device) == !actual_positions);
    if (!actual_positions)
      CHECK(working_bytes == unfiltered_bytes);
    else
      CHECK(working_bytes > unfiltered_bytes);
  }
  CHECK(split.requires_row_positions == reader_positions);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto batch = ingestible->materialize_metadata_to_table(*owned, *space, stream, false, nullptr);
  CHECK_FALSE(batch.row_positions);
  CHECK(row_requests == (actual_positions || virtual_kind == 1 ? 1 : 0));
  CHECK(source_requests == 0);  // One file's context is sufficient, even for filename output.
  if (mode == "parquet" && !virtual_kind) {
    REQUIRE(batch.execution);
    batch.execution->authoritative = true;
    auto width                     = batch.execution->layout_width;
    batch.execution->layout_width  = batch.table.n_columns() + 1;
    REQUIRE_THROWS_AS(ingestible->view_layout(batch, {}), std::invalid_argument);
    batch.execution->layout_width = width;
    auto layout                   = batch.execution->layout;
    batch.execution->layout       = std::make_shared<int const>(0);
    REQUIRE_THROWS_AS(ingestible->view_layout(batch, {}), std::invalid_argument);
    batch.execution->layout = std::move(layout);
  }
  if (dynamic) {
    REQUIRE(batch.execution);
    CHECK_FALSE(batch.execution->whole_static_complete);
    CHECK(batch.execution->predicates.size() == (static_pending && actual_positions ? 1 : 0));
    CHECK(batch.execution->visibility_complete);
    batch.execution->authoritative = true;
    CHECK(batch.table.num_rows() == (actual_positions ? (static_pending ? 898 : 299) : 999));
  }
  auto output = ingestible->post_filter_and_project(std::move(batch), *space, stream);
  CHECK(output->num_rows() == (mode == "iceberg_all"          ? 0
                               : mode == "iceberg_partial"    ? 898
                               : dynamic                      ? (static_pending     ? 898
                                                                 : actual_positions ? 299
                                                                                    : 999)
                               : mode == "iceberg_unfiltered" ? 999
                               : mode == "iceberg_cast"       ? 898
                                                              : 899));
  CHECK(output->num_columns() == (virtual_kind ? 34 : 33));
  stream.synchronize();
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Unfiltered Iceberg virtual rows use one reader across row groups",
                 "[scan][execution_record][row_positions][iceberg][integration]")
{
  bool const multi        = GENERATE(false, true);
  auto const delete_files = GENERATE(0, 1, 2, 3);
  CAPTURE(multi, delete_files);
  sirius::test::scratch_dir directory("iceberg_virtual_reader");
  auto info        = std::make_unique<iceberg_ingestible_table_info>();
  info->table_path = "virtual-reader";
  info->delete_sets.emplace();
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    for (int file = 0; file < (multi ? 2 : 1); ++file) {
      auto name = std::to_string(file) + ".parquet";
      auto written =
        con->Query("COPY (SELECT i::INTEGER id FROM range(6144) t(i)) TO " +
                   directory.file_literal(name) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
      REQUIRE(written);
      REQUIRE_FALSE(written->HasError());
      auto path = directory.file(name);
      info->resolved_file_paths.push_back(path);
      auto positions = std::make_shared<std::vector<int64_t> const>(
        delete_files & (1 << file) ? std::vector<int64_t>{11, 4097} : std::vector<int64_t>{});
      info->delete_sets->emplace(path, std::make_shared<iceberg_delete_set const>(path, positions));
    }
  }
  auto const row_id            = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  info->names                  = {"id"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(row_id)};
  info->virtual_columns        = {{row_id,
                                   "file_row_number",
                                   sirius::logical_type::make(sirius::type_id::BIGINT),
                                   scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER}};
  info->projected_output_count = 2;
  info->approximate_batch_size = std::size_t{1} << 30;
  auto counters                = std::make_shared<physical_check_counters>();
  counters->track_units        = true;
  info->profiles->counters     = counters;
  int requests = 0, decoded_sources = 0;
  counters->parquet_positions_for_testing = [&](bool row, bool source) {
    ++requests;
    CHECK(row);
    CHECK_FALSE(source);
  };
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    decoded_sources += !footer;
  };
  auto reader    = make_ingestible(std::move(info));
  auto ioctx     = std::make_shared<sirius::io::kvikio_context>();
  auto coalescer = reader->create_batch_coalescer();
  std::vector<std::unique_ptr<scan_info>> splits;
  std::size_t groups = 0;
  while (auto prepare = reader->next_split_provider(
           [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; })) {
    auto metadata = prepare({});
    groups += dynamic_cast<parquet_file_scan_info const&>(*metadata).row_groups.size();
    for (auto& split : coalescer->push(std::move(metadata)))
      splits.push_back(std::move(split));
  }
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(groups == (multi ? 6 : 3));
  REQUIRE(splits.size() == 1);
  CHECK_FALSE(dynamic_cast<parquet_split_info const&>(*splits.front()).requires_row_positions);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto batch =
    reader->materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
  CHECK_FALSE(batch.row_positions);
  auto output = reader->post_filter_and_project(std::move(batch), *space, stream);
  CHECK(requests == 1);
  CHECK(decoded_sources == (multi ? 2 : 1));
  auto ids  = host_column<int32_t>(output->view().column(0), stream);
  auto rows = host_column<int64_t>(output->view().column(1), stream);
  std::vector<int64_t> expected;
  for (int file = 0; file < (multi ? 2 : 1); ++file)
    for (int64_t row = 0; row < 6144; ++row)
      if (!(delete_files & (1 << file)) || (row != 11 && row != 4097)) expected.push_back(row);
  CHECK(rows == expected);
  CHECK(std::equal(ids.begin(), ids.end(), expected.begin(), expected.end()));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Iceberg split ranges restrict positions from eager legacy deletes",
                 "[scan][execution_record][row_positions][iceberg][eager_delete][integration]")
{
  auto const virtual_kind = GENERATE(0, 1, 2);  // none, row number, filename
  auto const grouping     = std::string(GENERATE("merged", "split", "footer_pruned"));
  bool const separate     = grouping == "split";
  bool const pruned       = grouping == "footer_pruned";
  CAPTURE(virtual_kind, grouping);
  sirius::test::scratch_dir directory("split_delete_ranges");
  auto const path = directory.file("input.parquet");
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    auto data = con->Query(
      "COPY (SELECT i::INTEGER id, (i % 2)::INTEGER amount FROM range(4096) t(i)) TO " +
      directory.file_literal("input.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    REQUIRE(data);
    REQUIRE_FALSE(data->HasError());
    auto deletes =
      con->Query("COPY (SELECT " + directory.file_literal("input.parquet") +
                 " file_path, pos FROM (VALUES (11::BIGINT), (-1::BIGINT), (11::BIGINT), "
                 "(-3::BIGINT), (-1::BIGINT)) t(pos)) TO " +
                 directory.file_literal("deletes.parquet") + " (FORMAT PARQUET)");
    REQUIRE(deletes);
    REQUIRE_FALSE(deletes->HasError());
  }
  auto ioctx = std::make_shared<sirius::io::kvikio_context>();
  iceberg_delete_discovery discovery;
  discovery.positional_delete_files.push_back(directory.file("deletes.parquet"));
  // Exercise the real V2 eager loader with unsorted, duplicate negative and valid
  // positions before either visibility implementation sees the sorted list.
  auto payload = load_delete_payload(*con->context, "split-ranges", ioctx.get(), {}, discovery);
  REQUIRE(payload);
  REQUIRE(payload->positional_deletes.at(path) == std::vector<int64_t>{-3, -1, -1, 11, 11});
  auto info                 = std::make_unique<iceberg_ingestible_table_info>();
  info->table_path          = "split-ranges";
  info->delete_data         = payload;
  info->resolved_file_paths = {path};
  info->names               = {"id", "amount"};
  info->returned_types.assign(2, sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projected_output_count = virtual_kind ? 3 : 2;
  info->approximate_batch_size = separate ? 1 : std::size_t{1} << 30;
  if (virtual_kind) {
    auto const id = virtual_kind == 1 ? duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER
                                      : duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME;
    info->column_ids.emplace_back(id);
    info->virtual_columns = {
      {id,
       virtual_kind == 1 ? "file_row_number" : "filename",
       sirius::logical_type::make(virtual_kind == 1 ? sirius::type_id::BIGINT
                                                    : sirius::type_id::VARCHAR),
       virtual_kind == 1 ? scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER
                         : scan_plan::parquet_virtual_column_kind::FILENAME}};
  }
  info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(pruned ? 0 : 1),
    sirius::test::constant_filter(pruned ? duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO
                                         : duckdb::ExpressionType::COMPARE_EQUAL,
                                  duckdb::Value::INTEGER(pruned ? 3000 : 1)));
  auto counters            = std::make_shared<physical_check_counters>();
  counters->track_units    = true;
  info->profiles->counters = counters;
  std::vector<bool> row_requests;
  counters->parquet_positions_for_testing = [&](bool row, bool source) {
    row_requests.push_back(row);
    CHECK_FALSE(source);
  };
  int reader_calls                     = 0;
  counters->parquet_filter_for_testing = [&](bool filtered, uint64_t rows) {
    ++reader_calls;
    if (pruned) {
      CHECK(filtered == bool(virtual_kind));
      CHECK(rows == (virtual_kind ? 1096 : 2048));
    }
  };
  auto reader  = make_ingestible(std::move(info));
  auto prepare = reader->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  auto metadata    = prepare({});
  auto const& file = dynamic_cast<parquet_file_scan_info const&>(*metadata);
  REQUIRE(file.row_groups.size() == (pruned ? 1 : 2));
  CHECK(file.row_groups.front().index == (pruned ? 1 : 0));
  CHECK(file.row_groups.front().requires_row_positions == (virtual_kind && !pruned));
  CHECK_FALSE(file.row_groups.back().requires_row_positions);
  CHECK(file.disable_filter_pushdown == !virtual_kind);
  auto const preparation_bytes = file.preparation_bytes();
  auto const last_base         = file.row_groups.back().unfiltered_working_bytes.value_or(
    file.row_groups.back().decode_working_bytes);
  auto coalescer = reader->create_batch_coalescer();
  auto splits    = coalescer->push(std::move(metadata));
  CHECK(coalescer->retained_bytes() == (separate ? last_base : preparation_bytes));
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == (separate ? 2 : 1));
  std::size_t consumed_preparation = 0;
  auto context                     = sirius::test::get_registered_sirius_context(*con);
  auto* space                      = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  for (std::size_t i = 0; i < splits.size(); ++i) {
    auto& split                = dynamic_cast<parquet_split_info&>(*splits[i]);
    bool const needs_positions = virtual_kind && !pruned && i == 0;
    bool const no_deletes      = pruned || i == 1;
    CHECK(split.requires_row_positions == needs_positions);
    consumed_preparation += split.preparation_bytes();
    auto const execution_bytes =
      split.estimated_working_set_bytes_for_device(space->get_device_id());
    CHECK(execution_bytes == (virtual_kind && no_deletes ? last_base : split.preparation_bytes()));
    if (virtual_kind == 2 && i == 1) CHECK(execution_bytes < split.preparation_bytes());
    if (needs_positions) {
      split.requires_row_positions = false;
      REQUIRE_THROWS_AS(
        reader->materialize_metadata_to_table(split, *space, stream, false, nullptr),
        sirius::internal_exception);
      split.requires_row_positions = true;
    }
    // A no-delete virtual split must not even reach the delete gather. Requested
    // row-number columns remain independent of this internal position demand.
    std::optional<detail::read_retirement_test_hooks> fault;
    if (virtual_kind && no_deletes) fault.emplace(detail::read_checkpoint::PAYLOAD_GATHER);
    auto batch = reader->materialize_metadata_to_table(split, *space, stream, false, nullptr);
    if (fault) CHECK_FALSE(fault->checkpoint_reached);
    fault.reset();
    REQUIRE(batch.execution);
    CHECK(batch.execution->visibility_complete);
    REQUIRE(batch.required_visibility);
    REQUIRE(batch.required_visibility->size() == 1);
    REQUIRE(batch.execution->visibility_applied.size() == 1);
    CHECK(batch.execution->visibility_applied.front().input ==
          batch.required_visibility->front().input);
    CHECK(batch.execution->visibility_applied.front().file == path);
    CHECK_FALSE(batch.row_positions);
    if (virtual_kind && no_deletes) CHECK(batch.execution->positions == row_position_state::none);
    auto output = reader->post_filter_and_project(std::move(batch), *space, stream);
    std::vector<int32_t> expected;
    for (int row = pruned     ? 3000
                   : separate ? static_cast<int>(i) * 2048
                              : 0;
         row < (separate ? static_cast<int>(i + 1) * 2048 : 4096);
         ++row)
      if (pruned || (row % 2 && row != 11)) expected.push_back(row);
    CHECK(host_column<int32_t>(output->view().column(0), stream) == expected);
    if (virtual_kind == 1)
      CHECK(host_column<int64_t>(output->view().column(2), stream) ==
            std::vector<int64_t>(expected.begin(), expected.end()));
  }
  CHECK(reader_calls == splits.size());
  CHECK(consumed_preparation == preparation_bytes);
  if (!virtual_kind)
    CHECK(row_requests.empty());
  else
    CHECK(row_requests == (pruned     ? std::vector<bool>{virtual_kind == 1}
                           : separate ? std::vector<bool>{true, virtual_kind == 1}
                                      : std::vector<bool>{true}));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Iceberg eager duplicate deletes match unique deletes across merged files",
                 "[scan][execution_record][row_positions][iceberg][eager_delete][integration]")
{
  auto const virtual_kind = GENERATE(0, 1, 2);  // none, row number, filename
  auto const multi        = GENERATE(false, true);
  CAPTURE(virtual_kind, multi);
  sirius::test::scratch_dir directory("eager_duplicate_deletes");
  std::vector<std::string> const names = multi ? std::vector<std::string>{"a.parquet", "b.parquet"}
                                               : std::vector<std::string>{"a.parquet"};
  std::vector<std::string> paths;
  std::vector<int32_t> expected_ids;
  std::vector<int64_t> expected_rows;
  for (std::size_t file = 0; file < names.size(); ++file) {
    paths.push_back(directory.file(names[file]));
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    auto written =
      con->Query("COPY (SELECT (i + " + std::to_string(file * 1000) +
                 ")::INTEGER id, (i % 2)::INTEGER amount FROM range(64) t(i) ORDER BY i) TO " +
                 directory.file_literal(names[file]) + " (FORMAT PARQUET)");
    REQUIRE(written);
    REQUIRE_FALSE(written->HasError());
    // Both files delete row 11, but row 5 belongs only to A's delete set and row
    // 17 only to B's. Distinct payload IDs expose swapped or combined bindings.
    for (int row = 1; row < 64; row += 2) {
      if (row == 11 || row == (file == 0 ? 5 : 17)) continue;
      expected_ids.push_back(static_cast<int32_t>(file * 1000 + row));
      expected_rows.push_back(row);
    }
  }
  auto ioctx   = std::make_shared<sirius::io::kvikio_context>();
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  std::unique_ptr<cudf::table> unique_output;
  // Keep every position positive so this comparison tests duplicate handling
  // independently of the signed-bound regression above. Both inputs go through
  // real delete files and the eager loader; the test does not sort or deduplicate.
  for (bool duplicates : {false, true}) {
    CAPTURE(duplicates);
    auto const delete_name = duplicates ? "duplicates.parquet" : "unique.parquet";
    std::string values;
    for (std::size_t file = 0; file < paths.size(); ++file) {
      auto const positions =
        file == 0
          ? (duplicates ? std::vector<int64_t>{11, 5, 11, 5} : std::vector<int64_t>{5, 11})
          : (duplicates ? std::vector<int64_t>{17, 11, 17, 11} : std::vector<int64_t>{11, 17});
      for (auto pos : positions) {
        if (!values.empty()) values += ", ";
        values +=
          "(" + sirius::test::sql_literal(paths[file]) + ", " + std::to_string(pos) + "::BIGINT)";
      }
    }
    {
      sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
      auto written =
        con->Query("COPY (SELECT * FROM (VALUES " + values + ") t(file_path, pos)) TO " +
                   directory.file_literal(delete_name) + " (FORMAT PARQUET)");
      REQUIRE(written);
      REQUIRE_FALSE(written->HasError());
    }
    iceberg_delete_discovery discovery;
    discovery.positional_delete_files.push_back(directory.file(delete_name));
    auto payload = load_delete_payload(*con->context, delete_name, ioctx.get(), {}, discovery);
    REQUIRE(payload);
    REQUIRE(payload->positional_deletes.size() == paths.size());
    for (std::size_t file = 0; file < paths.size(); ++file) {
      CHECK(
        payload->positional_deletes.at(paths[file]) ==
        (file == 0
           ? (duplicates ? std::vector<int64_t>{5, 5, 11, 11} : std::vector<int64_t>{5, 11})
           : (duplicates ? std::vector<int64_t>{11, 11, 17, 17} : std::vector<int64_t>{11, 17})));
    }
    auto info                 = std::make_unique<iceberg_ingestible_table_info>();
    info->table_path          = delete_name;
    info->delete_data         = payload;
    info->resolved_file_paths = paths;
    info->names               = {"id", "amount"};
    info->returned_types.assign(2, sirius::logical_type::make(sirius::type_id::INTEGER));
    info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
    info->projected_output_count = virtual_kind ? 3 : 2;
    info->approximate_batch_size = std::size_t{1} << 30;
    if (virtual_kind) {
      auto const id = virtual_kind == 1 ? duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER
                                        : duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME;
      info->column_ids.emplace_back(id);
      info->virtual_columns = {
        {id,
         virtual_kind == 1 ? "file_row_number" : "filename",
         sirius::logical_type::make(virtual_kind == 1 ? sirius::type_id::BIGINT
                                                      : sirius::type_id::VARCHAR),
         virtual_kind == 1 ? scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER
                           : scan_plan::parquet_virtual_column_kind::FILENAME}};
    }
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(1),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL,
                                    duckdb::Value::INTEGER(1)));
    auto counters                           = std::make_shared<physical_check_counters>();
    counters->track_units                   = true;
    info->profiles->counters                = counters;
    int position_calls                      = 0;
    counters->parquet_positions_for_testing = [&](bool row, bool source) {
      ++position_calls;
      CHECK(row);
      CHECK(source == multi);
    };
    int reader_calls                     = 0;
    counters->parquet_filter_for_testing = [&](bool filtered, uint64_t rows) {
      ++reader_calls;
      CHECK(filtered == bool(virtual_kind));
      CHECK(rows == (virtual_kind ? 32 : 64) * paths.size());
    };
    auto reader    = make_ingestible(std::move(info));
    auto coalescer = reader->create_batch_coalescer();
    std::vector<std::unique_ptr<scan_info>> splits;
    while (auto prepare = reader->next_split_provider(
             [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; })) {
      auto ready = coalescer->push(prepare({}));
      for (auto& split : ready)
        splits.push_back(std::move(split));
    }
    for (auto& split : coalescer->flush())
      splits.push_back(std::move(split));
    REQUIRE(splits.size() == 1);
    auto const& split = dynamic_cast<parquet_split_info const&>(*splits.front());
    REQUIRE(split.rg_slices.size() == paths.size());
    CHECK(split.requires_row_positions == bool(virtual_kind));
    for (std::size_t file = 0; file < paths.size(); ++file)
      CHECK(split.rg_slices[file].file_path == paths[file]);
    auto batch = reader->materialize_metadata_to_table(split, *space, stream, false, nullptr);
    CHECK(reader_calls == 1);
    CHECK(position_calls == (virtual_kind ? 1 : 0));
    REQUIRE(batch.execution);
    CHECK(batch.execution->visibility_complete);
    REQUIRE(batch.required_visibility);
    REQUIRE(batch.required_visibility->size() == paths.size());
    REQUIRE(batch.execution->visibility_applied.size() == paths.size());
    for (auto const& path : paths) {
      auto const matches   = [&](auto const& binding) { return binding.file == path; };
      auto const& applied  = batch.execution->visibility_applied;
      auto const& required = *batch.required_visibility;
      auto actual          = std::find_if(applied.begin(), applied.end(), matches);
      auto expected        = std::find_if(required.begin(), required.end(), matches);
      REQUIRE(actual != applied.end());
      REQUIRE(expected != required.end());
      CHECK(actual->input == expected->input);
    }
    CHECK_FALSE(batch.row_positions);
    CHECK(batch.table.num_rows() == (virtual_kind ? 30 : 62) * paths.size());
    auto output = reader->post_filter_and_project(std::move(batch), *space, stream);
    REQUIRE(output->num_columns() == (virtual_kind ? 3 : 2));
    CHECK(host_column<int32_t>(output->view().column(0), stream) == expected_ids);
    CHECK(host_column<int32_t>(output->view().column(1), stream) ==
          std::vector<int32_t>(expected_ids.size(), 1));
    if (virtual_kind == 1)
      CHECK(host_column<int64_t>(output->view().column(2), stream) == expected_rows);
    if (virtual_kind == 2) {
      for (std::size_t file = 0; file < paths.size(); ++file) {
        auto expected =
          cudf::make_column_from_scalar(cudf::string_scalar(paths[file], true, stream), 30, stream);
        std::vector<cudf::size_type> const range{static_cast<cudf::size_type>(file * 30),
                                                 static_cast<cudf::size_type>((file + 1) * 30)};
        CHECK(sirius::test::columns_logically_equal(
          cudf::slice(output->view().column(2), range, stream).front(), expected->view(), stream));
      }
    }
    if (duplicates) {
      REQUIRE(unique_output);
      for (int column = 0; column < output->num_columns(); ++column)
        CHECK(sirius::test::columns_logically_equal(
          output->view().column(column), unique_output->view().column(column), stream));
    } else {
      unique_output = std::move(output);
    }
  }
}

TEST_CASE("Shared materialization consumes only pending delete bindings",
          "[scan][execution_record][materialization]")
{
  auto set_a = std::make_shared<int const>(1), set_b = std::make_shared<int const>(2);
  auto required =
    std::make_shared<visibility_bindings const>(visibility_bindings{{"a", set_a}, {"b", set_b}});
  materialization_program program;
  program.contract = program.generation = 1;
  program.validate_output_schema        = false;
  program.required_visibility           = required;
  materialized_batch batch;
  batch.contract = batch.generation = 1;
  batch.logical_schema_complete = batch.projection_complete = true;
  batch.visibility_applied                                  = {{"a", set_a}};
  int calls                                                 = 0;
  materialization_steps steps;
  steps.bound_visibility = [&](materialized_batch&, auto pending) {
    ++calls;
    REQUIRE(pending.size() == 1);
    CHECK(pending.front().file == "b");
    CHECK(pending.front().input == set_b);
  };
  auto done = finish_materialization(std::move(batch), program, steps);
  CHECK(done.visibility_complete);
  CHECK(done.visibility_applied.size() == 2);
  done = finish_materialization(std::move(done), program, steps);
  CHECK(calls == 1);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Reader and Hive partial coverage respect qualified route authority",
                 "[scan][execution_record][hive][integration]")
{
  bool const hive = GENERATE(false, true);
  bool const bulk = GENERATE(false, true);
  // No predicate, full/partial pushdown, residual only, dynamic only and zero survivors.
  int const filter_kind = GENERATE(0, 1, 2, 3, 4, 5);
  CAPTURE(hive, bulk, filter_kind);
  sirius::test::scratch_dir directory("hive_execution_record");
  std::filesystem::create_directories(directory.path() / "year=2024");
  auto path = directory.file("year=2024/input.parquet");
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    auto written = con->Query(
      "COPY (SELECT i::INTEGER id, CASE WHEN i IN (11,301) THEN 200 ELSE 0 END amount, CASE WHEN "
      "i=301 THEN NULL ELSE 1 END region FROM range(1000) t(i)) TO " +
      directory.file_literal("year=2024/input.parquet") + " (FORMAT PARQUET)");
    REQUIRE(written);
    REQUIRE_FALSE(written->HasError());
  }
  auto info                 = std::make_unique<parquet_ingestible_table_info>();
  info->resolved_file_paths = {path};
  info->names               = {"id", "amount", "region", "year"};
  info->returned_types.assign(4, sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids     = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2)};
  info->projection_ids = {0, 1, 2};
  info->projected_output_count = 1;
  if (hive) {
    info->column_ids.emplace_back(3);
    info->projection_ids         = {0, 3, 1, 2};
    info->projected_output_count = 2;
    info->partition_indices      = {duckdb::HivePartitioningIndex("2024", 3)};
  }
  if (filter_kind) {
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    if (filter_kind != 3 && filter_kind != 4)
      info->table_filters->PushFilter(
        duckdb::ProjectionIndex(1),
        sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                      duckdb::Value::INTEGER(filter_kind == 5 ? 10000 : 100)));
    if (filter_kind >= 2 && filter_kind != 5)
      info->table_filters->PushFilter(
        duckdb::ProjectionIndex(2), sirius::test::null_filter(duckdb::LogicalType::INTEGER, true));
  }
  std::optional<sirius::op::sirius_dynamic_filter_set::producer> dynamic_producer;
  if (filter_kind == 4) {
    using namespace sirius::op;
    info->sirius_dynamic_filters = std::make_shared<sirius_dynamic_filter_set>();
    dynamic_producer.emplace(info->sirius_dynamic_filters->register_producer({0}));
    std::vector<zone_map_entry> zones;
    zones.push_back({std::make_unique<cudf::numeric_scalar<int32_t>>(11, true),
                     std::make_unique<cudf::numeric_scalar<int32_t>>(301, true)});
    REQUIRE(dynamic_producer->push_filter(
      0, std::make_shared<sirius_dynamic_zone_map_filter>(std::move(zones))));
    dynamic_producer->finish(sirius_dynamic_filter_set::completion::PUBLISHED);
  }
  auto counters            = std::make_shared<physical_check_counters>();
  counters->track_units    = true;
  info->profiles->counters = counters;
  int residual_calls = 0, decisions = 0;
  counters->parquet_filter_for_testing = [&](bool filtered, uint64_t rows) {
    CHECK(filtered == (filter_kind == 1 || filter_kind == 2 || filter_kind == 4));
    CHECK(rows == (filter_kind == 5                       ? 0
                   : filter_kind == 4                     ? 291
                   : filter_kind == 1 || filter_kind == 2 ? 2
                                                          : 1000));
  };
  counters->scan_residual_for_testing = [&](auto const& pending) {
    ++residual_calls;
    REQUIRE(pending.size() == 1);
  };
  counters->scan_execution_for_testing = [&](bool authority, auto const& facts) {
    ++decisions;
    CHECK(authority);
    CHECK(facts.whole_static_complete == (hive || filter_kind < 2));
    CHECK(facts.projection_complete == hive);
  };
  auto reader  = make_ingestible(std::move(info));
  auto ioctx   = std::make_shared<sirius::test::tracking_kvikio_context>(bulk);
  auto prepare = reader->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  auto coalescer = reader->create_batch_coalescer();
  auto splits    = coalescer->push(prepare({}));
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto batch =
    reader->materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
  REQUIRE(batch.execution);
  CHECK(batch.execution->authoritative);
  auto const input_rows = filter_kind == 5           ? 0
                          : filter_kind == 4         ? (hive ? 290 : 291)
                          : filter_kind == 0         ? 1000
                          : filter_kind == 3         ? (hive ? 999 : 1000)
                          : hive && filter_kind == 2 ? 1
                                                     : 2;
  CHECK(batch.table.num_rows() == input_rows);
  auto output = reader->post_filter_and_project(std::move(batch), *space, stream);
  CHECK(residual_calls == (filter_kind >= 2 ? 1 : 0));
  CHECK(decisions == 1);
  REQUIRE(output->num_columns() == (hive ? 2 : 1));
  std::vector<int32_t> expected =
    filter_kind == 2 ? std::vector<int32_t>{11} : std::vector<int32_t>{11, 301};
  if (filter_kind == 0 || filter_kind == 3) {
    expected.resize(1000);
    std::iota(expected.begin(), expected.end(), 0);
    if (filter_kind == 3) expected.erase(expected.begin() + 301);
  }
  if (filter_kind == 4) {
    expected.resize(290);
    std::iota(expected.begin(), expected.end(), 11);
  }
  if (filter_kind == 5) expected.clear();
  CHECK(host_column<int32_t>(output->view().column(0), stream) == expected);
  if (hive)
    CHECK(host_column<int32_t>(output->view().column(1), stream) ==
          std::vector<int32_t>(expected.size(), 2024));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Iceberg residual errors preserve deletion before unsafe evaluation",
                 "[scan][execution_record][iceberg][integration]")
{
  auto const deleted_row = GENERATE(-1, 0, 9);  // Empty set, bad row, or an unrelated row.
  auto const authority   = GENERATE(false, true);
  auto const virtual_row = GENERATE(false, true);
  // Without virtual columns or deletes, preserve the already-enabled reader filter.
  bool const reader_filtered = deleted_row == -1 && !virtual_row;
  CAPTURE(deleted_row, authority, virtual_row);
  sirius::test::scratch_dir directory("iceberg_execution_error");
  auto path = directory.file("input.parquet");
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    std::string payload;
    for (int i = 0; i < 32; ++i)
      payload += ", i::BIGINT payload" + std::to_string(i);
    auto written = con->Query(
      "COPY (SELECT i::INTEGER id, CASE WHEN i=0 THEN DATE '10000-01-01' ELSE DATE '2024-01-01' "
      "END d" +
      payload + " FROM range(10) t(i)) TO " + directory.file_literal("input.parquet") +
      " (FORMAT PARQUET)");
    REQUIRE(written);
    REQUIRE_FALSE(written->HasError());
  }
  auto info                 = std::make_unique<iceberg_ingestible_table_info>();
  info->resolved_file_paths = {path};
  info->table_path          = "error-order";
  info->names               = {"id", "d"};
  info->returned_types      = {sirius::logical_type::make(sirius::type_id::INTEGER),
                               sirius::logical_type::make(sirius::type_id::DATE)};
  info->column_ids          = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projection_ids      = {0};
  for (int i = 0; i < 32; ++i) {
    info->names.push_back("payload" + std::to_string(i));
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    info->column_ids.emplace_back(2 + i);
    info->projection_ids.push_back(2 + i);
  }
  info->projected_output_count = 33;
  info->approximate_batch_size = std::size_t{1} << 30;
  // The unsafe cast must preserve deletes-before-filter independently of payload
  // width and whether a virtual column consumes original reader positions.
  if (virtual_row) {
    auto const row_id = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
    info->column_ids.emplace_back(row_id);
    info->projection_ids.push_back(34);
    ++info->projected_output_count;
    info->virtual_columns = {{row_id,
                              "file_row_number",
                              sirius::logical_type::make(sirius::type_id::BIGINT),
                              scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER}};
  }
  info->projection_ids.push_back(1);
  info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(0),
    sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                  duckdb::Value::INTEGER(0)));
  auto cast = duckdb::BoundCastExpression::AddDefaultCastToType(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::DATE, 0),
    duckdb::LogicalType::TIMESTAMP_NS);
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(1),
    duckdb::make_uniq<duckdb::ExpressionFilter>(duckdb::ExpressionFilter::CreateNullCheckExpression(
      std::move(cast), duckdb::ExpressionType::OPERATOR_IS_NOT_NULL)));
  info->delete_sets.emplace();
  info->delete_sets->emplace(
    path,
    std::make_shared<iceberg_delete_set const>(
      path,
      std::make_shared<std::vector<int64_t> const>(
        deleted_row < 0 ? std::vector<int64_t>{} : std::vector<int64_t>{deleted_row})));
  auto reader  = make_ingestible(std::move(info));
  auto ioctx   = std::make_shared<sirius::io::kvikio_context>();
  auto prepare = reader->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  auto metadata    = prepare({});
  auto const& file = dynamic_cast<parquet_file_scan_info const&>(*metadata);
  CHECK_FALSE(file.position_filter_eligible);
  CHECK_FALSE(file.requires_row_positions);
  CHECK(file.disable_filter_pushdown == !reader_filtered);
  auto coalescer = reader->create_batch_coalescer();
  auto splits    = coalescer->push(std::move(metadata));
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto batch =
    reader->materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
  REQUIRE(batch.execution);
  batch.execution->authoritative = authority;
  if (deleted_row == 0 || reader_filtered) {
    auto output = reader->post_filter_and_project(std::move(batch), *space, stream);
    CHECK(output->num_rows() == 9);
  } else {
    REQUIRE_THROWS_MATCHES(reader->post_filter_and_project(std::move(batch), *space, stream),
                           sirius::invalid_input_exception,
                           Catch::Matchers::Message("DATE value is out of range for TIMESTAMP"));
  }
  stream.synchronize();
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Iceberg deletion vectors retain interval deletion without virtual columns",
                 "[scan][execution_record][row_positions][iceberg][integration]")
{
  using namespace sirius::scan_manager;
  auto const virtual_row = GENERATE(false, true);
  sirius::test::scratch_dir directory("iceberg_internal_dv");
  auto path   = directory.file("input.parquet");
  auto dvpath = directory.file("input.puffin");
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    std::string payload;
    for (int i = 0; i < 16; ++i)
      payload +=
        ", (i+" + std::to_string((i + 1) * 1000000) + ")::BIGINT payload" + std::to_string(i);
    auto written = con->Query("COPY (SELECT i::INTEGER id, (i%100)::INTEGER amount" + payload +
                              " FROM range(1000) t(i)) TO " +
                              directory.file_literal("input.parquet") + " (FORMAT PARQUET)");
    REQUIRE(written);
    REQUIRE_FALSE(written->HasError());
  }
  // Reuse a real, checksummed DV bitmap for rows 1 and 3, but bind its Puffin
  // footer to this data file. Both the manifest and footer are validated below.
  {
    std::ifstream fixture(
      "test/cpp/integration/data/iceberg_v3_deletion_vector/data/"
      "dv-00000-0-d7e8f9a0-0007-0007-0007-000000000002-00001.puffin",
      std::ios::binary);
    REQUIRE(fixture);
    fixture.seekg(4);
    std::array<char, 44> blob{};
    fixture.read(blob.data(), blob.size());
    REQUIRE(fixture.gcount() == blob.size());
    auto footer =
      "{\"blobs\":[{\"type\":\"deletion-vector-v1\",\"fields\":[],"
      "\"snapshot-id\":-1,\"sequence-number\":-1,\"offset\":4,\"length\":44,"
      "\"properties\":{\"referenced-data-file\":\"" +
      path + "\",\"cardinality\":\"2\"}}],\"properties\":{}}";
    std::ofstream output(dvpath, std::ios::binary);
    output.write("PFA1", 4);
    output.write(blob.data(), blob.size());
    output.write("PFA1", 4);
    output.write(footer.data(), footer.size());
    for (int shift = 0; shift < 32; shift += 8)
      output.put(static_cast<char>((footer.size() >> shift) & 0xff));
    for (int i = 0; i < 4; ++i)
      output.put(0);
    output.write("PFA1", 4);
    REQUIRE(output.good());
  }
  test::test_reservation_provider provider;
  provider.grant_bytes = 64 * 1024;
  auto ledger          = std::make_shared<preparation_ledger>(provider);
  auto discovery       = std::make_shared<iceberg_delete_discovery>();
  IcebergDeleteFileEntry dv;
  dv.file_path             = dvpath;
  dv.file_format           = "puffin";
  dv.referenced_data_file  = path;
  dv.content_offset        = 4;
  dv.content_size_in_bytes = 44;
  dv.file_size_in_bytes    = std::filesystem::file_size(dvpath);
  dv.record_count          = 2;
  discovery->deletion_vector_entries.push_back(std::move(dv));
  std::vector<std::string> paths{path};
  auto envelope = iceberg_dv_preparation::envelope(*discovery, paths);
  REQUIRE(ledger->admit(std::array{envelope}, memory_space_id(cucascade::memory::Tier::HOST, 0))
            .deferred);
  auto views    = std::make_shared<sirius::transparent::read_view_registry>();
  auto contract = allocate_scan_contract(
    *views, {}, 0, 1, nullptr, {}, {}, {materializer_kind::iceberg, "iceberg.v1"});
  auto plan =
    std::make_shared<iceberg_dv_preparation>(contract, "internal-dv", paths, discovery, *ledger);
  auto info                 = std::make_unique<iceberg_ingestible_table_info>();
  info->contract_id         = contract;
  info->read_views          = views;
  info->table_path          = "internal-dv";
  info->resolved_file_paths = paths;
  info->deferred            = plan;
  info->names               = {"id", "amount"};
  info->returned_types.assign(2, sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids     = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projection_ids = {0};
  for (int i = 0; i < 16; ++i) {
    info->names.push_back("payload" + std::to_string(i));
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    info->column_ids.emplace_back(2 + i);
    info->projection_ids.push_back(2 + i);
  }
  info->projection_ids.push_back(1);
  info->projected_output_count = 17;
  if (virtual_row) {
    auto const id = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
    info->column_ids.emplace_back(id);
    info->projection_ids.insert(info->projection_ids.begin() + 17, 18);
    ++info->projected_output_count;
    info->virtual_columns = {{id,
                              "file_row_number",
                              sirius::logical_type::make(sirius::type_id::BIGINT),
                              scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER}};
  }
  info->approximate_batch_size = std::size_t{1} << 30;
  info->table_filters          = duckdb::make_uniq<duckdb::TableFilterSet>();
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(1),
    sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                  duckdb::Value::INTEGER(3)));
  auto counters                           = std::make_shared<physical_check_counters>();
  counters->track_units                   = true;
  info->profiles->counters                = counters;
  int position_calls                      = 0;
  counters->parquet_positions_for_testing = [&](bool row, bool source) {
    ++position_calls;
    CHECK(row);
    CHECK_FALSE(source);
  };
  auto reader = make_ingestible(std::move(info));
  reader->attach_preparation_ledger(ledger);
  auto ioctx   = std::make_shared<sirius::io::kvikio_context>();
  auto claimed = reader->try_claim_split(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  REQUIRE(std::holds_alternative<preparation_job>(claimed));
  auto work = std::move(std::get<preparation_job>(claimed));
  work.unit = std::make_shared<preparation_unit>(work.requirement->key, work.requirement->required);
  auto metadata = work.run({});
  CHECK(counters->iceberg_delete_payload_loads == 1);
  auto& file = dynamic_cast<parquet_file_scan_info&>(*metadata);
  REQUIRE(file.row_groups.size() == 1);
  CHECK(file.requires_row_positions == virtual_row);
  CHECK(file.disable_filter_pushdown == !virtual_row);
  auto coalescer = reader->create_batch_coalescer();
  auto splits    = coalescer->push(std::move(metadata));
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  CHECK(dynamic_cast<parquet_split_info const&>(*splits.front()).requires_row_positions ==
        virtual_row);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto batch =
    reader->materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
  REQUIRE(batch.execution);
  CHECK(batch.execution->authoritative);
  REQUIRE(batch.required_visibility);
  REQUIRE(batch.required_visibility->size() == 1);
  REQUIRE(batch.execution->visibility_applied.size() == 1);
  CHECK(batch.execution->visibility_applied.front().file ==
        batch.required_visibility->front().file);
  CHECK(batch.execution->visibility_applied.front().input ==
        batch.required_visibility->front().input);
  CHECK(batch.execution->visibility_complete);
  CHECK(batch.execution->whole_static_complete == virtual_row);
  CHECK_FALSE(batch.row_positions);
  CHECK(position_calls == (virtual_row ? 1 : 0));
  CHECK(batch.table.num_rows() == (virtual_row ? 29 : 998));
  auto output = reader->post_filter_and_project(std::move(batch), *space, stream);
  std::vector<int32_t> expected;
  for (int i = 0; i < 1000; ++i)
    if (i % 100 < 3 && i != 1 && i != 3) expected.push_back(i);
  REQUIRE(output->num_columns() == (virtual_row ? 18 : 17));
  CHECK(host_column<int32_t>(output->view().column(0), stream) == expected);
  if (virtual_row)
    CHECK(host_column<int64_t>(output->view().column(17), stream) ==
          std::vector<int64_t>(expected.begin(), expected.end()));
  for (int col = 0; col < 16; ++col) {
    std::vector<int64_t> payload;
    for (auto id : expected)
      payload.push_back(id + (col + 1) * 1000000);
    CHECK(host_column<int64_t>(output->view().column(col + 1), stream) == payload);
  }
  output.reset();
  splits.clear();
  work = {};
  reader->finish_preparation();
  CHECK(ledger->tracked_units() == 0);
  reader.reset();
  plan.reset();
  ledger->close();
  ledger.reset();
  CHECK(provider.outstanding() == 0);
}

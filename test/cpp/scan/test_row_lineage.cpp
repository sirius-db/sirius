/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/materialization_contract.hpp"
#include "op/scan/row_lineage.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/static_predicate_catalogue.hpp"

#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <utils/scan_resource_test_utils.hpp>
#include <utils/test_validation_utility.hpp>

#include <array>

using namespace sirius::op::scan;
namespace {
row_lineage identity(std::string name, int64_t rows, int64_t first = 0, int64_t count = -1)
{
  auto domain = std::make_shared<lineage_domain>();
  domain->files.push_back({std::move(name), rows});
  domain->file_runs.push_back({0, first, count < 0 ? rows - first : count});
  row_lineage result;
  result.state  = row_position_state::identity;
  result.domain = std::move(domain);
  return result;
}
}  // namespace

TEST_CASE("Lineage concatenation advances by input domain rather than survivors", "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr       = cudf::get_current_device_resource_ref();
  auto original = identity("A", 1000);
  auto filtered =
    original.select(std::array<int64_t, 2>{11, 301}, static_cast<::cuda::stream_ref>(stream), mr);
  REQUIRE(filtered.input_domain_length() == 1000);
  REQUIRE(filtered.output_length() == 2);
  REQUIRE(filtered.state == row_position_state::per_row);
  auto second = identity("B", 4, 2, 2);
  std::array parts{filtered, second};
  auto joined = row_lineage::concatenate(parts, static_cast<::cuda::stream_ref>(stream), mr);
  REQUIRE(joined.ordinals_on(static_cast<::cuda::stream_ref>(stream)) ==
          std::vector<int64_t>{11, 301, 1000, 1001});
  auto compatibility = legacy_row_domain(joined, static_cast<::cuda::stream_ref>(stream));
  REQUIRE(compatibility.positions);
  CHECK(compatibility.positions->at(2).file == 1);
  CHECK(compatibility.positions->at(2).row == 2);
  REQUIRE(joined.resolve(1000) == std::pair<std::size_t, int64_t>{1, 2});
  auto again =
    joined.select(std::array<int64_t, 2>{1, 3}, static_cast<::cuda::stream_ref>(stream), mr);
  REQUIRE(again.input_domain_length() == 1002);
  REQUIRE(again.ordinals_on(static_cast<::cuda::stream_ref>(stream)) ==
          std::vector<int64_t>{301, 1001});
  REQUIRE(again.resolve(1001) == std::pair<std::size_t, int64_t>{1, 3});
}

TEST_CASE("Lineage resolves pruned gaps and composes only matching delivered domains",
          "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr           = cudf::get_current_device_resource_ref();
  auto domain       = std::make_shared<lineage_domain>();
  domain->files     = {{"A", 100}, {"B", 1000}};
  domain->file_runs = {{0, 20, 10}, {1, 300, 5}};
  row_lineage before{row_position_state::runs, domain, {{1, 2}, {11, 2}}, {}};
  before.validate_on(4, static_cast<::cuda::stream_ref>(stream));
  REQUIRE(before.resolve(11) == std::pair<std::size_t, int64_t>{1, 301});
  auto delivered       = std::make_shared<lineage_domain>();
  delivered->files     = domain->files;
  delivered->file_runs = {{0, 21, 2}, {1, 301, 2}};
  row_lineage step{row_position_state::runs, delivered, {{1, 2}}, {}};
  auto result = row_lineage::compose(before, step, static_cast<::cuda::stream_ref>(stream), mr);
  REQUIRE(result.domain == domain);
  REQUIRE(result.ordinals_on(static_cast<::cuda::stream_ref>(stream)) ==
          std::vector<int64_t>{2, 11});
  delivered->file_runs[1].first = 400;
  REQUIRE_THROWS(row_lineage::compose(before, step, static_cast<::cuda::stream_ref>(stream), mr));
  REQUIRE_THROWS(
    before.select(std::array<int64_t, 2>{0, 0}, static_cast<::cuda::stream_ref>(stream), mr));
  REQUIRE_THROWS(
    before.select(std::array<int64_t, 1>{4}, static_cast<::cuda::stream_ref>(stream), mr));
}

TEST_CASE("Lineage keeps compact domains compact and never invents lost positions",
          "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::array parts{identity("A", 1000000000), identity("B", 1000000000)};
  auto joined = row_lineage::concatenate(parts, static_cast<::cuda::stream_ref>(stream), mr);
  REQUIRE(joined.state == row_position_state::identity);
  REQUIRE_FALSE(joined.per_row);
  REQUIRE(joined.runs.empty());
  REQUIRE(joined.input_domain_length() == 2000000000);
  for (auto state : {row_position_state::none, row_position_state::lost}) {
    row_lineage unavailable;
    unavailable.state = state;
    unavailable.validate(100);
    REQUIRE(
      unavailable.select(std::array<int64_t, 1>{0}, static_cast<::cuda::stream_ref>(stream), mr)
        .state == state);
    std::array mixed{parts.front(), unavailable};
    REQUIRE(row_lineage::concatenate(mixed, static_cast<::cuda::stream_ref>(stream), mr).state ==
            row_position_state::lost);
  }
  auto bad                  = identity("A", 2);
  auto wrong                = std::make_shared<lineage_domain>(*bad.domain);
  wrong->file_runs[0].count = 3;
  bad.domain                = wrong;
  REQUIRE_THROWS(bad.validate(3));
}

TEST_CASE("Shared predicate catalogue owns full expressions and stable slot bindings",
          "[scan][lineage][execution_record]")
{
  std::vector<sirius::op::table_filter_conjunct> terms;
  for (int i = 0; i < 2; ++i) {
    auto comparison = duckdb::BoundComparisonExpression::Create(
      duckdb::ExpressionType::COMPARE_GREATERTHAN,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, i),
      duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(100)));
    terms.push_back(
      {static_cast<std::size_t>(i), static_cast<std::size_t>(i), std::move(comparison)});
  }
  auto catalogue = std::make_shared<static_predicate_catalogue const>(
    7, 1, std::move(terms), std::unordered_set<std::size_t>{1});
  REQUIRE(catalogue->size() == 2);
  REQUIRE(catalogue->columns(0).front() == 0);
  REQUIRE(catalogue->columns(1).front() == 1);
  REQUIRE_FALSE(catalogue->answerable(0));
  REQUIRE(catalogue->answerable(1));
  REQUIRE(catalogue->expression(0).ToString().find("100") != std::string::npos);
  std::array<sirius::op::predicate_progress, 1> progress{
    {{0, sirius::op::predicate_stage::enforced, {}}}};
  REQUIRE(catalogue->residual().against_progress(progress) != nullptr);
  progress[0].conjunct = 2;
  REQUIRE_THROWS(catalogue->residual().validate_progress(progress));
}

TEST_CASE("Reader position adaptation reuses row storage and respects pruned intervals",
          "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr            = cudf::get_current_device_resource_ref();
  auto positions     = std::make_unique<gpu_row_positions>();
  positions->sources = {{"A", 100, {{20, 30}}}, {"B", 1000, {{300, 305}}}};
  std::vector<uint64_t> rows{21, 301, 303};
  std::vector<int32_t> sources{0, 1, 1};
  positions->rows = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT64}, rows.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  positions->source_indices = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                        sources.size(),
                                                        cudf::mask_state::UNALLOCATED,
                                                        stream,
                                                        mr);
  REQUIRE(cudaMemcpyAsync(positions->rows->mutable_view().data<uint64_t>(),
                          rows.data(),
                          rows.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(positions->source_indices->mutable_view().data<int32_t>(),
                          sources.data(),
                          sources.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  void const* original_storage = positions->rows->view().head<uint64_t>();
  auto lineage =
    consume_reader_positions(std::move(positions), static_cast<::cuda::stream_ref>(stream), mr);
  CHECK(lineage.per_row->view().head<int64_t>() == original_storage);
  CHECK(lineage.input_domain_length() == 15);
  CHECK(lineage.ordinals_on(static_cast<::cuda::stream_ref>(stream)) ==
        std::vector<int64_t>{1, 11, 13});
  CHECK(lineage.resolve(11) == std::pair<std::size_t, int64_t>{1, 301});
  CHECK(lineage_conversion_workspace_bytes(3, 2) >= 3 * sizeof(int64_t));
}

TEST_CASE("GPU lineage composition matches an independent file-row oracle", "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr           = cudf::get_current_device_resource_ref();
  auto domain       = std::make_shared<lineage_domain>();
  domain->files     = {{"A", 3000}, {"B", 7000}};
  domain->file_runs = {{0, 13, 1000}, {1, 300, 1000}};
  row_lineage original{row_position_state::identity, domain};
  std::vector<int64_t> first_indices;
  std::vector<std::pair<std::size_t, int64_t>> oracle;
  for (int64_t i = 0; i < 2000; ++i) {
    if (i % 17 != 3) continue;
    first_indices.push_back(i);
    oracle.emplace_back(i < 1000 ? 0 : 1, i < 1000 ? 13 + i : 300 + i - 1000);
  }
  auto first = original.select(first_indices, stream, mr);
  REQUIRE(first.state == row_position_state::per_row);
  auto next_domain = std::make_shared<lineage_domain>();
  // A different file ordinal namespace must be remapped explicitly.
  next_domain->files = {{"B", 7000}, {"A", 3000}};
  for (auto const& [file, row] : oracle)
    next_domain->file_runs.push_back({1 - file, row, 1});
  std::vector<int64_t> second_indices;
  std::vector<std::pair<std::size_t, int64_t>> expected;
  for (int64_t i = static_cast<int64_t>(oracle.size()) - 1; i >= 0; i -= 5) {
    second_indices.push_back(i);
    expected.push_back(oracle[i]);
  }
  row_lineage next{row_position_state::identity, next_domain};
  next          = next.select(second_indices, stream, mr);
  auto composed = row_lineage::compose(first, next, stream, mr);
  CHECK(composed.domain == domain);
  auto ordinals = composed.ordinals_on(stream);
  REQUIRE(ordinals.size() == expected.size());
  for (std::size_t i = 0; i < ordinals.size(); ++i)
    CHECK(composed.resolve(ordinals[i]) == expected[i]);
  CHECK(composed.input_domain_length() == 2000);
  // A legal-sized but semantically foreign step must fail before use.
  auto foreign = std::make_shared<lineage_domain>(*next_domain);
  ++foreign->file_runs[0].first;
  next.domain = foreign;
  REQUIRE_THROWS(row_lineage::compose(first, next, stream, mr));
}

TEST_CASE("Compact lineage allocates no device rows and rejects domain overflow", "[scan][lineage]")
{
  rmm::cuda_stream stream;
  sirius::test::tracked_scan_resource resource(cudf::get_current_device_resource_ref(), stream);
  resource.state->fail_at = 0;
  rmm::device_async_resource_ref mr{resource};
  auto a = identity("A", 1000000000);
  auto b = identity("B", 1000000000);
  std::array parts{a, b};
  auto joined = row_lineage::concatenate(parts, stream, mr);
  CHECK(joined.state == row_position_state::identity);
  auto selected = joined.select(std::array<int64_t, 3>{0, 1, 2}, stream, mr);
  CHECK(selected.state == row_position_state::runs);
  CHECK(resource.state->attempts == 0);
  for (auto unavailable : {row_position_state::lost, row_position_state::none}) {
    row_lineage before;
    before.state = unavailable;
    CHECK(row_lineage::compose(before, a, stream, mr).state == unavailable);
  }
  auto huge       = std::make_shared<lineage_domain>();
  huge->files     = {{"A", INT64_MAX}, {"B", 1}};
  huge->file_runs = {{0, 0, INT64_MAX}, {1, 0, 1}};
  row_lineage overflow{row_position_state::identity, huge};
  REQUIRE_THROWS(overflow.validate(0));
}

TEST_CASE("Canonical conversion budgets unique scratch above the transferred reader buffer",
          "[scan][lineage][lineage_workspace]")
{
  auto rows        = GENERATE(std::size_t{0},
                       std::size_t{1},
                       std::size_t{2},
                       std::size_t{3},
                       std::size_t{16},
                       std::size_t{32},
                       std::size_t{128},
                       std::size_t{1024},
                       std::size_t{1000000});
  bool const multi = GENERATE(false, true);
  rmm::cuda_stream stream;
  sirius::test::tracked_scan_resource resource(cudf::get_current_device_resource_ref(), stream);
  rmm::device_async_resource_ref mr{resource};
  auto positions         = std::make_unique<gpu_row_positions>();
  auto const first_count = multi ? rows / 2 : rows;
  positions->sources     = {{"A", first_count, {{0, first_count}}}};
  if (multi) positions->sources.push_back({"B", rows - first_count, {{0, rows - first_count}}});
  positions->rows = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT64}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
  std::vector<uint64_t> file_rows(rows);
  std::vector<int32_t> sources(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    file_rows[row] = row < first_count ? row : row - first_count;
    sources[row]   = row < first_count ? 0 : 1;
  }
  if (rows)
    REQUIRE(cudaMemcpyAsync(positions->rows->mutable_view().data<uint64_t>(),
                            file_rows.data(),
                            rows * sizeof(uint64_t),
                            cudaMemcpyHostToDevice,
                            stream.value()) == cudaSuccess);
  if (multi) {
    positions->source_indices = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    if (rows)
      REQUIRE(cudaMemcpyAsync(positions->source_indices->mutable_view().data<int32_t>(),
                              sources.data(),
                              rows * sizeof(int32_t),
                              cudaMemcpyHostToDevice,
                              stream.value()) == cudaSuccess);
  }
  stream.synchronize();
  auto const buffer      = positions->rows->view().head();
  auto const transferred = resource.state->live;
  resource.state->peak   = transferred;
  {
    auto converted      = consume_reader_positions(std::move(positions), stream, mr);
    auto const scratch  = resource.state->peak - transferred;
    auto const reserved = lineage_conversion_workspace_bytes(rows, multi ? 2 : 1);
    CAPTURE(rows, multi, transferred, scratch, reserved);
    CHECK(scratch <= reserved);
    CHECK(converted.per_row->view().head() == buffer);
    CHECK(resource.state->live == rows * sizeof(int64_t));
    CHECK_FALSE(resource.state->wrong_stream);
    if (rows) {
      CHECK(converted.resolve(rows - 1) ==
            std::pair<std::size_t, int64_t>{
              multi ? 1 : 0, static_cast<int64_t>(multi ? rows - first_count - 1 : rows - 1)});
    }
  }
  stream.synchronize();
  CHECK(resource.state->live == 0);
}

TEST_CASE("Reader conversion remaps repeated file sources without merging input occurrences",
          "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr            = cudf::get_current_device_resource_ref();
  auto positions     = std::make_unique<gpu_row_positions>();
  positions->sources = {{"A", 1000, {{10, 12}}}, {"A", 1000, {{300, 302}}}};
  positions->rows    = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT64}, 4, cudf::mask_state::UNALLOCATED, stream, mr);
  positions->source_indices = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 4, cudf::mask_state::UNALLOCATED, stream, mr);
  std::array<uint64_t, 4> file_rows{11, 10, 301, 300};
  std::array<int32_t, 4> sources{0, 0, 1, 1};
  REQUIRE(cudaMemcpyAsync(positions->rows->mutable_view().data<uint64_t>(),
                          file_rows.data(),
                          sizeof(file_rows),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(positions->source_indices->mutable_view().data<int32_t>(),
                          sources.data(),
                          sizeof(sources),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  SECTION("same file namespace retains both selected domains")
  {
    auto converted = consume_reader_positions(std::move(positions), stream, mr);
    REQUIRE(converted.domain->files.size() == 1);
    REQUIRE(converted.domain->file_runs.size() == 2);
    CHECK(converted.input_domain_length() == 4);
    CHECK(converted.ordinals_on(stream) == std::vector<int64_t>{1, 0, 3, 2});
    CHECK(converted.resolve(3) == std::pair<std::size_t, int64_t>{0, 301});
  }
  SECTION("different lengths cannot share a file identity")
  {
    positions->sources[1].file_rows = 999;
    stream.synchronize();
    REQUIRE_THROWS(consume_reader_positions(std::move(positions), stream, mr));
  }
}

TEST_CASE("Canonical virtual columns resolve each selected file and original row",
          "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr           = cudf::get_current_device_resource_ref();
  auto domain       = std::make_shared<lineage_domain>();
  domain->files     = {{"A", 1000}, {"B", 1000}};
  domain->file_runs = {{0, 10, 2}, {1, 300, 2}};
  row_lineage selected{row_position_state::runs, domain, {{1, 1}, {3, 1}}};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 2, cudf::mask_state::UNALLOCATED, stream, mr));
  owning_table_view table{std::make_unique<cudf::table>(std::move(columns))};
  scan_plan plan;
  plan.data_columns    = {{0, "payload"}};
  using kind           = scan_plan::parquet_virtual_column_kind;
  plan.virtual_columns = {
    {997, "filename", sirius::logical_type::make(sirius::type_id::VARCHAR), kind::FILENAME, 1},
    {998, "file_index", sirius::logical_type::make(sirius::type_id::UBIGINT), kind::FILE_INDEX, 2},
    {999, "row", sirius::logical_type::make(sirius::type_id::BIGINT), kind::FILE_ROW_NUMBER, 3}};
  std::array<uint64_t, 2> indices{5, 9};
  append_lineage_virtual_columns(table, selected, plan, indices, stream, mr);
  REQUIRE(table.n_columns() == 4);
  auto view = table.view();
  CHECK(sirius::test::detail::copy_device_values<uint64_t>(
          view.column(2).data<uint64_t>(), 2, stream) == std::vector<uint64_t>{5, 9});
  CHECK(sirius::test::detail::copy_device_values<int64_t>(
          view.column(3).data<int64_t>(), 2, stream) == std::vector<int64_t>{11, 301});
  cudf::strings_column_view names{view.column(1)};
  auto offsets = names.offsets();
  std::vector<int64_t> host_offsets;
  if (offsets.type().id() == cudf::type_id::INT64)
    host_offsets =
      sirius::test::detail::copy_device_values<int64_t>(offsets.data<int64_t>(), 3, stream);
  else {
    auto narrow =
      sirius::test::detail::copy_device_values<int32_t>(offsets.data<int32_t>(), 3, stream);
    host_offsets.assign(narrow.begin(), narrow.end());
  }
  CHECK(host_offsets == std::vector<int64_t>{0, 1, 2});
  CHECK(sirius::test::detail::copy_device_values<char>(names.chars_begin(stream), 2, stream) ==
        std::vector<char>{'A', 'B'});
}

TEST_CASE("Empty fileless lineage assembles typed empty virtual columns",
          "[scan][lineage][decoder_boundary]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  row_lineage lineage{row_position_state::identity, std::make_shared<lineage_domain const>()};
  scan_plan plan;
  using kind           = scan_plan::parquet_virtual_column_kind;
  plan.virtual_columns = {
    {1, "file", sirius::logical_type::make(sirius::type_id::VARCHAR), kind::FILENAME, 0},
    {2, "index", sirius::logical_type::make(sirius::type_id::UBIGINT), kind::FILE_INDEX, 1},
    {3, "row", sirius::logical_type::make(sirius::type_id::BIGINT), kind::FILE_ROW_NUMBER, 2}};
  owning_table_view table;
  append_lineage_virtual_columns(table, lineage, plan, {}, stream, mr);
  REQUIRE(table.n_columns() == 3);
  CHECK(table.num_rows() == 0);
  CHECK(table.view().column(0).type().id() == cudf::type_id::STRING);
  CHECK(table.view().column(1).type().id() == cudf::type_id::UINT64);
  CHECK(table.view().column(2).type().id() == cudf::type_id::INT64);
}

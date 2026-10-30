/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/materialization_contract.hpp"
#include "op/scan/row_lineage.hpp"
#include "op/scan/static_predicate_catalogue.hpp"

#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

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
  auto filtered = original.select(std::array<int64_t, 2>{11, 301}, stream.view(), mr);
  REQUIRE(filtered.input_domain_length() == 1000);
  REQUIRE(filtered.output_length() == 2);
  REQUIRE(filtered.state == row_position_state::per_row);
  auto second = identity("B", 4, 2, 2);
  std::array parts{filtered, second};
  auto joined = row_lineage::concatenate(parts, stream.view(), mr);
  REQUIRE(joined.ordinals_on(stream.view()) == std::vector<int64_t>{11, 301, 1000, 1001});
  auto compatibility = legacy_row_domain(joined, stream.view());
  REQUIRE(compatibility.positions);
  CHECK(compatibility.positions->at(2).file == 1);
  CHECK(compatibility.positions->at(2).row == 2);
  REQUIRE(joined.resolve(1000) == std::pair<std::size_t, int64_t>{1, 2});
  auto again = joined.select(std::array<int64_t, 2>{1, 3}, stream.view(), mr);
  REQUIRE(again.input_domain_length() == 1002);
  REQUIRE(again.ordinals_on(stream.view()) == std::vector<int64_t>{301, 1001});
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
  before.validate_on(4, stream.view());
  REQUIRE(before.resolve(11) == std::pair<std::size_t, int64_t>{1, 301});
  auto delivered       = std::make_shared<lineage_domain>();
  delivered->files     = domain->files;
  delivered->file_runs = {{0, 21, 2}, {1, 301, 2}};
  row_lineage step{row_position_state::runs, delivered, {{1, 2}}, {}};
  auto result = row_lineage::compose(before, step, stream.view(), mr);
  REQUIRE(result.domain == domain);
  REQUIRE(result.ordinals_on(stream.view()) == std::vector<int64_t>{2, 11});
  delivered->file_runs[1].first = 400;
  REQUIRE_THROWS(row_lineage::compose(before, step, stream.view(), mr));
  REQUIRE_THROWS(before.select(std::array<int64_t, 2>{0, 0}, stream.view(), mr));
  REQUIRE_THROWS(before.select(std::array<int64_t, 1>{4}, stream.view(), mr));
}

TEST_CASE("Lineage keeps compact domains compact and never invents lost positions",
          "[scan][lineage]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::array parts{identity("A", 1000000000), identity("B", 1000000000)};
  auto joined = row_lineage::concatenate(parts, stream.view(), mr);
  REQUIRE(joined.state == row_position_state::identity);
  REQUIRE_FALSE(joined.per_row);
  REQUIRE(joined.runs.empty());
  REQUIRE(joined.input_domain_length() == 2000000000);
  for (auto state : {row_position_state::none, row_position_state::lost}) {
    row_lineage unavailable;
    unavailable.state = state;
    unavailable.validate(100);
    REQUIRE(unavailable.select(std::array<int64_t, 1>{0}, stream.view(), mr).state == state);
    std::array mixed{parts.front(), unavailable};
    REQUIRE(row_lineage::concatenate(mixed, stream.view(), mr).state == row_position_state::lost);
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
  auto lineage                 = consume_reader_positions(std::move(positions), stream.view(), mr);
  CHECK(lineage.per_row->view().head<int64_t>() == original_storage);
  CHECK(lineage.input_domain_length() == 15);
  CHECK(lineage.ordinals_on(stream.view()) == std::vector<int64_t>{1, 11, 13});
  CHECK(lineage.resolve(11) == std::pair<std::size_t, int64_t>{1, 301});
  CHECK(lineage_conversion_workspace_bytes(3, 2) >= 3 * sizeof(int64_t));
}

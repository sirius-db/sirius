/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/detail/read_retirement.hpp"
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
#include <limits>

using namespace sirius::op::scan;

TEST_CASE("Validated lineage shares immutable storage and rechecks reconstructed inputs",
          "[scan][lineage][validated_lineage]")
{
  rmm::cuda_stream stream;
  sirius::test::tracked_scan_resource tracked(cudf::get_current_device_resource_ref(), stream);
  rmm::device_async_resource_ref mr{tracked};
  auto domain                            = std::make_shared<lineage_domain>();
  domain->files                          = {{"A", 100}};
  domain->file_runs                      = {{0, 20, 10}};
  std::shared_ptr<cudf::column> external = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  std::array<int64_t, 3> values{7, 1, 5};
  REQUIRE(cudaMemcpyAsync(external->mutable_view().data<int64_t>(),
                          values.data(),
                          sizeof(values),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  row_lineage raw{row_position_state::per_row, domain, {}, external};
  auto validated = raw.validated_copy(3, stream, mr);
  CHECK(validated.per_row() != external);
  CHECK(validated.domain() != domain);
  CHECK(validated.matches_domain(domain));
  auto const validated_allocations = tracked.state->attempts;
  for (int i = 0; i < 3; ++i)
    validated.validate_on(3, stream, mr);
  CHECK(tracked.state->attempts == validated_allocations);
  auto imported_again = validated.validated_copy(3, stream, mr);
  CHECK(tracked.state->attempts > validated_allocations);
  CHECK(imported_again.per_row() == validated.per_row());
  // Reconstructing the same read-only fields does not carry private provenance.
  row_lineage reconstructed{
    validated.state(), validated.domain(), validated.runs(), validated.per_row()};
  auto const before_reconstructed = tracked.state->attempts;
  reconstructed.validate_on(3, stream, mr);
  CHECK(tracked.state->attempts > before_reconstructed);
  auto copy              = validated;
  auto const before_copy = tracked.state->attempts;
  copy.validate_on(3, stream, mr);
  CHECK(tracked.state->attempts == before_copy);
  CHECK_THROWS(validated.validate_on(4, stream, mr));
  // Mutating the decoder's aliases cannot change already validated storage.
  values = {-1, 1, 1};
  REQUIRE(cudaMemcpyAsync(external->mutable_view().data<int64_t>(),
                          values.data(),
                          sizeof(values),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  domain->file_runs[0].first = 40;
  stream.synchronize();
  CHECK(validated.ordinals_on(stream) == std::vector<int64_t>{7, 1, 5});
  CHECK(validated.resolve(7) == std::pair<std::size_t, int64_t>{0, 27});
  CHECK_FALSE(validated.matches_domain(domain));
  CHECK_THROWS(raw.validate_on(3, stream, mr));
  row_lineage changed{validated.state(), validated.domain(), validated.runs(), external};
  CHECK_THROWS(changed.validate_on(3, stream, mr));
  changed = {validated.state(), validated.domain(), {{0, 3}}, validated.per_row()};
  CHECK_THROWS(changed.validate_on(3, stream, mr));
  changed = {row_position_state::identity, validated.domain(), {}, validated.per_row()};
  CHECK_THROWS(changed.validate_on(3, stream, mr));
  changed = {validated.state(),
             std::make_shared<lineage_domain const>(lineage_domain{{{"foreign", 1}}, {{0, 0, 1}}}),
             validated.runs(),
             validated.per_row()};
  CHECK_THROWS(changed.validate_on(3, stream, mr));
  auto selected = validated.select(std::array<int64_t, 2>{2, 0}, stream, mr);
  CHECK(selected.ordinals_on(stream) == std::vector<int64_t>{5, 7});
  auto const selected_allocations = tracked.state->attempts;
  selected.validate_on(2, stream, mr);
  CHECK(tracked.state->attempts == selected_allocations);
  CHECK_THROWS(selected.select(std::array<int64_t, 2>{0, 0}, stream, mr));
  CHECK_THROWS(selected.select(std::array<int64_t, 1>{2}, stream, mr));
  auto again = selected.select(std::array<int64_t, 1>{1}, stream, mr);
  CHECK(again.ordinals_on(stream) == std::vector<int64_t>{7});
  std::array parts{validated, again};
  auto joined = row_lineage::concatenate(parts, stream, mr);
  CHECK(joined.ordinals_on(stream) == std::vector<int64_t>{7, 1, 5, 17});
  auto const joined_allocations = tracked.state->attempts;
  joined.validate_on(4, stream, mr);
  CHECK(tracked.state->attempts == joined_allocations);
}

TEST_CASE("Canonical lineage import retires every failed allocation",
          "[scan][lineage][validated_lineage][retirement]")
{
  rmm::cuda_stream stream;
  auto mr     = cudf::get_current_device_resource_ref();
  auto domain = std::make_shared<lineage_domain>(lineage_domain{{{"A", 8}}, {{0, 0, 8}}});
  auto column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  std::array<int64_t, 3> values{7, 1, 5};
  REQUIRE(cudaMemcpyAsync(column->mutable_view().data<int64_t>(),
                          values.data(),
                          sizeof(values),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  row_lineage raw{row_position_state::per_row, domain, {}, std::move(column)};
  stream.synchronize();
  sirius::test::tracked_scan_resource reference(mr, stream);
  auto imported          = raw.validated_copy(3, stream, rmm::device_async_resource_ref{reference});
  auto const allocations = reference.state->attempts;
  REQUIRE(allocations > 0);
  CHECK(imported.ordinals_on(stream) == std::vector<int64_t>{7, 1, 5});
  imported = {};
  CHECK(reference.state->live == 0);
  for (std::size_t failure = 0; failure < allocations; ++failure) {
    CAPTURE(failure, allocations);
    sirius::test::tracked_scan_resource failing(mr, stream);
    failing.state->fail_at = failure;
    CHECK_THROWS(raw.validated_copy(3, stream, rmm::device_async_resource_ref{failing}));
    CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
    CHECK(failing.state->live == 0);
    CHECK_FALSE(failing.state->wrong_stream);
  }
  CHECK(raw.ordinals_on(stream) == std::vector<int64_t>{7, 1, 5});
}

TEST_CASE("Lineage selection retires sources on failure and retains unjoined owners",
          "[scan][lineage][validated_lineage][retirement]")
{
  rmm::cuda_stream stream;
  auto mr           = cudf::get_current_device_resource_ref();
  auto domain       = std::make_shared<lineage_domain>();
  domain->files     = {{"A", 4}};
  domain->file_runs = {{0, 0, 4}};
  row_lineage source{row_position_state::identity, domain};
  auto validated =
    source.validated_copy(4, stream, mr).select(std::array<int64_t, 2>{3, 1}, stream, mr);
  rmm::cuda_stream other_stream;
  auto on_other = validated.select(std::array<int64_t, 1>{1}, other_stream, mr);
  CHECK(on_other.ordinals_on(stream) == std::vector<int64_t>{1});
  std::weak_ptr<cudf::column const> life = validated.per_row();
  bool const unjoined                    = GENERATE(false, true);
  {
    detail::read_retirement_test_hooks hooks(detail::read_checkpoint::PAYLOAD_GATHER, unjoined);
    if (unjoined)
      CHECK_THROWS_AS(validated.select(std::array<int64_t, 1>{1}, stream, mr),
                      sirius::unjoined_gpu_work);
    else
      CHECK_THROWS_AS(validated.select(std::array<int64_t, 1>{1}, stream, mr), rmm::out_of_memory);
    CHECK(hooks.checkpoint_reached);
    CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
    validated = {};
    CHECK(life.expired() == !unjoined);
    CHECK(hooks.retained.has_value() == unjoined);
  }
  CHECK(life.expired());
}

TEST_CASE("Every lineage selection allocation failure retires partial GPU work",
          "[scan][lineage][validated_lineage][retirement]")
{
  rmm::cuda_stream stream;
  auto mr           = cudf::get_current_device_resource_ref();
  auto domain       = std::make_shared<lineage_domain>();
  domain->files     = {{"A", 16}};
  domain->file_runs = {{0, 0, 16}};
  row_lineage source{row_position_state::identity, domain};
  auto validated =
    source.validated_copy(16, stream, mr).select(std::array<int64_t, 3>{15, 3, 7}, stream, mr);
  sirius::test::tracked_scan_resource reference(mr, stream);
  auto successful = validated.select(
    std::array<int64_t, 2>{2, 0}, stream, rmm::device_async_resource_ref{reference});
  auto allocations = reference.state->attempts;
  REQUIRE(allocations > 0);
  CHECK(successful.ordinals_on(stream) == std::vector<int64_t>{7, 15});
  successful = {};
  CHECK(reference.state->live == 0);
  for (std::size_t failure = 0; failure < allocations; ++failure) {
    CAPTURE(failure, allocations);
    sirius::test::tracked_scan_resource failing(mr, stream);
    failing.state->fail_at = failure;
    CHECK_THROWS(validated.select(
      std::array<int64_t, 2>{2, 0}, stream, rmm::device_async_resource_ref{failing}));
    CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
    CHECK(failing.state->live == 0);
    CHECK_FALSE(failing.state->wrong_stream);
  }
}

namespace {
row_lineage identity(std::string name, int64_t rows, int64_t first = 0, int64_t count = -1)
{
  auto domain = std::make_shared<lineage_domain>();
  domain->files.push_back({std::move(name), rows});
  domain->file_runs.push_back({0, first, count < 0 ? rows - first : count});
  return {row_position_state::identity, std::move(domain)};
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
  REQUIRE(filtered.state() == row_position_state::per_row);
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

TEST_CASE("Host lineage transfers retain storage until completion is established",
          "[scan][lineage][retirement]")
{
  auto const operation = GENERATE(0, 1, 2);     // compact upload, selection upload, download
  auto const failure   = GENERATE(0, 1, 2, 3);  // success, post-copy, failed cleanup, failed join
  CAPTURE(operation, failure);
  rmm::cuda_stream stream;
  sirius::test::tracked_scan_resource resource(cudf::get_current_device_resource_ref(), stream);
  rmm::device_async_resource_ref mr{resource};
  auto source = identity("A", 8).validated_copy(8, stream, mr);
  if (operation != 0) source = source.select(std::array<int64_t, 3>{7, 1, 5}, stream, mr);
  std::weak_ptr<cudf::column const> source_life = source.per_row();
  bool const unjoined                           = failure >= 2;
  row_lineage result;
  std::vector<int64_t> values;
  {
    detail::read_retirement_test_hooks hooks(
      failure == 1 || failure == 2 ? std::optional{detail::read_checkpoint::COPY} : std::nullopt,
      unjoined);
    auto transfer = [&] {
      if (operation == 0)
        result = source.select(std::array<int64_t, 3>{7, 1, 5}, stream, mr);
      else if (operation == 1)
        result = source.select(std::array<int64_t, 2>{2, 0}, stream, mr);
      else
        values = source.ordinals_on(stream);
    };
    if (unjoined)
      CHECK_THROWS_AS(transfer(), sirius::unjoined_gpu_work);
    else if (failure)
      CHECK_THROWS_AS(transfer(), rmm::out_of_memory);
    else
      REQUIRE_NOTHROW(transfer());
    CHECK(hooks.checkpoint_reached == (failure == 1 || failure == 2));
    CHECK(hooks.retained.has_value() == unjoined);
    CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
    source = {};
    if (operation == 2) CHECK(source_life.expired() == !unjoined);
    if (failure) CHECK((resource.state->live != 0) == unjoined);
  }
  CHECK(source_life.expired());
  if (!failure) {
    if (operation == 1)
      CHECK(result.ordinals_on(stream) == std::vector<int64_t>{5, 7});
    else if (operation == 0)
      CHECK(result.ordinals_on(stream) == std::vector<int64_t>{7, 1, 5});
    else
      CHECK(values == std::vector<int64_t>{7, 1, 5});
  }
  result = {};
  CHECK(resource.state->live == 0);
  CHECK_FALSE(resource.state->wrong_stream);
}

TEST_CASE("Lineage launch sizing rejects grids that cannot cover every row", "[scan][lineage]")
{
  for (auto rows :
       {int64_t{0}, int64_t{1}, int64_t{255}, int64_t{256}, int64_t{257}, int64_t{INT32_MAX}}) {
    CAPTURE(rows);
    auto blocks = detail::lineage_grid_size(rows);
    CHECK(int64_t{blocks} * 256 >= rows);
    CHECK((blocks == 0 || int64_t{blocks - 1} * 256 < rows));
  }
  int device, max_grid;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  REQUIRE(cudaDeviceGetAttribute(&max_grid, cudaDevAttrMaxGridDimX, device) == cudaSuccess);
  auto const limit = int64_t{max_grid} * 256;
  CHECK(detail::lineage_grid_size(limit) == static_cast<unsigned int>(max_grid));
  CHECK_THROWS_AS(detail::lineage_grid_size(limit + 1), std::invalid_argument);
  CHECK_THROWS_AS(detail::lineage_grid_size((int64_t{1} << 40) + 256), std::invalid_argument);
  CHECK_THROWS_AS(detail::lineage_grid_size(INT64_MAX), std::invalid_argument);
  CHECK_THROWS_AS(detail::lineage_grid_size(-1), std::invalid_argument);
}

TEST_CASE("Compact composition validates the entire sequence without device allocation",
          "[scan][lineage]")
{
  auto const rows = GENERATE(int64_t{0},
                             int64_t{1},
                             int64_t{255},
                             int64_t{256},
                             int64_t{257},
                             (int64_t{1} << 40) + 256,
                             int64_t{INT64_MAX - 2});
  CAPTURE(rows);
  rmm::cuda_stream stream;
  sirius::test::tracked_scan_resource resource(cudf::get_current_device_resource_ref(), stream);
  resource.state->fail_at = 0;
  rmm::device_async_resource_ref mr{resource};
  auto before          = identity("A", rows + 2, 0, rows);
  auto delivered       = std::make_shared<lineage_domain>(*before.domain());
  auto const split     = std::min(int64_t{256}, rows);
  delivered->file_runs = {{0, 0, split}, {0, split, rows - split}, {0, rows, 0}};
  row_lineage step{row_position_state::identity, delivered};
  auto result = row_lineage::compose(before, step, stream, mr);
  CHECK(result.output_length() == rows);
  CHECK(result.state() == row_position_state::identity);
  CHECK_FALSE(result.per_row());
  if (rows > 256) {
    // The old narrowed grid checked only the first 256 rows for 2^40 + 256.
    ++delivered->file_runs[1].first;
    CHECK_THROWS_AS(row_lineage::compose(before, step, stream, mr), std::invalid_argument);
  }
  CHECK(resource.state->attempts == 0);
}

TEST_CASE("Compact composition preserves reordered selections across files and intervals",
          "[scan][lineage]")
{
  rmm::cuda_stream stream;
  sirius::test::tracked_scan_resource resource(cudf::get_current_device_resource_ref(), stream);
  resource.state->fail_at = 0;
  rmm::device_async_resource_ref mr{resource};
  auto const large  = (int64_t{1} << 40);
  auto domain       = std::make_shared<lineage_domain>();
  domain->files     = {{"A", INT64_MAX}, {"B", 1000}};
  domain->file_runs = {{0, INT64_MAX - large, large}, {1, 10, 100}};
  row_lineage before{row_position_state::runs, domain, {{large, 100}, {0, large}}};
  auto delivered       = std::make_shared<lineage_domain>();
  delivered->files     = {{"B", 1000}, {"A", INT64_MAX}};
  delivered->file_runs = {
    {0, 10, 20}, {0, 30, 80}, {1, INT64_MAX - large, large - 1}, {1, INT64_MAX - 1, 1}};
  row_lineage step{row_position_state::runs, delivered, {{90, large + 10}, {0, 10}}};
  auto result = row_lineage::compose(before, step, stream, mr);
  CHECK(result.domain() == domain);
  std::array<int64_t, 6> output_indices{0, 9, 10, large + 9, large + 10, large + 19};
  std::array<int64_t, 6> expected_ordinals{large + 90, large + 99, 0, large - 1, large, large + 9};
  for (std::size_t i = 0; i < output_indices.size(); ++i) {
    auto index = output_indices[i];
    bool found = false;
    for (auto const& [first, count] : result.runs()) {
      if (index < count) {
        CHECK(first + index == expected_ordinals[i]);
        found = true;
        break;
      }
      index -= count;
    }
    CHECK(found);
  }
  CHECK(result.output_length() == large + 20);
  CHECK_FALSE(result.per_row());
  // A mismatch in the last interval must not hide behind the enormous preceding interval.
  --delivered->file_runs.back().first;
  CHECK_THROWS_AS(row_lineage::compose(before, step, stream, mr), std::invalid_argument);
  CHECK(resource.state->attempts == 0);
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
  REQUIRE(result.domain() == domain);
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
  REQUIRE(joined.state() == row_position_state::identity);
  REQUIRE_FALSE(joined.per_row());
  REQUIRE(joined.runs().empty());
  REQUIRE(joined.input_domain_length() == 2000000000);
  for (auto state : {row_position_state::none, row_position_state::lost}) {
    row_lineage unavailable;
    unavailable = {state, {}};
    unavailable.validate(100);
    REQUIRE(
      unavailable.select(std::array<int64_t, 1>{0}, static_cast<::cuda::stream_ref>(stream), mr)
        .state() == state);
    std::array mixed{parts.front(), unavailable};
    REQUIRE(row_lineage::concatenate(mixed, static_cast<::cuda::stream_ref>(stream), mr).state() ==
            row_position_state::lost);
  }
  auto bad                  = identity("A", 2);
  auto wrong                = std::make_shared<lineage_domain>(*bad.domain());
  wrong->file_runs[0].count = 3;
  bad                       = {bad.state(), wrong, bad.runs(), bad.per_row()};
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
  CHECK(lineage.per_row()->view().head<int64_t>() == original_storage);
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
  REQUIRE(first.state() == row_position_state::per_row);
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
  CHECK(composed.domain() == domain);
  auto ordinals = composed.ordinals_on(stream);
  REQUIRE(ordinals.size() == expected.size());
  for (std::size_t i = 0; i < ordinals.size(); ++i)
    CHECK(composed.resolve(ordinals[i]) == expected[i]);
  CHECK(composed.input_domain_length() == 2000);
  // A legal-sized but semantically foreign step must fail before use.
  auto foreign = std::make_shared<lineage_domain>(*next_domain);
  ++foreign->file_runs[0].first;
  next = {next.state(), foreign, next.runs(), next.per_row()};
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
  CHECK(joined.state() == row_position_state::identity);
  auto selected = joined.select(std::array<int64_t, 3>{0, 1, 2}, stream, mr);
  CHECK(selected.state() == row_position_state::runs);
  CHECK(resource.state->attempts == 0);
  for (auto unavailable : {row_position_state::lost, row_position_state::none}) {
    row_lineage before;
    before = {unavailable, before.domain()};
    CHECK(row_lineage::compose(before, a, stream, mr).state() == unavailable);
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
    CHECK(converted.per_row()->view().head() == buffer);
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
    REQUIRE(converted.domain()->files.size() == 1);
    REQUIRE(converted.domain()->file_runs.size() == 2);
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

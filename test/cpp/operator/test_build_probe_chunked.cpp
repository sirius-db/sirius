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

// A BUILD_PROBE join whose build partition arrives as several chunks (a build above
// operator_params::build_chunk_max_rows rows): every chunk is built before any probe, and the
// probe combines the chunks' results as if the build were one table.

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"
#include "utils/test_physical_operator.hpp"

#include <cudf/concatenate.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;

namespace {

struct finishable_pipeline : sirius::pipeline::sirius_pipeline {
  explicit finishable_pipeline(const sirius::pipeline::pipeline_build_context& ctx)
    : sirius_pipeline(ctx)
  {
  }

  [[nodiscard]] bool is_pipeline_finished() const override { return finished; }

  bool finished = false;
};

//! Depth-first, root-first numbering of a bare operator tree, standing in for
//! pipeline::assign_operator_ids in fixtures that never build pipelines.
void number_operator_tree(sirius_physical_operator& op, size_t& next_id)
{
  op.operator_id = next_id++;
  for (auto& child : op.children) {
    if (child) { number_operator_tree(*child, next_id); }
  }
}

cucascade::memory::memory_space& gpu_space() { return *get_default_gpu_space(); }

/// A join on probe (key INT, payload INT) = build (key INT), in BUILD_PROBE mode with one
/// partition, its ports fed by pipelines the test finishes.
struct chunked_join {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> join;
  cucascade::shared_data_repository build_repo;
  cucascade::shared_data_repository probe_repo;
  std::shared_ptr<finishable_pipeline> build_pipeline;
  std::shared_ptr<finishable_pipeline> probe_pipeline;
  sirius::pipeline::sirius_pipeline_build_state build_state;
};

std::unique_ptr<chunked_join> make_chunked_join(duckdb::JoinType join_type)
{
  using duckdb::LogicalType;
  auto f                 = std::make_unique<chunked_join>();
  f->logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  f->logical_join->types = {LogicalType::INTEGER, LogicalType::INTEGER};
  if (join_type == duckdb::JoinType::MARK) {
    f->logical_join->types.push_back(LogicalType::BOOLEAN);
  }
  if (join_type == duckdb::JoinType::INNER) {
    f->logical_join->types.push_back(LogicalType::INTEGER);
  }
  auto probe_child =
    duckdb::make_uniq<test_physical_operator>(SiriusPhysicalOperatorType::PROJECTION,
                                              sirius::from_duckdb_vec(duckdb::vector<LogicalType>{
                                                LogicalType::INTEGER, LogicalType::INTEGER}),
                                              0);
  auto build_child = duckdb::make_uniq<test_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<LogicalType>{LogicalType::INTEGER}),
    0);
  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.emplace_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::INTEGER, 0),
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::INTEGER, 0),
    duckdb::ExpressionType::COMPARE_EQUAL);
  f->join = duckdb::make_uniq<sirius_physical_hash_join>(
    *f->logical_join,
    std::move(probe_child),
    std::move(build_child),
    sirius::wrap_join_conditions(std::move(conditions)),
    join_type,
    duckdb::vector<duckdb::idx_t>{},
    duckdb::vector<duckdb::idx_t>{},
    sirius::from_duckdb_vec(duckdb::vector<LogicalType>{}),
    1000);
  size_t next_id = 0;
  number_operator_tree(*f->join, next_id);

  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  f->build_pipeline = std::make_shared<finishable_pipeline>(ctx);
  f->probe_pipeline = std::make_shared<finishable_pipeline>(ctx);
  f->build_state.add_pipeline_operator(*f->build_pipeline, *f->join->children[1]);
  f->build_state.add_pipeline_operator(*f->probe_pipeline, *f->join->children[0]);
  auto attach = [&](std::string_view name,
                    cucascade::shared_data_repository& repo,
                    std::shared_ptr<finishable_pipeline> source) {
    auto port          = std::make_unique<sirius_physical_operator::port>();
    port->type         = MemoryBarrierType::PARTIAL;
    port->repo         = &repo;
    port->src_pipeline = std::move(source);
    f->join->add_port(name, std::move(port));
  };
  attach("build", f->build_repo, f->build_pipeline);
  attach("default", f->probe_repo, f->probe_pipeline);

  // Above the small-table threshold, so no broadcast; well under the BUILD_PROBE budget.
  // BUILD_PROBE admission needs a measured row count; model an INTEGER build.
  uint64_t const build_bytes = 64ULL << 20;
  uint64_t const build_rows  = build_bytes / sizeof(int32_t);
  auto const strategy        = f->join->get_partition_strategy(partition_sizing_input{
    build_bytes, true, true, build_bytes, build_rows, build_rows, build_bytes});
  REQUIRE(strategy.build_probe);
  REQUIRE_FALSE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 1);
  f->join->set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));
  return f;
}

std::shared_ptr<cucascade::data_batch> int_batch(std::vector<int32_t> const& values)
{
  return make_numeric_batch<int32_t>(gpu_space(), values, cudf::type_id::INT32);
}

std::shared_ptr<cucascade::data_batch> int_batch_with_nulls(std::vector<int32_t> const& values,
                                                            std::vector<bool> const& valids)
{
  return make_numeric_batch_with_nulls<int32_t>(gpu_space(), values, valids, cudf::type_id::INT32);
}

std::shared_ptr<cucascade::data_batch> probe_batch(std::shared_ptr<cucascade::data_batch> keys,
                                                   std::vector<int32_t> const& payload)
{
  return concatenate_batches_horizontal({std::move(keys), int_batch(payload)}, gpu_space());
}

struct probe_result {
  std::unique_ptr<operator_data> output;
  std::size_t num_batches = 0;
  std::unique_ptr<cudf::table> merged;  // the output batches' rows, when there are several
  cudf::table_view view;
};

/// Delivers @p chunks as a split build and @p probe as the first probe batch, checks that the
/// join waits for the build producer to finish before building, then returns the build+probe task.
std::unique_ptr<operator_data> deliver_chunked(
  chunked_join& f,
  std::vector<std::shared_ptr<cucascade::data_batch>> const& chunks,
  std::shared_ptr<cucascade::data_batch> probe)
{
  f.join->note_build_chunked();
  for (auto const& chunk : chunks) {
    f.build_repo.add_data_batch(chunk, 0);
  }
  f.probe_repo.add_data_batch(std::move(probe), 0);

  // More chunks may still come until the build producer finishes.
  auto hint = f.join->get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK(hint->producer == f.join->children[1].get());

  f.build_pipeline->finished = true;
  hint                       = f.join->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  auto input = f.join->get_next_task_input_data();
  REQUIRE(input != nullptr);
  REQUIRE(dynamic_cast<pipelineable_operator_data&>(*input).get_data_batches().size() ==
          chunks.size() + 1);
  CHECK(f.build_repo.total_size() == 0);
  return input;
}

probe_result collect(std::unique_ptr<operator_data> output)
{
  probe_result result;
  result.output = std::move(output);
  auto const& batches =
    dynamic_cast<pipelineable_operator_data const&>(*result.output).get_data_batches();
  REQUIRE_FALSE(batches.empty());
  result.num_batches = batches.size();
  if (batches.size() == 1) {
    result.view = sirius::get_cudf_table_view(*batches[0]);
    return result;
  }
  std::vector<cudf::table_view> views;
  for (auto const& batch : batches) {
    views.push_back(sirius::get_cudf_table_view(*batch));
  }
  result.merged = cudf::concatenate(views);
  result.view   = result.merged->view();
  return result;
}

probe_result run_chunked(chunked_join& f,
                         std::vector<std::shared_ptr<cucascade::data_batch>> const& chunks,
                         std::shared_ptr<cucascade::data_batch> probe)
{
  auto input = deliver_chunked(f, chunks, std::move(probe));
  return collect(f.join->execute(*input, cudf::get_default_stream()));
}

/// A memory manager whose allocator is the current device resource, which the join allocates
/// from, while the caller holds it. Each manager installs its allocator over the previous one, and
/// the shared fixture space and other suites keep theirs: create the shared one first.
std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_current_manager()
{
  gpu_space();
  return sirius::test::operator_utils::initialize_memory_manager();
}

/// Counts @p stream's allocations against a reservation of @p bytes on @p manager's GPU, past
/// which an allocation throws rmm::out_of_memory, as under a task's fail policy.
struct reserved_stream {
  rmm::cuda_stream_view stream;
  cucascade::memory::reservation_aware_resource_adaptor* allocator = nullptr;

  reserved_stream(sirius::memory::sirius_memory_reservation_manager& manager,
                  rmm::cuda_stream_view stream,
                  std::size_t bytes)
    : stream(stream)
  {
    auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, 0);
    REQUIRE(space);
    auto reservation = space->make_reservation_or_null(bytes);
    REQUIRE(reservation);
    allocator =
      reservation->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(allocator);
    REQUIRE(allocator->attach_reservation_to_tracker(
      stream,
      std::move(reservation),
      std::make_unique<cucascade::memory::fail_reservation_limit_policy>(),
      nullptr));
  }
  ~reserved_stream() { release(); }

  [[nodiscard]] std::size_t allocated_bytes() const
  {
    return allocator->get_allocated_bytes(stream);
  }

  /// Drop the reservation, as a retry that was granted more memory would run without this cap.
  void release()
  {
    if (allocator == nullptr) { return; }
    allocator->reset_stream_reservation(stream);
    allocator = nullptr;
  }
};

std::vector<int32_t> iota_keys(int32_t n)
{
  std::vector<int32_t> keys(n);
  std::iota(keys.begin(), keys.end(), 0);
  return keys;
}

std::vector<int32_t> times_ten(std::vector<int32_t> values)
{
  for (auto& value : values) {
    value *= 10;
  }
  return values;
}

std::vector<int32_t> sorted(std::vector<int32_t> values)
{
  std::sort(values.begin(), values.end());
  return values;
}

}  // namespace

TEST_CASE("chunked BUILD_PROBE MARK join marks a row matched by any chunk", "[build_probe]")
{
  auto f      = make_chunked_join(duckdb::JoinType::MARK);
  auto result = run_chunked(
    *f, {int_batch({2}), int_batch({4})}, probe_batch(int_batch({1, 2, 3, 4}), {10, 20, 30, 40}));
  REQUIRE(result.view.num_rows() == 4);
  CHECK(copy_column_to_host<int32_t>(result.view.column(1)) ==
        std::vector<int32_t>{10, 20, 30, 40});
  auto const mark = result.view.column(2);
  CHECK(mark.null_count() == 0);
  CHECK(copy_column_to_host<bool>(mark) == std::vector<bool>{false, true, false, true});
}

TEST_CASE("chunked BUILD_PROBE MARK join: a NULL key in one chunk makes every non-match NULL",
          "[build_probe]")
{
  auto f          = make_chunked_join(duckdb::JoinType::MARK);
  auto result     = run_chunked(*f,
                                {int_batch({2}), int_batch_with_nulls({4, 0}, {true, false})},
                            probe_batch(int_batch({1, 2, 3, 4}), {10, 20, 30, 40}));
  auto const mark = result.view.column(2);
  CHECK(mark.null_count() == 2);
  CHECK(copy_validity_to_host(mark) == std::vector<bool>{false, true, false, true});
  auto const values = copy_column_to_host<bool>(mark);
  CHECK(values[1]);
  CHECK(values[3]);
}

TEST_CASE("chunked BUILD_PROBE MARK join: an empty chunk does not make the build empty",
          "[build_probe]")
{
  auto f = make_chunked_join(duckdb::JoinType::MARK);
  auto result =
    run_chunked(*f,
                {int_batch({}), int_batch({2})},
                probe_batch(int_batch_with_nulls({1, 0, 2}, {true, false, true}), {10, 20, 30}));
  // A NULL probe key against a nonempty build is NULL, not false.
  auto const mark = result.view.column(2);
  CHECK(copy_validity_to_host(mark) == std::vector<bool>{true, false, true});
  auto const values = copy_column_to_host<bool>(mark);
  CHECK_FALSE(values[0]);
  CHECK(values[2]);
}

TEST_CASE("chunked BUILD_PROBE SEMI join emits a row matched by two chunks once", "[build_probe]")
{
  auto f      = make_chunked_join(duckdb::JoinType::SEMI);
  auto result = run_chunked(*f,
                            {int_batch({2, 4}), int_batch({4, 6})},
                            probe_batch(int_batch({1, 2, 4, 6, 7}), {10, 20, 40, 60, 70}));
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(0))) ==
        std::vector<int32_t>{2, 4, 6});
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(1))) ==
        std::vector<int32_t>{20, 40, 60});
}

TEST_CASE("chunked BUILD_PROBE ANTI join emits only rows no chunk matches", "[build_probe]")
{
  auto f      = make_chunked_join(duckdb::JoinType::ANTI);
  auto result = run_chunked(*f,
                            {int_batch({2, 4}), int_batch({4, 6})},
                            probe_batch(int_batch({1, 2, 4, 6, 7}), {10, 20, 40, 60, 70}));
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(0))) == std::vector<int32_t>{1, 7});
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(1))) ==
        std::vector<int32_t>{10, 70});
}

TEST_CASE("chunked BUILD_PROBE INNER join joins every chunk's build rows", "[build_probe]")
{
  auto f      = make_chunked_join(duckdb::JoinType::INNER);
  auto result = run_chunked(
    *f, {int_batch({2, 4}), int_batch({4, 6})}, probe_batch(int_batch({2, 4, 7}), {20, 40, 70}));
  // One batch per chunk's join, not one concatenated batch.
  CHECK(result.num_batches == 2);
  // Build key 4 appears once in each chunk, so probe row 4 joins twice, as with one table.
  REQUIRE(result.view.num_columns() == 3);
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(0))) ==
        std::vector<int32_t>{2, 4, 4});
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(1))) ==
        std::vector<int32_t>{20, 40, 40});
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(2))) ==
        std::vector<int32_t>{2, 4, 4});
}

TEST_CASE("a chunked BUILD_PROBE build that fails on a later chunk frees the chunks it built",
          "[build_probe]")
{
  // Chunk 2's table needs far more than the reservation; chunk 1's fits in it.
  constexpr int32_t big_chunk_rows = 1 << 20;
  auto manager                     = make_current_manager();
  rmm::cuda_stream stream;  // outlives every table built on it
  auto f     = make_chunked_join(duckdb::JoinType::SEMI);
  auto input = deliver_chunked(*f,
                               {int_batch({2, 4}), int_batch(iota_keys(big_chunk_rows))},
                               probe_batch(int_batch({1, 2, 4, 7}), {10, 20, 40, 70}));
  reserved_stream capped(*manager, stream, 1 << 20);
  REQUIRE_THROWS_AS(f->join->execute(*input, stream.view()), std::bad_alloc);
  // An OOM retry reserves its memory before it runs again: chunk 1's table must not still count.
  CHECK(capped.allocated_bytes() == 0);
  CHECK(f->join->snapshot_build_probe_slots()[0].state == BUILD_HASH_TABLE_STATE::SCHEDULED);

  capped.release();
  auto output = f->join->execute(*input, stream.view());
  stream.synchronize();  // the checks below copy on the default stream
  auto result = collect(std::move(output));
  CHECK(sorted(copy_column_to_host<int32_t>(result.view.column(0))) ==
        std::vector<int32_t>{1, 2, 4, 7});
}

TEST_CASE("a chunked BUILD_PROBE probe that fails keeps the built chunks for the retry",
          "[build_probe]")
{
  // The build fits in the reservation; probing this many rows does not.
  constexpr int32_t probe_rows = 1 << 20;
  auto manager                 = make_current_manager();
  rmm::cuda_stream stream;  // outlives every table built on it
  auto keys  = iota_keys(probe_rows);
  auto f     = make_chunked_join(duckdb::JoinType::SEMI);
  auto input = deliver_chunked(
    *f, {int_batch({2, 4}), int_batch({4, 6})}, probe_batch(int_batch(keys), times_ten(keys)));
  reserved_stream capped(*manager, stream, 1 << 20);
  REQUIRE_THROWS_AS(f->join->execute(*input, stream.view()), std::bad_alloc);
  CHECK(f->join->snapshot_build_probe_slots()[0].state == BUILD_HASH_TABLE_STATE::BUILT);
  CHECK(capped.allocated_bytes() > 0);

  capped.release();
  auto output = f->join->execute(*input, stream.view());
  stream.synchronize();  // the checks below copy on the default stream
  auto result        = collect(std::move(output));
  auto const payload = copy_column_to_host<int32_t>(result.view.column(1));
  CHECK(copy_column_to_host<int32_t>(result.view.column(0)) == std::vector<int32_t>{2, 4, 6});
  CHECK(payload == std::vector<int32_t>{20, 40, 60});
}

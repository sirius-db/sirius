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

#include "catch.hpp"
#include "helper/type_conversions.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/merge/merge_chunk_position.hpp>
#include <op/merge/merge_config.hpp>
#include <op/merge/windowed_merge_input.hpp>
#include <op/sirius_physical_merge_sort.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

namespace {

constexpr std::size_t kGiB = std::size_t{1} << 30;

/// A fixed device budget, so window sizes do not depend on the test memory manager.
merge::device_budget large_budget()
{
  return {.gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 1};
}

duckdb::vector<duckdb::BoundOrderByNode> key_on_column_0()
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(duckdb::BoundOrderByNode(
    duckdb::OrderType::ASCENDING,
    duckdb::OrderByNullType::NULLS_LAST,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0)));
  return orders;
}

duckdb::vector<sirius::logical_type> bigint_types(std::size_t n)
{
  duckdb::vector<duckdb::LogicalType> types;
  for (std::size_t i = 0; i < n; ++i) {
    types.push_back(duckdb::LogicalType::BIGINT);
  }
  return sirius::from_duckdb_vec(std::move(types));
}

template <class Op = sirius_physical_merge_sort>
std::unique_ptr<Op> make_merge(data_repository& repo, std::size_t num_columns = 1)
{
  duckdb::vector<std::size_t> projections(num_columns);
  std::iota(projections.begin(), projections.end(), std::size_t{0});
  auto op =
    std::make_unique<Op>(bigint_types(num_columns), key_on_column_0(), std::move(projections), 0);
  op->add_port("default",
               std::make_unique<sirius_physical_operator::port>(
                 MemoryBarrierType::FULL, &repo, nullptr, nullptr));
  return op;
}

/// Sorted single-column INT64 batch of @p rows rows.
std::shared_ptr<data_batch> make_sorted_batch(memory_space& space, std::size_t rows)
{
  std::vector<int64_t> values(rows);
  std::iota(values.begin(), values.end(), int64_t{0});
  return make_numeric_batch<int64_t>(space, values, cudf::type_id::INT64);
}

/// INT64 key column plus a LIST<INT64> column with one element per row.
std::shared_ptr<data_batch> make_key_and_list_batch(memory_space& space, std::size_t rows)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();
  auto size   = static_cast<cudf::size_type>(rows);

  std::vector<int64_t> values(rows);
  std::iota(values.begin(), values.end(), int64_t{0});
  std::vector<int32_t> offsets(rows + 1);
  std::iota(offsets.begin(), offsets.end(), int32_t{0});

  auto make_int64 = [&] {
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, size, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpy(col->mutable_view().data<int64_t>(),
               values.data(),
               sizeof(int64_t) * values.size(),
               cudaMemcpyHostToDevice);
    return col;
  };
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, size + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
             offsets.data(),
             sizeof(int32_t) * offsets.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_int64());
  cols.push_back(
    cudf::make_lists_column(size, std::move(offsets_col), make_int64(), 0, rmm::device_buffer{}));
  return ::sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                   space,
                                   stream,
                                   ::sirius::telemetry::batch_telemetry_info{});
}

std::vector<std::uint64_t> ids_of(const std::vector<std::shared_ptr<data_batch>>& batches)
{
  std::vector<std::uint64_t> ids;
  for (auto const& batch : batches) {
    ids.push_back(batch->get_batch_id());
  }
  return ids;
}

std::vector<std::uint64_t> cursor_ids(const merge::windowed_merge_input& input)
{
  std::vector<std::uint64_t> ids;
  for (auto const& cursor : input.cursors()) {
    REQUIRE(cursor.run.batch_ids.size() == 1);
    ids.push_back(cursor.run.batch_ids.front());
  }
  return ids;
}

/// Requires that @p input is a pop-all task holding @p batches at @p position.
void require_one_shot_input(const std::unique_ptr<operator_data>& input,
                            const std::vector<std::shared_ptr<data_batch>>& batches,
                            std::size_t position)
{
  REQUIRE(input);
  REQUIRE(input->get_type() == operator_data_type::PIPELINEABLE);
  auto& data = dynamic_cast<pipelineable_operator_data&>(*input);
  REQUIRE(data.get_data_batches() == batches);
  REQUIRE(data.get_position() == position);
  REQUIRE(data.materializes_on_gpu());
}

merge::windowed_merge_input& require_windowed_input(const std::unique_ptr<operator_data>& input)
{
  REQUIRE(input);
  REQUIRE(input->get_type() == operator_data_type::WINDOWED_MERGE);
  return dynamic_cast<merge::windowed_merge_input&>(*input);
}

class late_mat_merge_sort : public sirius_physical_merge_sort {
 public:
  using sirius_physical_merge_sort::sirius_physical_merge_sort;

 protected:
  [[nodiscard]] bool has_late_materialization() const noexcept override { return true; }
};

}  // namespace

TEST_CASE("merge sort one_shot mode pops each partition at position = partition",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  repo.set_num_partitions(4);
  auto a = make_sorted_batch(*space, 4);
  auto b = make_sorted_batch(*space, 6);
  auto c = make_sorted_batch(*space, 2);
  repo.add_data_batch(a, 1);
  repo.add_data_batch(b, 1);
  repo.add_data_batch(c, 3);

  auto merge = make_merge(repo);
  REQUIRE(merge->get_merge_settings().mode == sirius::sort_merge_mode::one_shot);

  auto first = merge->get_next_task_input_data();
  require_one_shot_input(first, {a, b}, 1);
  REQUIRE(repo.size(1) == 0);
  REQUIRE(repo.size(3) == 1);

  auto second = merge->get_next_task_input_data();
  require_one_shot_input(second, {c}, 3);
  REQUIRE(repo.total_size() == 0);
  REQUIRE_FALSE(merge->get_next_task_input_data());

  // A partition that was drained is not armed again.
  repo.add_data_batch(make_sorted_batch(*space, 3), 1);
  REQUIRE_FALSE(merge->get_next_task_input_data());
}

TEST_CASE("merge sort automatic mode merges small GPU partitions one-shot at chunk positions",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  repo.set_num_partitions(3);
  auto a = make_sorted_batch(*space, 8);
  auto b = make_sorted_batch(*space, 8);
  auto c = make_sorted_batch(*space, 8);
  repo.add_data_batch(a, 0);
  repo.add_data_batch(b, 0);
  repo.add_data_batch(c, 2);

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic});
  merge->set_device_budget_provider(large_budget);

  auto first = merge->get_next_task_input_data();
  require_one_shot_input(first, {a, b}, merge::make_merge_chunk_position(0, 0));
  auto second = merge->get_next_task_input_data();
  require_one_shot_input(second, {c}, merge::make_merge_chunk_position(2, 0));
  REQUIRE(repo.total_size() == 0);
  REQUIRE_FALSE(merge->get_next_task_input_data());
}

TEST_CASE("merge sort automatic mode goes windowed when the partition exceeds the threshold",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  auto a = make_sorted_batch(*space, 1024);
  auto b = make_sorted_batch(*space, 1024);
  repo.add_data_batch(a);
  repo.add_data_batch(b);

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic});
  // 16 KiB of runs plus their output do not fit the one-shot share of a 4 KiB device.
  merge->set_device_budget_provider([] {
    return merge::device_budget{
      .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 4096, .concurrent_tasks = 1};
  });

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.cursors().size() == 2);
  REQUIRE(repo.size(0) == 2);
}

TEST_CASE("merge sort automatic mode goes windowed when a run is not on the GPU",
          "[merge_sort_arming]")
{
  tiered_memory_fixture fixture;
  REQUIRE(fixture.setup());
  data_repository repo;
  // The host conversion needs a stream of the space, not the default stream.
  auto const stream = fixture.gpu0->acquire_stream();
  auto gpu_batch    = fixture.make_gpu_batch(stream);
  auto host_batch   = fixture.make_host_batch(stream);
  REQUIRE(host_batch->to_read_only().get_current_tier() == Tier::HOST);
  repo.add_data_batch(gpu_batch);
  repo.add_data_batch(host_batch);

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic});
  merge->set_device_budget_provider(large_budget);

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(cursor_ids(windowed) == ids_of({gpu_batch, host_batch}));
  REQUIRE(windowed.final_pass());
  REQUIRE(windowed.get_estimated_size_in_bytes() > 0);
  // Arming reads sizes only: the spilled run stays where it is.
  REQUIRE(host_batch->to_read_only().get_current_tier() == Tier::HOST);
  REQUIRE(repo.size(0) == 2);
}

TEST_CASE("merge sort windowed mode arms by peeking, one task per partition", "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  repo.set_num_partitions(3);
  auto a = make_sorted_batch(*space, 100);
  auto b = make_sorted_batch(*space, 300);
  auto c = make_sorted_batch(*space, 200);
  auto d = make_sorted_batch(*space, 50);
  repo.add_data_batch(a, 0);
  repo.add_data_batch(b, 0);
  repo.add_data_batch(c, 0);
  repo.add_data_batch(d, 2);

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  merge->set_device_budget_provider(large_budget);

  auto first     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(first);
  REQUIRE(windowed.partition() == 0);
  REQUIRE(windowed.first_chunk() == 0);
  REQUIRE(windowed.final_pass());
  REQUIRE(windowed.config().window_rows > 0);
  REQUIRE(windowed.config().output_target_bytes > 0);
  REQUIRE(windowed.get_estimated_size_in_bytes() > 0);
  REQUIRE(windowed.get_preferred_device_id() == space->get_device_id());

  // One single-batch cursor per batch, by ascending batch id, at the start of its run.
  REQUIRE(cursor_ids(windowed) == ids_of({a, b, c}));
  const std::vector<std::size_t> rows{100, 300, 200};
  for (std::size_t i = 0; i < windowed.cursors().size(); ++i) {
    auto const& cursor = windowed.cursors()[i];
    REQUIRE(cursor.run.batch_rows == std::vector<std::size_t>{rows[i]});
    REQUIRE(cursor.batch == 0);
    REQUIRE(cursor.row == 0);
    REQUIRE(cursor.remaining() == rows[i]);
  }

  // The runs were peeked, not popped.
  REQUIRE(repo.size(0) == 3);
  for (auto const& batch : {a, b, c}) {
    REQUIRE(repo.get_data_batch_by_id(batch->get_batch_id(), 0) == batch);
  }

  // Partition 0 is in flight: the next arm skips it and takes partition 2.
  auto second       = merge->get_next_task_input_data();
  auto& second_data = require_windowed_input(second);
  REQUIRE(second_data.partition() == 2);
  REQUIRE(cursor_ids(second_data) == ids_of({d}));
  REQUIRE(second_data.final_pass());
  REQUIRE(repo.size(2) == 1);

  // Both partitions are in flight.
  REQUIRE_FALSE(merge->get_next_task_input_data());
  REQUIRE(repo.size(0) == 3);
  REQUIRE(repo.size(2) == 1);
}

TEST_CASE("merge sort with more runs than the fan-in cap plans an intermediate pass",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  const std::vector<std::size_t> rows{500, 100, 400, 200, 300};
  std::vector<std::shared_ptr<data_batch>> batches;
  for (auto n : rows) {
    batches.push_back(make_sorted_batch(*space, n));
    repo.add_data_batch(batches.back());
  }

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed, .max_fan_in = 3});
  merge->set_device_budget_provider(large_budget);

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE_FALSE(windowed.final_pass());
  REQUIRE(windowed.partition() == 0);

  // The three smallest runs, still in batch id order.
  REQUIRE(cursor_ids(windowed) == ids_of({batches[1], batches[3], batches[4]}));
  REQUIRE(windowed.cursors()[0].run.batch_rows == std::vector<std::size_t>{100});
  REQUIRE(windowed.cursors()[1].run.batch_rows == std::vector<std::size_t>{200});
  REQUIRE(windowed.cursors()[2].run.batch_rows == std::vector<std::size_t>{300});

  REQUIRE(repo.size(0) == rows.size());
  REQUIRE_FALSE(merge->get_next_task_input_data());
}

TEST_CASE("merge sort with at most fan-in runs plans a final pass", "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  for (std::size_t n : {30, 10, 20}) {
    repo.add_data_batch(make_sorted_batch(*space, n));
  }

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed, .max_fan_in = 3});
  merge->set_device_budget_provider(large_budget);

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.final_pass());
  REQUIRE(windowed.cursors().size() == 3);
}

TEST_CASE("merge sort merges one-shot when a fetched column is nested", "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  repo.set_num_partitions(2);
  auto a = make_key_and_list_batch(*space, 16);
  auto b = make_key_and_list_batch(*space, 16);
  repo.add_data_batch(a, 1);
  repo.add_data_batch(b, 1);

  auto merge = make_merge(repo, 2);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  merge->set_device_budget_provider(large_budget);

  auto input = merge->get_next_task_input_data();
  require_one_shot_input(input, {a, b}, merge::make_merge_chunk_position(1, 0));
  REQUIRE(repo.total_size() == 0);
  REQUIRE_FALSE(merge->get_next_task_input_data());
}

TEST_CASE("merge sort merges one-shot when late materialization is installed",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  auto a = make_sorted_batch(*space, 16);
  auto b = make_sorted_batch(*space, 16);
  repo.add_data_batch(a);
  repo.add_data_batch(b);

  auto merge = make_merge<late_mat_merge_sort>(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  merge->set_device_budget_provider(large_budget);

  auto input = merge->get_next_task_input_data();
  require_one_shot_input(input, {a, b}, merge::make_merge_chunk_position(0, 0));
  REQUIRE(repo.total_size() == 0);
}

TEST_CASE("merge sort input residency policy follows the merge mode", "[merge_sort_arming]")
{
  data_repository repo;
  auto merge = make_merge(repo);

  REQUIRE(merge->input_residency_policy("default") == port_residency_policy{});

  const port_residency_policy on_demand{.materialize_inputs_on_gpu = false,
                                        .downgrade_priority        = 100};
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic});
  REQUIRE(merge->input_residency_policy("default") == on_demand);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  REQUIRE(merge->input_residency_policy("default") == on_demand);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::one_shot});
  REQUIRE(merge->input_residency_policy("default") == port_residency_policy{});
}

TEST_CASE("merge sort windowed arm pops zero-row batches and gives them no cursor",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  auto a     = make_sorted_batch(*space, 40);
  auto empty = make_sorted_batch(*space, 0);
  auto b     = make_sorted_batch(*space, 60);
  repo.add_data_batch(a);
  repo.add_data_batch(empty);
  repo.add_data_batch(b);

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  merge->set_device_budget_provider(large_budget);

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(cursor_ids(windowed) == ids_of({a, b}));
  REQUIRE(repo.size(0) == 2);
  REQUIRE_FALSE(repo.get_data_batch_by_id(empty->get_batch_id(), 0));
  REQUIRE(repo.get_data_batch_by_id(a->get_batch_id(), 0) == a);
  REQUIRE(repo.get_data_batch_by_id(b->get_batch_id(), 0) == b);
}

TEST_CASE("merge sort merges a partition with no rows one-shot", "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  auto a = make_sorted_batch(*space, 0);
  auto b = make_sorted_batch(*space, 0);
  repo.add_data_batch(a);
  repo.add_data_batch(b);

  auto merge = make_merge(repo);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  merge->set_device_budget_provider(large_budget);

  auto input = merge->get_next_task_input_data();
  require_one_shot_input(input, {a, b}, merge::make_merge_chunk_position(0, 0));
  REQUIRE(repo.total_size() == 0);
}

TEST_CASE("merge sort ignores a nested column the merge does not fetch", "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  SECTION("final pass: the nested column is projected away, so the route stays windowed")
  {
    data_repository repo;
    auto a = make_key_and_list_batch(*space, 16);
    auto b = make_key_and_list_batch(*space, 16);
    repo.add_data_batch(a);
    repo.add_data_batch(b);

    auto merge = make_merge(repo, 2);
    merge->set_final_projections({0}, bigint_types(1));
    merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
    merge->set_device_budget_provider(large_budget);

    auto input     = merge->get_next_task_input_data();
    auto& windowed = require_windowed_input(input);
    REQUIRE(windowed.final_pass());
    REQUIRE(cursor_ids(windowed) == ids_of({a, b}));
    REQUIRE(repo.size(0) == 2);
  }

  SECTION("more runs than the fan-in cap: an intermediate pass rewrites it, so one-shot")
  {
    data_repository repo;
    std::vector<std::shared_ptr<data_batch>> batches;
    for (int i = 0; i < 3; ++i) {
      batches.push_back(make_key_and_list_batch(*space, 16));
      repo.add_data_batch(batches.back());
    }

    auto merge = make_merge(repo, 2);
    merge->set_final_projections({0}, bigint_types(1));
    merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed, .max_fan_in = 2});
    merge->set_device_budget_provider(large_budget);

    auto input = merge->get_next_task_input_data();
    require_one_shot_input(input, batches, merge::make_merge_chunk_position(0, 0));
    REQUIRE(repo.total_size() == 0);
  }
}

TEST_CASE("merge sort task hint is withheld while every partition with runs is in flight",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  repo.set_num_partitions(2);
  repo.add_data_batch(make_sorted_batch(*space, 10), 0);
  repo.add_data_batch(make_sorted_batch(*space, 10), 1);

  // A pipeline-barrier port with no source pipeline: the base hint is READY whenever the
  // repository holds data, so what follows is the operator's own gating.
  auto merge = std::make_unique<sirius_physical_merge_sort>(
    bigint_types(1), key_on_column_0(), duckdb::vector<std::size_t>{0}, 0);
  merge->add_port("default",
                  std::make_unique<sirius_physical_operator::port>(
                    MemoryBarrierType::PIPELINE, &repo, nullptr, nullptr));
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  merge->set_device_budget_provider(large_budget);

  auto ready = merge->get_next_task_hint();
  REQUIRE(ready.has_value());
  REQUIRE(ready->hint == TaskCreationHint::READY);

  auto first = merge->get_next_task_input_data();
  REQUIRE(require_windowed_input(first).partition() == 0);
  // Partition 1 has not been armed yet.
  auto still_ready = merge->get_next_task_hint();
  REQUIRE(still_ready.has_value());
  REQUIRE(still_ready->hint == TaskCreationHint::READY);

  auto second = merge->get_next_task_input_data();
  REQUIRE(require_windowed_input(second).partition() == 1);
  // Both partitions are in flight and their runs are still in the repository.
  REQUIRE(repo.total_size() == 2);
  REQUIRE_FALSE(merge->get_next_task_hint().has_value());
}

TEST_CASE("merge sort no-history estimate of a windowed task is its input's size",
          "[merge_sort_arming]")
{
  data_repository repo;
  auto merge = make_merge(repo);

  input_stats windowed{
    .num_batches = 3, .bytes = 12345, .type = operator_data_type::WINDOWED_MERGE};
  REQUIRE(merge->no_history_peak_memory_estimate(windowed) == 12345);

  // Any other input keeps the base operator's estimate.
  input_stats pipelineable{
    .num_batches = 3, .bytes = 12345, .type = operator_data_type::PIPELINEABLE};
  REQUIRE(merge->no_history_peak_memory_estimate(pipelineable) == 2 * 12345);
}

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
#include "data/sirius_row_range_reader_registry.hpp"
#include "exec/config.hpp"
#include "helper/type_conversions.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/merge/merge_chunk_position.hpp>
#include <op/merge/merge_config.hpp>
#include <op/merge/windowed_merge_input.hpp>
#include <op/sirius_physical_merge_sort.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;
namespace merge_utils = sirius::test::merge_utils;

namespace {

constexpr std::size_t kGiB = std::size_t{1} << 30;

/// A fixed device budget, so window sizes do not depend on the test memory manager.
merge::device_budget large_budget(const memory_space* = nullptr)
{
  return {.gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 1};
}

/// large_budget for two concurrent tasks, so two windowed partitions may be open at once.
merge::device_budget two_task_budget(const memory_space* = nullptr)
{
  return {.gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 2};
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

/// Longest a call that must not wait for a batch lock may take.
constexpr std::chrono::seconds kPrompt{5};

/// Runs @p call on another thread and requires that it returns within kPrompt.
template <typename Call>
auto require_prompt(Call call)
{
  auto result = std::async(std::launch::async, std::move(call));
  if (result.wait_for(kPrompt) != std::future_status::ready) {
    FAIL_CHECK("call did not return within " << kPrompt.count() << " s");
    std::abort();  // ~future would block forever on the stuck call
  }
  return result.get();
}

/// A MERGE_SORT over @p repo, windowed unless @p settings say otherwise, whose base task hint is
/// READY whenever the repository holds data, so what a test sees is the operator's own gating.
std::unique_ptr<sirius_physical_merge_sort> make_windowed_merge(
  data_repository& repo,
  merge::merge_settings settings = {.mode = sirius::sort_merge_mode::windowed})
{
  auto merge = std::make_unique<sirius_physical_merge_sort>(
    bigint_types(1), key_on_column_0(), duckdb::vector<std::size_t>{0}, 0);
  merge->add_port("default",
                  std::make_unique<sirius_physical_operator::port>(
                    MemoryBarrierType::PIPELINE, &repo, nullptr, nullptr));
  merge->set_merge_settings(settings);
  merge->set_device_budget_provider(large_budget);
  return merge;
}

/// Decoded bytes of @p batch as its current representation reports them.
std::size_t decoded_bytes(data_batch& batch)
{
  return batch.to_read_only().get_data()->get_uncompressed_data_size_in_bytes();
}

/// The estimate of a windowed task over runs of BIGINT columns only, @p row_width bytes per row,
/// with an INT64 key: @p unbuffered_runs of its runs are charged no window buffer, and
/// @p fallback_bytes is the sum of the batches that may be converted whole.
std::size_t expected_task_bytes(const merge::windowed_merge_input& input,
                                std::size_t unbuffered_runs,
                                std::size_t fallback_bytes,
                                std::size_t row_width = sizeof(int64_t))
{
  return merge::estimate_task_bytes(input.cursors().size(),
                                    unbuffered_runs,
                                    input.config().window_rows,
                                    row_width,
                                    sizeof(int64_t),
                                    input.config().output_target_bytes,
                                    fallback_bytes,
                                    row_width / sizeof(int64_t),
                                    0,
                                    input.config().round_rows);
}

/// sort_merge_disabled_rules with only @p rule switched off.
constexpr std::uint64_t without(merge::merge_rule rule) { return static_cast<std::uint64_t>(rule); }

/// Runs the task of @p input on @p space, as a GPU pipeline task does, and drops its output.
void run_task(sirius_physical_merge_sort& merge,
              operator_data& input,
              memory_space& space,
              ::cuda::stream_ref stream)
{
  input.prepare_for_processing(&space, stream);
  REQUIRE(merge.execute(input, stream));
}

/// Empties @p repo, so no batch outlives the memory spaces of a test's fixture.
void drain(data_repository& repo)
{
  for (std::size_t partition = 0; partition < repo.num_partitions(); ++partition) {
    while (repo.pop_next_data_batch(partition)) {}
  }
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
  merge->set_device_budget_provider([](const memory_space*) {
    return merge::device_budget{
      .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 4096, .concurrent_tasks = 1};
  });

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.cursors().size() == 2);
  REQUIRE(repo.size(0) == 2);
}

TEST_CASE("merge sort automatic mode windows a small partition of a sort that does not fit",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  // 1024 INT64 rows are 8 KiB. On a 128 KiB device the one-shot threshold is 25.6 KiB, so the
  // partition of 8 KiB is under it and one of 16 KiB or more is not.
  auto const small_device = [](const memory_space*) {
    return merge::device_budget{
      .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 128 * 1024, .concurrent_tasks = 4};
  };
  auto const fill = [&](data_repository& repo, std::size_t large_runs) {
    repo.set_num_partitions(2);
    repo.add_data_batch(make_sorted_batch(*space, 512), 0);
    repo.add_data_batch(make_sorted_batch(*space, 512), 0);
    for (std::size_t i = 0; i < large_runs; ++i) {
      repo.add_data_batch(make_sorted_batch(*space, 1024), 1);
    }
  };

  SECTION("the runs of the sort and the small partition's merge exceed the device")
  {
    // 120 KiB of runs, and 16 KiB + 8 bytes per row for the merge of the small partition.
    data_repository repo;
    fill(repo, 14);
    auto merge = make_merge(repo);
    merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic});
    merge->set_device_budget_provider(small_device);

    auto input  = merge->get_next_task_input_data();
    auto& first = require_windowed_input(input);
    REQUIRE(first.partition() == 0);
    REQUIRE(first.cursors().size() == 2);
    REQUIRE(repo.size(0) == 2);
    REQUIRE(merge->counters().one_shot_partitions == 0);
    drain(repo);
  }

  SECTION("they fit: the small partition is merged one-shot")
  {
    // 24 KiB of runs.
    data_repository repo;
    fill(repo, 2);
    auto merge = make_merge(repo);
    merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic});
    merge->set_device_budget_provider(small_device);

    auto first = merge->get_next_task_input_data();
    REQUIRE(first);
    REQUIRE(first->get_type() == operator_data_type::PIPELINEABLE);
    REQUIRE(repo.size(0) == 0);
    // The large partition is windowed by its own size, also after the small one left.
    auto input   = merge->get_next_task_input_data();
    auto& second = require_windowed_input(input);
    REQUIRE(second.partition() == 1);
    REQUIRE(merge->counters().one_shot_partitions == 1);
    drain(repo);
  }

  SECTION("without the rule the small partition is judged alone")
  {
    data_repository repo;
    fill(repo, 14);
    auto merge = make_merge(repo);
    merge->set_merge_settings({.mode           = sirius::sort_merge_mode::automatic,
                               .disabled_rules = without(merge::merge_rule::one_shot_fit)});
    merge->set_device_budget_provider(small_device);

    auto first = merge->get_next_task_input_data();
    REQUIRE(first);
    REQUIRE(first->get_type() == operator_data_type::PIPELINEABLE);
    REQUIRE(merge->counters().one_shot_partitions == 1);
    drain(repo);
  }
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
  // Two concurrent tasks: both partitions may be open at once.
  merge->set_device_budget_provider(two_task_budget);

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

  auto merge = make_windowed_merge(repo);
  // Two concurrent tasks: both partitions may be open at once.
  merge->set_device_budget_provider(two_task_budget);

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

  // By default a one-shot input keeps the base operator's estimate too: the rule is off.
  input_stats pipelineable{
    .num_batches = 3, .bytes = 12345, .type = operator_data_type::PIPELINEABLE};
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic});
  REQUIRE(merge->no_history_peak_memory_estimate(pipelineable) == 2 * 12345);

  // With the rule: a one-shot merge of several batches peaks at 2.5 x its input on top of it.
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::automatic, .disabled_rules = 0});
  REQUIRE(merge->no_history_peak_memory_estimate(pipelineable) == 2 * 12345 + 12345 / 2);
  input_stats huge{.num_batches = 2,
                   .bytes       = std::numeric_limits<std::size_t>::max() / 2,
                   .type        = operator_data_type::PIPELINEABLE};
  REQUIRE(merge->no_history_peak_memory_estimate(huge) == std::numeric_limits<std::size_t>::max());

  // A single batch is forwarded: the base operator's estimate.
  input_stats single{.num_batches = 1, .bytes = 12345, .type = operator_data_type::PIPELINEABLE};
  REQUIRE(merge->no_history_peak_memory_estimate(single) == 2 * 12345);

  // Without the rule every one-shot input keeps the base operator's estimate.
  merge->set_merge_settings({.mode           = sirius::sort_merge_mode::automatic,
                             .disabled_rules = without(merge::merge_rule::one_shot_estimate)});
  REQUIRE(merge->no_history_peak_memory_estimate(pipelineable) == 2 * 12345);
}

TEST_CASE("merge sort arming does not wait for a run locked by a tier move while a task runs",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  if (!sirius::row_range_reader_registry::is_initialized()) {
    sirius::row_range_reader_registry::initialize();
  }
  data_repository repo;
  repo.set_num_partitions(2);
  repo.add_data_batch(make_sorted_batch(*space, 10), 0);
  repo.add_data_batch(make_sorted_batch(*space, 10), 0);
  auto locked_run = make_sorted_batch(*space, 10);
  repo.add_data_batch(locked_run, 1);
  repo.add_data_batch(make_sorted_batch(*space, 10), 1);
  auto merge = make_windowed_merge(repo);

  auto first = merge->get_next_task_input_data();
  REQUIRE(require_windowed_input(first).partition() == 0);

  // What a downgrade holds for the whole move of a batch to another tier.
  std::optional<mutable_data_batch> tier_move = locked_run->to_mutable();

  // Arming partition 1 needs the size of the locked run: it is left for later, at once.
  auto const while_locked = require_prompt([&] { return merge->get_next_task_input_data(); });
  REQUIRE_FALSE(while_locked);
  // The partition comes back when the task in flight commits, so no hint is given until then.
  auto const hint_while_locked = require_prompt([&] { return merge->get_next_task_hint(); });
  REQUIRE_FALSE(hint_while_locked.has_value());
  REQUIRE(merge->counters().windowed_partitions == 1);

  // The commit of partition 0's task does not wait for the lock either.
  rmm::cuda_stream stream;
  ::cuda::stream_ref const stream_ref{stream.value()};
  first->prepare_for_processing(space, stream_ref);
  auto const committed = require_prompt([&] { return merge->execute(*first, stream_ref); });
  REQUIRE(committed);
  REQUIRE(repo.size(0) == 0);
  auto const hint_after_commit = require_prompt([&] { return merge->get_next_task_hint(); });
  REQUIRE(hint_after_commit.has_value());
  REQUIRE(hint_after_commit->hint == TaskCreationHint::READY);

  // With no task in flight nothing else would arm the partition again, so the call waits for
  // the tier move. It waits without the operator's mutex: the hint is still answered.
  auto second = std::async(std::launch::async, [&] { return merge->get_next_task_input_data(); });
  REQUIRE(second.wait_for(std::chrono::milliseconds{200}) == std::future_status::timeout);
  auto const hint_while_waiting = require_prompt([&] { return merge->get_next_task_hint(); });
  REQUIRE(hint_while_waiting.has_value());

  tier_move.reset();
  REQUIRE(second.wait_for(kPrompt) == std::future_status::ready);
  auto const armed = second.get();
  auto& windowed   = require_windowed_input(armed);
  REQUIRE(windowed.partition() == 1);
  REQUIRE(windowed.cursors().size() == 2);
  REQUIRE(merge->counters().windowed_partitions == 2);
  REQUIRE(merge->counters().one_shot_partitions == 0);
}

TEST_CASE("merge sort arms around a locked run whose size was recorded when it arrived",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  auto merge = make_windowed_merge(repo);
  auto a     = make_sorted_batch(*space, 10);
  auto b     = make_sorted_batch(*space, 20);
  // As the upstream operator pushes its runs.
  merge->push_data_batch_partitioned("default", a, 0);
  merge->push_data_batch_partitioned("default", b, 0);
  auto const b_bytes = decoded_bytes(*b);

  auto tier_move = b->to_mutable();
  auto const input =
    require_prompt([&] { return std::shared_ptr{merge->get_next_task_input_data()}; });
  REQUIRE(input);
  auto& windowed = dynamic_cast<merge::windowed_merge_input&>(*input);
  REQUIRE(windowed.cursors().size() == 2);
  REQUIRE(windowed.cursors()[0].run.batch_rows == std::vector<std::size_t>{10});
  REQUIRE(windowed.cursors()[1].run.batch_rows == std::vector<std::size_t>{20});
  REQUIRE(repo.size(0) == 2);
  // What the locked run will be when the task reads it is not known: it is charged as a batch
  // that is converted whole.
  REQUIRE(b_bytes > 0);
  REQUIRE(windowed.get_estimated_size_in_bytes() == expected_task_bytes(windowed, 0, b_bytes));
}

TEST_CASE("merge sort device budget provider reports the GPU spaces and the executor width",
          "[merge_sort_arming]")
{
  auto manager    = initialize_memory_manager(1,
                                              {.gpu_bytes                = std::size_t{128} << 20,
                                               .gpu_reservation_fraction = 0.75,
                                               .divide_among_gpus        = false,
                                               .per_stream_tracking      = false});
  auto const* gpu = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu != nullptr);
  auto const provider = sirius_physical_merge_sort::make_device_budget_provider(*manager, 6);

  // No run on a GPU: the smallest GPU space.
  auto const unknown = provider(nullptr);
  REQUIRE(unknown.concurrent_tasks == 6);
  REQUIRE(unknown.gpu_total_bytes == gpu->get_max_memory());
  REQUIRE(unknown.gpu_available_bytes ==
          gpu->get_available_memory() + gpu->get_total_reserved_memory());
  REQUIRE(unknown.gpu_total_bytes > 0);
  // The space holding most of the partition.
  auto const named = provider(gpu);
  REQUIRE(named.gpu_total_bytes == gpu->get_max_memory());
  REQUIRE(sirius_physical_merge_sort::make_device_budget_provider(*manager, 0)(nullptr)
            .concurrent_tasks == 1);

  // The operator asks its provider with the space that holds the partition's GPU runs.
  auto* space = const_cast<memory_space*>(gpu);
  data_repository repo;
  repo.add_data_batch(make_sorted_batch(*space, 10));
  repo.add_data_batch(make_sorted_batch(*space, 10));
  auto merge = make_merge(repo);
  REQUIRE_FALSE(merge->has_device_budget_provider());
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  const memory_space* asked = nullptr;
  std::size_t calls         = 0;
  merge->set_device_budget_provider([&](const memory_space* preferred) {
    asked = preferred;
    ++calls;
    return provider(preferred);
  });
  REQUIRE(merge->has_device_budget_provider());
  auto input = merge->get_next_task_input_data();
  require_windowed_input(input);
  REQUIRE(calls == 1);
  REQUIRE(asked == gpu);
  input.reset();
  merge.reset();
  while (repo.pop_next_data_batch(0)) {}
}

TEST_CASE("merge sort sizes a partition armed while another one's task holds a reservation alike",
          "[merge_sort_arming]")
{
  auto manager = initialize_memory_manager(1,
                                           {.gpu_bytes                = std::size_t{128} << 20,
                                            .gpu_reservation_fraction = 0.75,
                                            .divide_among_gpus        = false,
                                            .per_stream_tracking      = false});
  auto* space  = const_cast<memory_space*>(manager->get_memory_space(Tier::GPU, 0));
  REQUIRE(space != nullptr);
  // With the operator's provider (two tasks per device) and with none (the default width).
  auto const with_provider = GENERATE(true, false);
  CAPTURE(with_provider);
  REQUIRE(sirius::exec::default_gpu_pipeline_num_threads >= 2);

  data_repository repo;
  repo.set_num_partitions(2);
  for (std::size_t partition = 0; partition < 2; ++partition) {
    repo.add_data_batch(make_sorted_batch(*space, 10), partition);
    repo.add_data_batch(make_sorted_batch(*space, 10), partition);
  }
  const merge::merge_settings settings{.mode = sirius::sort_merge_mode::windowed};
  auto merge = make_merge(repo);
  merge->set_merge_settings(settings);
  if (with_provider) {
    merge->set_device_budget_provider(
      sirius_physical_merge_sort::make_device_budget_provider(*manager, 2));
  }

  auto first        = merge->get_next_task_input_data();
  auto const config = require_windowed_input(first).config();
  REQUIRE(require_windowed_input(first).partition() == 0);

  // What the executor holds for the first partition's task while the second one is armed.
  auto const reservation = space->make_reservation(std::size_t{32} << 20);
  REQUIRE(reservation);
  // Sized from the free bytes alone, the second partition would get smaller windows.
  auto const tasks = with_provider
                       ? std::size_t{2}
                       : static_cast<std::size_t>(sirius::exec::default_gpu_pipeline_num_threads);
  REQUIRE(merge::derive_merge_config(settings,
                                     {.gpu_available_bytes = space->get_available_memory(),
                                      .gpu_total_bytes     = space->get_max_memory(),
                                      .concurrent_tasks    = tasks},
                                     2,
                                     sizeof(int64_t))
            .window_rows < config.window_rows);

  auto second        = merge->get_next_task_input_data();
  auto& second_input = require_windowed_input(second);
  REQUIRE(second_input.partition() == 1);
  REQUIRE(second_input.config().window_rows == config.window_rows);
  REQUIRE(second_input.config().output_target_bytes == config.output_target_bytes);

  first.reset();
  second.reset();
  merge.reset();
  drain(repo);
}

TEST_CASE("merge sort one-shot commit makes a partition deferred by a locked run armable again",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  // Declared before the batches: the task's reads of them are recorded on this stream.
  rmm::cuda_stream stream;
  ::cuda::stream_ref const stream_ref{stream.value()};
  data_repository repo;
  repo.set_num_partitions(2);
  repo.add_data_batch(make_sorted_batch(*space, 10), 0);
  repo.add_data_batch(make_sorted_batch(*space, 10), 0);
  auto locked_run = make_sorted_batch(*space, 10);
  auto other_run  = make_sorted_batch(*space, 10);
  repo.add_data_batch(locked_run, 1);
  repo.add_data_batch(other_run, 1);
  auto merge = make_windowed_merge(repo, {.mode = sirius::sort_merge_mode::automatic});

  // Partition 0 is small and on the GPU: one task pops it whole.
  auto first = merge->get_next_task_input_data();
  REQUIRE(first);
  REQUIRE(first->get_type() == operator_data_type::PIPELINEABLE);
  REQUIRE(repo.size(0) == 0);

  std::optional<mutable_data_batch> tier_move = locked_run->to_mutable();
  auto const while_locked = require_prompt([&] { return merge->get_next_task_input_data(); });
  REQUIRE_FALSE(while_locked);
  auto const hint_while_locked = require_prompt([&] { return merge->get_next_task_hint(); });
  REQUIRE_FALSE(hint_while_locked.has_value());

  // The one-shot task is the only one in flight: its commit has to offer partition 1 again.
  first->prepare_for_processing(space, stream_ref);
  auto const committed = require_prompt([&] { return merge->execute(*first, stream_ref); });
  REQUIRE(committed);
  auto const hint_after_commit = require_prompt([&] { return merge->get_next_task_hint(); });
  REQUIRE(hint_after_commit.has_value());
  REQUIRE(hint_after_commit->hint == TaskCreationHint::READY);

  tier_move.reset();
  auto second = merge->get_next_task_input_data();
  require_one_shot_input(second, {locked_run, other_run}, merge::make_merge_chunk_position(1, 0));
  REQUIRE(merge->counters().one_shot_partitions == 2);
}

TEST_CASE("merge sort without a budget provider sizes its windows from the GPU holding its runs",
          "[merge_sort_arming]")
{
  tiered_memory_fixture fixture;
  REQUIRE(fixture.setup());
  auto const stream = fixture.gpu0->acquire_stream();
  data_repository repo;
  repo.set_num_partitions(2);
  repo.add_data_batch(fixture.make_gpu_batch(stream), 0);
  repo.add_data_batch(fixture.make_host_batch(stream), 0);
  repo.add_data_batch(fixture.make_host_batch(stream), 1);
  repo.add_data_batch(fixture.make_host_batch(stream), 1);

  const merge::merge_settings settings{.mode = sirius::sort_merge_mode::windowed};
  auto merge = make_merge(repo);
  merge->set_merge_settings(settings);
  REQUIRE_FALSE(merge->has_device_budget_provider());

  // The default executor width, lowered to the two partitions the operator has.
  auto const tasks = std::min<std::size_t>(
    static_cast<std::size_t>(sirius::exec::default_gpu_pipeline_num_threads), 2);
  auto const from_space = [&] {
    return merge::derive_merge_config(
      settings,
      {.gpu_available_bytes =
         fixture.gpu0->get_available_memory() + fixture.gpu0->get_total_reserved_memory(),
       .gpu_total_bytes  = fixture.gpu0->get_max_memory(),
       .concurrent_tasks = tasks},
      2,
      sizeof(int64_t));
  };
  // The fixture's GPU is not the size assumed for an unknown device, so the two configs differ.
  auto const unknown = merge::derive_merge_config(
    settings,
    {.gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = tasks},
    2,
    sizeof(int64_t));
  REQUIRE(from_space().window_rows != unknown.window_rows);

  // Partition 0 has a run on the GPU: its space gives the budget.
  auto expected = from_space();
  auto first    = merge->get_next_task_input_data();
  auto& on_gpu  = require_windowed_input(first);
  REQUIRE(on_gpu.partition() == 0);
  REQUIRE(on_gpu.config().window_rows == expected.window_rows);
  REQUIRE(on_gpu.config().output_target_bytes == expected.output_target_bytes);

  // Partition 1 has none: the space last seen holding runs is used again.
  expected       = from_space();
  auto second    = merge->get_next_task_input_data();
  auto& all_host = require_windowed_input(second);
  REQUIRE(all_host.partition() == 1);
  REQUIRE(all_host.config().window_rows == expected.window_rows);
  REQUIRE(all_host.config().output_target_bytes == expected.output_target_bytes);

  first.reset();
  second.reset();
  merge.reset();
  drain(repo);
}

TEST_CASE("merge sort without a budget provider or a GPU run assumes an 8 GiB device",
          "[merge_sort_arming]")
{
  tiered_memory_fixture fixture;
  REQUIRE(fixture.setup());
  auto const stream = fixture.gpu0->acquire_stream();
  data_repository repo;
  repo.add_data_batch(fixture.make_host_batch(stream));
  repo.add_data_batch(fixture.make_host_batch(stream));

  const merge::merge_settings settings{.mode = sirius::sort_merge_mode::windowed};
  auto merge = make_merge(repo);
  merge->set_merge_settings(settings);

  auto const expected = merge::derive_merge_config(
    settings,
    // One partition: it has the device's budget to itself, whatever the executor width.
    {.gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 1},
    2,
    sizeof(int64_t));
  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.config().window_rows == expected.window_rows);
  REQUIRE(windowed.config().output_target_bytes == expected.output_target_bytes);

  input.reset();
  merge.reset();
  drain(repo);
}

TEST_CASE("merge sort reserves window buffers for the runs that are on the GPU when it plans",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  repo.add_data_batch(make_sorted_batch(*space, 100));
  repo.add_data_batch(make_sorted_batch(*space, 200));
  auto merge = make_windowed_merge(repo);

  // Both runs are tables on the task's device: it slices them in place. They stay spillable
  // until the task locks them, so the window buffers of one of them are reserved as a spare.
  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.gpu_runs() == 2);
  REQUIRE(windowed.get_estimated_size_in_bytes() == expected_task_bytes(windowed, 1, 0));
  REQUIRE(windowed.get_estimated_size_in_bytes() > expected_task_bytes(windowed, 2, 0));
  REQUIRE(windowed.get_estimated_size_in_bytes() < expected_task_bytes(windowed, 0, 0));
  // A round is sized from the output target, not from the window.
  REQUIRE(windowed.config().round_rows ==
          merge::round_rows_for(
            windowed.config().window_rows, windowed.config().output_target_bytes, sizeof(int64_t)));

  // A retry after an OOM charges both runs, at the same sizes.
  auto const sizes = windowed.config();
  REQUIRE(windowed.shrink_for_retry());
  REQUIRE(windowed.gpu_runs() == 0);
  REQUIRE(windowed.config().same_sizes(sizes));
  REQUIRE(windowed.get_estimated_size_in_bytes() == expected_task_bytes(windowed, 0, 0));
  REQUIRE(merge->counters().recharges == 1);
  REQUIRE(merge->counters().shrinks == 0);
}

TEST_CASE("merge sort charges every run its window buffers without the in-place reservation",
          "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  data_repository repo;
  repo.add_data_batch(make_sorted_batch(*space, 100));
  repo.add_data_batch(make_sorted_batch(*space, 200));
  auto merge =
    make_windowed_merge(repo,
                        {.mode           = sirius::sort_merge_mode::windowed,
                         .disabled_rules = without(merge::merge_rule::in_place_reservation)});

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.gpu_runs() == 0);
  REQUIRE(windowed.get_estimated_size_in_bytes() == expected_task_bytes(windowed, 0, 0));
}

TEST_CASE("merge sort counts a run in place only when every batch it has left is a GPU table",
          "[merge_sort_arming]")
{
  merge_utils::tiers_fixture fx;
  data_repository repo;
  auto on_gpu  = make_sorted_batch(*fx.gpu, 500);
  auto on_gpu2 = make_sorted_batch(*fx.gpu, 700);
  auto on_host = make_sorted_batch(*fx.gpu, 600);
  merge_utils::place_batch(fx, on_host, merge_utils::placement::host);
  for (auto const& batch : {on_gpu, on_gpu2, on_host}) {
    repo.add_data_batch(batch);
  }
  auto merge = make_windowed_merge(repo);

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.cursors().size() == 3);
  // Two runs in place (one of them the spare) and the host run: two runs are charged.
  REQUIRE(windowed.gpu_runs() == 2);
  REQUIRE(windowed.get_estimated_size_in_bytes() == expected_task_bytes(windowed, 1, 0));
  REQUIRE(windowed.get_preferred_device_id() == fx.gpu->get_device_id());

  input.reset();
  merge.reset();
  drain(repo);
}

TEST_CASE("merge sort shares the window budget among the partitions it has", "[merge_sort_arming]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  // An executor of eight threads on an 8 GiB device.
  auto const eight_threads = [](const memory_space*) {
    return merge::device_budget{
      .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 8};
  };
  auto const window_with = [&](std::size_t partitions, std::uint64_t disabled_rules) {
    data_repository repo;
    repo.set_num_partitions(partitions + 2);  // the last two stay empty
    for (std::size_t partition = 0; partition < partitions; ++partition) {
      repo.add_data_batch(make_sorted_batch(*space, 100), partition);
      repo.add_data_batch(make_sorted_batch(*space, 100), partition);
    }
    auto merge = make_windowed_merge(
      repo, {.mode = sirius::sort_merge_mode::windowed, .disabled_rules = disabled_rules});
    merge->set_device_budget_provider(eight_threads);
    std::vector<std::unique_ptr<operator_data>> inputs;
    std::vector<std::size_t> windows;
    while (auto input = merge->get_next_task_input_data()) {
      windows.push_back(require_windowed_input(input).config().window_rows);
      inputs.push_back(std::move(input));
    }
    // Every partition is sized alike, whenever it is armed, and all of them may be open.
    REQUIRE(windows.size() == std::min<std::size_t>(partitions, 8));
    REQUIRE(std::all_of(windows.begin(), windows.end(), [&](auto w) { return w == windows[0]; }));
    inputs.clear();
    merge.reset();
    drain(repo);
    return windows[0];
  };
  auto const derived = [](std::size_t tasks) {
    return merge::derive_merge_config({},
                                      {.gpu_available_bytes = 8 * kGiB,
                                       .gpu_total_bytes     = 8 * kGiB,
                                       .concurrent_tasks    = tasks},
                                      2,
                                      sizeof(int64_t))
      .window_rows;
  };
  // T = 1 is bounded by the ceiling, so the comparisons use two partitions and more.
  REQUIRE(derived(2) > derived(4));
  REQUIRE(derived(4) > derived(8));

  // Two partitions under eight threads are sized for two open partitions, not for eight.
  REQUIRE(window_with(2, 0) == derived(2));
  REQUIRE(window_with(4, 0) == derived(4));
  // More partitions than threads: the executor width, as before.
  REQUIRE(window_with(12, 0) == derived(8));
  // Without the rule the width is the divisor whatever the partition count.
  REQUIRE(window_with(2, without(merge::merge_rule::partition_budget)) == derived(8));
}

TEST_CASE("merge sort opens no more partitions than the budget was shared among",
          "[merge_sort_arming]")
{
  // Three partitions of host runs under an executor of two threads: sized for two open
  // partitions, and no third one opens while two keep their mergers.
  merge_utils::tiers_fixture fx;
  data_repository repo;
  repo.set_num_partitions(3);
  for (std::size_t partition = 0; partition < 3; ++partition) {
    for (int run = 0; run < 2; ++run) {
      auto batch = make_sorted_batch(*fx.gpu, 3000);
      merge_utils::place_batch(fx, batch, merge_utils::placement::host);
      repo.add_data_batch(batch, partition);
    }
  }
  auto merge = make_windowed_merge(
    repo, {.mode = sirius::sort_merge_mode::windowed, .window_rows = 1024, .output_bytes = 1024});
  merge->set_device_budget_provider(two_task_budget);

  auto first  = merge->get_next_task_input_data();
  auto second = merge->get_next_task_input_data();
  REQUIRE(require_windowed_input(first).partition() == 0);
  REQUIRE(require_windowed_input(second).partition() == 1);
  REQUIRE_FALSE(merge->get_next_task_input_data());
  REQUIRE_FALSE(merge->get_next_task_hint().has_value());

  // With one partition per thread and a thread to spare, every partition opens.
  data_repository small;
  small.set_num_partitions(2);
  for (std::size_t partition = 0; partition < 2; ++partition) {
    for (int run = 0; run < 2; ++run) {
      auto batch = make_sorted_batch(*fx.gpu, 3000);
      merge_utils::place_batch(fx, batch, merge_utils::placement::host);
      small.add_data_batch(batch, partition);
    }
  }
  auto roomy = make_windowed_merge(
    small, {.mode = sirius::sort_merge_mode::windowed, .window_rows = 1024, .output_bytes = 1024});
  roomy->set_device_budget_provider([](const memory_space*) {
    return merge::device_budget{
      .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 16};
  });
  auto a = roomy->get_next_task_input_data();
  auto b = roomy->get_next_task_input_data();
  REQUIRE(require_windowed_input(a).partition() == 0);
  REQUIRE(require_windowed_input(b).partition() == 1);

  first.reset();
  second.reset();
  a.reset();
  b.reset();
  merge.reset();
  roomy.reset();
  drain(repo);
  drain(small);
}

TEST_CASE("merge sort in automatic mode forwards a partition whose only run is on the GPU",
          "[merge_sort_arming]")
{
  merge_utils::tiers_fixture fx;
  // A threshold no partition fits under: the automatic mode would window every partition.
  const merge::merge_settings always_windowed{.mode = sirius::sort_merge_mode::automatic,
                                              .one_shot_max_fraction = 1e-12};

  SECTION("one GPU run: the one-shot route gets the batch alone, the empty ones are dropped")
  {
    data_repository repo;
    auto empty_a = make_sorted_batch(*fx.gpu, 0);
    auto run     = make_sorted_batch(*fx.gpu, 500);
    auto empty_b = make_sorted_batch(*fx.gpu, 0);
    for (auto const& batch : {empty_a, run, empty_b}) {
      repo.add_data_batch(batch);
    }
    auto merge = make_windowed_merge(repo, always_windowed);

    auto input = merge->get_next_task_input_data();
    require_one_shot_input(input, {run}, merge::make_merge_chunk_position(0, 0));
    REQUIRE(repo.total_size() == 0);
    REQUIRE(merge->counters().one_shot_partitions == 1);
    REQUIRE(merge->counters().windowed_partitions == 0);

    // The task forwards the batch itself: no row is copied.
    input->prepare_for_processing(fx.gpu, fx.stream_ref());
    auto const output = merge->execute(*input, fx.stream_ref());
    REQUIRE(output);
    auto const& data = dynamic_cast<const pipelineable_operator_data&>(*output);
    REQUIRE(data.get_data_batches().size() == 1);
    REQUIRE(data.get_data_batches()[0] == run);
    REQUIRE(data.get_position() == merge::make_merge_chunk_position(0, 0));
    REQUIRE_FALSE(merge->get_next_task_input_data());
  }

  SECTION("a run on another tier is read window by window")
  {
    data_repository repo;
    auto run = make_sorted_batch(*fx.gpu, 500);
    merge_utils::place_batch(fx, run, merge_utils::placement::host);
    repo.add_data_batch(run);
    auto merge = make_windowed_merge(repo, always_windowed);

    auto input = merge->get_next_task_input_data();
    REQUIRE(require_windowed_input(input).cursors().size() == 1);
    REQUIRE(repo.size(0) == 1);
    input.reset();
    merge.reset();
    drain(repo);
  }

  SECTION("two runs are merged")
  {
    data_repository repo;
    repo.add_data_batch(make_sorted_batch(*fx.gpu, 500));
    repo.add_data_batch(make_sorted_batch(*fx.gpu, 500));
    auto merge = make_windowed_merge(repo, always_windowed);
    auto input = merge->get_next_task_input_data();
    REQUIRE(require_windowed_input(input).cursors().size() == 2);
    input.reset();
    merge.reset();
    drain(repo);
  }

  SECTION("windowed mode means a windowed task, also for a single run")
  {
    data_repository repo;
    auto run = make_sorted_batch(*fx.gpu, 500);
    repo.add_data_batch(run);
    auto merge = make_windowed_merge(repo);
    auto input = merge->get_next_task_input_data();
    REQUIRE(require_windowed_input(input).cursors().size() == 1);
    REQUIRE(merge->counters().windowed_partitions == 1);
    REQUIRE(merge->counters().one_shot_partitions == 0);
    input.reset();
    merge.reset();
    drain(repo);
  }

  SECTION("without the rule the single run takes a windowed task in automatic mode too")
  {
    data_repository repo;
    auto run = make_sorted_batch(*fx.gpu, 500);
    repo.add_data_batch(run);
    auto settings           = always_windowed;
    settings.disabled_rules = without(merge::merge_rule::single_run_passthrough);
    auto merge              = make_windowed_merge(repo, settings);
    auto input              = merge->get_next_task_input_data();
    REQUIRE(require_windowed_input(input).cursors().size() == 1);
    REQUIRE(merge->counters().windowed_partitions == 1);
    input.reset();
    merge.reset();
    drain(repo);
  }
}

TEST_CASE("merge sort reserves the sum of the batches its runs may convert whole",
          "[merge_sort_arming]")
{
  merge_utils::tiers_fixture fx;
  data_repository repo;
  auto on_gpu   = make_sorted_batch(*fx.gpu, 500);
  auto packed_a = make_sorted_batch(*fx.gpu, 1000);
  auto packed_b = make_sorted_batch(*fx.gpu, 3000);
  cudf::get_default_stream().sync();
  auto merge = make_windowed_merge(repo);
  // As the upstream operator pushes its runs: their rows are recorded while they are tables.
  for (auto const& batch : {on_gpu, packed_a, packed_b}) {
    merge->push_data_batch_partitioned("default", batch, 0);
  }
  auto const fallback_bytes = decoded_bytes(*packed_a) + decoded_bytes(*packed_b);
  REQUIRE(decoded_bytes(*packed_a) > 0);
  // A packed host batch has no row-range reader: the merger converts it whole to the GPU.
  for (auto const& batch : {packed_a, packed_b}) {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_packed_representation>(
      sirius::converter_registry::get(), fx.host, fx.stream_ref());
  }
  fx.stream.synchronize();

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.cursors().size() == 3);
  // Both converted batches are on the GPU until the task commits: their sizes add up.
  REQUIRE(windowed.get_estimated_size_in_bytes() ==
          expected_task_bytes(windowed, 0, fallback_bytes));

  input.reset();
  merge.reset();
  drain(repo);
}

TEST_CASE("merge sort reserves a compressed batch whose plan cannot be read by row range",
          "[merge_sort_arming]")
{
  merge_utils::tiers_fixture fx;
  merge_utils::scoped_spill_compression compression;
  data_repository repo;

  // (key INT64, id INT64) rows, sorted by key.
  auto const make_rows = [](std::size_t rows, std::int64_t first) {
    std::vector<int64_t> keys(rows);
    std::iota(keys.begin(), keys.end(), first);
    return merge_utils::make_table(merge_utils::make_column(keys, cudf::type_id::INT64),
                                   merge_utils::make_column(keys, cudf::type_id::INT64));
  };
  auto const add_gpu_run = [&](std::size_t rows, std::int64_t first) {
    auto batch = ::sirius::make_data_batch(
      std::make_unique<cudf::table>(make_rows(rows, first)->view(), default_stream(), fx.mr()),
      *fx.gpu,
      default_stream(),
      ::sirius::telemetry::batch_telemetry_info{});
    repo.add_data_batch(batch);
    return batch;
  };
  auto on_gpu   = add_gpu_run(4000, 0);
  auto readable = add_gpu_run(5000, 100000);
  merge_utils::compress_batch(fx, readable, merge_utils::placement::compressed_host);

  // Column 0 with a spill plan, column 1 with a plan the spill never picks and the row-range
  // reader refuses.
  auto const rows = make_rows(6000, 200000);
  const std::string plan =
    "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
    "---\n"
    "input -> bitcomp\n";
  auto blob   = std::make_shared<sirius::compressed_device_blob>();
  blob->table = simpatico::compress_with_plan(rows->view(), plan, fx.stream_ref(), fx.mr(), {});
  fx.stream.synchronize();
  auto refused =
    cucascade::data_batch::make(sirius::get_next_batch_id(),
                                std::make_unique<sirius::compressed_device_representation>(
                                  *fx.gpu,
                                  std::move(blob),
                                  std::vector<std::string>{"key", "id"},
                                  rows->alloc_size(),
                                  rows->alloc_size(),
                                  rows->num_rows()));
  repo.add_data_batch(refused);
  REQUIRE(sirius::row_range_reader_registry::get().has_reader(*refused->to_read_only().get_data()));

  auto merge = make_merge(repo, 2);
  merge->set_merge_settings({.mode = sirius::sort_merge_mode::windowed});
  merge->set_device_budget_provider(large_budget);

  auto input     = merge->get_next_task_input_data();
  auto& windowed = require_windowed_input(input);
  REQUIRE(windowed.cursors().size() == 3);
  // The refused batch is decoded whole, so it is charged; the readable compressed run is not.
  auto const reader_state =
    merge::estimate_reader_state_bytes(2, 2, 0, windowed.config().window_rows);
  REQUIRE(windowed.get_estimated_size_in_bytes() ==
          expected_task_bytes(windowed, 0, decoded_bytes(*refused), 2 * sizeof(int64_t)) +
            reader_state);

  input.reset();
  merge.reset();
  drain(repo);
}

namespace {

/// A windowed merge over two sorted HOST runs of @p rows rows each, which its tasks read through
/// window buffers; each task emits about one window.
struct host_runs_merge {
  // Declared first: destroyed after the batches and the operator.
  merge_utils::tiers_fixture fx;
  data_repository repo;
  std::unique_ptr<sirius_physical_merge_sort> merge;

  explicit host_runs_merge(std::size_t rows,
                           std::size_t window_rows  = 1024,
                           int runs                 = 2,
                           std::uint64_t max_fan_in = merge::merge_settings{}.max_fan_in)
  {
    for (int run = 0; run < runs; ++run) {
      auto batch = make_sorted_batch(*fx.gpu, rows);
      merge_utils::place_batch(fx, batch, merge_utils::placement::host);
      repo.add_data_batch(batch);
    }
    merge = make_windowed_merge(repo,
                                {.mode         = sirius::sort_merge_mode::windowed,
                                 .window_rows  = window_rows,
                                 .output_bytes = 1024,
                                 .max_fan_in   = max_fan_in});
  }

  ~host_runs_merge()
  {
    merge.reset();
    drain(repo);
  }

  /// Runs the task of @p input and waits for the device work it left on the stream.
  void run(operator_data& input)
  {
    run_task(*merge, input, *fx.gpu, fx.stream_ref());
    fx.stream.synchronize();
  }
};

}  // namespace

TEST_CASE("merge sort does not reserve again the window buffers its kept merger holds",
          "[merge_sort_arming]")
{
  host_runs_merge test(5000);

  auto first        = test.merge->get_next_task_input_data();
  auto& first_input = require_windowed_input(first);
  REQUIRE(first_input.get_estimated_size_in_bytes() == expected_task_bytes(first_input, 0, 0));
  // No run is on a GPU and no merger is kept yet.
  REQUIRE_FALSE(first_input.get_preferred_device_id().has_value());
  test.run(*first);

  // The partition kept its merger, and with it both runs' window buffers: the next task is
  // charged the merge scratch and the output only, and goes to the merger's device.
  auto second        = test.merge->get_next_task_input_data();
  auto& second_input = require_windowed_input(second);
  REQUIRE(second_input.get_estimated_size_in_bytes() == expected_task_bytes(second_input, 2, 0));
  REQUIRE(second_input.get_estimated_size_in_bytes() < first_input.get_estimated_size_in_bytes());
  REQUIRE(second_input.get_preferred_device_id() == test.fx.gpu->get_device_id());

  // A retry after an OOM starts from a new merger, which allocates every buffer.
  (void)second_input.shrink_for_retry();
  REQUIRE(second_input.get_estimated_size_in_bytes() == expected_task_bytes(second_input, 0, 0));

  test.run(*second);
  REQUIRE(test.merge->counters().mergers_built == 1);
}

TEST_CASE("merge sort keeps the sizes of a task that shrank for the partition's later tasks",
          "[merge_sort_arming]")
{
  host_runs_merge test(20000, 4096);

  auto first        = test.merge->get_next_task_input_data();
  auto& first_input = require_windowed_input(first);
  REQUIRE(first_input.config().window_rows == 4096);
  // What the executor does before it retries a task that ran out of memory.
  REQUIRE(first_input.shrink_for_retry());
  REQUIRE(first_input.config().window_rows == 2048);
  test.run(*first);

  // The next task starts at the shrunk sizes, so it continues the merger the first one left.
  auto second        = test.merge->get_next_task_input_data();
  auto& second_input = require_windowed_input(second);
  REQUIRE(second_input.config().window_rows == 2048);
  REQUIRE(second_input.config().output_target_bytes == first_input.config().output_target_bytes);
  test.run(*second);
  REQUIRE(test.merge->counters().windowed_tasks == 2);
  REQUIRE(test.merge->counters().mergers_built == 1);
  REQUIRE(test.merge->counters().shrinks == 1);
}

TEST_CASE("merge sort starts every pass from the derived sizes after a task shrank",
          "[merge_sort_arming]")
{
  // Four runs under a fan-in of 2: two intermediate passes, then the final one.
  host_runs_merge test(20000, 4096, 4, 2);

  auto first        = test.merge->get_next_task_input_data();
  auto& first_input = require_windowed_input(first);
  REQUIRE_FALSE(first_input.final_pass());
  REQUIRE(first_input.config().window_rows == 4096);
  auto const derived    = first_input.config();
  auto const first_pass = cursor_ids(first_input);
  // What the executor does before it retries a task that ran out of memory.
  REQUIRE(first_input.shrink_for_retry());
  REQUIRE(first_input.config().window_rows == 2048);
  auto const shrunk = first_input.config();
  test.run(*first);

  // The later tasks of the pass continue the first one's merger at the shrunk sizes.
  std::size_t later_tasks = 0;
  std::unique_ptr<operator_data> next;
  while (true) {
    next           = test.merge->get_next_task_input_data();
    auto& input    = require_windowed_input(next);
    auto const ids = cursor_ids(input);
    if (!std::all_of(ids.begin(), ids.end(), [&](auto id) {
          return std::find(first_pass.begin(), first_pass.end(), id) != first_pass.end();
        })) {
      break;
    }
    REQUIRE_FALSE(input.final_pass());
    REQUIRE(input.config().window_rows == shrunk.window_rows);
    REQUIRE(input.config().output_target_bytes == shrunk.output_target_bytes);
    test.run(*next);
    ++later_tasks;
  }
  REQUIRE(later_tasks > 1);
  REQUIRE(test.merge->counters().mergers_built == 1);

  // The next pass reads other runs with a new merger, at the derived sizes again.
  auto& second_pass = require_windowed_input(next);
  REQUIRE_FALSE(second_pass.final_pass());
  REQUIRE(second_pass.config().window_rows == derived.window_rows);
  REQUIRE(second_pass.config().output_target_bytes == derived.output_target_bytes);
  test.run(*next);
  REQUIRE(test.merge->counters().mergers_built == 2);
  REQUIRE(test.merge->counters().shrinks == 1);
}

TEST_CASE("merge sort frees the window buffers of its kept mergers when it is finalized",
          "[merge_sort_arming]")
{
  host_runs_merge test(5000);
  cudf::get_default_stream().sync();
  auto const available_before = test.fx.gpu->get_available_memory();

  auto first = test.merge->get_next_task_input_data();
  REQUIRE(first);
  test.run(*first);
  // The partition's merger holds the window buffers of both runs between two tasks.
  REQUIRE(test.fx.gpu->get_available_memory() < available_before);

  // A LIMIT downstream finishes the pipeline while the partition still has runs.
  test.merge->finalize_operator();
  REQUIRE(test.fx.gpu->get_available_memory() == available_before);

  // A task that still runs builds its own merger and leaves none behind.
  auto second = test.merge->get_next_task_input_data();
  REQUIRE(second);
  test.run(*second);
  REQUIRE(test.merge->counters().mergers_built == 2);
  REQUIRE(test.fx.gpu->get_available_memory() == available_before);
}

TEST_CASE("merge sort opens no more windowed partitions than its budget has concurrent tasks",
          "[merge_sort_arming]")
{
  constexpr std::size_t num_partitions = 16;
  constexpr std::size_t cap            = 2;
  constexpr std::size_t rows           = 3000;
  // Declared first: destroyed after the batches and the operator.
  merge_utils::tiers_fixture fx;
  data_repository repo;
  repo.set_num_partitions(num_partitions);
  for (std::size_t partition = 0; partition < num_partitions; ++partition) {
    for (int run = 0; run < 2; ++run) {
      auto batch = make_sorted_batch(*fx.gpu, rows);
      merge_utils::place_batch(fx, batch, merge_utils::placement::host);
      repo.add_data_batch(batch, partition);
    }
  }
  // About one window a task, so every partition takes a chain of tasks and keeps its merger
  // between them.
  auto merge = make_windowed_merge(
    repo, {.mode = sirius::sort_merge_mode::windowed, .window_rows = 1024, .output_bytes = 1024});
  merge->set_device_budget_provider(two_task_budget);

  auto const unfinished = [&] {
    std::size_t count = 0;
    for (std::size_t partition = 0; partition < num_partitions; ++partition) {
      if (repo.size(partition) > 0) { ++count; }
    }
    return count;
  };

  // The tasks handed out and not run yet, by partition.
  std::map<std::size_t, std::unique_ptr<operator_data>> pending;
  // The partitions that were given a task and still have runs: each has a task in flight or
  // keeps its merger.
  std::set<std::size_t> open;
  std::vector<std::size_t> opened;
  std::vector<std::vector<int64_t>> merged(num_partitions);
  std::size_t tasks = 0;
  while (true) {
    // What a task creation worker does: ask for the hint, then take every task there is.
    auto const hinted = merge->get_next_task_hint().has_value();
    std::size_t taken = 0;
    while (auto input = merge->get_next_task_input_data()) {
      auto const partition = require_windowed_input(input).partition();
      REQUIRE(pending.emplace(partition, std::move(input)).second);
      if (open.insert(partition).second) { opened.push_back(partition); }
      REQUIRE(open.size() <= cap);
      ++taken;
    }
    // The hint announces work exactly when there is some: a partition the cap holds back does
    // not make the operator look ready.
    REQUIRE(hinted == (taken > 0));
    REQUIRE_FALSE(merge->get_next_task_hint().has_value());
    // A partition that finished gave its place to the next one at once.
    REQUIRE(open.size() == std::min(cap, unfinished()));
    REQUIRE(pending.size() == open.size());
    if (pending.empty()) { break; }

    // Run one task, of the lower or of the higher open partition in turn.
    auto const it        = tasks % 3 == 0 ? pending.begin() : std::prev(pending.end());
    auto const partition = it->first;
    it->second->prepare_for_processing(fx.gpu, fx.stream_ref());
    auto const output = merge->execute(*it->second, fx.stream_ref());
    fx.stream.synchronize();
    REQUIRE(output);
    for (auto const& batch :
         dynamic_cast<const pipelineable_operator_data&>(*output).get_data_batches()) {
      auto const values =
        copy_column_to_host<int64_t>(sirius::get_cudf_table_view(*batch).column(0));
      merged[partition].insert(merged[partition].end(), values.begin(), values.end());
    }
    pending.erase(it);
    if (repo.size(partition) == 0) { open.erase(partition); }
    ++tasks;
  }

  // Every partition was merged, each from one merger, in full and in order.
  REQUIRE(opened.size() == num_partitions);
  REQUIRE(repo.total_size() == 0);
  std::vector<int64_t> expected;
  for (std::size_t row = 0; row < rows; ++row) {
    expected.insert(expected.end(), 2, static_cast<int64_t>(row));
  }
  for (std::size_t partition = 0; partition < num_partitions; ++partition) {
    CAPTURE(partition);
    REQUIRE(merged[partition] == expected);
  }
  auto const counters = merge->counters();
  REQUIRE(counters.windowed_partitions == num_partitions);
  REQUIRE(counters.one_shot_partitions == 0);
  REQUIRE(counters.windowed_tasks == tasks);
  REQUIRE(tasks > num_partitions);
  REQUIRE(counters.mergers_built == num_partitions);

  merge.reset();
  drain(repo);
}

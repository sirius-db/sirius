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
#include "data/data_batch_utils.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "helper/type_conversions.hpp"
#include "op/merge/merge_config.hpp"
#include "op/merge/merge_sort_stats_log.hpp"
#include "op/merge/windowed_merge.hpp"
#include "op/merge/windowed_merge_input.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/log_test_utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::merge_utils;

namespace {

namespace utils = sirius::test::operator_utils;

constexpr std::size_t kMiB        = std::size_t{1} << 20;
constexpr std::size_t kGiB        = std::size_t{1} << 30;
constexpr std::size_t kOperatorId = 7;

/// MERGE_SORT that fails its next `ooms` tasks with a GPU out-of-memory error before they merge
/// anything, as a task does when an allocation of its merge fails.
class oom_merge_sort final : public sirius_physical_merge_sort {
 public:
  using sirius_physical_merge_sort::sirius_physical_merge_sort;

  std::size_t ooms{0};

  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         ::cuda::stream_ref stream) override
  {
    if (ooms > 0) {
      --ooms;
      throw rmm::out_of_memory("injected OOM in a merge task");
    }
    return sirius_physical_merge_sort::execute(input, stream);
  }
};

/// A GPU memory space, an input repository of sorted (key INT64, id INT64) runs and the
/// MERGE_SORT over it, driven task by task on the calling thread.
struct stats_log_fixture {
  // Declared first: its destructor shuts the manager down after the batches are gone.
  utils::tiered_memory_fixture memory;
  cucascade::shared_data_repository repo;
  std::unique_ptr<oom_merge_sort> op;
  rmm::cuda_stream stream;

  stats_log_fixture(merge::merge_settings settings, std::size_t num_partitions)
  {
    REQUIRE(memory.setup());
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
    if (num_partitions > 1) { repo.set_num_partitions(num_partitions); }

    duckdb::vector<duckdb::LogicalType> logical{duckdb::LogicalType::BIGINT,
                                                duckdb::LogicalType::BIGINT};
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(duckdb::BoundOrderByNode(
      duckdb::OrderType::ASCENDING,
      duckdb::OrderByNullType::NULLS_LAST,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0)));
    op              = std::make_unique<oom_merge_sort>(sirius::from_duckdb_vec(std::move(logical)),
                                          std::move(orders),
                                          duckdb::vector<std::size_t>{0, 1},
                                          0);
    op->operator_id = kOperatorId;
    op->add_port("default",
                 std::make_unique<sirius_physical_operator::port>(
                   MemoryBarrierType::FULL, &repo, nullptr, nullptr));
    op->set_merge_settings(settings);
    // A fixed device budget, so the route does not depend on the test's GPU limit.
    op->set_device_budget_provider([](const cucascade::memory::memory_space*) {
      return merge::device_budget{
        .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 1};
    });
  }

  ~stats_log_fixture()
  {
    op.reset();
    for (std::size_t partition = 0; partition < repo.num_partitions(); ++partition) {
      while (repo.pop_next_data_batch(partition)) {}
    }
  }

  [[nodiscard]] ::cuda::stream_ref stream_ref() const { return ::cuda::stream_ref{stream.value()}; }

  /// Adds one sorted single-batch GPU run of @p rows rows to @p partition.
  void add_run(std::size_t partition, std::size_t rows, std::uint32_t seed)
  {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<std::int64_t> dist(0, 49999);
    std::vector<std::int64_t> keys(rows);
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      keys[i] = dist(gen);
      ids[i]  = static_cast<std::int64_t>(seed) * 1'000'000 + static_cast<std::int64_t>(i);
    }
    auto const unsorted =
      make_table(make_column(keys, cudf::type_id::INT64), make_column(ids, cudf::type_id::INT64));
    auto const sorted   = stable_sort(unsorted->view(),
                                      {.key_cols        = {0},
                                       .column_order    = {cudf::order::ASCENDING},
                                       .null_precedence = {cudf::null_order::AFTER}});
    auto default_stream = utils::default_stream();
    auto batch          = ::sirius::make_data_batch(
      std::make_unique<cudf::table>(
        sorted->view(), default_stream, utils::get_resource_ref(*memory.gpu0)),
      *memory.gpu0,
      default_stream,
      ::sirius::telemetry::batch_telemetry_info{});
    repo.add_data_batch(batch, partition);
  }

  /// Runs one task the way a GPU pipeline task runs the operator; returns the rows it emitted.
  std::size_t run_task(operator_data& input)
  {
    input.prepare_for_processing(memory.gpu0, stream_ref());
    auto const output = op->execute(input, stream_ref());
    REQUIRE(output);
    std::size_t rows = 0;
    for (auto const& batch :
         dynamic_cast<const pipelineable_operator_data&>(*output).get_data_batches()) {
      rows += sirius::get_batch_rows_and_bytes(*batch).rows.value_or(0);
    }
    return rows;
  }

  /// Runs the operator's remaining tasks one after the other; returns how many there were.
  std::size_t run_to_end()
  {
    std::size_t tasks = 0;
    while (auto input = op->get_next_task_input_data()) {
      REQUIRE(tasks < 10000);
      run_task(*input);
      ++tasks;
    }
    return tasks;
  }
};

/// The merge-sort statistics lines among the records of @p log, which must all be at INFO.
std::vector<std::string> stats_lines(const sirius::test::scoped_recording_log_sink& log)
{
  std::vector<std::string> lines;
  for (auto const& record : log.records()) {
    if (!record.message.starts_with(merge::kMergeSortStatsTag)) { continue; }
    REQUIRE(record.level == sirius::log::level::info);
    lines.push_back(record.message);
  }
  return lines;
}

bool has_token(const std::string& line, const std::string& token)
{
  return (" " + line + " ").find(" " + token + " ") != std::string::npos;
}

/// What threads driving one operator to its end saw.
struct concurrent_drive {
  std::size_t tasks{0};        ///< tasks executed
  std::size_t rows{0};         ///< rows they emitted
  std::size_t early_lines{0};  ///< times a statistics line existed while a task was still to run
  bool timed_out{false};
  std::string error;  ///< what() of the first exception a thread threw
};

/// Creates and executes the operator's tasks from @p num_threads threads, each on its own stream,
/// until the operator has no task and its repository is empty. Catch2 assertions are not
/// thread-safe, so the threads only record; the caller asserts.
concurrent_drive drive_concurrently(stats_log_fixture& test,
                                    const sirius::test::scoped_recording_log_sink& log,
                                    std::size_t num_threads)
{
  std::atomic<std::size_t> tasks{0};
  std::atomic<std::size_t> rows{0};
  std::atomic<std::size_t> early_lines{0};
  std::atomic<bool> timed_out{false};
  std::mutex error_mutex;
  std::string error;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);

  auto const worker = [&] {
    try {
      rmm::cuda_stream stream;
      ::cuda::stream_ref stream_ref{stream.value()};
      while (true) {
        auto input = test.op->get_next_task_input_data();
        if (!input) {
          // Null also while every partition with runs has its task on another thread.
          if (test.repo.total_size() == 0) { return; }
          if (std::chrono::steady_clock::now() > deadline) {
            timed_out = true;
            return;
          }
          std::this_thread::yield();
          continue;
        }
        // This task has not committed, so no commit can have been the operator's last.
        for (auto const& record : log.records()) {
          if (record.message.starts_with(merge::kMergeSortStatsTag)) { ++early_lines; }
        }
        input->prepare_for_processing(test.memory.gpu0, stream_ref);
        auto const output = test.op->execute(*input, stream_ref);
        for (auto const& batch :
             dynamic_cast<const pipelineable_operator_data&>(*output).get_data_batches()) {
          rows += sirius::get_batch_rows_and_bytes(*batch).rows.value_or(0);
        }
        ++tasks;
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lg(error_mutex);
      if (error.empty()) { error = e.what(); }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (std::size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  return {.tasks       = tasks,
          .rows        = rows,
          .early_lines = early_lines,
          .timed_out   = timed_out,
          .error       = error};
}

}  // namespace

TEST_CASE("format_merge_sort_stats writes every counter in the documented order",
          "[merge_sort_stats_log]")
{
  merge_sort_counters counters;
  counters.windowed_partitions            = 1;
  counters.one_shot_partitions            = 2;
  counters.windowed_tasks                 = 3;
  counters.intermediate_tasks             = 4;
  counters.mergers_built                  = 5;
  counters.rounds.rounds                  = 6;
  counters.rounds.rows_out                = 7;
  counters.rounds.round_rows_min          = 8;
  counters.rounds.round_rows_max          = 9;
  counters.rounds.refill_rows             = 10;
  counters.rounds.refill_bytes            = 11;
  counters.rounds.refill_bytes_host       = 12;
  counters.rounds.refill_bytes_disk       = 13;
  counters.rounds.refill_bytes_compressed = 14;
  counters.rounds.runs_read_compressed    = 15;
  counters.rounds.window_growths          = 16;
  counters.rounds.compacted_rows          = 17;
  counters.rounds.fallback_conversions    = 18;
  counters.shrinks                        = 19;
  // Not part of the line.
  counters.rounds.refill_truncated = 99;

  REQUIRE(merge::format_merge_sort_stats(42, counters) ==
          "[merge_sort][stats] op=42 windowed_partitions=1 one_shot_partitions=2 windowed_tasks=3 "
          "intermediate_tasks=4 mergers_built=5 rounds=6 rows_out=7 round_rows_min=8 "
          "round_rows_max=9 refill_rows=10 refill_bytes=11 refill_bytes_host=12 "
          "refill_bytes_disk=13 refill_bytes_compressed=14 runs_read_compressed=15 "
          "window_growths=16 compacted_rows=17 fallback_conversions=18 shrinks=19");

  REQUIRE(merge::format_merge_sort_stats(0, {}) ==
          "[merge_sort][stats] op=0 windowed_partitions=0 one_shot_partitions=0 windowed_tasks=0 "
          "intermediate_tasks=0 mergers_built=0 rounds=0 rows_out=0 round_rows_min=0 "
          "round_rows_max=0 refill_rows=0 refill_bytes=0 refill_bytes_host=0 refill_bytes_disk=0 "
          "refill_bytes_compressed=0 runs_read_compressed=0 window_growths=0 compacted_rows=0 "
          "fallback_conversions=0 shrinks=0");
}

TEST_CASE("windowed merge sort logs one statistics line when its last task commits",
          "[merge_sort_stats_log]")
{
  constexpr std::size_t num_partitions = 2;
  constexpr std::size_t runs_per_part  = 3;
  constexpr std::size_t rows_per_run   = 3000;
  stats_log_fixture test(
    {.mode = sirius::sort_merge_mode::windowed, .window_rows = 1024, .output_bytes = 64 * 1024},
    num_partitions);
  for (std::size_t p = 0; p < num_partitions; ++p) {
    for (std::size_t r = 0; r < runs_per_part; ++r) {
      test.add_run(p, rows_per_run, static_cast<std::uint32_t>(1 + p * runs_per_part + r));
    }
  }
  sirius::test::scoped_recording_log_sink log{"info"};

  // Every task but the last leaves work behind, and logs nothing.
  std::size_t tasks = 0;
  while (auto input = test.op->get_next_task_input_data()) {
    REQUIRE(tasks < 10000);
    REQUIRE(stats_lines(log).empty());
    test.run_task(*input);
    ++tasks;
  }
  REQUIRE(tasks > num_partitions);
  REQUIRE(test.repo.total_size() == 0);

  auto const counters = test.op->counters();
  REQUIRE(counters.windowed_partitions == num_partitions);
  REQUIRE(counters.one_shot_partitions == 0);
  REQUIRE(counters.windowed_tasks == tasks);
  REQUIRE(counters.rounds.rounds > 0);
  REQUIRE(counters.rounds.rows_out == num_partitions * runs_per_part * rows_per_run);
  REQUIRE(counters.shrinks == 0);

  auto const lines = stats_lines(log);
  REQUIRE(lines.size() == 1);
  REQUIRE(lines.front() == merge::format_merge_sort_stats(kOperatorId, counters));
  REQUIRE(has_token(lines.front(), "op=7"));
  REQUIRE(has_token(lines.front(), "windowed_partitions=2"));
  REQUIRE(has_token(lines.front(), "rounds=" + std::to_string(counters.rounds.rounds)));
  REQUIRE_FALSE(has_token(lines.front(), "rounds=0"));

  // Asking a drained operator for more work does not log again.
  REQUIRE_FALSE(test.op->get_next_task_input_data());
  REQUIRE(stats_lines(log).size() == 1);
}

TEST_CASE("one-shot merge sort logs one statistics line when its last task commits",
          "[merge_sort_stats_log]")
{
  constexpr std::size_t num_partitions = 3;
  stats_log_fixture test({.mode = sirius::sort_merge_mode::one_shot}, num_partitions);
  // Partition 1 stays empty.
  for (std::size_t p : {std::size_t{0}, std::size_t{2}}) {
    for (std::size_t r = 0; r < 3; ++r) {
      test.add_run(p, 500, static_cast<std::uint32_t>(20 + p * 3 + r));
    }
  }
  sirius::test::scoped_recording_log_sink log{"info"};

  // Both tasks exist before either runs, as when the task creator drains the operator.
  auto first  = test.op->get_next_task_input_data();
  auto second = test.op->get_next_task_input_data();
  REQUIRE(first);
  REQUIRE(second);
  REQUIRE_FALSE(test.op->get_next_task_input_data());
  REQUIRE(first->get_type() != operator_data_type::WINDOWED_MERGE);

  REQUIRE(test.run_task(*first) == 1500);
  REQUIRE(stats_lines(log).empty());
  REQUIRE(test.run_task(*second) == 1500);

  auto const counters = test.op->counters();
  REQUIRE(counters.one_shot_partitions == 2);
  REQUIRE(counters.windowed_partitions == 0);
  REQUIRE(counters.windowed_tasks == 0);

  auto const lines = stats_lines(log);
  REQUIRE(lines.size() == 1);
  REQUIRE(lines.front() == merge::format_merge_sort_stats(kOperatorId, counters));
  REQUIRE(has_token(lines.front(), "one_shot_partitions=2"));
  REQUIRE(has_token(lines.front(), "rounds=0"));
}

TEST_CASE("merge sort in automatic mode logs one line for one-shot and windowed partitions",
          "[merge_sort_stats_log]")
{
  // 0.0001 of the 8 GiB budget is about 840 KiB: the small partition merges one-shot, the large
  // one windowed.
  stats_log_fixture test({.mode                  = sirius::sort_merge_mode::automatic,
                          .window_rows           = 1024,
                          .output_bytes          = 64 * 1024,
                          .one_shot_max_fraction = 0.0001},
                         2);
  for (std::size_t r = 0; r < 3; ++r) {
    test.add_run(0, 200, static_cast<std::uint32_t>(40 + r));
    test.add_run(1, 30000, static_cast<std::uint32_t>(50 + r));
  }
  sirius::test::scoped_recording_log_sink log{"info"};

  // The one-shot task is created first and committed last, after the windowed partition is done.
  auto one_shot = test.op->get_next_task_input_data();
  REQUIRE(one_shot);
  REQUIRE(one_shot->get_type() != operator_data_type::WINDOWED_MERGE);
  REQUIRE(test.run_to_end() > 1);
  REQUIRE(stats_lines(log).empty());
  REQUIRE(test.run_task(*one_shot) == 600);

  auto const counters = test.op->counters();
  REQUIRE(counters.one_shot_partitions == 1);
  REQUIRE(counters.windowed_partitions == 1);
  REQUIRE(counters.rounds.rows_out == 90000);
  auto const lines = stats_lines(log);
  REQUIRE(lines.size() == 1);
  REQUIRE(lines.front() == merge::format_merge_sort_stats(kOperatorId, counters));
}

TEST_CASE("merge sort counts the window shrinks of its tasks retried after an OOM",
          "[merge_sort_stats_log]")
{
  // One halving away from the window floor, and already at the 16 MiB output floor.
  constexpr std::size_t window_rows = 2 * merge::kShrinkWindowRowsFloor;
  stats_log_fixture test({.mode         = sirius::sort_merge_mode::windowed,
                          .window_rows  = window_rows,
                          .output_bytes = 16 * kMiB},
                         1);
  for (std::size_t r = 0; r < 4; ++r) {
    test.add_run(0, 5000, static_cast<std::uint32_t>(60 + r));
  }
  sirius_physical_merge_sort::reset_process_counters();
  sirius::test::scoped_recording_log_sink log{"info"};

  auto input     = test.op->get_next_task_input_data();
  auto& windowed = dynamic_cast<merge::windowed_merge_input&>(*input);
  REQUIRE(windowed.config().window_rows == window_rows);

  // The task runs out of memory twice. After each OOM the executor asks the input to shrink
  // (gpu_pipeline_executor's retry path) and runs the task again.
  test.op->ooms = 2;
  REQUIRE_THROWS_AS(test.run_task(*input), rmm::out_of_memory);
  REQUIRE(input->shrink_for_retry());
  REQUIRE(windowed.config().window_rows == merge::kShrinkWindowRowsFloor);
  REQUIRE(test.op->counters().shrinks == 1);

  // At its floors the input does not shrink, and nothing is counted.
  REQUIRE_THROWS_AS(test.run_task(*input), rmm::out_of_memory);
  REQUIRE_FALSE(input->shrink_for_retry());
  REQUIRE(test.op->counters().shrinks == 1);
  REQUIRE(stats_lines(log).empty());

  REQUIRE(test.run_task(*input) > 0);
  test.run_to_end();
  REQUIRE(test.repo.total_size() == 0);

  auto const counters = test.op->counters();
  REQUIRE(counters.shrinks == 1);
  REQUIRE(counters.rounds.rows_out == 20000);
  REQUIRE(sirius_physical_merge_sort::process_counters().shrinks == 1);
  auto const lines = stats_lines(log);
  REQUIRE(lines.size() == 1);
  REQUIRE(lines.front() == merge::format_merge_sort_stats(kOperatorId, counters));
  REQUIRE(lines.front().ends_with(" shrinks=1"));
}

TEST_CASE("merge sort with no input logs one all-zero statistics line when it is finalized",
          "[merge_sort_stats_log]")
{
  stats_log_fixture test(
    {.mode = sirius::sort_merge_mode::windowed, .window_rows = 1024, .output_bytes = 64 * 1024}, 2);
  sirius::test::scoped_recording_log_sink log{"info"};

  REQUIRE_FALSE(test.op->get_next_task_input_data());
  REQUIRE(stats_lines(log).empty());

  test.op->finalize_operator();
  auto const lines = stats_lines(log);
  REQUIRE(lines.size() == 1);
  REQUIRE(lines.front() == merge::format_merge_sort_stats(kOperatorId, {}));

  test.op->finalize_operator();
  REQUIRE(stats_lines(log).size() == 1);
}

TEST_CASE("merge sort that logged its statistics at its last commit logs nothing when finalized",
          "[merge_sort_stats_log]")
{
  auto const mode = GENERATE(sirius::sort_merge_mode::windowed, sirius::sort_merge_mode::one_shot);
  stats_log_fixture test({.mode = mode, .window_rows = 1024, .output_bytes = 64 * 1024}, 2);
  for (std::size_t p = 0; p < 2; ++p) {
    for (std::size_t r = 0; r < 3; ++r) {
      test.add_run(p, 2000, static_cast<std::uint32_t>(70 + p * 3 + r));
    }
  }
  sirius::test::scoped_recording_log_sink log{"info"};

  REQUIRE(test.run_to_end() >= 2);
  auto const lines = stats_lines(log);
  REQUIRE(lines.size() == 1);

  test.op->finalize_operator();
  REQUIRE(stats_lines(log) == lines);
}

TEST_CASE("merge sort finalized before its last task commits logs once, with the counters so far",
          "[merge_sort_stats_log]")
{
  // What a LIMIT downstream does: the pipeline finishes while the operator still has runs.
  stats_log_fixture test(
    {.mode = sirius::sort_merge_mode::windowed, .window_rows = 1024, .output_bytes = 64 * 1024}, 1);
  for (std::size_t r = 0; r < 3; ++r) {
    test.add_run(0, 3000, static_cast<std::uint32_t>(80 + r));
  }
  sirius::test::scoped_recording_log_sink log{"info"};

  auto first = test.op->get_next_task_input_data();
  REQUIRE(first);
  test.run_task(*first);
  auto const counters_at_finalize = test.op->counters();
  REQUIRE(counters_at_finalize.windowed_tasks == 1);
  REQUIRE(stats_lines(log).empty());

  test.op->finalize_operator();
  auto const lines = stats_lines(log);
  REQUIRE(lines.size() == 1);
  REQUIRE(lines.front() == merge::format_merge_sort_stats(kOperatorId, counters_at_finalize));

  // Tasks that still run afterwards commit without a second line.
  REQUIRE(test.run_to_end() > 0);
  REQUIRE(test.repo.total_size() == 0);
  REQUIRE(stats_lines(log) == lines);
}

TEST_CASE("merge sort driven from four threads logs exactly one statistics line",
          "[merge_sort_stats_log]")
{
  constexpr std::size_t num_threads = 4;
  constexpr std::size_t repetitions = 10;
  constexpr std::size_t small_rows  = 200;
  constexpr std::size_t large_rows  = 30000;
  constexpr std::size_t runs        = 3;

  struct scenario {
    sirius::sort_merge_mode mode;
    std::vector<std::size_t> rows_per_run;  // of each partition
    std::size_t one_shot_partitions;
    std::size_t windowed_partitions;
  };
  // Automatic: 0.0001 of the 8 GiB budget is about 840 KiB, so the small partitions merge
  // one-shot and the large ones windowed.
  auto const which = GENERATE(as<std::string>{}, "automatic", "windowed", "one_shot");
  auto const scen =
    which == "automatic" ? scenario{sirius::sort_merge_mode::automatic,
                                    {small_rows, large_rows, small_rows, large_rows, small_rows},
                                    3,
                                    2}
    : which == "windowed"
      ? scenario{sirius::sort_merge_mode::windowed,
                 {large_rows, large_rows, large_rows, large_rows},
                 0,
                 4}
      : scenario{sirius::sort_merge_mode::one_shot, {large_rows, small_rows, large_rows}, 3, 0};
  CAPTURE(which);

  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    CAPTURE(repetition);
    stats_log_fixture test({.mode                  = scen.mode,
                            .window_rows           = 1024,
                            .output_bytes          = 64 * 1024,
                            .one_shot_max_fraction = 0.0001},
                           scen.rows_per_run.size());
    std::size_t total_rows = 0;
    for (std::size_t p = 0; p < scen.rows_per_run.size(); ++p) {
      for (std::size_t r = 0; r < runs; ++r) {
        test.add_run(p, scen.rows_per_run[p], static_cast<std::uint32_t>(100 + p * runs + r));
        total_rows += scen.rows_per_run[p];
      }
    }
    sirius::test::scoped_recording_log_sink log{"info"};

    auto const drive = drive_concurrently(test, log, num_threads);
    REQUIRE(drive.error.empty());
    REQUIRE_FALSE(drive.timed_out);
    REQUIRE(drive.early_lines == 0);
    REQUIRE(drive.rows == total_rows);
    REQUIRE(test.repo.total_size() == 0);
    REQUIRE_FALSE(test.op->get_next_task_input_data());

    auto const counters = test.op->counters();
    REQUIRE(counters.one_shot_partitions == scen.one_shot_partitions);
    REQUIRE(counters.windowed_partitions == scen.windowed_partitions);
    REQUIRE(counters.windowed_tasks + counters.one_shot_partitions == drive.tasks);

    // The line holds the final counters: it was logged by the last commit, not an earlier one.
    auto const lines = stats_lines(log);
    REQUIRE(lines.size() == 1);
    REQUIRE(lines.front() == merge::format_merge_sort_stats(kOperatorId, counters));

    test.op->finalize_operator();
    REQUIRE(stats_lines(log).size() == 1);
  }
}

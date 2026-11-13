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
#include "creator/task_creator.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "exec/config.hpp"
#include "helper/type_conversions.hpp"
#include "operator/merge/merge_pipeline_harness.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/query_retirement_test_utils.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "sirius_config.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_device.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/merge/merge_chunk_position.hpp>
#include <op/merge/merge_config.hpp>
#include <op/merge/windowed_merge_input.hpp>
#include <op/sirius_physical_merge_sort.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

namespace {

namespace merge_utils = sirius::test::merge_utils;
namespace pl          = sirius::pipeline;

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kGiB = std::size_t{1} << 30;

/// Set while the calling thread is inside the MERGE_SORT's execute().
thread_local bool t_in_merge_execute = false;

/// (key, payload) rows on the host. The payload is `run * kPayloadStride + row`, so the order a
/// merge with ties broken by run index produces is the plain (key, payload) order.
using host_rows                       = std::vector<std::pair<std::int64_t, std::int64_t>>;
constexpr std::int64_t kPayloadStride = 1'000'000;

// ---------------------------------------------------------------------------
// Observers
// ---------------------------------------------------------------------------

/// What one MERGE_SORT task did.
struct task_record {
  std::size_t partition{0};
  bool windowed{false};
  bool final_pass{true};
  std::size_t first_chunk{0};
  std::size_t runs{0};  ///< cursors of a windowed task, batches of a one-shot one
  std::optional<std::size_t> position;
  std::size_t output_batches{0};
  std::size_t output_rows{0};
};

/// MERGE_SORT that records every task it executes, in completion order.
class recording_merge_sort final : public sirius_physical_merge_sort {
 public:
  using sirius_physical_merge_sort::sirius_physical_merge_sort;

  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         ::cuda::stream_ref stream) override
  {
    struct scope {
      scope() { t_in_merge_execute = true; }
      ~scope() { t_in_merge_execute = false; }
    } in_execute;

    auto const running = _running.fetch_add(1) + 1;
    auto peak          = _peak_running.load();
    while (running > peak && !_peak_running.compare_exchange_weak(peak, running)) {}
    struct done {
      std::atomic<std::size_t>& running;
      ~done() { running.fetch_sub(1); }
    } leaving{_running};

    task_record record;
    if (auto const* windowed = dynamic_cast<const merge::windowed_merge_input*>(&input)) {
      record.windowed    = true;
      record.partition   = windowed->partition();
      record.final_pass  = windowed->final_pass();
      record.first_chunk = windowed->first_chunk();
      record.runs        = windowed->cursors().size();
    } else {
      auto const& one_shot = dynamic_cast<const pipelineable_operator_data&>(input);
      record.runs          = one_shot.get_data_batches().size();
      if (auto const position = one_shot.get_position()) {
        record.partition = merge::split_merge_chunk_position(*position).first;
      }
    }

    auto output           = sirius_physical_merge_sort::execute(input, stream);
    auto const& published = dynamic_cast<const pipelineable_operator_data&>(*output);
    record.position       = published.get_position();
    record.output_batches = published.get_data_batches().size();
    for (auto const& batch : published.get_data_batches()) {
      record.output_rows += sirius::get_batch_rows_and_bytes(*batch).rows.value_or(0);
    }
    std::lock_guard<std::mutex> lock(_mutex);
    _records.push_back(record);
    return output;
  }

  [[nodiscard]] std::vector<task_record> records() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _records;
  }

  /// Most tasks of this operator that were inside execute() at once.
  [[nodiscard]] std::size_t peak_running() const { return _peak_running.load(); }

 private:
  mutable std::mutex _mutex;
  std::vector<task_record> _records;
  std::atomic<std::size_t> _running{0};
  std::atomic<std::size_t> _peak_running{0};
};

/// The MERGE_SORT's input repository; records the batches a merge task adds to it.
class observed_input_repository final : public shared_data_repository {
 public:
  void add_data_batch_unlocked(const std::shared_ptr<data_batch>& batch,
                               std::size_t partition = 0) override
  {
    shared_data_repository::add_data_batch_unlocked(batch, partition);
    if (!t_in_merge_execute) { return; }
    std::lock_guard<std::mutex> lock(_mutex);
    _pushed_back.emplace_back(batch->get_batch_id(), partition);
  }

  /// (batch id, partition) of every batch added from inside the MERGE_SORT's execute().
  [[nodiscard]] std::vector<std::pair<std::uint64_t, std::size_t>> pushed_back() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _pushed_back;
  }

 private:
  mutable std::mutex _mutex;
  std::vector<std::pair<std::uint64_t, std::size_t>> _pushed_back;
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct harness_options {
  std::size_t num_partitions   = 1;
  std::size_t executor_threads = 1;
  /// GPU memory spaces, each with its own executor of `executor_threads` threads.
  std::size_t num_gpus = 1;
};

/**
 * @brief A recording MERGE_SORT over (key BIGINT, payload BIGINT) in a merge_utils::merge_pipeline,
 * with runs added per partition and per device.
 */
struct merge_harness {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  memory_space* gpu  = nullptr;
  memory_space* host = nullptr;
  /// The GPU memory space of each device; gpus[0] == gpu.
  std::vector<memory_space*> gpus;
  observed_input_repository input;
  std::unique_ptr<recording_merge_sort> merge;
  /// Requests for the MERGE_SORT, and how many of them came from inside its own execute().
  std::atomic<std::size_t> requests{0};
  std::atomic<std::size_t> rearms_from_execute{0};
  /// Declared last: it is torn down (and the input popped) before anything above.
  std::unique_ptr<merge_utils::merge_pipeline> pipe;

  explicit merge_harness(harness_options options = {})
    : manager{initialize_memory_manager(options.num_gpus,
                                        {.gpu_bytes                = 512 * kMiB,
                                         .gpu_reservation_fraction = 0.9,
                                         .divide_among_gpus        = false,
                                         .per_stream_tracking      = false})}
  {
    for (std::size_t device = 0; device < options.num_gpus; ++device) {
      gpus.push_back(
        const_cast<memory_space*>(manager->get_memory_space(Tier::GPU, static_cast<int>(device))));
      REQUIRE(gpus.back() != nullptr);
    }
    gpu        = gpus.front();
    auto hosts = manager->get_memory_spaces_for_tier(Tier::HOST);
    REQUIRE_FALSE(hosts.empty());
    host = const_cast<memory_space*>(hosts.front());
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
    if (options.num_partitions > 1) { input.set_num_partitions(options.num_partitions); }

    duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                              duckdb::LogicalType::BIGINT};
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(duckdb::BoundOrderByNode(
      duckdb::OrderType::ASCENDING,
      duckdb::OrderByNullType::NULLS_LAST,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0)));
    merge = std::make_unique<recording_merge_sort>(sirius::from_duckdb_vec(std::move(types)),
                                                   std::move(orders),
                                                   duckdb::vector<std::size_t>{0, 1},
                                                   0);
    pipe  = std::make_unique<merge_utils::merge_pipeline>(
      *merge,
      input,
      *manager,
      merge_utils::merge_pipeline_options{
         .executor_threads   = options.executor_threads,
         .thread_name_prefix = "merge-continuation",
         .on_request =
          [this] {
            requests.fetch_add(1);
            if (t_in_merge_execute) { rearms_from_execute.fetch_add(1); }
          },
         .describe =
          [this] { return std::to_string(merge->records().size()) + " tasks executed, "; }});
  }

  /// Applies the sort_merge_* settings the way the plan generator does, from operator_params.
  void configure(const sirius::operator_params& params,
                 merge::device_budget budget = {.gpu_available_bytes = 8 * kGiB,
                                                .gpu_total_bytes     = 8 * kGiB,
                                                .concurrent_tasks    = 1})
  {
    merge->set_merge_settings(merge::merge_settings::from(params));
    merge->set_device_budget_provider([budget](const memory_space*) { return budget; });
  }

  /**
   * @brief Add one sorted run of @p rows rows to @p partition: keys drawn from
   * [key_base, key_base + key_range), so runs share keys, and the payload described at host_rows.
   *
   * The run is created in the GPU memory space of @p device.
   *
   * @return The run's rows, in run order.
   */
  host_rows add_run(std::size_t partition,
                    std::size_t run_index,
                    std::size_t rows,
                    std::int64_t key_base  = 0,
                    std::int64_t key_range = 5000,
                    std::size_t device     = 0)
  {
    auto& space = *gpus.at(device);
    rmm::cuda_set_device_raii on_device{rmm::cuda_device_id{space.get_device_id()}};
    std::vector<std::int64_t> keys(rows);
    auto seed = static_cast<std::uint64_t>(partition * 7919 + run_index * 104729 + 12345);
    for (auto& key : keys) {
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      key =
        key_base + static_cast<std::int64_t>((seed >> 33) % static_cast<std::uint64_t>(key_range));
    }
    std::sort(keys.begin(), keys.end());
    std::vector<std::int64_t> payload(rows);
    host_rows result(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      payload[i] =
        static_cast<std::int64_t>(run_index) * kPayloadStride + static_cast<std::int64_t>(i);
      result[i] = {keys[i], payload[i]};
    }
    auto table  = merge_utils::make_table(merge_utils::make_column(keys, cudf::type_id::INT64),
                                         merge_utils::make_column(payload, cudf::type_id::INT64));
    auto stream = default_stream();
    auto batch  = sirius::make_data_batch(
      std::make_unique<cudf::table>(table->view(), stream, get_resource_ref(space)),
      space,
      stream,
      sirius::telemetry::batch_telemetry_info{});
    input.add_data_batch(batch, partition);
    _input_ids.emplace_back(batch->get_batch_id(), partition);
    _inputs.push_back(batch);
    return result;
  }

  /// Downgrade the @p index-th added run to the HOST tier, as a spill would.
  void downgrade_to_host(std::size_t index)
  {
    auto batch = _inputs.at(index).lock();
    REQUIRE(batch);
    // The conversion needs a stream of the space, not the default stream.
    auto const stream = gpu->acquire_stream();
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(
        sirius::converter_registry::get(), host, stream);
    }
    stream.sync();
    REQUIRE(batch->to_read_only().get_current_tier() == Tier::HOST);
  }

  [[nodiscard]] Tier tier_of(std::size_t index) const
  {
    auto batch = _inputs.at(index).lock();
    REQUIRE(batch);
    return batch->to_read_only().get_current_tier();
  }

  /// Runs the merge to its end; see merge_utils::merge_pipeline::run.
  void run() { pipe->run(); }

  /// Requires that every input batch left the repository and was freed.
  void require_inputs_popped()
  {
    REQUIRE(input.total_size() == 0);
    for (auto const& [id, partition] : _input_ids) {
      REQUIRE_FALSE(input.get_data_batch_by_id(id, partition));
    }
    for (auto const& batch : _inputs) {
      REQUIRE(batch.expired());
    }
  }

  /// The collected chunks of @p partition (all partitions when nullopt), in position order.
  [[nodiscard]] host_rows collected(std::optional<std::size_t> partition = std::nullopt) const
  {
    host_rows rows;
    for (auto const& [position, batch] : pipe->collector.chunks()) {
      if (partition && merge::split_merge_chunk_position(position).first != *partition) {
        continue;
      }
      // A chunk is read on the device of the task that produced it.
      auto const device = batch->to_read_only().get_memory_space()->get_device_id();
      rmm::cuda_set_device_raii on_device{rmm::cuda_device_id{device}};
      auto const view = sirius::get_cudf_table_view(*batch);
      REQUIRE(view.num_columns() == 2);
      auto const keys    = copy_column_to_host<std::int64_t>(view.column(0));
      auto const payload = copy_column_to_host<std::int64_t>(view.column(1));
      for (std::size_t i = 0; i < keys.size(); ++i) {
        rows.emplace_back(keys[i], payload[i]);
      }
    }
    return rows;
  }

 private:
  std::vector<std::pair<std::uint64_t, std::size_t>> _input_ids;
  std::vector<std::weak_ptr<data_batch>> _inputs;
};

/// windowed mode, 1024-row windows and 64 KiB output chunks: 4096 two-column rows a task.
sirius::operator_params small_window_params()
{
  sirius::operator_params params;
  params.sort_merge_mode_setting = sirius::sort_merge_mode::windowed;
  params.sort_merge_window_rows  = 1024;
  params.sort_merge_output_bytes = 64 * 1024;
  return params;
}

/// @p rows in the order a merge with ties broken by run index emits them.
host_rows full_sort(host_rows rows)
{
  std::sort(rows.begin(), rows.end());
  return rows;
}

host_rows concatenate(const std::vector<host_rows>& runs)
{
  host_rows all;
  for (auto const& run : runs) {
    all.insert(all.end(), run.begin(), run.end());
  }
  return all;
}

bool sorted_by_key(const host_rows& rows)
{
  return std::is_sorted(
    rows.begin(), rows.end(), [](auto const& a, auto const& b) { return a.first < b.first; });
}

/// Requires @p actual to be the rows of @p expected in key order; rows with equal keys may come
/// in any order (cudf::merge and a fan-in pass do not keep the run order of ties).
void require_sorted_rows(const host_rows& actual, const host_rows& expected)
{
  REQUIRE(actual.size() == expected.size());
  REQUIRE(sorted_by_key(actual));
  REQUIRE(full_sort(actual) == full_sort(expected));
}

/**
 * @brief Requires the per-partition task chain: at most one task at a time (so the completion
 * order is the chain), final-pass chunks numbered 0, 1, 2, ... and published at
 * make_merge_chunk_position(partition, chunk), one batch each.
 *
 * @return The number of final-pass tasks of @p partition.
 */
std::size_t require_chunk_chain(const std::vector<task_record>& records, std::size_t partition)
{
  std::size_t next_chunk = 0;
  for (auto const& record : records) {
    if (record.partition != partition) { continue; }
    REQUIRE(record.windowed);
    if (!record.final_pass) { continue; }
    REQUIRE(record.first_chunk == next_chunk);
    REQUIRE(record.output_batches == 1);
    REQUIRE(record.output_rows > 0);
    REQUIRE(record.position == merge::make_merge_chunk_position(partition, next_chunk));
    ++next_chunk;
  }
  return next_chunk;
}

}  // namespace

TEST_CASE("windowed merge sort continues one partition through a chain of positioned tasks",
          "[merge_sort_continuation]")
{
  merge_harness test;
  test.configure(small_window_params());
  std::vector<host_rows> runs;
  for (std::size_t run = 0; run < 5; ++run) {
    runs.push_back(test.add_run(0, run, 3000 + 500 * run));
  }
  auto const expected = full_sort(concatenate(runs));

  test.run();

  auto const records = test.merge->records();
  REQUIRE(records.size() >= 3);
  for (auto const& record : records) {
    REQUIRE(record.final_pass);
    REQUIRE(record.runs <= 5);
  }
  REQUIRE(records.front().runs == 5);
  REQUIRE(require_chunk_chain(records, 0) == records.size());

  // Positions are unique (the collector rejects a duplicate) and were published in order, as
  // every task of the partition starts only after the previous one committed.
  auto const arrival = test.pipe->collector.arrival();
  REQUIRE(arrival.size() == records.size());
  for (std::size_t chunk = 0; chunk < arrival.size(); ++chunk) {
    REQUIRE(arrival[chunk] == merge::make_merge_chunk_position(0, chunk));
  }
  // No chunk exceeds the output target: 64 KiB of 16-byte rows.
  for (auto const& record : records) {
    REQUIRE(record.output_rows <= 4096);
  }

  // The first task came from the request run() makes; every other one from the operator
  // scheduling itself when its task committed.
  REQUIRE(test.rearms_from_execute.load() == records.size() - 1);
  REQUIRE(test.requests.load() == records.size());
  REQUIRE(test.pipe->pipeline->get_tasks_completed() == records.size());
  REQUIRE(test.pipe->pipeline->is_pipeline_finished());

  REQUIRE(test.collected() == expected);
  REQUIRE(test.input.pushed_back().empty());
  test.require_inputs_popped();
}

TEST_CASE("windowed merge sort merges several partitions in parallel", "[merge_sort_continuation]")
{
  // More executor threads than partitions with runs, so two tasks of one partition could overlap.
  merge_harness test{{.num_partitions = 4, .executor_threads = 4}};
  test.configure(small_window_params());
  // Partition 2 stays empty; the keys of partition p are all below those of partition p + 1.
  const std::vector<std::size_t> partitions{0, 1, 3};
  std::map<std::size_t, std::vector<host_rows>> runs;
  for (auto partition : partitions) {
    for (std::size_t run = 0; run < 4; ++run) {
      runs[partition].push_back(test.add_run(partition,
                                             run,
                                             2500 + 300 * run + 100 * partition,
                                             static_cast<std::int64_t>(partition) * 10000,
                                             4000));
    }
  }

  test.run();

  auto const records = test.merge->records();
  host_rows expected_all;
  std::size_t total_chunks = 0;
  for (auto partition : partitions) {
    auto const chunks = require_chunk_chain(records, partition);
    REQUIRE(chunks >= 3);
    total_chunks += chunks;
    auto const expected = full_sort(concatenate(runs[partition]));
    REQUIRE(test.collected(partition) == expected);
    expected_all.insert(expected_all.end(), expected.begin(), expected.end());
  }
  REQUIRE(records.size() == total_chunks);
  for (auto const& record : records) {
    REQUIRE(record.partition != 2);
  }

  // Unique across partitions: one collected chunk per task, each at its own position.
  auto const chunks = test.pipe->collector.chunks();
  REQUIRE(chunks.size() == total_chunks);
  std::set<std::size_t> positions;
  for (auto const& record : records) {
    REQUIRE(record.position.has_value());
    REQUIRE(positions.insert(*record.position).second);
    REQUIRE(chunks.contains(*record.position));
  }
  // Within a partition the chunks were published in order, whatever the other partitions did.
  std::map<std::size_t, std::size_t> last_chunk;
  for (auto position : test.pipe->collector.arrival()) {
    auto const [partition, chunk] = merge::split_merge_chunk_position(position);
    auto const it                 = last_chunk.find(partition);
    REQUIRE(chunk == (it == last_chunk.end() ? 0 : it->second + 1));
    last_chunk[partition] = chunk;
  }

  // Read in position order, the partitions concatenate into the query's order.
  REQUIRE(test.collected() == expected_all);
  REQUIRE(sorted_by_key(expected_all));
  // One partition never has two tasks, so the peak is bounded by the partitions with runs, below
  // the executor's four threads.
  REQUIRE(test.merge->peak_running() <= partitions.size());
  REQUIRE(test.pipe->pipeline->is_pipeline_finished());
  test.require_inputs_popped();
}

TEST_CASE("windowed merge sort with more runs than the fan-in cap merges in passes",
          "[merge_sort_continuation]")
{
  merge_harness test{{.num_partitions = 2}};
  auto params                  = small_window_params();
  params.sort_merge_max_fan_in = 2;
  test.configure(params);
  constexpr std::size_t partition = 1;
  std::vector<host_rows> runs;
  for (std::size_t run = 0; run < 7; ++run) {
    runs.push_back(test.add_run(partition, run, 1500 + 250 * run));
  }
  auto const expected = full_sort(concatenate(runs));

  test.run();

  auto const records             = test.merge->records();
  std::size_t intermediate_tasks = 0;
  std::size_t final_tasks        = 0;
  for (auto const& record : records) {
    REQUIRE(record.windowed);
    REQUIRE(record.partition == partition);
    REQUIRE(record.runs <= 2);
    if (record.final_pass) {
      ++final_tasks;
      continue;
    }
    // Every intermediate task precedes the final pass and publishes nothing downstream.
    REQUIRE(final_tasks == 0);
    REQUIRE_FALSE(record.position.has_value());
    REQUIRE(record.output_batches == 0);
    ++intermediate_tasks;
  }
  // 7 runs take five two-run intermediate merges before two runs are left for the final pass.
  REQUIRE(intermediate_tasks >= 5);
  REQUIRE(final_tasks >= 3);
  REQUIRE(require_chunk_chain(records, partition) == final_tasks);
  REQUIRE(test.pipe->collector.chunks().size() == final_tasks);

  // Each intermediate task wrote one batch of its run back into the partition it reads.
  auto const pushed_back = test.input.pushed_back();
  REQUIRE(pushed_back.size() == intermediate_tasks);
  std::set<std::uint64_t> intermediate_ids;
  for (auto const& [id, pushed_partition] : pushed_back) {
    REQUIRE(pushed_partition == partition);
    REQUIRE(intermediate_ids.insert(id).second);
  }

  REQUIRE(test.rearms_from_execute.load() == records.size() - 1);
  REQUIRE(test.pipe->pipeline->is_pipeline_finished());

  // A pass reorders runs, so equal keys need not keep the run order.
  require_sorted_rows(test.collected(), expected);

  // A merger lasts for one intermediate pass, or for the final pass.
  auto const counters = test.merge->counters();
  REQUIRE(counters.windowed_partitions == 1);
  REQUIRE(counters.windowed_tasks == records.size());
  REQUIRE(counters.intermediate_tasks == intermediate_tasks);
  REQUIRE(counters.mergers_built >= 2);
  REQUIRE(counters.mergers_built <= records.size());

  // The intermediate runs were consumed and popped too.
  test.require_inputs_popped();
  for (auto id : intermediate_ids) {
    REQUIRE_FALSE(test.input.get_data_batch_by_id(id, partition));
  }
}

TEST_CASE("windowed merge sort merges runs spread over the GPU and HOST tiers",
          "[merge_sort_continuation]")
{
  merge_harness test;
  test.configure(small_window_params());
  std::vector<host_rows> runs;
  for (std::size_t run = 0; run < 5; ++run) {
    runs.push_back(test.add_run(0, run, 3000 + 400 * run));
  }
  test.downgrade_to_host(1);
  test.downgrade_to_host(3);
  REQUIRE(test.tier_of(0) == Tier::GPU);
  REQUIRE(test.tier_of(1) == Tier::HOST);
  auto const expected = full_sort(concatenate(runs));

  test.run();

  auto const records = test.merge->records();
  REQUIRE(records.size() >= 3);
  REQUIRE(require_chunk_chain(records, 0) == records.size());
  REQUIRE(test.rearms_from_execute.load() == records.size() - 1);
  REQUIRE(test.collected() == expected);
  REQUIRE(test.pipe->pipeline->is_pipeline_finished());
  test.require_inputs_popped();
}

TEST_CASE("windowed merge sort reads each HOST row once across the tasks of a partition",
          "[merge_sort_continuation]")
{
  constexpr std::size_t num_runs     = 8;
  constexpr std::size_t rows_per_run = 3000;
  merge_harness test;
  test.configure(small_window_params());
  std::vector<host_rows> runs;
  for (std::size_t run = 0; run < num_runs; ++run) {
    runs.push_back(test.add_run(0, run, rows_per_run));
    test.downgrade_to_host(run);
  }
  auto const total_rows = num_runs * rows_per_run;
  auto const expected   = full_sort(concatenate(runs));

  test.run();

  auto const records = test.merge->records();
  REQUIRE(records.size() >= 4);
  REQUIRE(require_chunk_chain(records, 0) == records.size());
  REQUIRE(test.collected() == expected);

  // One merger served every task, so the rows a task buffered and did not emit were not read
  // again by the next one: a merger per task would read about (k - 1) windows more per task.
  auto const counters = test.merge->counters();
  REQUIRE(counters.windowed_partitions == 1);
  REQUIRE(counters.one_shot_partitions == 0);
  REQUIRE(counters.windowed_tasks == records.size());
  REQUIRE(counters.intermediate_tasks == 0);
  REQUIRE(counters.mergers_built == 1);
  REQUIRE(counters.rounds.rows_out == total_rows);
  REQUIRE(counters.rounds.refill_rows >= total_rows);
  REQUIRE(counters.rounds.refill_rows <= total_rows + total_rows / 100);
  REQUIRE(counters.rounds.fallback_conversions == 0);
  test.require_inputs_popped();
}

TEST_CASE("windowed merge sort keeps its merger when a run ends before the others",
          "[merge_sort_continuation]")
{
  merge_harness test;
  test.configure(small_window_params());
  std::vector<host_rows> runs;
  // Run 0 holds only the smallest keys, so it is read to its end by the first tasks.
  runs.push_back(test.add_run(0, 0, 2000, 0, 500));
  for (std::size_t run = 1; run < 5; ++run) {
    runs.push_back(test.add_run(0, run, 4000, 0, 5000));
  }
  test.downgrade_to_host(0);
  test.downgrade_to_host(2);
  test.downgrade_to_host(4);
  std::size_t host_rows_total = runs[0].size() + runs[2].size() + runs[4].size();
  auto const expected         = full_sort(concatenate(runs));

  test.run();

  auto const records = test.merge->records();
  REQUIRE(records.size() >= 4);
  // The first task still read run 0; the last ones no longer did.
  REQUIRE(records.front().runs == 5);
  REQUIRE(records.back().runs < 5);
  REQUIRE(test.collected() == expected);
  auto const counters = test.merge->counters();
  REQUIRE(counters.mergers_built == 1);
  REQUIRE(counters.rounds.refill_rows >= host_rows_total);
  REQUIRE(counters.rounds.refill_rows <= host_rows_total + host_rows_total / 100);
  test.require_inputs_popped();
}

TEST_CASE("merge sort automatic mode routes by the one-shot share of the GPU",
          "[merge_sort_continuation]")
{
  sirius::operator_params params;
  params.sort_merge_mode_setting = sirius::sort_merge_mode::automatic;
  params.sort_merge_window_rows  = 1024;
  params.sort_merge_output_bytes = 64 * 1024;

  SECTION("tiny data on a large device merges one-shot")
  {
    merge_harness test{{.num_partitions = 2}};
    test.configure(params);
    std::vector<host_rows> runs;
    for (std::size_t run = 0; run < 3; ++run) {
      runs.push_back(test.add_run(1, run, 64, 0, 50));
    }

    test.run();

    auto const records = test.merge->records();
    REQUIRE(records.size() == 1);
    REQUIRE_FALSE(records.front().windowed);
    REQUIRE(records.front().runs == 3);
    REQUIRE(records.front().output_batches == 1);
    REQUIRE(records.front().output_rows == 3 * 64);
    // Every mode but one_shot encodes the partition's position the same way.
    REQUIRE(records.front().position == merge::make_merge_chunk_position(1, 0));
    REQUIRE(test.rearms_from_execute.load() == 0);
    require_sorted_rows(test.collected(), concatenate(runs));
    REQUIRE(test.pipe->pipeline->is_pipeline_finished());
    test.require_inputs_popped();
  }

  SECTION("a one-shot fraction of 0.0001 of a small device forces the windowed route")
  {
    merge_harness test;
    params.sort_merge_one_shot_max_fraction = 0.0001;
    // 0.0001 of 64 MiB is under 7 KiB; the partition is about 300 KiB.
    test.configure(
      params,
      {.gpu_available_bytes = 64 * kMiB, .gpu_total_bytes = 64 * kMiB, .concurrent_tasks = 1});
    std::vector<host_rows> runs;
    for (std::size_t run = 0; run < 5; ++run) {
      runs.push_back(test.add_run(0, run, 3500 + 200 * run));
    }

    test.run();

    auto const records = test.merge->records();
    REQUIRE(records.size() >= 3);
    REQUIRE(require_chunk_chain(records, 0) == records.size());
    REQUIRE(test.rearms_from_execute.load() == records.size() - 1);
    REQUIRE(test.collected() == full_sort(concatenate(runs)));
    REQUIRE(test.pipe->pipeline->is_pipeline_finished());
    test.require_inputs_popped();
  }
}

TEST_CASE("windowed merge sort merges runs spread over two GPUs, one partition per device",
          "[merge_sort_mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs two GPUs"); }
  constexpr std::size_t num_partitions = 4;
  constexpr std::size_t runs_per_part  = 6;
  merge_harness test{{.num_partitions = num_partitions, .executor_threads = 2, .num_gpus = 2}};
  test.configure(small_window_params());
  // Four runs of a partition on its own device and two on the other one: the task goes to the
  // device holding most of the partition, and the other two runs are moved to it whole.
  std::vector<std::vector<host_rows>> runs(num_partitions);
  for (std::size_t partition = 0; partition < num_partitions; ++partition) {
    for (std::size_t run = 0; run < runs_per_part; ++run) {
      auto const home   = partition % 2;
      auto const device = run % 3 == 2 ? 1 - home : home;
      runs[partition].push_back(test.add_run(partition,
                                             run,
                                             4000 + 300 * run,
                                             static_cast<std::int64_t>(partition) * 10000,
                                             5000,
                                             device));
    }
  }

  test.run();

  auto const records = test.merge->records();
  for (std::size_t partition = 0; partition < num_partitions; ++partition) {
    CAPTURE(partition);
    REQUIRE(require_chunk_chain(records, partition) >= 3);
    REQUIRE(test.collected(partition) == full_sort(concatenate(runs[partition])));
  }
  REQUIRE(sorted_by_key(test.collected()));
  REQUIRE(test.pipe->pipeline->is_pipeline_finished());

  // Partitions were merged on both devices, at the same time.
  std::map<int, std::size_t> tasks_per_device;
  test.pipe->scheduler->visit_executors(
    [&](int device, const sirius::pipeline::gpu_pipeline_executor& executor) {
      tasks_per_device[device] = executor.get_metrics().tasks_executed;
    });
  REQUIRE(tasks_per_device.size() == 2);
  for (auto const& [device, tasks] : tasks_per_device) {
    CAPTURE(device);
    REQUIRE(tasks >= 3);
  }
  REQUIRE(test.merge->peak_running() >= 2);

  // The two runs of each partition on the other device were converted whole to the merging
  // device, once each; nothing was read by row range.
  auto const counters = test.merge->counters();
  REQUIRE(counters.windowed_partitions == num_partitions);
  REQUIRE(counters.windowed_tasks == records.size());
  REQUIRE(counters.rounds.fallback_conversions == num_partitions * (runs_per_part / 3));
  REQUIRE(counters.rounds.refill_rows == 0);
  test.require_inputs_popped();
}

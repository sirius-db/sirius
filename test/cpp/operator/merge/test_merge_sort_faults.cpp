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

// The windowed MERGE_SORT under the failures it is designed to survive: a run that changes tier
// while it is being merged, a downgrade racing the merge tasks, a run no reader can window, and
// GPU out-of-memory inside a task (with and without room left to shrink the windows).

#include "catch.hpp"
#include "creator/task_creator.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "data/sirius_converter_registry.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/config.hpp"
#include "helper/type_conversions.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/merge_chunk_position.hpp"
#include "op/merge/merge_config.hpp"
#include "op/merge/windowed_merge.hpp"
#include "op/merge/windowed_merge_input.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/query_retirement_test_utils.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_scheduler.hpp"
#include "utils/telemetry_utils.hpp"
#include "utils/utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <compression/compressed_representation.hpp>
#include <compression/spill_compression.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::merge_utils;
using cucascade::memory::Tier;

namespace {

namespace utils = sirius::test::operator_utils;

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kGiB = std::size_t{1} << 30;

bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ---------------------------------------------------------------------------
// Memory: one GPU space with a software limit, a host space and a disk space
// ---------------------------------------------------------------------------

/// Declare it before anything that holds batches of its spaces: the destructor shuts the manager
/// and the converter registry down.
struct fault_fixture {
  std::filesystem::path tmp_dir;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  /// The host converter rejects the default stream, so conversions run on this one.
  rmm::cuda_stream stream;

  explicit fault_fixture(std::size_t gpu_bytes = 256 * kMiB)
  {
    sirius::converter_registry::reset_for_testing();
    std::string tmp_template =
      (std::filesystem::temp_directory_path() / "sirius_merge_faults_XXXXXX").string();
    REQUIRE(mkdtemp(tmp_template.data()) != nullptr);
    tmp_dir = tmp_template;

    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(gpu_bytes)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(kGiB)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .track_reservation_per_stream(false)
      .set_disk_mounting_point(0, kGiB, tmp_dir.string());
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
    gpu  = manager->get_memory_space(Tier::GPU, 0);
    host = manager->get_memory_space(Tier::HOST, 0);
    disk = manager->get_memory_space(Tier::DISK, 0);
    REQUIRE(gpu != nullptr);
    REQUIRE(host != nullptr);
    REQUIRE(disk != nullptr);
  }

  ~fault_fixture()
  {
    manager->shutdown();
    sirius::converter_registry::shutdown();
    std::error_code ignored;
    std::filesystem::remove_all(tmp_dir, ignored);
  }

  fault_fixture(const fault_fixture&)            = delete;
  fault_fixture& operator=(const fault_fixture&) = delete;

  [[nodiscard]] ::cuda::stream_ref stream_ref() const { return ::cuda::stream_ref{stream.value()}; }
  [[nodiscard]] rmm::cuda_stream_view stream_view() const
  {
    return rmm::cuda_stream_view{stream.value()};
  }
  [[nodiscard]] rmm::device_async_resource_ref mr() const { return utils::get_resource_ref(*gpu); }
};

Tier tier_of(cucascade::data_batch& batch) { return batch.to_read_only().get_current_tier(); }

template <typename Representation>
bool holds(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return dynamic_cast<Representation const*>(ro.get_data()) != nullptr;
}

/// Converts @p batch in place; blocks until no task holds a lock on it.
template <typename Representation>
void convert_batch(fault_fixture& fx,
                   cucascade::data_batch& batch,
                   cucascade::memory::memory_space* space)
{
  {
    auto mut = batch.to_mutable();
    mut.convert_to<Representation>(sirius::converter_registry::get(), space, fx.stream_ref());
  }
  fx.stream.synchronize();
}

/// Same, but fails the test instead of waiting when the batch is locked.
template <typename Representation>
void convert_unlocked_batch(fault_fixture& fx,
                            cucascade::data_batch& batch,
                            cucascade::memory::memory_space* space)
{
  {
    auto mut = batch.try_to_mutable();
    REQUIRE(mut.has_value());
    mut->convert_to<Representation>(sirius::converter_registry::get(), space, fx.stream_ref());
  }
  fx.stream.synchronize();
}

/// Switches spill compression on for one test and off afterwards.
struct scoped_spill_compression {
  scoped_spill_compression()
  {
    sirius::compression::spill_compression_settings settings;
    settings.enabled         = true;
    settings.min_batch_bytes = 0;
    sirius::compression::set_spill_compression_settings(settings);
  }
  ~scoped_spill_compression() { sirius::compression::set_spill_compression_settings({}); }
};

// ---------------------------------------------------------------------------
// Rows and runs
// ---------------------------------------------------------------------------

merge::merge_key_spec ascending_key()
{
  return {.key_cols        = {0},
          .column_order    = {cudf::order::ASCENDING},
          .null_precedence = {cudf::null_order::AFTER}};
}

/// An unsorted (key INT64 in [0, key_range), id INT64) table; ids are unique from @p first_id on.
std::unique_ptr<cudf::table> random_rows(std::size_t rows,
                                         std::int64_t key_range,
                                         std::int64_t first_id,
                                         std::uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_int_distribution<std::int64_t> dist(0, key_range - 1);
  std::vector<std::int64_t> keys(rows);
  std::vector<std::int64_t> ids(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    keys[i] = dist(gen);
    ids[i]  = first_id + static_cast<std::int64_t>(i);
  }
  return make_table(make_column(keys, cudf::type_id::INT64),
                    make_column(ids, cudf::type_id::INT64));
}

const std::vector<std::size_t> kTwoCols{0, 1};
const std::vector<cudf::data_type> kTwoInt64{cudf::data_type{cudf::type_id::INT64},
                                             cudf::data_type{cudf::type_id::INT64}};

/// Sorts @p unsorted and adds it to @p partition of @p repository as one single-batch GPU run.
test_run add_sorted_run(cucascade::shared_data_repository& repository,
                        cucascade::memory::memory_space& space,
                        const cudf::table_view& unsorted,
                        std::size_t partition = 0)
{
  auto stream = utils::default_stream();
  test_run out;
  out.rows   = stable_sort(unsorted, ascending_key());
  auto batch = ::sirius::make_data_batch(
    std::make_unique<cudf::table>(out.rows->view(), stream, utils::get_resource_ref(space)),
    space,
    stream,
    ::sirius::telemetry::batch_telemetry_info{});
  repository.add_data_batch(batch, partition);
  out.run.batch_ids.push_back(batch->get_batch_id());
  out.run.batch_rows.push_back(static_cast<std::size_t>(out.rows->num_rows()));
  out.batches.push_back(std::move(batch));
  return out;
}

bool same_cursors(const std::vector<merge::merge_input_cursor>& lhs,
                  const std::vector<merge::merge_input_cursor>& rhs)
{
  return std::equal(
    lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), [](const auto& a, const auto& b) {
      return a.batch == b.batch && a.row == b.row && a.run.batch_ids == b.run.batch_ids &&
             a.run.batch_rows == b.run.batch_rows;
    });
}

// ---------------------------------------------------------------------------
// The operator, driven task by task
// ---------------------------------------------------------------------------

std::unique_ptr<sirius_physical_merge_sort> make_merge(cucascade::shared_data_repository& repo,
                                                       merge::merge_settings settings)
{
  duckdb::vector<duckdb::LogicalType> logical{duckdb::LogicalType::BIGINT,
                                              duckdb::LogicalType::BIGINT};
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(duckdb::BoundOrderByNode(
    duckdb::OrderType::ASCENDING,
    duckdb::OrderByNullType::NULLS_LAST,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0)));
  auto op =
    std::make_unique<sirius_physical_merge_sort>(sirius::from_duckdb_vec(std::move(logical)),
                                                 std::move(orders),
                                                 duckdb::vector<std::size_t>{0, 1},
                                                 0);
  op->add_port("default",
               std::make_unique<sirius_physical_operator::port>(
                 MemoryBarrierType::FULL, &repo, nullptr, nullptr));
  op->set_merge_settings(settings);
  // A fixed device budget, so the route and the fan-in do not depend on the test's GPU limit.
  op->set_device_budget_provider([] {
    return merge::device_budget{
      .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 1};
  });
  return op;
}

merge::windowed_merge_input& as_windowed(const std::unique_ptr<operator_data>& input)
{
  REQUIRE(input);
  REQUIRE(input->get_type() == operator_data_type::WINDOWED_MERGE);
  return dynamic_cast<merge::windowed_merge_input&>(*input);
}

/// The output chunks of a merge, keyed by their position. Thread-safe.
class merge_output {
 public:
  /// Copies the batches of one task's output out of the GPU space.
  void add(const operator_data& output)
  {
    auto const& data = dynamic_cast<const pipelineable_operator_data&>(output);
    if (data.get_data_batches().empty()) { return; }
    std::vector<std::unique_ptr<cudf::table>> tables;
    for (auto const& batch : data.get_data_batches()) {
      auto const ro = batch->to_read_only();
      tables.push_back(std::make_unique<cudf::table>(sirius::get_cudf_table_view(ro)));
    }
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& table : tables) {
      _chunks.push_back({data.get_position(), std::move(table)});
    }
  }

  [[nodiscard]] std::size_t num_chunks() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _chunks.size();
  }

  /// Requires that every chunk has its own position and that the chunks of each partition are
  /// numbered 0, 1, 2, ... without a gap; returns the chunks of @p partition in chunk order.
  std::unique_ptr<cudf::table> partition_table(std::size_t partition)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::sort(_chunks.begin(), _chunks.end(), [](const auto& a, const auto& b) {
      return a.position < b.position;
    });
    std::vector<cudf::table_view> views;
    std::optional<std::size_t> previous;
    std::size_t next_chunk = 0;
    for (auto const& chunk : _chunks) {
      REQUIRE(chunk.position.has_value());
      REQUIRE(chunk.position != previous);
      previous                   = chunk.position;
      auto const [owner, number] = merge::split_merge_chunk_position(*chunk.position);
      if (owner != partition) { continue; }
      REQUIRE(number == next_chunk);
      ++next_chunk;
      views.push_back(chunk.table->view());
    }
    REQUIRE_FALSE(views.empty());
    return cudf::concatenate(views);
  }

 private:
  struct chunk {
    std::optional<std::size_t> position;
    std::unique_ptr<cudf::table> table;
  };
  mutable std::mutex _mutex;
  std::vector<chunk> _chunks;
};

/// Runs one task on the calling thread, the way a GPU pipeline task runs the operator.
void run_task_inline(sirius_physical_merge_sort& op,
                     operator_data& input,
                     cucascade::memory::memory_space& space,
                     ::cuda::stream_ref stream,
                     merge_output& output)
{
  input.prepare_for_processing(&space, stream);
  auto result = op.execute(input, stream);
  REQUIRE(result);
  output.add(*result);
}

// ---------------------------------------------------------------------------
// The GPU task scheduler around the operator, for the OOM tests
// ---------------------------------------------------------------------------

constexpr std::size_t kTaskReservation = 8 * kMiB;

class fault_global_state : public sirius::pipeline::sirius_pipeline_task_global_state {
 public:
  fault_global_state()
    : sirius_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context())
  {
  }
};

/// What one attempt of a task saw when it started.
struct task_attempt {
  std::size_t retry_floor{0};
  std::uint32_t retry_count{0};
  merge::window_config config;
  std::vector<merge::merge_input_cursor> cursors;
  bool under_pressure{false};
  bool oom{false};
  std::size_t usage_before_merge{0};  ///< GPU bytes allocated when the merge started
  std::size_t usage_at_oom{0};        ///< GPU bytes allocated when the allocation failed
};

/// Script and record of one task and its retries.
struct fault_probe {
  sirius_physical_merge_sort* op = nullptr;
  merge_output* output           = nullptr;
  /// The first `pressured_attempts` attempts run with the GPU space filled up to
  /// `free_room_bytes`, so the merge runs out of memory once its output has grown past that.
  std::size_t pressured_attempts{0};
  std::size_t free_room_bytes{0};

  std::atomic<bool> done{false};
  std::mutex mutex;
  std::vector<task_attempt> attempts;
  std::string error;

  std::vector<task_attempt> observed()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return attempts;
  }
};

/// A GPU pipeline task whose compute step is one sirius_physical_merge_sort::execute(), with the
/// OOM handling of gpu_pipeline_task: raise the retry floor, hand the input back for a retry.
class merge_fault_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  merge_fault_task(uint64_t task_id,
                   std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state> local_state,
                   std::shared_ptr<fault_global_state> global_state,
                   std::shared_ptr<fault_probe> probe)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state)),
      _probe(std::move(probe))
  {
  }

  std::unique_ptr<operator_data> compute_task(::cuda::stream_ref) override { return nullptr; }
  void publish_output(operator_data&, ::cuda::stream_ref) override {}
  std::vector<sirius_physical_operator*> get_output_consumers() override { return {}; }

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space*) const override
  {
    auto const floor = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>()
                         .get_retry_reservation_floor();
    sirius::pipeline::reservation_size_info info;
    info.retry_reservation_floor = floor;
    info.reservation_size        = std::max(kTaskReservation, floor);
    return info;
  }

  void execute(::cuda::stream_ref stream) override
  {
    using cucascade::memory::reservation_aware_resource_adaptor;
    auto& local = _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();

    auto reservation = local.release_reservation();
    if (!reservation) { throw std::runtime_error("merge fault task: no GPU reservation"); }
    auto* space = const_cast<cucascade::memory::memory_space*>(&reservation->get_memory_space());
    auto* allocator = reservation->get_memory_resource_as<reservation_aware_resource_adaptor>();
    if (allocator == nullptr) { throw std::runtime_error("merge fault task: no allocator"); }
    if (!allocator->attach_reservation_to_tracker(
          stream,
          std::move(reservation),
          std::make_unique<cucascade::memory::ignore_reservation_limit_policy>(),
          nullptr)) {
      throw std::runtime_error("merge fault task: could not attach the reservation");
    }
    struct reservation_guard {
      reservation_aware_resource_adaptor* allocator;
      ::cuda::stream_ref stream;
      ~reservation_guard() { allocator->reset_stream_reservation(stream); }
    } guard{allocator, stream};

    auto& input       = dynamic_cast<merge::windowed_merge_input&>(*local._input_data);
    std::size_t index = 0;
    bool pressure     = false;
    {
      std::lock_guard<std::mutex> lock(_probe->mutex);
      index    = _probe->attempts.size();
      pressure = index < _probe->pressured_attempts;
      _probe->attempts.push_back({.retry_floor    = local.get_retry_reservation_floor(),
                                  .retry_count    = local.retry_count,
                                  .config         = input.config(),
                                  .cursors        = input.cursors(),
                                  .under_pressure = pressure});
    }
    auto const record = [&](auto&& update) {
      std::lock_guard<std::mutex> lock(_probe->mutex);
      update(_probe->attempts[index]);
    };

    // Memory pressure: fill the GPU space until only `free_room_bytes` are left.
    struct hog_guard {
      reservation_aware_resource_adaptor* allocator;
      ::cuda::stream_ref stream;
      void* data{nullptr};
      std::size_t bytes{0};
      void release()
      {
        if (data != nullptr) {
          allocator->deallocate(stream, data, bytes, alignof(std::max_align_t));
          data = nullptr;
        }
      }
      ~hog_guard() { release(); }
    } hog{allocator, stream};
    if (pressure) {
      auto const available = allocator->get_available_memory(stream);
      hog.bytes = available > _probe->free_room_bytes ? available - _probe->free_room_bytes : 0;
      while (hog.bytes > 0 && hog.data == nullptr) {
        try {
          hog.data = allocator->allocate(stream, hog.bytes, alignof(std::max_align_t));
        } catch (const rmm::out_of_memory&) {
          hog.bytes = hog.bytes > kMiB ? hog.bytes - kMiB : 0;
        }
      }
    }
    record([&](task_attempt& a) { a.usage_before_merge = allocator->get_total_allocated_bytes(); });

    try {
      input.prepare_for_processing(space, stream);
      auto result = _probe->op->execute(input, stream);
      hog.release();
      if (result) { _probe->output->add(*result); }
      _probe->done.store(true, std::memory_order_release);
    } catch (const rmm::out_of_memory& oom) {
      hog.release();
      std::size_t usage_at_oom = 0;
      if (auto const* detail =
            dynamic_cast<const cucascade::memory::cucascade_out_of_memory*>(&oom)) {
        usage_at_oom = detail->global_usage;
      }
      record([&](task_attempt& a) {
        a.oom          = true;
        a.usage_at_oom = usage_at_oom;
      });
      // As gpu_pipeline_task does at its OOM throw sites.
      local.update_retry_reservation_floor_after_oom(
        std::max(kTaskReservation, local.get_retry_reservation_floor()), 0, std::nullopt);
      throw sirius::pipeline::oom_reschedule_exception(
        std::move(local._input_data), 0, "OOM in a windowed merge task");
    }
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed_local = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<merge_fault_task>(
      task_id,
      std::move(typed_local),
      std::dynamic_pointer_cast<fault_global_state>(_global_state),
      _probe);
  }

 private:
  std::shared_ptr<fault_probe> _probe;
};

/// The GPU task scheduler of one query, with one worker thread.
struct scheduler_fixture {
  sirius::exec::query_lifecycle_registry lifecycle;
  std::unique_ptr<sirius::pipeline::task_scheduler> executor;
  std::unique_ptr<sirius::creator::task_creator> creator;
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();
  std::shared_ptr<fault_global_state> global_state;
  uint64_t next_task_id{0};

  explicit scheduler_fixture(fault_fixture& fx)
  {
    sirius::exec::thread_pool_config config;
    config.num_threads        = 1;
    config.thread_name_prefix = "merge-fault";
    lifecycle.open_query(sirius::make_query_id(0));
    executor = std::make_unique<sirius::pipeline::task_scheduler>(
      lifecycle, config, *fx.manager, sirius::test::make_test_telemetry_context());
    creator = std::make_unique<sirius::creator::task_creator>(
      lifecycle, sirius::creator::task_creator_config{}, *fx.manager);
    creator->set_task_scheduler(*executor);
    executor->set_task_creator(*creator);
    global_state = std::make_shared<fault_global_state>();
    global_state->set_completion_handler(completion);
    executor->start();
  }

  // GPU workers borrow the creator, so stop them before member destruction.
  ~scheduler_fixture() { executor->stop(); }

  /// Schedules one task over @p input and waits for it (and its retries). False when the task
  /// did not complete: the query failed, or the deadline passed.
  bool run(const std::shared_ptr<fault_probe>& probe,
           std::unique_ptr<operator_data> input,
           std::chrono::seconds timeout = std::chrono::seconds(180))
  {
    executor->schedule(std::make_unique<merge_fault_task>(
      next_task_id++,
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input)),
      global_state,
      probe));
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!probe->done.load(std::memory_order_acquire) && !completion->has_error() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return probe->done.load(std::memory_order_acquire);
  }
};

/// A repository with four single-batch runs of (key, id) rows, the second and fourth spilled to
/// the host tier, and the windowed operator over it.
struct oom_scenario {
  fault_fixture fx{512 * kMiB};
  cucascade::shared_data_repository repo;
  std::vector<test_run> runs;
  std::unique_ptr<sirius_physical_merge_sort> op;
  merge_output output;

  oom_scenario(std::size_t rows_per_run, merge::merge_settings settings)
  {
    for (std::size_t i = 0; i < 4; ++i) {
      auto const first_id = static_cast<std::int64_t>(i * rows_per_run);
      runs.push_back(add_sorted_run(
        repo,
        *fx.gpu,
        random_rows(rows_per_run, 1'000'000, first_id, static_cast<std::uint32_t>(41 + i))
          ->view()));
    }
    convert_batch<cucascade::host_data_representation>(fx, *runs[1].batches[0], fx.host);
    convert_batch<cucascade::host_data_representation>(fx, *runs[3].batches[0], fx.host);
    op = make_merge(repo, settings);
  }

  std::shared_ptr<fault_probe> make_probe(std::size_t pressured_attempts,
                                          std::size_t free_room_bytes)
  {
    auto probe                = std::make_shared<fault_probe>();
    probe->op                 = op.get();
    probe->output             = &output;
    probe->pressured_attempts = pressured_attempts;
    probe->free_room_bytes    = free_room_bytes;
    return probe;
  }

  /// Runs the partition's remaining tasks through @p scheduler, without faults.
  void finish(scheduler_fixture& scheduler)
  {
    for (std::size_t task = 0; task < 10000; ++task) {
      auto input = op->get_next_task_input_data();
      if (!input) { return; }
      REQUIRE(scheduler.run(make_probe(0, 0), std::move(input)));
    }
    FAIL("the merge did not finish");
  }

  void require_exact_output()
  {
    auto const expected = expected_full_sort(runs, ascending_key());
    auto const merged   = output.partition_table(0);
    REQUIRE(merged->num_rows() == expected->num_rows());
    REQUIRE(tables_equal(merged->view(), expected->view()));
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// 1. Tier change between rounds
// ---------------------------------------------------------------------------

TEST_CASE("windowed merge reads a HOST run from DISK after it is spilled between two rounds",
          "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  fault_fixture fx;
  cucascade::shared_data_repository repo;
  std::vector<test_run> runs;
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(6000, 50000, 0, 1)->view()));
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(5000, 50000, 10000, 2)->view()));
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(4000, 50000, 20000, 3)->view()));
  convert_batch<cucascade::host_data_representation>(fx, *runs[0].batches[0], fx.host);
  convert_batch<cucascade::host_data_representation>(fx, *runs[2].batches[0], fx.host);
  REQUIRE(tier_of(*runs[0].batches[0]) == Tier::HOST);

  constexpr std::size_t window_rows = 512;
  constexpr std::size_t spill_round = 3;
  std::size_t round                 = 0;
  std::size_t run0_rows_before      = 0;  // rows of run 0 merged before it changed tier
  bool gpu_run_was_locked           = false;
  // The merge kernel, plus the fault: once round 3 has merged, run 0 moves from HOST to DISK.
  merge::bounded_merge_fn spilling_merge = [&](std::span<const cudf::table_view> windows,
                                               const merge::merge_key_spec& keys,
                                               std::span<const std::uint8_t> truncated,
                                               cudf::size_type max_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr) {
    auto result = merge::bounded_kway_merge(windows, keys, truncated, max_rows, stream, mr);
    ++round;
    if (round <= spill_round) {
      // No run can be exhausted this early, so window 0 is run 0.
      REQUIRE(windows.size() == 3);
      run0_rows_before += static_cast<std::size_t>(result.consumed[0]);
    }
    if (round == spill_round) {
      // A GPU run is sliced under a read lock held for the whole task: it cannot move now.
      gpu_run_was_locked = !runs[1].batches[0]->try_to_mutable().has_value();
      // A spilled run is locked only while a window is refilled.
      convert_unlocked_batch<cucascade::disk_data_representation>(fx, *runs[0].batches[0], fx.disk);
    }
    return result;
  };

  auto cursors = make_cursors(runs);
  merge::windowed_merger merger(run_batches(runs),
                                cursors,
                                kTwoCols,
                                kTwoInt64,
                                ascending_key(),
                                {.window_rows = window_rows, .output_target_bytes = kGiB},
                                sirius::row_range_reader_registry::get(),
                                *fx.gpu,
                                spilling_merge);
  merge::merge_round_stats stats;
  auto const merged = merger.run(fx.stream_view(), fx.mr(), &stats);

  REQUIRE(all_exhausted(cursors));
  REQUIRE(round > spill_round);
  CHECK(gpu_run_was_locked);
  CHECK(tier_of(*runs[0].batches[0]) == Tier::DISK);
  // Run 0 still had unread rows beyond its buffered window, so later refills read the disk copy.
  REQUIRE(run0_rows_before + window_rows < runs[0].run.total_rows());
  CHECK(stats.rounds == round);
  CHECK(stats.fallback_conversions == 0);
  // Every row of the two spilled runs went through a window buffer.
  CHECK(stats.refill_rows >= runs[0].run.total_rows() + runs[2].run.total_rows());

  auto const expected = expected_full_sort(runs, ascending_key());
  REQUIRE(merged->num_rows() == expected->num_rows());
  REQUIRE(tables_equal(merged->view(), expected->view()));
}

TEST_CASE("windowed merge follows a run across GPU, HOST and DISK between tasks",
          "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  fault_fixture fx;
  cucascade::shared_data_repository repo;
  std::vector<test_run> runs;
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(6000, 50000, 0, 4)->view()));
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(5000, 50000, 10000, 5)->view()));
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(4000, 50000, 20000, 6)->view()));
  convert_batch<cucascade::host_data_representation>(fx, *runs[0].batches[0], fx.host);
  auto& gpu_batch  = *runs[1].batches[0];
  auto& host_batch = *runs[0].batches[0];

  auto cursors = make_cursors(runs);
  // A 1-byte output target ends every run() after one round, as a task boundary would.
  merge::windowed_merger merger(run_batches(runs),
                                cursors,
                                kTwoCols,
                                kTwoInt64,
                                ascending_key(),
                                {.window_rows = 512, .output_target_bytes = 1},
                                sirius::row_range_reader_registry::get(),
                                *fx.gpu);
  merge::merge_round_stats stats;
  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::table_view> views;
  std::size_t tier_changes = 0;
  while (!all_exhausted(cursors)) {
    REQUIRE(chunks.size() < 1000);
    chunks.push_back(merger.run(fx.stream_view(), fx.mr(), &stats));
    views.push_back(chunks.back()->view());
    // Between two tasks no batch is locked, so each of these conversions goes through.
    switch (chunks.size()) {
      case 2:  // the GPU run is downgraded while it is half read
        REQUIRE(cursors[1].row > 0);
        REQUIRE_FALSE(cursors[1].exhausted());
        convert_unlocked_batch<cucascade::host_data_representation>(fx, gpu_batch, fx.host);
        ++tier_changes;
        break;
      case 4:  // and spilled further
        REQUIRE_FALSE(cursors[1].exhausted());
        convert_unlocked_batch<cucascade::disk_data_representation>(fx, gpu_batch, fx.disk);
        ++tier_changes;
        break;
      case 6:  // a spilled run comes back to the GPU: it is sliced again
        REQUIRE_FALSE(cursors[0].exhausted());
        convert_unlocked_batch<cucascade::gpu_table_representation>(fx, host_batch, fx.gpu);
        ++tier_changes;
        break;
      default: break;
    }
  }
  REQUIRE(tier_changes == 3);
  CHECK(tier_of(gpu_batch) == Tier::DISK);
  CHECK(tier_of(host_batch) == Tier::GPU);
  CHECK(stats.fallback_conversions == 0);
  // Rows of the downgraded run were read through its window buffer afterwards.
  CHECK(stats.refill_rows > 0);

  auto const merged   = cudf::concatenate(views);
  auto const expected = expected_full_sort(runs, ascending_key());
  REQUIRE(merged->num_rows() == expected->num_rows());
  REQUIRE(tables_equal(merged->view(), expected->view()));
}

// ---------------------------------------------------------------------------
// 2. Downgrade during a merge
// ---------------------------------------------------------------------------

TEST_CASE("merge sort stays exact while the downgrade executor spills its runs",
          "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  constexpr std::size_t num_partitions = 3;
  constexpr std::size_t runs_per_part  = 4;
  constexpr std::size_t rows_per_run   = 60000;
  auto const query                     = sirius::make_query_id(1);

  fault_fixture fx;
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(query);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(query);

  // A repository of some other operator, registered first: with equal priorities the sweep
  // would spill it before the merge's input.
  auto bystander_owner = std::make_unique<cucascade::shared_data_repository>();
  std::vector<test_run> bystanders;
  for (std::size_t i = 0; i < 4; ++i) {
    bystanders.push_back(add_sorted_run(
      *bystander_owner,
      *fx.gpu,
      random_rows(rows_per_run, 1000, 0, static_cast<std::uint32_t>(90 + i))->view()));
  }
  auto merge_owner = std::make_unique<cucascade::shared_data_repository>();
  auto& repo       = *merge_owner;
  repo.set_num_partitions(num_partitions);
  std::vector<std::vector<test_run>> runs(num_partitions);
  std::size_t run_bytes = 0;
  for (std::size_t p = 0; p < num_partitions; ++p) {
    for (std::size_t r = 0; r < runs_per_part; ++r) {
      auto const index = p * runs_per_part + r;
      runs[p].push_back(add_sorted_run(repo,
                                       *fx.gpu,
                                       random_rows(rows_per_run,
                                                   20000,
                                                   static_cast<std::int64_t>(index * rows_per_run),
                                                   static_cast<std::uint32_t>(100 + index))
                                         ->view(),
                                       p));
      run_bytes = sirius::get_batch_rows_and_bytes(*runs[p].back().batches[0]).bytes;
    }
  }

  auto op = make_merge(
    repo,
    {.mode = sirius::sort_merge_mode::windowed, .window_rows = 2048, .output_bytes = 256 * 1024});
  // What repository wiring copies onto the input repository of a windowed sort.
  auto const policy = op->input_residency_policy("default");
  REQUIRE(policy.downgrade_priority == 100);
  REQUIRE_FALSE(policy.materialize_inputs_on_gpu);
  repo.set_downgrade_priority(policy.downgrade_priority);
  repo_mgr.add_new_repository(1, "out", std::move(bystander_owner));
  repo_mgr.add_new_repository(2, "default", std::move(merge_owner));

  auto const count_on = [](const std::vector<test_run>& group, Tier tier) {
    return static_cast<std::size_t>(std::count_if(
      group.begin(), group.end(), [tier](auto& run) { return tier_of(*run.batches[0]) == tier; }));
  };
  auto const merge_runs_off_gpu = [&] {
    std::size_t total = 0;
    for (auto const& group : runs) {
      total += group.size() - count_on(group, Tier::GPU);
    }
    return total;
  };

  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period = std::chrono::milliseconds{0}};
  sirius::parallel::downgrade_executor downgrader(lifecycle,
                                                  config,
                                                  repo_registry,
                                                  cucascade::memory::memory_space_id(Tier::GPU, 0),
                                                  fx.gpu,
                                                  *fx.manager);
  downgrader.start();

  // Before the merge starts: the sweep takes the merge's runs, not the bystander's batches.
  REQUIRE(downgrader.request_free_memory_and_wait(3 * run_bytes) > 0);
  REQUIRE(merge_runs_off_gpu() >= 1);
  REQUIRE(count_on(bystanders, Tier::GPU) == bystanders.size());

  merge_output output;
  std::atomic<std::size_t> tasks_started{0};
  std::atomic<std::size_t> tasks_done{0};
  std::atomic<int> in_flight{0};
  std::atomic<bool> merge_done{false};
  std::mutex error_mutex;
  std::string error;
  auto const worker = [&] {
    try {
      rmm::cuda_stream stream;
      ::cuda::stream_ref const stream_ref{stream.value()};
      while (true) {
        in_flight.fetch_add(1);
        auto input = op->get_next_task_input_data();
        if (!input) {
          // Nothing to arm: either every partition has a task in flight, or the merge is done.
          if (in_flight.fetch_sub(1) == 1) { break; }
          std::this_thread::yield();
          continue;
        }
        tasks_started.fetch_add(1);
        input->prepare_for_processing(fx.gpu, stream_ref);
        auto result = op->execute(*input, stream_ref);
        output.add(*result);
        tasks_done.fetch_add(1);
        in_flight.fetch_sub(1);
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(error_mutex);
      error = e.what();
    }
  };

  // The downgrade requests race the merge tasks from here on.
  std::atomic<std::size_t> freed_during_merge{0};
  std::atomic<std::size_t> requests_during_merge{0};
  std::thread spiller([&] {
    while (!merge_done.load()) {
      auto const freed = downgrader.request_free_memory_and_wait(run_bytes);
      if (tasks_started.load() > 0 && !merge_done.load()) {
        freed_during_merge.fetch_add(freed);
        requests_during_merge.fetch_add(1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  std::vector<std::thread> workers;
  for (int i = 0; i < 2; ++i) {
    workers.emplace_back(worker);
  }
  for (auto& thread : workers) {
    thread.join();
  }
  merge_done.store(true);
  spiller.join();
  downgrader.stop();

  INFO("merge error: " << error);
  REQUIRE(error.empty());
  // Several tasks per partition, with downgrades landing between and during them.
  CHECK(tasks_done.load() > num_partitions);
  CHECK(requests_during_merge.load() > 0);
  CHECK(freed_during_merge.load() > 0);
  // Every run was read to its end and popped.
  CHECK(repo.total_size() == 0);
  REQUIRE_FALSE(op->get_next_task_input_data());

  for (std::size_t p = 0; p < num_partitions; ++p) {
    CAPTURE(p);
    auto const expected = expected_full_sort(runs[p], ascending_key());
    auto const merged   = output.partition_table(p);
    REQUIRE(merged->num_rows() == expected->num_rows());
    REQUIRE(tables_equal(merged->view(), expected->view()));
  }
}

// ---------------------------------------------------------------------------
// 3. No reader: whole-batch fallback
// ---------------------------------------------------------------------------

namespace {

/// Merges a GPU run with a run that @p make_unreadable has put in a representation without a
/// row-range reader, over many run() calls.
template <typename Representation, typename MakeUnreadable>
void check_fallback_merge(fault_fixture& fx, MakeUnreadable&& make_unreadable)
{
  cucascade::shared_data_repository repo;
  std::vector<test_run> runs;
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(3000, 40000, 0, 7)->view()));
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(2500, 40000, 10000, 8)->view()));
  auto& batch = *runs[1].batches[0];
  make_unreadable(batch);
  REQUIRE(holds<Representation>(batch));
  auto const& readers = sirius::row_range_reader_registry::get();
  REQUIRE_FALSE(readers.has_reader(*batch.to_read_only().get_data()));
  auto const id = batch.get_batch_id();

  auto cursors = make_cursors(runs);
  merge::windowed_merger merger(run_batches(runs),
                                cursors,
                                kTwoCols,
                                kTwoInt64,
                                ascending_key(),
                                {.window_rows = 256, .output_target_bytes = 1},
                                readers,
                                *fx.gpu);
  merge::merge_round_stats stats;
  auto const collected = collect_full_sort(merger, cursors, fx.stream_view(), fx.mr(), &stats);

  // Converted on the first round, then sliced like any GPU run for the rest of the merge.
  REQUIRE(collected.chunks.size() > 2);
  CHECK(stats.fallback_conversions == 1);
  CHECK(stats.refill_rows == 0);
  CHECK(holds<cucascade::gpu_table_representation>(batch));
  CHECK(batch.to_read_only().get_memory_space() == fx.gpu);
  CHECK(batch.get_batch_id() == id);
  CHECK(repo.get_data_batch_by_id(id, 0).get() == &batch);

  auto const expected = expected_full_sort(runs, ascending_key());
  REQUIRE(collected.table->num_rows() == expected->num_rows());
  REQUIRE(tables_equal(collected.table->view(), expected->view()));
}

}  // namespace

TEST_CASE("windowed merge converts a packed host run to the GPU once", "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  fault_fixture fx;
  check_fallback_merge<cucascade::host_data_packed_representation>(
    fx, [&](cucascade::data_batch& batch) {
      convert_batch<cucascade::host_data_packed_representation>(fx, batch, fx.host);
    });
}

TEST_CASE("windowed merge converts a spill-compressed run to the GPU once", "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  fault_fixture fx;
  scoped_spill_compression compression;
  check_fallback_merge<sirius::compressed_host_representation>(
    fx, [&](cucascade::data_batch& batch) {
      // The spill path: the downgrade's own conversion, with compression on.
      auto shared = batch.shared_from_this();
      sirius::convertible_data_batch convertible(shared);
      REQUIRE(convertible.convert({fx.host}, fx.stream_ref(), *fx.manager, true).has_value());
      fx.stream.synchronize();
    });
}

TEST_CASE("merge sort falls back when a run loses its reader between two tasks",
          "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  fault_fixture fx;
  cucascade::shared_data_repository repo;
  std::vector<test_run> runs;
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(30000, 40000, 0, 9)->view()));
  runs.push_back(add_sorted_run(repo, *fx.gpu, random_rows(25000, 40000, 100000, 10)->view()));
  auto& spilled = *runs[1].batches[0];
  convert_batch<cucascade::host_data_representation>(fx, spilled, fx.host);

  auto op = make_merge(
    repo,
    {.mode = sirius::sort_merge_mode::windowed, .window_rows = 1024, .output_bytes = 128 * 1024});
  merge_output output;

  // The partition is armed, and its first task windows the spilled run through the host reader.
  auto first = op->get_next_task_input_data();
  REQUIRE(as_windowed(first).cursors().size() == 2);
  run_task_inline(*op, *first, *fx.gpu, fx.stream_ref(), output);
  REQUIRE(tier_of(spilled) == Tier::HOST);

  // The run is now re-spilled in a layout no reader can window.
  convert_batch<cucascade::gpu_table_representation>(fx, spilled, fx.gpu);
  convert_batch<cucascade::host_data_packed_representation>(fx, spilled, fx.host);
  REQUIRE(holds<cucascade::host_data_packed_representation>(spilled));

  auto second          = op->get_next_task_input_data();
  auto const& resumed  = as_windowed(second);
  auto const spill_pos = resumed.cursors()[1];
  REQUIRE(spill_pos.row > 0);
  REQUIRE_FALSE(spill_pos.exhausted());
  run_task_inline(*op, *second, *fx.gpu, fx.stream_ref(), output);
  // The task converted the whole batch and went on from the cursor.
  CHECK(holds<cucascade::gpu_table_representation>(spilled));
  CHECK(spilled.to_read_only().get_memory_space() == fx.gpu);

  for (std::size_t task = 0;; ++task) {
    REQUIRE(task < 1000);
    auto input = op->get_next_task_input_data();
    if (!input) { break; }
    run_task_inline(*op, *input, *fx.gpu, fx.stream_ref(), output);
  }
  CHECK(output.num_chunks() > 3);
  CHECK(repo.total_size() == 0);

  auto const expected = expected_full_sort(runs, ascending_key());
  auto const merged   = output.partition_table(0);
  REQUIRE(merged->num_rows() == expected->num_rows());
  REQUIRE(tables_equal(merged->view(), expected->view()));
}

// ---------------------------------------------------------------------------
// 4. OOM in execute
// ---------------------------------------------------------------------------

TEST_CASE("merge sort task that runs out of memory is retried with smaller windows",
          "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  constexpr std::size_t window_rows  = 8192;
  constexpr std::size_t output_bytes = 32 * kMiB;
  constexpr std::size_t rows_per_run = 650'000;  // 4 runs x 16 bytes: 41.6 MiB, two chunks
  constexpr std::size_t round_bytes  = window_rows * 16;
  constexpr std::size_t free_room    = 4 * kMiB;
  constexpr std::size_t oom_attempts = 2;
  oom_scenario s(rows_per_run,
                 {.mode         = sirius::sort_merge_mode::windowed,
                  .window_rows  = window_rows,
                  .output_bytes = output_bytes});
  scheduler_fixture scheduler(s.fx);

  // The first task completes untouched, so the faulted one starts from cursors inside the runs.
  auto first = s.op->get_next_task_input_data();
  REQUIRE(as_windowed(first).final_pass());
  REQUIRE(scheduler.run(s.make_probe(0, 0), std::move(first)));
  REQUIRE(s.output.num_chunks() == 1);

  auto second         = s.op->get_next_task_input_data();
  auto const& input   = as_windowed(second);
  auto const snapshot = input.cursors();
  REQUIRE(input.first_chunk() == 1);
  REQUIRE(input.config().window_rows == window_rows);
  REQUIRE(input.config().output_target_bytes == output_bytes);
  REQUIRE(std::any_of(snapshot.begin(), snapshot.end(), [](auto& c) { return c.row > 0; }));

  // Two attempts run with only 4 MiB of the GPU space free: the merge starts, and fails once its
  // rounds of output have used that up.
  auto probe = s.make_probe(oom_attempts, free_room);
  REQUIRE(scheduler.run(probe, std::move(second)));
  REQUIRE_FALSE(scheduler.completion->has_error());

  auto const attempts = probe->observed();
  REQUIRE(attempts.size() == oom_attempts + 1);
  for (std::size_t i = 0; i < attempts.size(); ++i) {
    CAPTURE(i);
    auto const& attempt = attempts[i];
    CHECK(attempt.oom == (i < oom_attempts));
    CHECK(attempt.retry_count == i);
    // The input halved its windows, so the reservation floor was not raised.
    CHECK(attempt.retry_floor == 0);
    CHECK(attempt.config.window_rows == window_rows >> i);
    CHECK(attempt.config.output_target_bytes == std::max(output_bytes >> i, 16 * kMiB));
    // Every attempt starts from the cursors of the task, not from where a failed one got to.
    CHECK(same_cursors(attempt.cursors, snapshot));
    if (attempt.oom) {
      // The failure came after rounds of the merge had produced output, not at its first step.
      CHECK(attempt.usage_at_oom >= attempt.usage_before_merge + 2 * (round_bytes >> i));
    }
  }
  // The failed attempts emitted nothing: one chunk per completed task.
  REQUIRE(s.output.num_chunks() == 2);

  s.finish(scheduler);
  REQUIRE_FALSE(scheduler.completion->has_error());
  CHECK(s.repo.total_size() == 0);
  s.require_exact_output();
}

// ---------------------------------------------------------------------------
// 5. OOM at the shrink floor
// ---------------------------------------------------------------------------

TEST_CASE("merge sort task at its shrink floor escalates the reservation floor and completes",
          "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  // One halving away from the window floor, and already at the 16 MiB output floor.
  constexpr std::size_t window_rows  = 2 * merge::kShrinkWindowRowsFloor;
  constexpr std::size_t output_floor = 16 * kMiB;
  oom_scenario s(150'000,
                 {.mode         = sirius::sort_merge_mode::windowed,
                  .window_rows  = window_rows,
                  .output_bytes = output_floor});
  scheduler_fixture scheduler(s.fx);

  auto input = s.op->get_next_task_input_data();
  REQUIRE(as_windowed(input).config().window_rows == window_rows);
  auto const snapshot = as_windowed(input).cursors();
  auto probe          = s.make_probe(3, kMiB);
  REQUIRE(scheduler.run(probe, std::move(input)));
  REQUIRE_FALSE(scheduler.completion->has_error());

  auto const attempts = probe->observed();
  REQUIRE(attempts.size() == 4);
  for (std::size_t i = 0; i < attempts.size(); ++i) {
    CAPTURE(i);
    CHECK(attempts[i].oom == (i < 3));
    CHECK(attempts[i].retry_count == i);
    CHECK(attempts[i].config.output_target_bytes == output_floor);
    CHECK(attempts[i].config.window_rows == (i == 0 ? window_rows : merge::kShrinkWindowRowsFloor));
    CHECK(same_cursors(attempts[i].cursors, snapshot));
  }
  // The first OOM is answered by the last possible shrink (floor kept); the next two find the
  // input at its floors, so the reservation floor escalates as for any other task.
  CHECK(attempts[0].retry_floor == 0);
  CHECK(attempts[1].retry_floor == 0);
  CHECK(attempts[2].retry_floor >= 2 * kTaskReservation);
  CHECK(attempts[3].retry_floor >= 2 * attempts[2].retry_floor);

  s.finish(scheduler);
  REQUIRE_FALSE(scheduler.completion->has_error());
  CHECK(s.repo.total_size() == 0);
  s.require_exact_output();
}

TEST_CASE("merge sort task that keeps running out of memory fails at the retry limit",
          "[merge_sort_faults]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  constexpr std::size_t retry_limit = 3;
  oom_scenario s(150'000,
                 {.mode         = sirius::sort_merge_mode::windowed,
                  .window_rows  = merge::kShrinkWindowRowsFloor,
                  .output_bytes = 16 * kMiB});
  scheduler_fixture scheduler(s.fx);
  auto injections                       = std::make_shared<sirius::op::scan::test_injections>();
  injections->gpu_task_retry_limit      = retry_limit;
  injections->gpu_task_retry_backoff_ms = 0;
  scheduler.completion->injections      = std::move(injections);

  auto input          = s.op->get_next_task_input_data();
  auto const snapshot = as_windowed(input).cursors();
  // Every attempt is starved, however large its reservation grows.
  auto probe         = s.make_probe(std::numeric_limits<std::size_t>::max(), kMiB);
  auto const started = std::chrono::steady_clock::now();
  REQUIRE_FALSE(scheduler.run(probe, std::move(input), std::chrono::seconds(120)));
  auto const elapsed = std::chrono::steady_clock::now() - started;

  // The query fails with the retry-limit error instead of retrying forever.
  REQUIRE(scheduler.completion->has_error());
  CHECK(elapsed < std::chrono::seconds(120));
  try {
    scheduler.completion->get_awaitable().get();
    FAIL("the retry limit must fail the query");
  } catch (std::exception const& e) {
    CHECK(std::string(e.what()).find("exceeded maximum retry limit") != std::string::npos);
  }
  sirius::test::retire_scheduler_query(scheduler.lifecycle,
                                       *scheduler.executor,
                                       sirius::make_query_id(0),
                                       false,
                                       scheduler.creator.get());

  auto const attempts = probe->observed();
  REQUIRE(attempts.size() == retry_limit + 1);
  std::size_t previous_floor = 0;
  for (std::size_t i = 0; i < attempts.size(); ++i) {
    CAPTURE(i);
    CHECK(attempts[i].oom);
    CHECK(attempts[i].config.window_rows == merge::kShrinkWindowRowsFloor);
    CHECK(attempts[i].config.output_target_bytes == 16 * kMiB);
    CHECK(same_cursors(attempts[i].cursors, snapshot));
    if (i > 0) { CHECK(attempts[i].retry_floor > previous_floor); }
    previous_floor = attempts[i].retry_floor;
  }
  // Nothing was emitted or consumed: the runs are all still in the repository, and the failed
  // task still owns the partition, so no second task can start on it.
  CHECK(s.output.num_chunks() == 0);
  CHECK(s.repo.total_size() == 4);
  CHECK_FALSE(s.op->get_next_task_input_data());
}

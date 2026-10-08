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

#include "catch.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "operator/partitioned_join_test_utils.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace acc      = sirius::test::accumulation;
namespace op       = sirius::op;
namespace op_utils = sirius::test::operator_utils;
namespace pl       = sirius::pipeline;
using acc::wait_until;
using operator_ptr                      = std::unique_ptr<op::operator_data>;
constexpr std::size_t small_reservation = 64 * 1024;
constexpr std::uint64_t bloom_cap       = 64 * acc::mib;

class observed_repository final : public cucascade::shared_data_repository {
 public:
  void add_data_batch(acc::batch_ptr batch, std::size_t partition = 0) override
  {
    if (before_deposit) { before_deposit(); }
    cucascade::shared_data_repository::add_data_batch(std::move(batch), partition);
    deposits.fetch_add(1);
  }

  void discard_batches()
  {
    for (std::size_t partition = 0; partition < num_partitions(); ++partition) {
      while (pop_next_data_batch(partition)) {}
    }
  }

  std::function<void()> before_deposit;
  std::atomic<std::size_t> deposits{0};
};

class observed_partition final : public op::sirius_physical_partition {
 public:
  using sirius_physical_partition::sirius_physical_partition;

  operator_ptr execute(op::operator_data const& input, ::cuda::stream_ref stream) override
  {
    if (before_execute) { before_execute(input, stream); }
    auto result = sirius_physical_partition::execute(input, stream);
    if (after_execute) { after_execute(input, stream); }
    return result;
  }

  std::function<void(op::operator_data const&, ::cuda::stream_ref)> before_execute;
  std::function<void(op::operator_data const&, ::cuda::stream_ref)> after_execute;
};

struct allocation_observer {
  std::function<void(int, ::cuda::stream_ref, std::size_t)> before_allocate;
};

// Interpose below reservation tracking so the task retains its ordinary allocator and policy.
class observed_resource {
 public:
  observed_resource(int device, std::size_t capacity, std::shared_ptr<allocation_observer> observer)
    : upstream_{cucascade::memory::make_default_gpu_memory_resource(device, capacity)},
      observer_{std::move(observer)},
      device_{device}
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return upstream_.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    upstream_.deallocate_sync(pointer, bytes, alignment);
  }
  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    if (observer_->before_allocate) { observer_->before_allocate(device_, stream, bytes); }
    return upstream_.allocate(stream, bytes, alignment);
  }
  void deallocate(::cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    upstream_.deallocate(stream, pointer, bytes, alignment);
  }
  friend void get_property(observed_resource const&, ::cuda::mr::device_accessible) noexcept {}
  bool operator==(observed_resource const& other) const noexcept
  {
    return observer_ == other.observer_ && device_ == other.device_;
  }

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> upstream_;
  std::shared_ptr<allocation_observer> observer_;
  int device_;
};

struct retry_record {
  bool constrain_first_attempt = false;
  std::atomic<std::size_t> retries{0};
  std::size_t floor         = 0;
  std::size_t resume_index  = 0;
  std::uint64_t original_id = 0;
};

class observed_task final : public pl::gpu_pipeline_task {
 public:
  observed_task(std::uint64_t id,
                std::unique_ptr<pl::sirius_pipeline_task_local_state> local,
                std::shared_ptr<pl::sirius_pipeline_task_global_state> global,
                std::shared_ptr<retry_record> record)
    : gpu_pipeline_task{id, {}, std::move(local), std::move(global)}, record_{std::move(record)}
  {
  }

  pl::reservation_size_info get_estimated_reservation_size_info(
    cucascade::memory::memory_space const* space) const override
  {
    auto info = gpu_pipeline_task::get_estimated_reservation_size_info(space);
    if (record_->constrain_first_attempt &&
        _local_state->cast<pl::gpu_pipeline_task_local_state>().retry_count == 0) {
      info.reservation_size = small_reservation;
    }
    return info;
  }

  std::unique_ptr<pl::gpu_pipeline_task> create_rescheduled_task(
    std::uint64_t id, std::unique_ptr<pl::sirius_pipeline_task_local_state> local) override
  {
    auto const& state     = local->cast<pl::gpu_pipeline_task_local_state>();
    auto const& input     = dynamic_cast<op::pipelineable_operator_data const&>(*state._input_data);
    record_->floor        = state.get_retry_reservation_floor();
    record_->resume_index = state._start_operator_index;
    record_->original_id  = input.original_batch_ids().front();
    record_->retries.fetch_add(1);
    return std::make_unique<observed_task>(
      id, std::move(local), get_shared_global_state(), record_);
  }

 private:
  std::shared_ptr<retry_record> record_;
};

struct harness {
  std::shared_ptr<allocation_observer> allocations = std::make_shared<allocation_observer>();
  acc::fixture memory;
  cucascade::shared_data_repository input;
  observed_repository output;
  std::shared_ptr<op_utils::controllable_pipeline> producer =
    std::make_shared<op_utils::controllable_pipeline>();
  op_utils::partitioned_join tree;
  observed_partition* partition;
  std::shared_ptr<pl::sirius_pipeline> pipeline;
  std::shared_ptr<pl::completion_handler> completion = std::make_shared<pl::completion_handler>();
  std::shared_ptr<pl::sirius_pipeline_task_global_state> global;
  std::unique_ptr<pl::task_scheduler> scheduler;
  std::uint64_t next_task_id = 1;

  explicit harness(std::size_t devices = 1, bool per_stream = false)
    : memory{{.devices             = devices,
              .per_stream_tracking = per_stream,
              .gpu_resource_factory =
                [observer = allocations](int device, std::size_t capacity) {
                  return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
                    observed_resource{device, capacity, observer}};
                }}},
      tree{op_utils::make_partitioned_join(
        {.side_types           = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::BIGINT},
         .filter_plan          = memory.make_plan(bloom_cap),
         .stats                = &memory.stats,
         .make_build_partition = [](duckdb::vector<sirius::logical_type> types,
                                    op::sirius_physical_hash_join& join)
           -> duckdb::unique_ptr<op::sirius_physical_partition> {
           return duckdb::make_uniq<observed_partition>(std::move(types), 1, &join, true);
         }})},
      partition{static_cast<observed_partition*>(tree.build_partition)}
  {
    auto* const receiver   = tree.join->children[1].get();
    tree.join->operator_id = 1;
    partition->operator_id = 2;
    receiver->operator_id  = 3;
    partition->set_num_partitions(2);

    pipeline = std::make_shared<pl::sirius_pipeline>(pl::pipeline_build_context{nullptr, true});
    pipeline->set_pipeline_id(7);
    pl::sirius_pipeline_build_state state;
    state.set_pipeline_source(*pipeline, *partition);
    state.set_pipeline_operators(*pipeline, {*partition});
    state.set_pipeline_sink(*pipeline, partition, 1);
    partition->set_pipeline(pipeline);
    auto upstream          = std::make_unique<op::sirius_physical_operator::port>();
    upstream->type         = op::MemoryBarrierType::FULL;
    upstream->repo         = &input;
    upstream->src_pipeline = producer;
    partition->add_port("default", std::move(upstream));
    auto downstream          = std::make_unique<op::sirius_physical_operator::port>();
    downstream->type         = op::MemoryBarrierType::FULL;
    downstream->repo         = &output;
    downstream->src_pipeline = pipeline;
    receiver->add_port("default", std::move(downstream));
    partition->add_next_port_after_sink({receiver, "default", {}});
    global = std::make_shared<pl::sirius_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    global->set_completion_handler(completion);
  }

  ~harness()
  {
    if (scheduler) { scheduler->stop(); }
    allocations->before_allocate = {};
    tree.join->cancel_dynamic_filter_publication();
    // Task output buffers retain executor streams, so free them before the scheduler's stream pool.
    output.discard_batches();
    for (std::size_t partition = 0; partition < input.num_partitions(); ++partition) {
      while (input.pop_next_data_batch(partition)) {}
    }
  }

  auto tasks(std::vector<acc::batch_ptr> const& batches,
             std::shared_ptr<retry_record> record = std::make_shared<retry_record>())
  {
    for (auto const& batch : batches) {
      partition->push_data_batch("default", batch);
    }
    std::vector<std::unique_ptr<observed_task>> result;
    for (std::size_t i = 0; i < batches.size(); ++i) {
      auto data = partition->get_next_task_input_data();
      REQUIRE(data);
      result.push_back(std::make_unique<observed_task>(
        next_task_id++,
        std::make_unique<pl::gpu_pipeline_task_local_state>(std::move(data)),
        global,
        record));
    }
    REQUIRE(memory.stats.snapshot().accumulations_started == 1);
    return result;
  }

  void start()
  {
    sirius::exec::thread_pool_config config;
    config.num_threads        = 1;
    config.thread_name_prefix = "publication-test";
    scheduler                 = std::make_unique<pl::task_scheduler>(
      config, *memory.manager, sirius::test::make_test_telemetry_context());
    scheduler->start();
  }

  void schedule(std::unique_ptr<observed_task> task, int device = 0)
  {
    task->local_state()->cast<pl::gpu_pipeline_task_local_state>().set_preferred_device_id(device);
    scheduler->schedule(std::move(task));
  }

  void run_inline(std::unique_ptr<observed_task> task, int device)
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto info        = task->get_estimated_reservation_size_info(&memory.gpu(device));
    auto reservation = memory.gpu(device).make_reservation_or_null(small_reservation +
                                                                   info.bytes_to_materialize_input);
    REQUIRE(reservation);
    task->local_state()->cast<pl::gpu_pipeline_task_local_state>().set_reservation(
      std::move(reservation), info);
    task->execute(memory.task_stream(device));
  }

  void wait_completed(std::size_t count)
  {
    wait_until([&] { return pipeline->get_tasks_completed() >= count || completion->has_error(); });
    REQUIRE_FALSE(completion->has_error());
  }
};

class reuse_operator final : public op::sirius_physical_operator {
 public:
  reuse_operator() : sirius_physical_operator{op::SiriusPhysicalOperatorType::FILTER, {}, 0} {}
  operator_ptr execute(op::operator_data const& input, ::cuda::stream_ref) override
  {
    return std::make_unique<op::pipelineable_operator_data>(
      dynamic_cast<op::pipelineable_operator_data const&>(input).get_data_batches());
  }
  void sink(op::operator_data const&, ::cuda::stream_ref) override { ++deposits; }
  std::atomic<std::size_t> deposits{0};
};

/**
 * @brief Whether the channel holds the terminal pair of Bloom filters and every row of @p batches
 * passes both on GPU 0. Asserts nothing, so the repository callback may call it.
 */
bool contains_every_build_row(acc::fixture const& memory,
                              std::vector<acc::batch_ptr> const& batches)
{
  auto snapshot = memory.channel->snapshot();
  if (!snapshot.terminal() || snapshot.entries().size() != 2) { return false; }
  auto& space       = memory.gpu(0);
  auto const stream = space.acquire_stream();
  for (auto const& entry : snapshot.entries()) {
    auto const* bloom = dynamic_cast<op::sirius_dynamic_bloom_filter const*>(entry.filter.get());
    if (bloom == nullptr || !bloom->is_available_on_device(0)) { return false; }
    for (auto const& batch : batches) {
      auto source = batch->to_read_only();
      auto column = sirius::get_cudf_table_view(source).column(
        static_cast<cudf::size_type>(entry.column_index));
      auto const mask =
        acc::probe_membership(*bloom, column, 0, stream, space.get_default_allocator());
      if (!mask || std::count(mask->begin(), mask->end(), true) != column.size()) { return false; }
    }
  }
  return true;
}

void require_worker_reuse(harness& test)
{
  reuse_operator next;
  next.operator_id = 10;
  auto pipeline = std::make_shared<pl::sirius_pipeline>(pl::pipeline_build_context{nullptr, true});
  pipeline->set_pipeline_id(8);
  pl::sirius_pipeline_build_state state;
  state.set_pipeline_source(*pipeline, next);
  state.set_pipeline_operators(*pipeline, {next});
  state.set_pipeline_sink(*pipeline, &next, 1);
  auto global = std::make_shared<pl::sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());
  auto completion = std::make_shared<pl::completion_handler>();
  global->set_completion_handler(completion);
  absl::Cleanup quiesce = [&] { test.scheduler->drain_after_error(pipeline->get_query_id()); };
  auto local            = std::make_unique<pl::gpu_pipeline_task_local_state>(
    std::make_unique<op::pipelineable_operator_data>(
      std::vector<acc::batch_ptr>{test.memory.make_batch(0, 0, 100)}));
  local->set_preferred_device_id(0);
  test.scheduler->schedule(std::make_unique<pl::gpu_pipeline_task>(
    100, std::vector<cucascade::shared_data_repository*>{}, std::move(local), global));
  wait_until([&] { return next.deposits.load() == 1 || completion->has_error(); });
  test.scheduler->wait_for_completion(pipeline->get_query_id());
  REQUIRE_FALSE(completion->has_error());
  REQUIRE(next.deposits.load() == 1);
  REQUIRE(pipeline->get_tasks_completed() == 1);
}

}  // namespace

TEST_CASE("the elected PARTITION task publishes before its first repository deposit",
          "[pipeline][dynamic_filter][in_task_publication]")
{
  bool const per_stream = GENERATE(false, true);
  harness test{1, per_stream};
  auto const first  = test.memory.make_batch(0, 0, 2000);
  auto const second = test.memory.make_batch(0, 5000, 2000);
  auto tasks        = test.tasks({first, second});
  std::vector<std::pair<bool, std::size_t>> snapshots;
  bool full_union_at_deposit = false;
  test.output.before_deposit = [&] {
    auto snapshot = test.memory.channel->snapshot();
    snapshots.emplace_back(snapshot.terminal(), snapshot.entries().size());
    if (snapshots.size() == 3) {
      full_union_at_deposit = contains_every_build_row(test.memory, {first, second});
    }
  };
  test.start();
  absl::Cleanup stop = [&] { test.scheduler->stop(); };
  // Reverse the certified input order: the final contributor is not the final inventory entry.
  test.schedule(std::move(tasks[1]));
  test.wait_completed(1);
  test.schedule(std::move(tasks[0]));
  test.wait_completed(2);
  test.scheduler->wait_for_completion(test.pipeline->get_query_id());
  REQUIRE(snapshots.size() == 4);
  REQUIRE((snapshots[0] == std::pair<bool, std::size_t>{false, 0}));
  REQUIRE((snapshots[1] == std::pair<bool, std::size_t>{false, 0}));
  REQUIRE((snapshots[2] == std::pair<bool, std::size_t>{true, 2}));
  REQUIRE((snapshots[3] == std::pair<bool, std::size_t>{true, 2}));
  REQUIRE(full_union_at_deposit);
  REQUIRE(test.memory.stats.snapshot().accumulation_completed_contributions == 2);
  REQUIRE(test.memory.stats.snapshot().accumulation_publications_finished == 1);
  test.memory.require_members({first, second});
}

TEST_CASE("mandatory scatter OOM retries the elected identity through the executor",
          "[pipeline][dynamic_filter][in_task_publication][memory]")
{
  harness test;
  auto const first                = test.memory.make_batch(0, 0, 100);
  auto const final                = test.memory.make_batch(0, 1000, 200'000);
  auto record                     = std::make_shared<retry_record>();
  record->constrain_first_attempt = true;
  auto tasks                      = test.tasks({first, final}, record);
  std::atomic<bool> fail_next{false};
  std::atomic<std::size_t> failures{0};
  test.allocations->before_allocate = [&](int, ::cuda::stream_ref, std::size_t) {
    if (fail_next.exchange(false)) {
      ++failures;
      throw rmm::out_of_memory{"refused mandatory scatter allocation"};
    }
  };
  std::atomic<bool> retry_deposit_ready{false};
  std::atomic<bool> retry_kept_build_open{false};
  test.partition->before_execute = [&](op::operator_data const&, ::cuda::stream_ref) {
    if (record->retries.load() != 0) {
      retry_kept_build_open.store(!test.partition->finalized.load() &&
                                  !test.memory.channel->snapshot().terminal());
    }
  };
  test.output.before_deposit = [&] {
    if (record->retries.load() != 0) {
      auto snapshot = test.memory.channel->snapshot();
      retry_deposit_ready.store(snapshot.terminal() && snapshot.entries().size() == 2);
    }
  };
  test.start();
  absl::Cleanup stop = [&] { test.scheduler->stop(); };
  test.schedule(std::move(tasks[0]));
  test.wait_completed(1);
  fail_next.store(true);
  test.schedule(std::move(tasks[1]));
  test.wait_completed(3);
  test.scheduler->wait_for_completion(test.pipeline->get_query_id());
  REQUIRE(failures.load() == 1);
  REQUIRE(record->retries.load() == 1);
  REQUIRE(record->resume_index == 0);
  REQUIRE(record->original_id == final->get_batch_id());
  REQUIRE(record->floor >= pl::gpu_pipeline_task_local_state::kDefaultRetryRequestBytes);
  REQUIRE(retry_deposit_ready.load());
  REQUIRE(retry_kept_build_open.load());
  REQUIRE(test.partition->finalized.load());
  REQUIRE(test.output.deposits.load() == 4);
  auto const counters = test.memory.stats.snapshot();
  REQUIRE(counters.accumulation_duplicate_contributions == 1);
  REQUIRE(counters.accumulation_completed_contributions == 2);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.filters_pushed == 2);
  REQUIRE(test.global->get_memory_history().totals().output_records == 2);
  test.memory.require_members({first, final});
}

TEST_CASE("a migrated final PARTITION accounts merge scratch in task history",
          "[pipeline][dynamic_filter][in_task_publication][memory][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  bool const per_stream = GENERATE(false, true);
  harness test{2, per_stream};
  auto const first = test.memory.make_batch(1, 0, 2000);
  auto const final = test.memory.make_batch(1, 5000, 2000);
  auto tasks       = test.tasks({first, final});
  test.run_inline(std::move(tasks[0]), 1);
  auto const scratch_bytes     = acc::merge_scratch_bytes(acc::bloom_geometry(4000, 2, bloom_cap));
  std::size_t scratch_requests = 0;
  std::size_t live_before      = 0;
  std::size_t live_after       = 0;
  std::size_t final_peak       = 0;
  bool cloned                  = false;
  test.allocations->before_allocate =
    [&](int device, ::cuda::stream_ref stream, std::size_t requested) {
      if (device == 0 && requested == scratch_bytes && test.output.deposits.load() == 2) {
        ++scratch_requests;
        live_before = test.memory.allocator(0).get_allocated_bytes(stream) - requested;
      }
    };
  test.partition->after_execute = [&](op::operator_data const& input, ::cuda::stream_ref stream) {
    auto const& data = dynamic_cast<op::pipelineable_operator_data const&>(input);
    cloned           = data.original_batch_ids().front() == final->get_batch_id() &&
             data.get_data_batches().front()->get_batch_id() != final->get_batch_id();
    live_after = test.memory.allocator(0).get_allocated_bytes(stream);
    final_peak = test.memory.allocator(0).get_peak_allocated_bytes(stream);
  };
  test.run_inline(std::move(tasks[1]), 0);
  REQUIRE(cloned);
  REQUIRE(scratch_requests == 1);
  REQUIRE(live_after == live_before);
  REQUIRE(final_peak >= live_before + scratch_bytes);
  auto const basis    = sirius::get_cudf_table_view(*final).num_rows() * 12;
  auto const estimate = test.global->get_memory_history().estimate_peak_memory(basis);
  REQUIRE(estimate);
  REQUIRE(*estimate >= scratch_bytes / 2);
  REQUIRE(test.global->get_memory_history().totals().output_records == 2);
  REQUIRE(test.memory.gpu(0).get_total_reserved_memory() == 0);
  REQUIRE_FALSE(test.memory.allocator(0).is_stream_tracked(test.memory.task_stream(0)));
  REQUIRE(test.memory.stats.snapshot().accumulation_completed_contributions == 2);
  test.memory.require_members({first, final});
}

TEST_CASE("scheduler error drain joins collecting and publishing PARTITION tasks before reuse",
          "[pipeline][dynamic_filter][in_task_publication][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  bool const during_publication = GENERATE(false, true);
  acc::load_accumulation_kernels(2);
  harness test{2};
  auto const first                = test.memory.make_batch(1, 0, 2000);
  auto const final                = test.memory.make_batch(0, 5000, 2000);
  auto const baseline             = test.memory.allocated_bytes();
  auto record                     = std::make_shared<retry_record>();
  record->constrain_first_attempt = true;
  auto tasks                      = test.tasks({first, final}, record);
  auto const scratch_bytes = acc::merge_scratch_bytes(acc::bloom_geometry(4000, 2, bloom_cap));
  std::atomic<std::size_t> scratch_requests{0};
  std::unique_ptr<acc::stream_gate> gate;
  std::atomic<bool> gated{false};
  if (during_publication) {
    test.run_inline(std::move(tasks[0]), 1);
    test.allocations->before_allocate =
      [&](int device, ::cuda::stream_ref stream, std::size_t requested) {
        if (device == 0 && requested == scratch_bytes && test.output.deposits.load() == 2) {
          if (scratch_requests.fetch_add(1) == 0) {
            gate = std::make_unique<acc::stream_gate>(stream);
            gated.store(true);
          }
        }
      };
  } else {
    test.partition->after_execute = [&](op::operator_data const&, ::cuda::stream_ref stream) {
      gate = std::make_unique<acc::stream_gate>(stream);
      gated.store(true);
    };
  }
  test.start();
  absl::Cleanup stop = [&] { test.scheduler->stop(); };
  test.schedule(std::move(tasks[during_publication ? 1 : 0]), during_publication ? 0 : 1);
  wait_until([&] { return gated.load() || test.completion->has_error(); });
  REQUIRE(gated.load());
  test.completion->report_error("another query task failed");
  test.tree.join->cancel_dynamic_filter_publication();
  auto drained = std::async(
    std::launch::async, [&] { test.scheduler->drain_after_error(test.pipeline->get_query_id()); });
  auto const blocked = drained.wait_for(std::chrono::milliseconds{100});
  gate->open();
  REQUIRE(blocked == std::future_status::timeout);
  drained.get();
  REQUIRE_FALSE(gate->timed_out());
  REQUIRE(scratch_requests.load() == (during_publication ? 1 : 0));
  tasks.clear();
  test.output.discard_batches();
  test.tree.join->dynamic_filter_session().finalize_input();
  REQUIRE(test.memory.channel->snapshot().terminal());
  REQUIRE(test.memory.channel->snapshot().empty());
  REQUIRE(test.memory.stats.snapshot().filters_pushed == 0);
  REQUIRE(test.memory.allocated_bytes() == baseline);
  require_worker_reuse(test);
}

TEST_CASE("optional publication scratch OOM deposits without rescheduling the PARTITION task",
          "[pipeline][dynamic_filter][in_task_publication][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  harness test{2};
  auto const first                = test.memory.make_batch(1, 0, 2000);
  auto const final                = test.memory.make_batch(0, 5000, 2000);
  auto record                     = std::make_shared<retry_record>();
  record->constrain_first_attempt = true;
  auto tasks                      = test.tasks({first, final}, record);
  test.run_inline(std::move(tasks[0]), 1);
  auto const scratch_bytes = acc::merge_scratch_bytes(acc::bloom_geometry(4000, 2, bloom_cap));
  std::atomic<std::size_t> scratch_requests{0};
  test.allocations->before_allocate = [&](int device, ::cuda::stream_ref, std::size_t requested) {
    if (device == 0 && requested == scratch_bytes && test.output.deposits.load() == 2) {
      ++scratch_requests;
      throw rmm::out_of_memory{"refused optional merge scratch"};
    }
  };
  test.start();
  absl::Cleanup stop = [&] { test.scheduler->stop(); };
  test.schedule(std::move(tasks[1]));
  test.wait_completed(2);
  test.scheduler->wait_for_completion(test.pipeline->get_query_id());
  REQUIRE(scratch_requests.load() == 1);
  REQUIRE(record->retries.load() == 0);
  REQUIRE(test.output.deposits.load() == 4);
  REQUIRE(test.memory.channel->snapshot().terminal());
  REQUIRE(test.memory.channel->snapshot().empty());
  REQUIRE(test.memory.stats.snapshot().filters_pushed == 0);
  REQUIRE(test.memory.stats.snapshot().accumulations_skipped_admission == 1);
  require_worker_reuse(test);
}

TEST_CASE("a fatal PARTITION task drains its collecting publication and leaves the worker reusable",
          "[pipeline][dynamic_filter][in_task_publication]")
{
  harness test;
  auto const first    = test.memory.make_batch(0, 0, 2000);
  auto const final    = test.memory.make_batch(0, 5000, 2000);
  auto const baseline = test.memory.allocated_bytes();
  auto record         = std::make_shared<retry_record>();
  auto tasks          = test.tasks({first, final}, record);
  test.run_inline(std::move(tasks[0]), 0);
  test.partition->before_execute = [](op::operator_data const&, ::cuda::stream_ref) {
    throw std::runtime_error{"mandatory PARTITION failure"};
  };
  test.start();
  absl::Cleanup stop = [&] { test.scheduler->stop(); };
  test.schedule(std::move(tasks[1]));
  wait_until([&] { return test.completion->has_error(); });
  test.tree.join->cancel_dynamic_filter_publication();
  test.scheduler->drain_after_error(test.pipeline->get_query_id());
  test.output.discard_batches();
  test.tree.join->dynamic_filter_session().finalize_input();
  REQUIRE(record->retries.load() == 0);
  REQUIRE(test.output.deposits.load() == 2);
  REQUIRE(test.memory.channel->snapshot().terminal());
  REQUIRE(test.memory.channel->snapshot().empty());
  REQUIRE(test.memory.allocated_bytes() == baseline);
  require_worker_reuse(test);
}

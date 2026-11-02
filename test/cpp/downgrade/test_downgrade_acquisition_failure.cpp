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
#include "data/sirius_converter_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_submission.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::parallel {

struct downgrade_executor_test_access {
  using site = downgrade_executor::stream_acquisition_site;

  static void fail_once(downgrade_executor& executor,
                        site acquisition_site,
                        size_t skip,
                        std::exception_ptr error,
                        std::function<void()> before_failure = {})
  {
    executor._stream_acquisition_fault.emplace(downgrade_executor::stream_acquisition_fault{
      acquisition_site, skip, std::move(error), std::move(before_failure)});
  }

  static std::thread::id processing_thread(downgrade_executor const& executor)
  {
    return executor._processing_thread.get_id();
  }

  static size_t available_streams(downgrade_executor const& executor)
  {
    return executor._stream_pool->size();
  }

  static bool monitor_request_enqueued(downgrade_executor const& executor)
  {
    return executor._monitor_request_enqueued.load(std::memory_order_relaxed);
  }

  static std::future<size_t> request_as_monitor(downgrade_executor& executor,
                                                std::function<bool()> predicate)
  {
    auto request                = std::make_unique<downgrade_request>();
    request->predicate          = std::move(predicate);
    request->is_monitor_request = true;
    auto result                 = request->result.get_future();
    executor._monitor_request_enqueued.store(true, std::memory_order_relaxed);
    if (!executor._request_queue.push(std::move(request))) {
      throw std::runtime_error("test monitor request queue is inactive");
    }
    return result;
  }
};

}  // namespace sirius::parallel

namespace {

using namespace std::chrono_literals;
using test_access = sirius::parallel::downgrade_executor_test_access;
using site        = test_access::site;
using cucascade::memory::Tier;
const auto query_id          = sirius::make_query_id(0);
constexpr size_t batch_bytes = 128 * sizeof(int32_t);

struct acquisition_failure : std::runtime_error {
  acquisition_failure() : std::runtime_error("one-shot downgrade stream acquisition failure") {}
};

class worker_gate {
 public:
  void wait()
  {
    std::unique_lock lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    if (!cv_.wait_for(lock, 10s, [&] { return released_; })) { timed_out_ = true; }
  }

  bool wait_until_entered()
  {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [&] { return entered_; });
  }

  void release()
  {
    std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

  bool timed_out() const
  {
    std::lock_guard lock(mutex_);
    return timed_out_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_{false};
  bool released_{false};
  bool timed_out_{false};
};

class failing_gpu_representation : public cucascade::gpu_table_representation {
 public:
  failing_gpu_representation(std::unique_ptr<cudf::table> table,
                             cucascade::memory::memory_space& space,
                             ::cuda::stream_ref stream,
                             worker_gate& gate)
    : gpu_table_representation(std::move(table), space, stream), gate_(gate)
  {
  }

  worker_gate& gate_;
};

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_memory_manager()
{
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(256ull << 20)
    .set_reservation_fraction_per_gpu(0.9)
    .set_per_numa_region_capacity(64ull << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(0.9)
    .set_host_pool_features(1ull << 20, 4, 1);
  return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
}

struct fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory = make_memory_manager();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::data::data_repository_manager_registry repositories;
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue{
    sirius::pipeline::index_keys_for};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  std::unique_ptr<sirius::parallel::downgrade_executor> executor;

  fixture()
  {
    sirius::converter_registry::initialize();
    lifecycle.open_query(query_id);
    repositories.create_for_query(query_id);
    sirius::exec::downgrade_executor_config config{
      .thread_pool    = {.num_threads = 2, .thread_name_prefix = "acquire_failure"},
      .monitor_period = 0ms};
    auto* gpu = memory->get_memory_space(Tier::GPU, 0);
    executor  = std::make_unique<sirius::parallel::downgrade_executor>(
      lifecycle, config, repositories, gpu->get_id(), gpu, *memory, &queue);
    executor->start();
  }

  std::shared_ptr<cucascade::data_batch> add_batch(site location,
                                                   uint64_t id,
                                                   worker_gate* failing_conversion = nullptr)
  {
    auto* gpu   = memory->get_memory_space(Tier::GPU, 0);
    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            128,
                                            cudf::mask_state::UNALLOCATED,
                                            producer,
                                            gpu->get_default_allocator());
    REQUIRE(cudaMemsetAsync(
              column->mutable_view().data<int32_t>(), 0x2a, batch_bytes, producer.value()) ==
            cudaSuccess);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    auto table = std::make_unique<cudf::table>(std::move(columns));
    std::unique_ptr<cucascade::idata_representation> representation;
    if (failing_conversion) {
      representation = std::make_unique<failing_gpu_representation>(
        std::move(table), *gpu, producer, *failing_conversion);
    } else {
      representation =
        cucascade::gpu_table_representation::make_written_on(std::move(table), *gpu, producer);
    }
    producer.synchronize();
    auto batch = cucascade::data_batch::make(id, std::move(representation));
    if (location == site::repository) {
      auto repository = std::make_unique<cucascade::shared_data_repository>();
      repository->add_data_batch(batch);
      repositories.get(query_id)->add_new_repository(id, "out", std::move(repository));
    } else {
      auto input = std::make_unique<sirius::op::pipelineable_operator_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
      auto local =
        std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input));
      auto global = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
        nullptr, sirius::test::make_test_telemetry_context());
      auto submission = lifecycle.try_begin_submission(query_id);
      auto task       = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
        id, std::vector<cucascade::shared_data_repository*>{}, std::move(local), std::move(global));
      task->retain_work(submission.take_work_lease());
      REQUIRE(queue.push(std::move(task)));
    }
    return batch;
  }
};

template <typename Predicate>
bool wait_until(Predicate predicate)
{
  auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

void require_original_exception(std::future<size_t>& result, std::exception_ptr expected)
{
  REQUIRE(result.wait_for(5s) == std::future_status::ready);
  try {
    static_cast<void>(result.get());
    FAIL("downgrade acquisition failure was reported as success");
  } catch (acquisition_failure const&) {
    REQUIRE(std::current_exception() == expected);
  }
}

void check_acquisition_failure(site location, bool after_worker, bool monitor_request = false)
{
  worker_gate gate;
  fixture f;
  auto failed_batch = f.add_batch(location, location == site::repository ? 2 : 1);
  std::shared_ptr<cucascade::data_batch> running_batch;
  if (after_worker) { running_batch = f.add_batch(location, location == site::repository ? 1 : 2); }
  auto const failed_references  = failed_batch.use_count();
  auto const running_references = running_batch.use_count();
  auto error                    = std::make_exception_ptr(acquisition_failure{});
  std::promise<void> fault_reached;
  auto reached = fault_reached.get_future();
  test_access::fail_once(*f.executor, location, after_worker ? 1 : 0, error, [&] {
    if (after_worker && !gate.wait_until_entered()) {
      throw std::runtime_error("downgrade worker did not reach its predicate");
    }
    fault_reached.set_value();
  });
  auto manager   = test_access::processing_thread(*f.executor);
  auto predicate = [&gate, manager] {
    if (std::this_thread::get_id() != manager) { gate.wait(); }
    return false;
  };
  auto result = monitor_request ? test_access::request_as_monitor(*f.executor, predicate)
                : after_worker  ? f.executor->request_downgrade(predicate)
                                : f.executor->request_free_memory(2 * batch_bytes);
  absl::Cleanup release_worker = [&] {
    gate.release();
    f.executor->stop();
  };
  REQUIRE(reached.wait_for(5s) == std::future_status::ready);
  reached.get();

  if (after_worker) {
    REQUIRE(wait_until([&] {
      return location == site::repository ? f.lifecycle.activity(query_id).work == 1
                                          : f.queue.size() == 1;
    }));
    CHECK(result.wait_for(50ms) == std::future_status::timeout);
    CHECK(test_access::available_streams(*f.executor) == 1);
    if (monitor_request) { CHECK(test_access::monitor_request_enqueued(*f.executor)); }
    CHECK(failed_batch.use_count() == failed_references);
    CHECK(running_batch.use_count() == running_references + (location == site::repository ? 1 : 0));
    gate.release();
  }
  require_original_exception(result, error);
  CHECK_FALSE(gate.timed_out());
  CHECK(test_access::available_streams(*f.executor) == 2);
  CHECK_FALSE(test_access::monitor_request_enqueued(*f.executor));
  CHECK(failed_batch.use_count() == failed_references);
  CHECK(running_batch.use_count() == running_references);
  CHECK(failed_batch->get_state() == cucascade::batch_state::idle);
  CHECK(failed_batch->to_read_only().get_memory_space()->get_tier() == Tier::GPU);
  if (running_batch) {
    CHECK(running_batch->to_read_only().get_memory_space()->get_tier() == Tier::HOST);
  }
  CHECK(f.lifecycle.activity(query_id).work ==
        (location == site::pipeline_queue ? 1 + after_worker : 0));

  auto next = monitor_request ? test_access::request_as_monitor(*f.executor, [] { return false; })
                              : f.executor->request_free_memory(2 * batch_bytes);
  REQUIRE(next.wait_for(5s) == std::future_status::ready);
  CHECK(next.get() == batch_bytes);
  CHECK(failed_batch->to_read_only().get_memory_space()->get_tier() == Tier::HOST);
  CHECK(test_access::available_streams(*f.executor) == 2);
  CHECK_FALSE(test_access::monitor_request_enqueued(*f.executor));
  f.queue.drain();
  CHECK(f.lifecycle.activity(query_id).work == 0);
  CHECK(f.lifecycle.activity(query_id).submissions == 0);
}

}  // namespace

TEST_CASE("Downgrade acquisition failure before dispatch preserves candidates",
          "[downgrade_acquisition_failure]")
{
  SECTION("repository") { check_acquisition_failure(site::repository, false); }
  SECTION("pipeline queue") { check_acquisition_failure(site::pipeline_queue, false); }
}

TEST_CASE("Downgrade acquisition failure drains live worker captures",
          "[downgrade_acquisition_failure]")
{
  SECTION("repository") { check_acquisition_failure(site::repository, true); }
  SECTION("pipeline queue") { check_acquisition_failure(site::pipeline_queue, true); }
}

TEST_CASE("Downgrade monitor acquisition failure releases its request gate",
          "[downgrade_acquisition_failure]")
{
  SECTION("repository") { check_acquisition_failure(site::repository, true, true); }
  SECTION("pipeline queue") { check_acquisition_failure(site::pipeline_queue, true, true); }
}

TEST_CASE("Downgrade acquisition failure keeps failed worker candidates alive through drain",
          "[downgrade_acquisition_failure]")
{
  bool const retire_query = GENERATE(false, true);
  worker_gate gate;
  fixture f;
  sirius::converter_registry::get()
    .register_converter<failing_gpu_representation, cucascade::host_data_representation>(
      [](cucascade::idata_representation const& source,
         cucascade::memory::memory_space const*,
         ::cuda::stream_ref,
         cucascade::memory::reservation*) -> std::unique_ptr<cucascade::idata_representation> {
        source.cast<failing_gpu_representation>().gate_.wait();
        throw std::runtime_error("worker conversion failed");
      });
  absl::Cleanup cleanup = [&] {
    sirius::converter_registry::get()
      .unregister_converter<failing_gpu_representation, cucascade::host_data_representation>();
  };
  auto failed_batch = f.add_batch(site::pipeline_queue, 1);
  auto worker_batch = f.add_batch(site::pipeline_queue, 2, &gate);
  auto error        = std::make_exception_ptr(acquisition_failure{});
  std::promise<void> fault_reached;
  auto reached = fault_reached.get_future();
  test_access::fail_once(*f.executor, site::pipeline_queue, 1, error, [&] {
    if (!gate.wait_until_entered()) {
      throw std::runtime_error("downgrade worker did not reach its converter");
    }
    fault_reached.set_value();
  });
  auto result                  = f.executor->request_downgrade([] { return false; });
  absl::Cleanup release_worker = [&] {
    gate.release();
    f.executor->drain();
    // Failed conversions leave GPU buffers bound to the borrowed stream. Rebind before destroying
    // its pool.
    {
      auto batch = worker_batch->to_mutable();
      batch.rebind_stream(f.producer);
    }
    f.executor->stop();
  };
  REQUIRE(reached.wait_for(5s) == std::future_status::ready);
  reached.get();
  REQUIRE(wait_until([&] { return f.queue.size() == 1; }));
  CHECK(result.wait_for(50ms) == std::future_status::timeout);
  CHECK(f.lifecycle.activity(query_id).work == 2);
  if (retire_query) {
    f.lifecycle.quiesce_and_wait_for_submissions(query_id);
    f.queue.drain();
    CHECK(f.lifecycle.activity(query_id).work == 1);
  }
  gate.release();
  require_original_exception(result, error);
  CHECK_FALSE(gate.timed_out());
  CHECK(test_access::available_streams(*f.executor) == 2);
  CHECK(f.queue.size() == (retire_query ? 0 : 2));
  CHECK(f.lifecycle.activity(query_id).work == (retire_query ? 0 : 2));
  CHECK(worker_batch->get_state() == cucascade::batch_state::idle);
  CHECK(worker_batch->to_read_only().get_memory_space()->get_tier() == Tier::GPU);
  CHECK(failed_batch->to_read_only().get_memory_space()->get_tier() == Tier::GPU);
  f.queue.drain();
  CHECK(f.lifecycle.activity(query_id).work == 0);
  CHECK(f.lifecycle.activity(query_id).submissions == 0);
}

TEST_CASE("Downgrade stop and drain wait for workers after acquisition failure",
          "[downgrade_acquisition_failure]")
{
  bool const stop = GENERATE(false, true);
  worker_gate gate;
  fixture f;
  auto first  = f.add_batch(site::repository, 1);
  auto second = f.add_batch(site::repository, 2);
  auto error  = std::make_exception_ptr(acquisition_failure{});
  std::promise<void> fault_reached;
  auto reached = fault_reached.get_future();
  test_access::fail_once(*f.executor, site::repository, 1, error, [&] {
    if (!gate.wait_until_entered()) {
      throw std::runtime_error("downgrade worker did not reach its predicate");
    }
    fault_reached.set_value();
  });
  auto manager = test_access::processing_thread(*f.executor);
  auto result  = f.executor->request_downgrade([&] {
    if (std::this_thread::get_id() != manager) { gate.wait(); }
    return false;
  });
  std::promise<void> shutdown_started;
  auto started = shutdown_started.get_future();
  std::promise<void> shutdown_finished;
  auto finished = shutdown_finished.get_future();
  std::jthread shutdown;
  absl::Cleanup release_worker = [&] {
    gate.release();
    if (shutdown.joinable()) { shutdown.join(); }
    f.executor->stop();
  };
  REQUIRE(reached.wait_for(5s) == std::future_status::ready);
  reached.get();
  shutdown = std::jthread([&] {
    shutdown_started.set_value();
    try {
      if (stop) {
        f.executor->stop();
      } else {
        f.executor->drain();
      }
      shutdown_finished.set_value();
    } catch (...) {
      shutdown_finished.set_exception(std::current_exception());
    }
  });
  REQUIRE(started.wait_for(5s) == std::future_status::ready);
  CHECK(finished.wait_for(50ms) == std::future_status::timeout);
  CHECK(result.wait_for(0ms) == std::future_status::timeout);
  gate.release();
  require_original_exception(result, error);
  REQUIRE(finished.wait_for(5s) == std::future_status::ready);
  REQUIRE_NOTHROW(finished.get());
  shutdown.join();
  CHECK_FALSE(gate.timed_out());
  CHECK(f.lifecycle.activity(query_id).work == 0);
  if (stop) { f.executor->start(); }
  auto next = f.executor->request_free_memory(2 * batch_bytes);
  REQUIRE(next.wait_for(5s) == std::future_status::ready);
  CHECK(next.get() == batch_bytes);
}

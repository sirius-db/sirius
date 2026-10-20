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

#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/memory_prefetcher.hpp"
#include "utils/cuda_status_fault.hpp"
#include "utils/dynamic_filter_publication_fixture.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <utility>

namespace {
using sirius::scan_manager::memory_prefetcher;
using sirius::scan_manager::memory_prefetcher_config;
using sirius::scan_manager::split_connector;
using sirius::test::scoped_cuda_status_fault;

bool await_condition(auto predicate)
{
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return predicate();
}

void check_admission(sirius::exec::query_lifecycle_registry& lifecycle, bool fatal)
{
  CHECK(lifecycle.runtime_failed() == fatal);
  CHECK(lifecycle.runtime_failure_message().empty() == !fatal);
  CHECK(lifecycle.accepts_work(sirius::make_query_id(1)) == !fatal);
  CHECK(lifecycle.accepts_work(sirius::make_query_id(2)) == !fatal);
  if (fatal)
    CHECK_THROWS(lifecycle.open_query(sirius::make_query_id(3)));
  else
    CHECK_NOTHROW(lifecycle.open_query(sirius::make_query_id(3)));
  // Only the status was injected: no test leaves a poisoned CUDA context.
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
}

struct one_batch_provider : sirius::scan_manager::databatch_provider {
  explicit one_batch_provider(std::shared_ptr<cucascade::data_batch> source)
    : source(std::move(source))
  {
    validation.identity = validation.layout = validation.structure = {true, true};
  }
  std::shared_ptr<cucascade::data_batch> source;
  batch get_next_batch() override { return {.data = std::exchange(source, {})}; }
};
}  // namespace

TEST_CASE("prefetch lease acquisition reports fatal errors and preserves ordinary failures",
          "[cuda_status_fault][device_health][memory_prefetcher]")
{
  auto status = GENERATE(
    cudaErrorIllegalAddress, cudaErrorUnknown, cudaErrorMemoryAllocation, cudaErrorInvalidValue);
  sirius::test::publication::claim_fixture fixture;
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(sirius::make_query_id(1));
  lifecycle.open_query(sirius::make_query_id(2));
  auto connector = std::make_shared<split_connector>();
  memory_prefetcher_config config;
  config.num_threads      = 1;
  config.poll_interval_ms = 1;
  scoped_cuda_status_fault fault(scoped_cuda_status_fault::acquire_stream, status);
  memory_prefetcher worker(config, {connector}, fixture.gpu_space, lifecycle);
  bool observed = await_condition([&] { return fault.consumed(); });
  worker.stop();  // joins the catch/report path before inspecting the latch
  REQUIRE(observed);
  check_admission(lifecycle, status == cudaErrorIllegalAddress || status == cudaErrorUnknown);
}

TEST_CASE("prefetch conversion reports fatal errors without treating OOM as runtime failure",
          "[cuda_status_fault][device_health][memory_prefetcher]")
{
  auto status = GENERATE(
    cudaErrorIllegalAddress, cudaErrorUnknown, cudaErrorMemoryAllocation, cudaErrorInvalidValue);
  auto exception_type = GENERATE(0, 1, 2);  // cuDF/RMM, CCCL, Thrust
  sirius::test::publication::claim_fixture fixture;
  auto batch     = fixture.make_host_build_batch();
  auto connector = std::make_shared<split_connector>();
  one_batch_provider provider(batch);
  std::stop_source stop;
  sirius::scan_manager::load_balancing_scan_batch_coalescer::drain_cached_provider(
    provider, *connector, stop.get_token(), false);

  auto calls       = std::make_shared<std::atomic<unsigned>>(0);
  auto& converters = sirius::converter_registry::get();
  REQUIRE((converters.unregister_converter<cucascade::host_data_representation,
                                           cucascade::gpu_table_representation>()));
  converters
    .register_converter<cucascade::host_data_representation, cucascade::gpu_table_representation>(
      [calls, status, exception_type](
        cucascade::idata_representation&,
        const cucascade::memory::memory_space*,
        ::cuda::stream_ref,
        cucascade::memory::reservation*) -> std::unique_ptr<cucascade::idata_representation> {
        calls->fetch_add(1);
        if (exception_type == 1) throw ::cuda::cuda_error(status, "prefetch conversion");
        if (exception_type == 2)
          throw thrust::system_error(status, thrust::cuda_category(), "prefetch conversion");
        if (status == cudaErrorMemoryAllocation)
          throw rmm::out_of_memory("injected conversion OOM");
        throw cudf::cuda_error("injected prefetch conversion error", status);
      });
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(sirius::make_query_id(1));
  lifecycle.open_query(sirius::make_query_id(2));
  memory_prefetcher_config config;
  config.num_threads       = 1;
  config.min_free_fraction = 0;
  memory_prefetcher worker(config, {connector}, fixture.gpu_space, lifecycle);
  bool observed = await_condition([&] { return calls->load() != 0; });
  worker.stop();
  REQUIRE(observed);
  bool fatal = status == cudaErrorIllegalAddress || status == cudaErrorUnknown;
  check_admission(lifecycle, fatal);
  if (fatal) CHECK(calls->load() == 1);
  CHECK(worker.batches_prefetched() == 0);
  CHECK(batch->to_read_only().get_current_tier() == cucascade::memory::Tier::HOST);
  CHECK(fixture.gpu_space->get_active_reservation_count() == 0);
}

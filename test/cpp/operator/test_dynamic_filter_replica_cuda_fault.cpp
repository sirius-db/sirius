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

#include "dynamic_filter_accumulation_test_utils.hpp"
#include "dynamic_filter_replica_test_utils.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator_test_utils.hpp"
#include "utils/cuda_status_fault.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime.h>

#include <absl/cleanup/cleanup.h>
#include <catch.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <memory/pool_peer_access.hpp>
#include <memory/pool_peer_access_detail.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

using sirius::test::replica::get_replica_spaces;
using sirius::test::replica::make_values;
using sirius::test::replica::mask_to_host;

namespace {

struct replica_allocation_observer {
  std::function<void(::cuda::stream_ref)> after;
  std::atomic<std::size_t> live_bytes{0};
};

class observed_replica_resource {
 public:
  observed_replica_resource(int device,
                            std::size_t capacity,
                            std::shared_ptr<replica_allocation_observer> fault)
    : upstream_{cucascade::memory::make_default_gpu_memory_resource(device, capacity)},
      fault_{std::move(fault)}
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return upstream_.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
  }
  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    auto* ptr = upstream_.allocate(stream, bytes, alignment);
    fault_->live_bytes.fetch_add(bytes);
    try {
      if (fault_->after) { fault_->after(stream); }
    } catch (...) {
      deallocate(stream, ptr, bytes, alignment);
      throw;
    }
    return ptr;
  }
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    fault_->live_bytes.fetch_sub(bytes);
  }
  friend void get_property(observed_replica_resource const&, ::cuda::mr::device_accessible) noexcept
  {
  }
  bool operator==(observed_replica_resource const& other) const noexcept
  {
    return fault_ == other.fault_;
  }

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> upstream_;
  std::shared_ptr<replica_allocation_observer> fault_;
};

}  // namespace

TEMPLATE_TEST_CASE("ordinary replica CUDA failures propagate after pending copies drain",
                   "[dynamic_filter][multi_gpu][replica_cuda_error][cuda_status_fault]",
                   sirius::op::sirius_dynamic_bloom_filter,
                   sirius::op::sirius_dynamic_in_list_filter,
                   sirius::op::sirius_dynamic_small_in_list_filter)
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count < 2) {
    WARN("test needs 2 GPUs, " << device_count << " visible; skipping");
    return;
  }
  bool const have_pending_copy = GENERATE(false, true);
  auto fault                   = std::make_shared<replica_allocation_observer>();
  auto manager                 = sirius::test::operator_utils::initialize_memory_manager(
    2, {.gpu_resource_factory = [fault](int device, std::size_t capacity) {
      if (device == 0) {
        return cucascade::memory::make_default_gpu_memory_resource(device, capacity);
      }
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        observed_replica_resource{device, capacity, fault}};
    }});
  auto spaces  = get_replica_spaces(*manager);
  auto& source = spaces.front().get_gpu_space();
  if (have_pending_copy && !sirius::memory::try_grant_pool_access(source, 1)) {
    WARN(
      "pending asynchronous replica case requires source-pool access; first-copy errors remain "
      "covered");
    return;
  }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  auto const stream = source.acquire_stream();
  auto keys   = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto filter = std::make_unique<TestType>(keys->view(), stream, source.get_default_allocator());
  stream.sync();
  keys.reset();
  std::string expected_diagnostic;
  try {
    (void)sirius::memory::detail::interpret_pool_access_result(
      cucascade::memory::pool_peer_access_status::CUDA_ERROR, cudaErrorInvalidResourceHandle, 0, 1);
  } catch (cucascade::cuda_error const& error) {
    expected_diagnostic = error.what();
  }
  REQUIRE_FALSE(expected_diagnostic.empty());

  std::atomic<std::shared_ptr<sirius::test::accumulation::stream_gate>> gate;
  cudaStream_t first_stream{};
  fault->after = [&](::cuda::stream_ref allocation_stream) {
    if (have_pending_copy && !first_stream) {
      first_stream = allocation_stream.get();
      gate.store(std::make_shared<sirius::test::accumulation::stream_gate>(allocation_stream));
    }
  };
  // A repeated target exercises two pending destinations with the two GPUs available in CI.
  // The first destination is not published until the drain, so both entries enqueue independently.
  if (have_pending_copy) { spaces.push_back(spaces.back()); }
  sirius::test::scoped_pool_access_fault query_fault{
    source.template get_memory_resource_of<cucascade::memory::Tier::GPU>()->pool_handle(),
    1,
    have_pending_copy ? 1U : 0U,
    cudaErrorInvalidResourceHandle};
  auto worker           = std::async(std::launch::async, [&] {
    try {
      filter->replicate_to_devices(spaces);
    } catch (...) {
      return std::current_exception();
    }
    return std::exception_ptr{};
  });
  absl::Cleanup release = [&] {
    if (auto pending_gate = gate.load()) { pending_gate->open(); }
    if (worker.valid()) { worker.wait(); }
  };
  sirius::test::accumulation::wait_until([&] { return query_fault.consumed(); });
  if (have_pending_copy) {
    REQUIRE(gate.load() != nullptr);
    REQUIRE(worker.wait_for(std::chrono::milliseconds{100}) == std::future_status::timeout);
    REQUIRE(fault->live_bytes.load() > 0);
    gate.load()->open();
  }
  auto const result = worker.get();
  INFO("replica call returned " << (result ? "an exception" : "normally"));
  REQUIRE(result != nullptr);
  try {
    std::rethrow_exception(result);
  } catch (cucascade::cuda_error const& error) {
    INFO("returned CUDA error: " << error.what());
    REQUIRE(typeid(error) == typeid(cucascade::cuda_error));
    REQUIRE(std::string{error.what()} == expected_diagnostic);
  } catch (std::exception const& error) {
    INFO("returned exception type: " << typeid(error).name() << "; message: " << error.what());
    FAIL("pool-grant CUDA error changed type during replica cleanup");
  }
  if (auto pending_gate = gate.load()) { REQUIRE_FALSE(pending_gate->timed_out()); }
  fault->after = {};
  REQUIRE(filter->replica_count() == (have_pending_copy ? 2 : 1));
  REQUIRE(spaces.back().get_gpu_space().get_total_reserved_memory() == 0);
  if (have_pending_copy) {
    rmm::cuda_set_device_raii target{rmm::cuda_device_id{1}};
    auto& target_space       = spaces.back().get_gpu_space();
    auto const target_stream = target_space.acquire_stream();
    auto probe =
      make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, target_stream);
    auto mask =
      filter->compute_mask(probe->view(), 1, target_stream, target_space.get_default_allocator());
    REQUIRE(mask != nullptr);
    REQUIRE(mask_to_host(mask->view(), target_stream) == std::vector<std::uint8_t>{1, 1, 1});
  }
  filter.reset();
  REQUIRE(fault->live_bytes.load() == 0);
}

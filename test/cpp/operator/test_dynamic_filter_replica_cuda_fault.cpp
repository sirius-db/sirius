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

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime.h>

#include <absl/cleanup/cleanup.h>
#include <catch.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
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
#include <optional>
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
  if (device_count < 2) { SKIP("replica fault requires two GPUs"); }
  bool const have_pending_copy = GENERATE(false, true);
  bool const fail_drain        = have_pending_copy && GENERATE(false, true);
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
  std::atomic<bool> drain_fault_consumed{false};
  auto worker           = std::async(std::launch::async, [&] {
    std::optional<sirius::test::scoped_cuda_status_fault> drain_fault;
    if (fail_drain) {
      drain_fault.emplace(sirius::test::scoped_cuda_status_fault::synchronize_stream,
                          cudaErrorUnknown);
    }
    try {
      filter->replicate_to_devices(spaces);
    } catch (...) {
      if (drain_fault) { drain_fault_consumed.store(drain_fault->consumed()); }
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
  REQUIRE(filter->replica_count() == 1);
  REQUIRE(drain_fault_consumed.load() == fail_drain);
  REQUIRE(spaces.back().get_gpu_space().get_total_reserved_memory() == 0);
  REQUIRE(fault->live_bytes.load() == 0);
  filter.reset();
  REQUIRE(fault->live_bytes.load() == 0);
}

TEMPLATE_TEST_CASE("replica completion errors drain every pending copy before propagation",
                   "[dynamic_filter][multi_gpu][replica_cuda_error][cuda_status_fault]",
                   sirius::op::sirius_dynamic_bloom_filter,
                   sirius::op::sirius_dynamic_in_list_filter,
                   sirius::op::sirius_dynamic_small_in_list_filter)
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count < 2) { SKIP("completion fault requires two GPUs"); }
  auto allocations = std::make_shared<replica_allocation_observer>();
  auto manager     = sirius::test::operator_utils::initialize_memory_manager(
    2, {.gpu_resource_factory = [allocations](int device, std::size_t capacity) {
      if (device == 0) {
        return cucascade::memory::make_default_gpu_memory_resource(device, capacity);
      }
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        observed_replica_resource{device, capacity, allocations}};
    }});
  auto spaces  = get_replica_spaces(*manager);
  auto& source = spaces.front().get_gpu_space();
  if (!sirius::memory::try_grant_pool_access(source, 1)) {
    SKIP("completion fault requires a direct copy so staging cannot consume it");
  }
  spaces.push_back(spaces.back());
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  auto const stream = source.acquire_stream();
  auto keys   = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto filter = std::make_unique<TestType>(keys->view(), stream, source.get_default_allocator());
  stream.sync();
  keys.reset();
  auto const* host_resource =
    spaces.front()
      .get_host_staging_space()
      .template get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  std::atomic<std::shared_ptr<sirius::test::accumulation::stream_gate>> second_gate;
  cudaStream_t first_stream{};
  allocations->after = [&](::cuda::stream_ref allocation_stream) {
    if (!first_stream) {
      first_stream = allocation_stream.get();
    } else if (allocation_stream.get() != first_stream && !second_gate.load()) {
      second_gate.store(
        std::make_shared<sirius::test::accumulation::stream_gate>(allocation_stream));
    }
  };
  sirius::test::cuda_transfer_observation observed;
  bool consumed         = false;
  auto worker           = std::async(std::launch::async, [&] {
    sirius::test::scoped_cuda_transfer_observer observer{*host_resource};
    sirius::test::scoped_cuda_status_fault fault{
      sirius::test::scoped_cuda_status_fault::synchronize_stream, cudaErrorInvalidValue};
    try {
      filter->replicate_to_devices(spaces);
    } catch (...) {
      consumed = fault.consumed();
      observed = observer.snapshot();
      return std::current_exception();
    }
    return std::exception_ptr{};
  });
  absl::Cleanup release = [&] {
    if (auto gate = second_gate.load()) { gate->open(); }
    if (worker.valid()) { worker.wait(); }
  };
  sirius::test::accumulation::wait_until([&] { return second_gate.load() != nullptr; });
  REQUIRE(worker.wait_for(std::chrono::milliseconds{100}) == std::future_status::timeout);
  REQUIRE(allocations->live_bytes.load() > 0);
  second_gate.load()->open();
  auto result = worker.get();
  REQUIRE(consumed);
  REQUIRE(observed.synchronizations == 2);
  REQUIRE(result != nullptr);
  REQUIRE_THROWS_AS(std::rethrow_exception(result), cucascade::cuda_error);
  REQUIRE(filter->replica_count() == 1);
  REQUIRE(allocations->live_bytes.load() == 0);
  REQUIRE_FALSE(second_gate.load()->timed_out());
}

TEST_CASE("zone-map replication skips allocation refusal but propagates CUDA failures",
          "[dynamic_filter][multi_gpu][replica_cuda_error][cuda_status_fault]")
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count < 2) { SKIP("zone-map replica fault requires two GPUs"); }
  int const failure_mode = GENERATE(0, 1, 2);
  auto allocations       = std::make_shared<replica_allocation_observer>();
  auto manager           = sirius::test::operator_utils::initialize_memory_manager(
    2, {.gpu_resource_factory = [allocations](int device, std::size_t capacity) {
      if (device == 0) {
        return cucascade::memory::make_default_gpu_memory_resource(device, capacity);
      }
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        observed_replica_resource{device, capacity, allocations}};
    }});
  auto spaces        = get_replica_spaces(*manager);
  auto const& source = spaces.front().get_gpu_space();
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  auto const stream = source.acquire_stream();
  std::vector<sirius::op::zone_map_entry> zones;
  zones.push_back({std::make_unique<cudf::numeric_scalar<std::int64_t>>(
                     2, true, stream, source.get_default_allocator()),
                   std::make_unique<cudf::numeric_scalar<std::int64_t>>(
                     6, true, stream, source.get_default_allocator())});
  sirius::op::sirius_dynamic_zone_map_filter filter{std::move(zones)};
  stream.sync();
  std::optional<sirius::test::scoped_cuda_status_fault> fault;
  std::size_t target_allocations = 0;
  allocations->after             = [&](::cuda::stream_ref target_stream) {
    ++target_allocations;
    if (failure_mode == 2 && target_allocations == 1) {
      fault.emplace(sirius::test::scoped_cuda_status_fault::synchronize_stream,
                    cudaErrorInvalidValue,
                    0,
                    target_stream.get());
    } else if (failure_mode != 2 && target_allocations == 3) {
      if (failure_mode == 0) { throw rmm::out_of_memory{"later zone scalar refused"}; }
      // libcudf constructors live in a shared library, so inject their error type through this
      // allocator callback after an earlier scalar queued work.
      throw cudf::cuda_error{"injected target scalar constructor: cudaErrorInvalidValue",
                             cudaErrorInvalidValue};
    }
  };
  if (failure_mode == 0) {
    REQUIRE_NOTHROW(filter.replicate_to_devices(spaces));
  } else {
    try {
      filter.replicate_to_devices(spaces);
      FAIL("zone-map CUDA failures must propagate");
    } catch (std::exception const& error) {
      REQUIRE((dynamic_cast<cucascade::cuda_error const*>(&error) != nullptr ||
               dynamic_cast<cudf::cuda_error const*>(&error) != nullptr ||
               dynamic_cast<rmm::cuda_error const*>(&error) != nullptr ||
               dynamic_cast<::cuda::cuda_error const*>(&error) != nullptr));
      REQUIRE(std::string{error.what()}.find("cudaErrorInvalidValue") != std::string::npos);
    }
    if (failure_mode == 2) {
      REQUIRE(fault.has_value());
      REQUIRE(fault->consumed());
    }
  }
  REQUIRE(filter.is_available_on_device(0));
  REQUIRE_FALSE(filter.is_available_on_device(1));
  REQUIRE(allocations->live_bytes.load() == 0);
  allocations->after = {};
  fault.reset();
  filter.replicate_to_devices(spaces);
  REQUIRE(filter.is_available_on_device(1));
}

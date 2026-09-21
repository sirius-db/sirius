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
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct ordering_test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  ordering_test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

ordering_test_env& env()
{
  static ordering_test_env e;
  return e;
}

constexpr std::size_t kRows      = 1 << 16;
constexpr std::int32_t kStale    = 0x2AAAAAAA;
constexpr std::int32_t kExpected = 0x1BBBBBBB;

struct delay_state {
  std::atomic<bool> release{false};
  std::atomic<bool> entered{false};

  ~delay_state() { release.store(true, std::memory_order_release); }
};

/// Host function that parks a stream until the test releases it.
void CUDART_CB block_until_released(void* userData)
{
  auto* state = static_cast<delay_state*>(userData);
  state->entered.store(true, std::memory_order_release);
  while (!state->release.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

bool wait_for_flag(std::atomic<bool> const& flag,
                   std::chrono::milliseconds timeout = std::chrono::seconds(5))
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

void throw_if_cuda_error(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
  }
}

/// Registers host memory and guarantees that a blocked callback is released before cleanup.
class registered_host_memory {
 public:
  registered_host_memory(void* data,
                         std::size_t bytes,
                         rmm::cuda_stream_view stream,
                         delay_state& gate)
    : _data(data), _stream(stream.value()), _gate(&gate)
  {
    throw_if_cuda_error(cudaHostRegister(_data, bytes, 0), "cudaHostRegister");
    _registered = true;
  }

  registered_host_memory(registered_host_memory const&)            = delete;
  registered_host_memory& operator=(registered_host_memory const&) = delete;

  ~registered_host_memory()
  {
    if (!_registered) { return; }
    _gate->release.store(true, std::memory_order_release);
    static_cast<void>(cudaStreamSynchronize(_stream));
    static_cast<void>(cudaHostUnregister(_data));
  }

  cudaError_t unregister()
  {
    if (!_registered) { return cudaSuccess; }
    auto const status = cudaHostUnregister(_data);
    if (status == cudaSuccess) { _registered = false; }
    return status;
  }

 private:
  void* _data;
  cudaStream_t _stream;
  delay_state* _gate;
  bool _registered{false};
};

/// Releases the CUDA callback gate before joining, including during test failure unwinding.
class gated_thread {
 public:
  template <typename Function>
  gated_thread(delay_state& gate, Function&& function)
    : _gate(&gate), _thread(std::forward<Function>(function))
  {
  }

  gated_thread(gated_thread const&)            = delete;
  gated_thread& operator=(gated_thread const&) = delete;

  ~gated_thread() { release_and_join(); }

  void release_and_join()
  {
    _gate->release.store(true, std::memory_order_release);
    if (_thread.joinable()) { _thread.join(); }
  }

 private:
  delay_state* _gate;
  std::jthread _thread;
};

/// Build a one-column INT32 batch on `stream`, filled with `value`, and settle it.
std::unique_ptr<cudf::column> make_settled_column(std::int32_t value, rmm::cuda_stream_view stream)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(kRows),
                                       cudf::mask_state::UNALLOCATED,
                                       stream);
  std::vector<std::int32_t> fill(kRows, value);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                          fill.data(),
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  return col;
}

std::shared_ptr<cucascade::data_batch> wrap_batch(std::unique_ptr<cudf::column> col,
                                                  rmm::cuda_stream_view writer_stream)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                 *env().gpu_space,
                                 writer_stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

/// Bring a (possibly spilled) batch back to the GPU and return column 0's bytes.
std::vector<std::int32_t> read_back(cucascade::data_batch& batch, rmm::cuda_stream_view stream)
{
  auto ro = batch.to_read_only();
  if (ro.get_memory_space()->get_tier() != cucascade::memory::Tier::GPU) {
    auto mut = cucascade::data_batch::readonly_to_mutable(std::move(ro));
    mut.convert_to<cucascade::gpu_table_representation>(
      sirius::converter_registry::get(), env().gpu_space, ::cuda::stream_ref{stream.value()});
    ro = cucascade::data_batch::mutable_to_readonly(std::move(mut));
  }
  auto view = ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  std::vector<std::int32_t> out(static_cast<std::size_t>(view.column(0).size()));
  REQUIRE(cudaMemcpyAsync(out.data(),
                          view.column(0).head<std::int32_t>(),
                          out.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  return out;
}

std::size_t count_not_expected(std::vector<std::int32_t> const& values)
{
  std::size_t n = 0;
  for (auto v : values) {
    if (v != kExpected) { ++n; }
  }
  return n;
}

/// Warm first-call costs on the conversion path so they do not eat into the race window.
void warm_conversion_path()
{
  rmm::cuda_stream stream;
  auto batch = wrap_batch(make_settled_column(kExpected, stream.view()), stream.view());
  sirius::convertible_data_batch wrapper(batch);
  REQUIRE(wrapper.convert({env().host_space}, stream.view(), *env().mgr, true).has_value());
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
}

}  // namespace

TEST_CASE("downgrade conversion orders after the producer's writer event",
          "[spill_converter_ordering]")
{
  warm_conversion_path();

  rmm::cuda_stream producer_stream;
  rmm::cuda_stream downgrade_stream;

  auto col = make_settled_column(kStale, producer_stream.view());

  delay_state gate;
  std::vector<std::int32_t> final_bytes(kRows, kExpected);
  registered_host_memory final_bytes_registration(
    final_bytes.data(), kRows * sizeof(std::int32_t), producer_stream.view(), gate);
  REQUIRE(cudaLaunchHostFunc(producer_stream.value(), block_until_released, &gate) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                          final_bytes.data(),
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice,
                          producer_stream.value()) == cudaSuccess);

  auto batch = wrap_batch(std::move(col), producer_stream.view());
  REQUIRE(wait_for_flag(gate.entered));

  sirius::convertible_data_batch wrapper(batch);
  std::optional<std::vector<std::size_t>> result;
  std::exception_ptr worker_error;
  std::atomic<bool> worker_started{false};
  std::atomic<bool> converted{false};
  gated_thread downgrader(gate, [&] {
    worker_started.store(true, std::memory_order_release);
    try {
      result = wrapper.convert({env().host_space}, downgrade_stream.view(), *env().mgr, true);
    } catch (...) {
      worker_error = std::current_exception();
    }
    converted.store(true, std::memory_order_release);
  });

  REQUIRE(wait_for_flag(worker_started));
  REQUIRE_FALSE(wait_for_flag(converted, std::chrono::milliseconds(100)));
  downgrader.release_and_join();
  if (worker_error) { std::rethrow_exception(worker_error); }
  REQUIRE(result.has_value());

  REQUIRE(cudaStreamSynchronize(producer_stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(downgrade_stream.value()) == cudaSuccess);
  REQUIRE(final_bytes_registration.unregister() == cudaSuccess);

  auto const out = read_back(*batch, downgrade_stream.view());
  REQUIRE(out.size() == kRows);
  auto const torn = count_not_expected(out);
  INFO("host image carries " << torn << " stale (torn) values of " << kRows);
  REQUIRE(torn == 0);
}

TEST_CASE("a recorded reader event holds off the downgrade until the read completes",
          "[spill_converter_ordering]")
{
  warm_conversion_path();

  rmm::cuda_stream reader_stream;
  rmm::cuda_stream downgrade_stream;

  auto batch =
    wrap_batch(make_settled_column(kExpected, reader_stream.view()), reader_stream.view());
  REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);

  delay_state gate;
  std::vector<std::int32_t> reader_out(kRows, 0);
  registered_host_memory reader_out_registration(
    reader_out.data(), kRows * sizeof(std::int32_t), reader_stream.view(), gate);
  {
    auto ro   = batch->to_read_only();
    auto view = ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
    REQUIRE(cudaLaunchHostFunc(reader_stream.value(), block_until_released, &gate) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(reader_out.data(),
                            view.column(0).head<std::int32_t>(),
                            kRows * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost,
                            reader_stream.value()) == cudaSuccess);
    ro.record_reader_event(::cuda::stream_ref{reader_stream.value()});
  }
  REQUIRE(wait_for_flag(gate.entered));
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);

  sirius::convertible_data_batch wrapper(batch);
  REQUIRE_FALSE(
    wrapper.convert({env().host_space}, downgrade_stream.view(), *env().mgr, false).has_value());
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  std::atomic<bool> converted{false};
  std::atomic<bool> worker_started{false};
  std::exception_ptr worker_error;
  gated_thread downgrader(gate, [&] {
    worker_started.store(true, std::memory_order_release);
    try {
      auto result = wrapper.convert({env().host_space}, downgrade_stream.view(), *env().mgr, true);
      if (!result.has_value()) { throw std::runtime_error("blocking downgrade did not convert"); }
      rmm::device_buffer poison(kRows * sizeof(std::int32_t), downgrade_stream.view());
      throw_if_cuda_error(
        cudaMemsetAsync(
          poison.data(), 0xEE, kRows * sizeof(std::int32_t), downgrade_stream.value()),
        "cudaMemsetAsync");
      throw_if_cuda_error(cudaStreamSynchronize(downgrade_stream.value()), "cudaStreamSynchronize");
    } catch (...) {
      worker_error = std::current_exception();
    }
    converted.store(true, std::memory_order_release);
  });

  REQUIRE(wait_for_flag(worker_started));
  REQUIRE_FALSE(wait_for_flag(converted, std::chrono::milliseconds(100)));
  downgrader.release_and_join();
  if (worker_error) { std::rethrow_exception(worker_error); }
  REQUIRE(converted.load(std::memory_order_acquire));
  REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);
  REQUIRE(reader_out_registration.unregister() == cudaSuccess);

  auto const scribbled = count_not_expected(reader_out);
  INFO("straggler reader observed " << scribbled << " scribbled values of " << kRows);
  REQUIRE(scribbled == 0);

  auto const out = read_back(*batch, downgrade_stream.view());
  REQUIRE(out.size() == kRows);
  auto const torn = count_not_expected(out);
  INFO("host image carries " << torn << " torn values of " << kRows);
  REQUIRE(torn == 0);
}

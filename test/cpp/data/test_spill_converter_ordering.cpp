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
#include <memory>
#include <thread>
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
};

/// Host function that parks a stream until the test releases it.
void CUDART_CB block_until_released(void* userData)
{
  auto* state = static_cast<delay_state*>(userData);
  while (!state->release.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

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
  REQUIRE(cudaHostRegister(final_bytes.data(), kRows * sizeof(std::int32_t), 0) == cudaSuccess);
  REQUIRE(cudaLaunchHostFunc(producer_stream.value(), block_until_released, &gate) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                          final_bytes.data(),
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice,
                          producer_stream.value()) == cudaSuccess);

  auto batch = wrap_batch(std::move(col), producer_stream.view());

  std::thread releaser([&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    gate.release.store(true, std::memory_order_release);
  });
  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({env().host_space}, downgrade_stream.view(), *env().mgr, true);
  releaser.join();
  REQUIRE(result.has_value());

  REQUIRE(cudaStreamSynchronize(producer_stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(downgrade_stream.value()) == cudaSuccess);
  REQUIRE(cudaHostUnregister(final_bytes.data()) == cudaSuccess);

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
  REQUIRE(cudaHostRegister(reader_out.data(), kRows * sizeof(std::int32_t), 0) == cudaSuccess);
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
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);

  sirius::convertible_data_batch wrapper(batch);
  REQUIRE_FALSE(
    wrapper.convert({env().host_space}, downgrade_stream.view(), *env().mgr, false).has_value());
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  std::atomic<bool> converted{false};
  std::thread downgrader([&] {
    auto result = wrapper.convert({env().host_space}, downgrade_stream.view(), *env().mgr, true);
    REQUIRE(result.has_value());
    rmm::device_buffer poison(kRows * sizeof(std::int32_t), downgrade_stream.view());
    REQUIRE(cudaMemsetAsync(
              poison.data(), 0xEE, kRows * sizeof(std::int32_t), downgrade_stream.value()) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(downgrade_stream.value()) == cudaSuccess);
    converted.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  CHECK_FALSE(converted.load(std::memory_order_acquire));

  gate.release.store(true, std::memory_order_release);
  downgrader.join();
  REQUIRE(converted.load(std::memory_order_acquire));
  REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);
  REQUIRE(cudaHostUnregister(reader_out.data()) == cudaSuccess);

  auto const scribbled = count_not_expected(reader_out);
  INFO("straggler reader observed " << scribbled << " scribbled values of " << kRows);
  REQUIRE(scribbled == 0);

  auto const out = read_back(*batch, downgrade_stream.view());
  REQUIRE(out.size() == kRows);
  auto const torn = count_not_expected(out);
  INFO("host image carries " << torn << " torn values of " << kRows);
  REQUIRE(torn == 0);
}

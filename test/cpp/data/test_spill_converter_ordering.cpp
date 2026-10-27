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
#include "data/spill_chunked_converters.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>
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
  sirius::test::operator_utils::ensure_converter_registry();
  static ordering_test_env e;
  return e;
}

constexpr std::size_t kRows      = 1 << 16;
constexpr std::int32_t kStale    = 0x2AAAAAAA;
constexpr std::int32_t kExpected = 0x1BBBBBBB;

struct delay_state {
  std::atomic<bool> release{false};
  std::atomic<bool> entered{false};
  std::atomic<bool> timed_out{false};

  ~delay_state() { release.store(true, std::memory_order_release); }
};

/// Host function that parks a stream until the test releases it.
void CUDART_CB block_until_released(void* userData)
{
  auto* state = static_cast<delay_state*>(userData);
  state->entered.store(true, std::memory_order_release);
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (!state->release.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      state->timed_out.store(true, std::memory_order_release);
      break;
    }
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
                         ::cuda::stream_ref stream,
                         delay_state& gate)
    : _data(data), _stream(stream.get()), _gate(&gate)
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
std::unique_ptr<cudf::column> make_settled_column(std::int32_t value, ::cuda::stream_ref stream)
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
                          stream.get()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  return col;
}

std::shared_ptr<cucascade::data_batch> wrap_batch(std::unique_ptr<cudf::column> col,
                                                  ::cuda::stream_ref writer_stream)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                 *env().gpu_space,
                                 writer_stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

/// Bring a (possibly spilled) batch back to the GPU and return column 0's bytes.
std::vector<std::int32_t> read_back(cucascade::data_batch& batch, ::cuda::stream_ref stream)
{
  {
    auto mut = batch.to_mutable();
    if (mut.get_memory_space()->get_tier() != cucascade::memory::Tier::GPU) {
      mut.convert_to<cucascade::gpu_table_representation>(
        sirius::converter_registry::get(), env().gpu_space, stream);
    }
  }
  auto ro   = batch.to_read_only();
  auto view = ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  std::vector<std::int32_t> out(static_cast<std::size_t>(view.column(0).size()));
  REQUIRE(cudaMemcpyAsync(out.data(),
                          view.column(0).head<std::int32_t>(),
                          out.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
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
  auto batch = wrap_batch(make_settled_column(kExpected, stream), stream);
  sirius::convertible_data_batch wrapper(batch);
  REQUIRE(wrapper.convert({env().host_space}, stream, *env().mgr, true).has_value());
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
}

}  // namespace

TEST_CASE("downgrade conversion orders after the producer's writer event",
          "[spill_converter_ordering]")
{
  warm_conversion_path();

  rmm::cuda_stream producer_stream;
  rmm::cuda_stream downgrade_stream;

  auto col = make_settled_column(kStale, producer_stream);

  delay_state gate;
  std::vector<std::int32_t> final_bytes(kRows, kExpected);
  registered_host_memory final_bytes_registration(
    final_bytes.data(), kRows * sizeof(std::int32_t), producer_stream, gate);
  REQUIRE(cudaLaunchHostFunc(producer_stream.value(), block_until_released, &gate) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                          final_bytes.data(),
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice,
                          producer_stream.value()) == cudaSuccess);

  auto batch = wrap_batch(std::move(col), producer_stream);
  REQUIRE(wait_for_flag(gate.entered));

  sirius::convertible_data_batch wrapper(batch);
  std::optional<std::vector<std::size_t>> result;
  std::exception_ptr worker_error;
  std::atomic<bool> worker_started{false};
  std::atomic<bool> converted{false};
  gated_thread downgrader(gate, [&] {
    worker_started.store(true, std::memory_order_release);
    try {
      result = wrapper.convert({env().host_space}, downgrade_stream, *env().mgr, true);
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

  auto const out = read_back(*batch, downgrade_stream);
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

  auto batch = wrap_batch(make_settled_column(kExpected, reader_stream), reader_stream);
  REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);

  delay_state gate;
  std::vector<std::int32_t> reader_out(kRows, 0);
  registered_host_memory reader_out_registration(
    reader_out.data(), kRows * sizeof(std::int32_t), reader_stream, gate);
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
    wrapper.convert({env().host_space}, downgrade_stream, *env().mgr, false).has_value());
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  std::atomic<bool> converted{false};
  std::atomic<bool> worker_started{false};
  std::exception_ptr worker_error;
  gated_thread downgrader(gate, [&] {
    worker_started.store(true, std::memory_order_release);
    try {
      auto result = wrapper.convert({env().host_space}, downgrade_stream, *env().mgr, true);
      if (!result.has_value()) { throw std::runtime_error("blocking downgrade did not convert"); }
      rmm::device_buffer poison(kRows * sizeof(std::int32_t), downgrade_stream);
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

  auto const out = read_back(*batch, downgrade_stream);
  REQUIRE(out.size() == kRows);
  auto const torn = count_not_expected(out);
  INFO("host image carries " << torn << " torn values of " << kRows);
  REQUIRE(torn == 0);
}

namespace {

std::unique_ptr<cudf::column> make_ordering_strings(::cuda::stream_ref stream)
{
  std::vector<std::int32_t> offsets{0, 4, 9, 9, 13, 19};
  std::string const chars{"skipalphaBETAsuffix"};
  auto offset_column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, offsets.size(), cudf::mask_state::UNALLOCATED, stream);
  REQUIRE(cudaMemcpyAsync(offset_column->mutable_view().head<void>(),
                          offsets.data(),
                          offsets.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  rmm::device_buffer data(chars.data(), chars.size(), stream);
  auto mask                      = cudf::create_null_mask(5, cudf::mask_state::ALL_VALID, stream);
  cudf::bitmask_type const valid = 0b11011;
  REQUIRE(
    cudaMemcpyAsync(mask.data(), &valid, sizeof(valid), cudaMemcpyHostToDevice, stream.get()) ==
    cudaSuccess);
  stream.sync();
  return cudf::make_strings_column(
    5, std::move(offset_column), std::move(data), 1, std::move(mask));
}

void require_column_bytes(cudf::column_view expected,
                          cudf::column_view actual,
                          ::cuda::stream_ref stream)
{
  REQUIRE(actual.type() == expected.type());
  REQUIRE(actual.size() == expected.size());
  REQUIRE(actual.null_count() == expected.null_count());
  auto compare = [&](void const* a, void const* b, std::size_t bytes) {
    std::vector<std::byte> first(bytes), second(bytes);
    REQUIRE(cudaMemcpyAsync(first.data(), a, bytes, cudaMemcpyDeviceToHost, stream.get()) ==
            cudaSuccess);
    REQUIRE(cudaMemcpyAsync(second.data(), b, bytes, cudaMemcpyDeviceToHost, stream.get()) ==
            cudaSuccess);
    stream.sync();
    REQUIRE(first == second);
  };
  if (expected.nullable()) {
    cudf::bitmask_type first{}, second{};
    REQUIRE(cudaMemcpyAsync(
              &first, expected.null_mask(), sizeof(first), cudaMemcpyDeviceToHost, stream.get()) ==
            cudaSuccess);
    REQUIRE(cudaMemcpyAsync(
              &second, actual.null_mask(), sizeof(second), cudaMemcpyDeviceToHost, stream.get()) ==
            cudaSuccess);
    stream.sync();
    auto const used_bits = (cudf::bitmask_type{1} << expected.size()) - 1;
    REQUIRE((first & used_bits) == (second & used_bits));
  }
  if (expected.type().id() == cudf::type_id::STRING) {
    auto const chars = cudf::strings_column_view(expected).chars_size(stream);
    REQUIRE(cudf::strings_column_view(actual).chars_size(stream) == chars);
    compare(expected.head<char>(), actual.head<char>(), chars);
    REQUIRE(actual.num_children() == 1);
    REQUIRE(expected.num_children() == 1);
    // The host-to-GPU converter promotes string offsets to INT64; compare their exact values.
    auto offsets = cudf::cast(expected.child(0), actual.child(0).type(), stream);
    require_column_bytes(offsets->view(), actual.child(0), stream);
    return;
  } else if (expected.type().id() != cudf::type_id::STRUCT) {
    compare(
      expected.head<void>(), actual.head<void>(), expected.size() * cudf::size_of(expected.type()));
  }
  REQUIRE(actual.num_children() == expected.num_children());
  for (cudf::size_type i = 0; i < expected.num_children(); ++i) {
    require_column_bytes(expected.child(i), actual.child(i), stream);
  }
}

void enqueue_column_overwrite(cudf::column_view original,
                              cudf::mutable_column_view target,
                              bool clear,
                              ::cuda::stream_ref stream)
{
  auto copy = [&](void* dst, void const* src, std::size_t bytes) {
    if (clear) {
      REQUIRE(cudaMemsetAsync(dst, 0, bytes, stream.get()) == cudaSuccess);
    } else {
      REQUIRE(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream.get()) ==
              cudaSuccess);
    }
  };
  if (original.nullable()) {
    copy(target.null_mask(),
         original.null_mask(),
         cudf::bitmask_allocation_size_bytes(original.size()));
  }
  if (original.type().id() == cudf::type_id::STRING) {
    // Geometry is already settled on the independent expected column.
    copy(target.head<char>(),
         original.head<char>(),
         cudf::strings_column_view(original).chars_size(cudf::get_default_stream()));
  } else if (original.type().id() != cudf::type_id::STRUCT) {
    copy(
      target.head<void>(), original.head<void>(), original.size() * cudf::size_of(original.type()));
  }
  for (cudf::size_type i = 0; i < original.num_children(); ++i) {
    enqueue_column_overwrite(original.child(i), target.child(i), clear, stream);
  }
}

}  // namespace

TEST_CASE("direct chunked spill waits before reading string geometry and null masks",
          "[spill_converter_ordering][cucascade_api_migration]")
{
  auto& setup = env();
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  sirius::spill::register_chunked_spill_converters(registry, 7);

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_ordering_strings(producer));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_ordering_strings(producer));
  columns.push_back(
    cudf::make_structs_column(5, std::move(children), 0, rmm::device_buffer{}, producer));
  cudf::table full{std::move(columns)};
  bool const sliced = GENERATE(false, true);
  // The converter requires zero offsets; exercise the canonicalized sliced scan input too.
  auto const view = sliced ? cudf::slice(full.view(), {1, 4}, producer).front() : full.view();
  cudf::table expected{view, producer};
  auto table = std::make_unique<cudf::table>(expected.view(), producer);
  producer.synchronize();
  for (cudf::size_type i = 0; i < expected.num_columns(); ++i) {
    enqueue_column_overwrite(
      expected.view().column(i), table->mutable_view().column(i), true, producer);
  }
  producer.synchronize();

  auto writable = table->mutable_view();
  auto source   = cucascade::gpu_table_representation::make_written_on(
    std::move(table), *setup.gpu_space, producer);
  delay_state gate;
  absl::Cleanup release_producer = [&] {
    gate.release.store(true, std::memory_order_release);
    producer.synchronize_no_throw();
  };
  REQUIRE(cudaLaunchHostFunc(producer.value(), block_until_released, &gate) == cudaSuccess);
  for (cudf::size_type i = 0; i < expected.num_columns(); ++i) {
    enqueue_column_overwrite(expected.view().column(i), writable.column(i), false, producer);
  }
  source->record_writer_event(producer);
  REQUIRE(wait_for_flag(gate.entered));
  std::unique_ptr<cucascade::host_data_representation> host;
  std::atomic<bool> started{false}, finished{false};
  std::exception_ptr failure;
  gated_thread worker{gate, [&] {
                        rmm::cuda_set_device_raii worker_device{rmm::cuda_device_id{0}};
                        started.store(true, std::memory_order_release);
                        try {
                          auto const& immutable = *source;
                          host = registry.convert<cucascade::host_data_representation>(
                            immutable, setup.host_space, reader);
                        } catch (...) {
                          failure = std::current_exception();
                        }
                        finished.store(true, std::memory_order_release);
                      }};
  REQUIRE(wait_for_flag(started));
  REQUIRE_FALSE(wait_for_flag(finished, std::chrono::milliseconds{100}));
  worker.release_and_join();
  REQUIRE_FALSE(gate.timed_out.load());
  if (failure) { std::rethrow_exception(failure); }
  REQUIRE(host != nullptr);
  REQUIRE(host->get_host_table()->data_size > 0);
  auto restored =
    registry.convert<cucascade::gpu_table_representation>(*host, setup.gpu_space, reader);
  for (cudf::size_type i = 0; i < expected.num_columns(); ++i) {
    require_column_bytes(expected.view().column(i), restored->get_table_view().column(i), reader);
  }
}

TEST_CASE("batch helpers record real default and nonblocking writers",
          "[spill_converter_ordering][cucascade_api_migration]")
{
  auto& setup = env();
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  rmm::cuda_stream nonblocking{rmm::cuda_stream::flags::non_blocking};
  bool const use_default = GENERATE(false, true);
  auto const writer      = ::cuda::stream_ref{use_default ? cudaStream_t{} : nonblocking.value()};
  auto const mode        = GENERATE(0, 1, 2);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_settled_column(kExpected, writer));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  std::shared_ptr<cucascade::data_batch> batch;
  if (mode == 0) {
    batch = sirius::make_data_batch(std::move(table), *setup.gpu_space, writer, {});
  } else if (mode == 1) {
    batch = sirius::make_data_batch(std::move(*table), *setup.gpu_space, writer, {});
  } else {
    std::shared_ptr<cudf::table> owner{std::move(table)};
    auto const view  = owner->view();
    auto const bytes = owner->alloc_size();
    batch            = sirius::make_data_batch_from_view(
      view, std::move(owner), bytes, *setup.gpu_space, writer, {});
  }
  auto source                = batch->to_read_only();
  auto const& representation = source.get_data()->cast<cucascade::gpu_table_representation>();
  REQUIRE(representation.get_writer_event() != nullptr);
  auto cloned = representation.clone(nonblocking);
  nonblocking.synchronize();
  auto const& gpu_clone = cloned->cast<cucascade::gpu_table_representation>();
  REQUIRE(gpu_clone.get_writer_event() != nullptr);
  REQUIRE(gpu_clone.get_table_view().num_rows() == kRows);
  std::vector<std::int32_t> values(kRows);
  REQUIRE(cudaMemcpyAsync(values.data(),
                          gpu_clone.get_table_view().column(0).head<void>(),
                          values.size() * sizeof(values[0]),
                          cudaMemcpyDeviceToHost,
                          nonblocking.value()) == cudaSuccess);
  nonblocking.synchronize();
  REQUIRE(count_not_expected(values) == 0);
}

TEST_CASE("unknown pinned views keep owners until release copies complete",
          "[spill_converter_ordering][cucascade_api_migration]")
{
  auto& setup = env();
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  rmm::cuda_stream writer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_settled_column(kExpected, writer));
  auto owner                      = std::make_shared<cudf::table>(std::move(columns));
  std::weak_ptr<cudf::table> weak = owner;
  auto const view                 = owner->view();
  auto const bytes                = owner->alloc_size();
  cucascade::gpu_table_representation source{
    view, std::move(owner), bytes, *setup.gpu_space, ::cuda::stream_ref{cudaStream_t{}}};
  REQUIRE(source.get_writer_event() == nullptr);
  delay_state gate;
  absl::Cleanup release = [&] {
    gate.release.store(true, std::memory_order_release);
    writer.synchronize_no_throw();
  };
  REQUIRE(cudaLaunchHostFunc(writer.value(), block_until_released, &gate) == cudaSuccess);
  std::unique_ptr<cudf::table> copy;
  std::exception_ptr failure;
  std::atomic<bool> started{false}, finished{false};
  gated_thread worker{gate, [&] {
                        rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{0}};
                        started.store(true, std::memory_order_release);
                        try {
                          copy = source.release_table(reader);
                        } catch (...) {
                          failure = std::current_exception();
                        }
                        finished.store(true, std::memory_order_release);
                      }};
  REQUIRE(wait_for_flag(started));
  REQUIRE_FALSE(wait_for_flag(finished, std::chrono::milliseconds{100}));
  REQUIRE_FALSE(weak.expired());
  worker.release_and_join();
  if (failure) { std::rethrow_exception(failure); }
  REQUIRE_FALSE(gate.timed_out.load());
  REQUIRE(weak.expired());
  REQUIRE(source.release_table(reader) == nullptr);
  std::vector<std::int32_t> values(kRows);
  REQUIRE(cudaMemcpyAsync(values.data(),
                          copy->view().column(0).head<void>(),
                          values.size() * sizeof(values[0]),
                          cudaMemcpyDeviceToHost,
                          reader.value()) == cudaSuccess);
  reader.synchronize();
  REQUIRE(count_not_expected(values) == 0);
}

TEST_CASE("batch helpers reject writers on another device including null streams",
          "[cucascade_api_migration][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto& setup = env();
  rmm::cuda_set_device_raii wrong_device{rmm::cuda_device_id{1}};
  rmm::cuda_stream other{rmm::cuda_stream::flags::non_blocking};
  bool const use_default = GENERATE(false, true);
  auto const stream      = ::cuda::stream_ref{use_default ? cudaStream_t{} : other.value()};
  REQUIRE_THROWS_AS(
    sirius::make_data_batch(std::make_unique<cudf::table>(), *setup.gpu_space, stream, {}),
    cucascade::logic_error);
}

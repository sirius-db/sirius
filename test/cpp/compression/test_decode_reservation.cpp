// SPDX-License-Identifier: Apache-2.0

#include "compression/compressed_scan.hpp"

#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <api/simpatico_codegen.hpp>
#include <catch.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

// The same lifetime ordering as a pipeline task: submitted work and result
// destruction finish while its calling-thread reservation remains attached.
struct attached_decode_reservation {
  cucascade::memory::reservation_aware_resource_adaptor& allocator;
  simpatico::stream_pool& streams;
  ::cuda::stream_ref attachment_stream;

  ~attached_decode_reservation()
  {
    streams.sync_all();
    allocator.reset_stream_reservation(attachment_stream);
  }
};

}  // namespace

TEST_CASE("Simpatico decode preserves reservation OOM and releases partial outputs",
          "[compression][decode][reservation]")
{
  using namespace cucascade::memory;
  constexpr std::size_t column_bytes = 1U << 20;
  constexpr cudf::size_type rows     = column_bytes / sizeof(std::int32_t);

  // Inputs belong to a separate resource; only decode allocations consume the
  // deliberately small engine budget. Three outputs cannot fit; two can.
  rmm::mr::cuda_async_memory_resource input_resource{8U << 20};
  rmm::cuda_stream input_stream;
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  for (int column = 0; column < 3; ++column) {
    auto values = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            rows,
                                            cudf::mask_state::UNALLOCATED,
                                            input_stream.view(),
                                            input_resource);
    REQUIRE(cudaMemsetAsync(values->mutable_view().head<std::int32_t>(),
                            column + 1,
                            column_bytes,
                            input_stream.value()) == cudaSuccess);
    input_columns.push_back(std::move(values));
  }
  cudf::table input{std::move(input_columns)};
  input_stream.synchronize();
  auto compressed = simpatico::compress_with_plan(input.view(),
                                                  "input -> identity\n---\n"
                                                  "input -> identity\n---\n"
                                                  "input -> identity\n",
                                                  input_stream.view(),
                                                  input_resource);

  int device = -1;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  gpu_memory_space_config config;
  config.device_id              = device;
  config.memory_capacity        = 2 * column_bytes + column_bytes / 2;
  config.per_stream_reservation = false;
  memory_space space{config};
  auto* allocator = space.get_memory_resource_as<reservation_aware_resource_adaptor>();
  REQUIRE(allocator != nullptr);
  simpatico::stream_pool streams;
  REQUIRE(streams.init(3));
  auto const attachment_stream = ::cuda::stream_ref{streams.streams.front()};
  auto reservation             = space.make_reservation(column_bytes);
  REQUIRE(reservation != nullptr);
  REQUIRE(
    allocator->attach_reservation_to_tracker(attachment_stream,
                                             std::move(reservation),
                                             std::make_unique<ignore_reservation_limit_policy>(),
                                             std::make_unique<throw_on_oom_policy>()));
  attached_decode_reservation attachment{*allocator, streams, attachment_stream};

  bool observed_engine_oom = false;
  try {
    auto result = simpatico::decompress(compressed, streams, space.get_default_allocator());
    FAIL("three decoded columns unexpectedly fit the two-and-a-half-column budget");
  } catch (cucascade_out_of_memory const& error) {
    observed_engine_oom = true;
    CHECK(static_cast<int>(error.error_kind) == static_cast<int>(MemoryError::LIMIT_EXCEEDED));
    CHECK(error.requested_bytes == column_bytes);
  }
  REQUIRE(observed_engine_oom);
  // Check before an extra synchronization: failure cleanup must destroy all
  // partial decode owners on this thread, while the reservation is attached.
  CHECK(allocator->get_allocated_bytes(attachment_stream) == 0);
  CHECK(allocator->is_stream_tracked(attachment_stream));

  std::array<std::size_t, 2> selected{2, 0};
  auto result = simpatico::decompress(
    compressed, std::span<std::size_t const>{selected}, streams, space.get_default_allocator());
  REQUIRE(result != nullptr);
  REQUIRE(result->num_columns() == 2);
  REQUIRE(result->num_rows() == rows);
  CHECK(allocator->get_allocated_bytes(attachment_stream) == 2 * column_bytes);

  // Read on an unrelated stream without a producer event or extra pool wait:
  // the public decode boundary must already have completed its output writes.
  std::vector<std::int32_t> host(rows);
  for (int column = 0; column < 2; ++column) {
    REQUIRE(cudaMemcpyAsync(host.data(),
                            result->view().column(column).head<std::int32_t>(),
                            column_bytes,
                            cudaMemcpyDeviceToHost,
                            input_stream.value()) == cudaSuccess);
    input_stream.synchronize();
    auto const byte     = static_cast<std::int32_t>(selected[column] + 1);
    auto const expected = byte * 0x01010101;
    bool equal          = true;
    for (auto value : host)
      equal = equal && value == expected;
    CHECK(equal);
  }
  result.reset();
  CHECK(allocator->get_allocated_bytes(attachment_stream) == 0);
  CHECK(streams.sync_all() == cudaSuccess);
  input_stream.synchronize();
}

TEST_CASE("Simpatico decode charges one request's temporaries beyond earlier outputs",
          "[compression][decode][reservation]")
{
  using namespace cucascade::memory;
  constexpr cudf::size_type rows     = 1 << 20;
  constexpr std::size_t column_bytes = rows * sizeof(std::int32_t);
  constexpr std::size_t columns      = 4;

  // Each column decodes its ANS-coded differences into a full-width intermediate column before the
  // delta launch reads it, so every request has temporaries about the size of its output.
  rmm::mr::cuda_async_memory_resource input_resource{64U << 20};
  rmm::cuda_stream input_stream;
  std::vector<std::vector<std::int32_t>> expected(columns, std::vector<std::int32_t>(rows));
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  std::string plan;
  for (std::size_t column = 0; column < columns; ++column) {
    for (cudf::size_type row = 0; row < rows; ++row)
      expected[column][row] =
        static_cast<std::int32_t>((static_cast<std::uint32_t>(row) + column) * 2654435761U >> 4);
    auto values = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            rows,
                                            cudf::mask_state::UNALLOCATED,
                                            input_stream.view(),
                                            input_resource);
    REQUIRE(cudaMemcpyAsync(values->mutable_view().head<std::int32_t>(),
                            expected[column].data(),
                            column_bytes,
                            cudaMemcpyHostToDevice,
                            input_stream.value()) == cudaSuccess);
    input_columns.push_back(std::move(values));
    plan += std::string{column ? "---\n" : ""} +
            "input -> delta -> differences\ndelta.differences -> ans\n";
  }
  cudf::table input{std::move(input_columns)};
  input_stream.synchronize();
  auto compressed =
    simpatico::compress_with_plan(input.view(), plan, input_stream.view(), input_resource);

  int device = -1;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  gpu_memory_space_config config;
  config.device_id              = device;
  config.memory_capacity        = 16 * columns * column_bytes;
  config.per_stream_reservation = false;
  memory_space space{config};
  auto* allocator = space.get_memory_resource_as<reservation_aware_resource_adaptor>();
  REQUIRE(allocator != nullptr);
  simpatico::stream_pool streams;
  REQUIRE(streams.init(columns));
  auto const attachment_stream = ::cuda::stream_ref{streams.streams.front()};
  auto reservation             = space.make_reservation(column_bytes);
  REQUIRE(reservation != nullptr);
  REQUIRE(
    allocator->attach_reservation_to_tracker(attachment_stream,
                                             std::move(reservation),
                                             std::make_unique<ignore_reservation_limit_policy>(),
                                             std::make_unique<throw_on_oom_policy>()));
  attached_decode_reservation attachment{*allocator, streams, attachment_stream};
  auto const decode = [&](std::span<std::size_t const> selected) {
    return simpatico::decompress(compressed, selected, streams, space.get_default_allocator());
  };

  // Decoded alone, each column's charge peaks at its output plus its temporaries.
  std::size_t outputs     = 0;
  std::size_t temporaries = 0;
  for (std::size_t column = 0; column < columns; ++column) {
    REQUIRE(allocator->get_allocated_bytes(attachment_stream) == 0);
    allocator->reset_peak_allocated_bytes(attachment_stream);
    std::array<std::size_t, 1> const one{column};
    auto result       = decode(one);
    auto const output = allocator->get_allocated_bytes(attachment_stream);
    auto const peak   = allocator->get_peak_allocated_bytes(attachment_stream);
    REQUIRE(peak > output);
    outputs += output;
    temporaries = std::max(temporaries, peak - output);
  }

  // Together, the charge never exceeds the earlier outputs plus one request's temporaries: each
  // request releases its temporaries before the next one is submitted.
  allocator->reset_peak_allocated_bytes(attachment_stream);
  std::array<std::size_t, columns> selected{};
  for (std::size_t column = 0; column < columns; ++column)
    selected[column] = column;
  auto result = decode(selected);
  REQUIRE(result != nullptr);
  REQUIRE(result->num_columns() == static_cast<cudf::size_type>(columns));
  CHECK(allocator->get_peak_allocated_bytes(attachment_stream) <= outputs + temporaries);
  CHECK(allocator->get_allocated_bytes(attachment_stream) == outputs);

  std::vector<std::int32_t> host(rows);
  for (std::size_t column = 0; column < columns; ++column) {
    REQUIRE(cudaMemcpyAsync(host.data(),
                            result->view().column(column).head<std::int32_t>(),
                            column_bytes,
                            cudaMemcpyDeviceToHost,
                            input_stream.value()) == cudaSuccess);
    input_stream.synchronize();
    CHECK(host == expected[column]);
  }
  result.reset();
  CHECK(allocator->get_allocated_bytes(attachment_stream) == 0);
  CHECK(streams.sync_all() == cudaSuccess);
  input_stream.synchronize();
}

TEST_CASE("Compressed facade admits private lanes against the caller and frees allocation origins",
          "[compression][decode][reservation][allocation_origin]")
{
  using namespace cucascade::memory;
  auto per_stream                = GENERATE(false, true);
  constexpr cudf::size_type rows = 1024;
  constexpr std::size_t bytes    = rows * sizeof(std::int32_t);
  rmm::mr::cuda_async_memory_resource input_resource{8U << 20};
  rmm::cuda_stream task_stream;
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                              rows,
                                              cudf::mask_state::UNALLOCATED,
                                              task_stream.view(),
                                              input_resource));
  REQUIRE(cudaMemsetAsync(
            columns.front()->mutable_view().head<std::int32_t>(), 7, bytes, task_stream.value()) ==
          cudaSuccess);
  cudf::table input{std::move(columns)};
  task_stream.synchronize();
  auto compressed = simpatico::compress_with_plan(
    input.view(), "input -> identity\n", task_stream.view(), input_resource);
  int device = -1;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  gpu_memory_space_config config;
  config.device_id              = device;
  config.memory_capacity        = 1U << 20;
  config.per_stream_reservation = per_stream;
  memory_space space{config};
  auto* allocator = space.get_memory_resource_as<reservation_aware_resource_adaptor>();
  REQUIRE(allocator != nullptr);
  auto& lanes = simpatico::thread_device_stream_pool(4);
  REQUIRE(lanes.streams.size() >= 4);
  auto lane = ::cuda::stream_ref{lanes.streams.front()};
  if (per_stream) {
    REQUIRE(allocator->attach_reservation_to_tracker(lane, space.make_reservation(4 * bytes)));
  }
  REQUIRE(
    allocator->attach_reservation_to_tracker(task_stream.view(),
                                             space.make_reservation(256),
                                             std::make_unique<fail_reservation_limit_policy>()));
  std::array<std::size_t, 1> const selected{0};
  CHECK_THROWS_AS(
    sirius::decompress_chunk(compressed,
                             selected,
                             nullptr,
                             {},
                             space,
                             task_stream.view(),
                             *space.get_memory_resource_of<cucascade::memory::Tier::GPU>()),
    rmm::out_of_memory);
  CHECK(allocator->get_allocated_bytes(task_stream.view()) == 0);
  if (per_stream) { CHECK(allocator->get_allocated_bytes(lane) == 0); }
  allocator->reset_stream_reservation(task_stream.view());
  REQUIRE(
    allocator->attach_reservation_to_tracker(task_stream.view(),
                                             space.make_reservation(bytes),
                                             std::make_unique<fail_reservation_limit_policy>()));
  auto decoded =
    sirius::decompress_chunk(compressed,
                             selected,
                             nullptr,
                             {},
                             space,
                             task_stream.view(),
                             *space.get_memory_resource_of<cucascade::memory::Tier::GPU>());
  REQUIRE(decoded.table != nullptr);
  CHECK(allocator->get_allocated_bytes(task_stream.view()) == bytes);
  CHECK(allocator->get_peak_allocated_bytes(task_stream.view()) == bytes);
  if (per_stream) { CHECK(allocator->get_allocated_bytes(lane) == 0); }
  std::vector<std::int32_t> expected(rows);
  REQUIRE(cudaMemcpyAsync(expected.data(),
                          decoded.table->view().column(0).head<std::int32_t>(),
                          bytes,
                          cudaMemcpyDeviceToHost,
                          task_stream.value()) == cudaSuccess);
  task_stream.synchronize();
  CHECK(
    std::all_of(expected.begin(), expected.end(), [](auto value) { return value == 0x07070707; }));

  // Match converter teardown ordering: the decoded buffers survive their original reservation.
  auto output_columns = decoded.table->release();
  auto contents       = output_columns.front()->release();
  contents.data->set_stream(task_stream.view());
  rmm::device_buffer null_mask =
    contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{};
  null_mask.set_stream(task_stream.view());
  output_columns.front() = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                          rows,
                                                          std::move(*contents.data),
                                                          std::move(null_mask),
                                                          0);
  decoded.table          = std::make_unique<cudf::table>(std::move(output_columns));
  allocator->reset_stream_reservation(task_stream.view());
  REQUIRE(allocator->attach_reservation_to_tracker(task_stream.view(),
                                                   space.make_reservation(2 * bytes)));
  std::jthread free_thread([&] {
    cudaSetDevice(device);
    decoded.table.reset();
  });
  free_thread.join();
  task_stream.synchronize();
  CHECK(allocator->get_allocated_bytes(task_stream.view()) == 0);
  if (per_stream) { allocator->reset_stream_reservation(lane); }
  allocator->reset_stream_reservation(task_stream.view());
  CHECK(allocator->get_total_allocated_bytes() == 0);
  CHECK(allocator->get_total_reserved_bytes() == 0);
  CHECK(allocator->get_active_reservation_count() == 0);
}

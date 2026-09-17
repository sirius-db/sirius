// SPDX-License-Identifier: Apache-2.0

#include "api/simpatico_codegen.hpp"

#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <catch.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace {

// The same lifetime ordering as a pipeline task: submitted work and result
// destruction finish while its calling-thread reservation remains attached.
struct attached_decode_reservation {
  cucascade::memory::reservation_aware_resource_adaptor& allocator;
  simpatico::stream_pool& streams;
  rmm::cuda_stream_view attachment_stream;

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
  auto const attachment_stream = rmm::cuda_stream_view{streams.streams.front()};
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

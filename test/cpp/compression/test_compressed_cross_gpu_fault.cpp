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

// A compressed GPU-tier chunk decoded on a GPU other than the one holding it is copied across
// and decoded on the target stream, which the converter then waits for. A CUDA status that
// leaves the GPU unusable must reach the executor in a form its runtime-health classifier
// recognizes, from that wait and from the wait on the failure path. Injected statuses do not
// poison the device, so these tests run in the isolated fault executable.

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "catch.hpp"
#include "compression/compressed_representation.hpp"
#include "compression/device_compressed_blob.hpp"
#include "cuda/device_health.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "utils/cuda_status_fault.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using fault = sirius::test::scoped_cuda_status_fault;

constexpr cudf::size_type kNumRows = 4096;

struct cross_gpu_decode_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* owner  = nullptr;
  cucascade::memory::memory_space* target = nullptr;

  bool setup()
  {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) { return false; }
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(2)
      .set_gpu_usage_limit(256ULL << 20)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1ULL << 30)
      .use_numa_id_as_host_id()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_numa_region(0.75);
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    auto const gpus = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    REQUIRE(gpus.size() >= 2);
    owner  = const_cast<cucascade::memory::memory_space*>(gpus[0]);
    target = const_cast<cucascade::memory::memory_space*>(gpus[1]);
    return true;
  }

  ~cross_gpu_decode_fixture()
  {
    if (manager) {
      manager->shutdown();
      sirius::converter_registry::shutdown();
    }
  }
};

/// A compressed chunk and the stream its buffers were allocated on. Those buffers free themselves
/// on that stream, so it must outlive them: `chunk` is declared last, so it is destroyed first.
struct owned_chunk {
  rmm::cuda_stream stream;
  std::unique_ptr<sirius::compressed_device_representation> chunk;
};

/// One bitpacked INT64 column compressed on @p space with its header kept, as pin_table keeps a
/// GPU-tier chunk. Without @p with_slices the header names payload buffers the chunk cannot
/// locate, so a cross-GPU decode fails at its first copy, before anything is copied.
owned_chunk make_compressed_chunk(cucascade::memory::memory_space& space, bool with_slices)
{
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{space.get_device_id()}};
  owned_chunk result;
  auto const& stream = result.stream;
  ::cuda::stream_ref const stream_ref{stream.value()};
  std::vector<std::int64_t> values(kNumRows);
  std::iota(values.begin(), values.end(), std::int64_t{1000});
  rmm::device_buffer data(
    values.data(), values.size() * sizeof(std::int64_t), stream, space.get_default_allocator());
  cudf::column column(
    cudf::data_type{cudf::type_id::INT64}, kNumRows, std::move(data), rmm::device_buffer{}, 0);

  auto blob   = std::make_shared<sirius::compressed_device_blob>();
  blob->table = simpatico::compress_with_plan(
    cudf::table_view{std::vector<cudf::column_view>{column.view()}},
    "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n",
    stream_ref,
    space.get_default_allocator());
  stream.synchronize();

  std::uint64_t payload_bytes = 0;
  std::vector<simpatico::payload_buffer_ref> buffers;
  REQUIRE(simpatico::build_compressed_table_header(
            blob->table, blob->header, buffers, payload_bytes, stream_ref)
            .empty());
  stream.synchronize();
  REQUIRE_FALSE(blob->header.empty());
  if (with_slices) {
    for (auto const& buffer : buffers) {
      if (buffer.size_bytes == 0) { continue; }
      blob->slices.push_back(sirius::compressed_payload_slice{
        buffer.offset, static_cast<std::byte const*>(buffer.device_ptr), buffer.size_bytes});
    }
  }
  result.chunk = std::make_unique<sirius::compressed_device_representation>(
    space,
    std::move(blob),
    std::vector<std::string>{"k"},
    static_cast<std::size_t>(payload_bytes),
    static_cast<std::size_t>(kNumRows) * sizeof(std::int64_t),
    static_cast<std::int64_t>(kNumRows));
  return result;
}

/// How a decode failed, as seen while its exception was being handled.
struct decode_failure {
  bool failed = false;
  std::optional<cudaError_t> fatal_status;
  std::string message;
  /// The message of the exception nested in the one thrown, if any.
  std::optional<std::string> nested_message;
};

/// Decodes @p chunk on @p target, the GPU current on the calling thread. The executor classifies
/// a failure while handling it (query_lifecycle_registry::report_failure), so this does the same
/// and keeps only what it found, not the exception.
decode_failure decode_on(sirius::compressed_device_representation const& chunk,
                         cucascade::memory::memory_space const& target,
                         rmm::cuda_stream const& stream)
{
  decode_failure result;
  try {
    auto decoded = sirius::converter_registry::get().convert<cucascade::gpu_table_representation>(
      chunk, &target, ::cuda::stream_ref{stream.value()});
    REQUIRE(decoded != nullptr);
    CHECK(decoded->get_table_view().num_rows() == kNumRows);
  } catch (std::exception const& error) {
    result.failed       = true;
    result.fatal_status = sirius::fatal_device_status(std::current_exception());
    result.message      = error.what();
    try {
      std::rethrow_if_nested(error);
    } catch (std::exception const& nested) {
      result.nested_message = nested.what();
    }
  }
  return result;
}

bool contains(std::string_view text, std::string_view part)
{
  return text.find(part) != std::string_view::npos;
}

}  // namespace

TEST_CASE("cross-GPU compressed decode reports a fatal status from its failure-path wait",
          "[compression][cuda_status_fault][multi_gpu]")
{
  cross_gpu_decode_fixture f;
  if (!f.setup()) {
    WARN("Skipping: needs two GPUs");
    return;
  }
  auto const injected = GENERATE(cudaErrorIllegalAddress, cudaErrorInvalidValue);
  CAPTURE(cudaGetErrorName(injected));

  auto const owned = make_compressed_chunk(*f.owner, /*with_slices=*/false);

  rmm::cuda_set_device_raii target_device{rmm::cuda_device_id{f.target->get_device_id()}};
  rmm::cuda_stream target_stream;
  decode_failure failure;
  {
    // The decode fails at its first copy, so the first wait on the target stream is the one on
    // the failure path.
    fault injection{fault::synchronize_stream, injected, 0, target_stream.value()};
    failure = decode_on(*owned.chunk, *f.target, target_stream);
    REQUIRE(injection.consumed());
  }
  REQUIRE(failure.failed);

  constexpr std::string_view original_failure = "has no payload buffer";
  if (injected == cudaErrorIllegalAddress) {
    CHECK(failure.fatal_status.value_or(cudaSuccess) == cudaErrorIllegalAddress);
    // The fatal status replaces the decode's own failure, which stays nested in it.
    REQUIRE(failure.nested_message.has_value());
    CHECK(contains(*failure.nested_message, original_failure));
  } else {
    // Any other status leaves the decode's own failure in place.
    CHECK_FALSE(failure.fatal_status.has_value());
    CHECK_FALSE(failure.nested_message.has_value());
    CHECK(contains(failure.message, original_failure));
  }
  target_stream.synchronize();
}

TEST_CASE("cross-GPU compressed decode reports a fatal status from its completion wait",
          "[compression][cuda_status_fault][multi_gpu]")
{
  cross_gpu_decode_fixture f;
  if (!f.setup()) {
    WARN("Skipping: needs two GPUs");
    return;
  }
  auto const injected = GENERATE(cudaErrorIllegalAddress, cudaErrorInvalidValue);
  CAPTURE(cudaGetErrorName(injected));

  auto const owned = make_compressed_chunk(*f.owner, /*with_slices=*/true);

  rmm::cuda_set_device_raii target_device{rmm::cuda_device_id{f.target->get_device_id()}};
  rmm::cuda_stream target_stream;
  // How many waits on the target stream precede the completion wait depends on the copy route
  // (a host-staged copy waits for each upload). Fail each wait in turn until a decode reaches
  // none: the last one failed is the completion wait. The others are not checked here.
  decode_failure last_failure;
  std::size_t waits = 0;
  for (;; ++waits) {
    REQUIRE(waits < 256);
    fault injection{fault::synchronize_stream, injected, waits, target_stream.value()};
    auto failure = decode_on(*owned.chunk, *f.target, target_stream);
    if (!injection.consumed()) {
      REQUIRE_FALSE(failure.failed);
      break;
    }
    last_failure = std::move(failure);
  }
  REQUIRE(waits > 0);
  REQUIRE(last_failure.failed);
  if (injected == cudaErrorIllegalAddress) {
    CHECK(last_failure.fatal_status.value_or(cudaSuccess) == cudaErrorIllegalAddress);
  } else {
    CHECK_FALSE(last_failure.fatal_status.has_value());
    CHECK(contains(last_failure.message, "cudaErrorInvalidValue"));
  }
  target_stream.synchronize();
}

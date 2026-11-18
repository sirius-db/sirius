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

#pragma once

#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator/operator_test_utils.hpp"

#include <rmm/cuda_stream.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/device_compressed_blob.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/sirius_converter_registry.hpp>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace sirius::test::compression_utils {

inline bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// GPU, host and disk spaces, with the converters registered. The disk space lives in a
/// directory of its own under the temp directory, named after @p name and the process.
struct tiered_env {
  std::filesystem::path dir;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  rmm::cuda_stream stream;

  explicit tiered_env(std::string const& name)
    : dir(std::filesystem::temp_directory_path() / (name + "_" + std::to_string(::getpid())))
  {
    std::filesystem::create_directories(dir);
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .track_reservation_per_stream(false)
      .set_gpu_usage_limit(1ull << 30)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1ull << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, 4ull << 30, dir.string());
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    gpu  = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
    host = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
    disk = const_cast<cucascade::memory::memory_space*>(
      mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).front());
  }

  // The env is a function-local static of its test file, destroyed at process exit: a failed
  // removal must not throw from there.
  ~tiered_env()
  {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }

  tiered_env(const tiered_env&)            = delete;
  tiered_env& operator=(const tiered_env&) = delete;
};

/// Converts @p batch to @p space, on the env's stream.
inline void move_to(tiered_env& env,
                    std::shared_ptr<cucascade::data_batch> const& batch,
                    cucascade::memory::memory_space* space)
{
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({space}, env.stream, *env.mgr, true).has_value());
}

/// The device tier of a compressed host batch: its table rebuilt in @p space from the pinned
/// payload. Waits for @p stream, so @p host need not outlive the call.
inline std::unique_ptr<sirius::compressed_device_representation> make_compressed_device(
  sirius::compressed_host_representation const& host,
  cucascade::memory::memory_space& space,
  ::cuda::stream_ref stream)
{
  auto const& blocks = host.payload();
  auto blob          = std::make_shared<sirius::compressed_device_blob>();
  std::string error;
  simpatico::payload_fetch_fn fetch =
    [&blocks](std::uint64_t off, std::size_t sz, void* dst, ::cuda::stream_ref s) {
      sirius::copy_pinned_blocks_to_device(blocks, off, dst, sz, s);
    };
  blob->table = simpatico::read_compressed_table_from_memory(
    host.header(), fetch, stream, operator_utils::get_resource_ref(space), &error);
  REQUIRE(error.empty());
  stream.sync();
  return std::make_unique<sirius::compressed_device_representation>(
    space,
    std::move(blob),
    host.column_names(),
    host.get_size_in_bytes(),
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());
}

}  // namespace sirius::test::compression_utils

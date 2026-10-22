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

#include "data/convertible_data_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

/// Batches, digests and memory spaces for the exchange's tests.
namespace sirius::test::exchange_utils {

namespace utils = sirius::test::operator_utils;
using cucascade::memory::Tier;

constexpr std::size_t KiB = std::size_t{1} << 10;

struct io_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;

  io_env()
  {
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(1ull << 30)
      .set_reservation_fraction_per_gpu(0.9)
      .set_per_numa_region_capacity(1ull << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.9);
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    gpu  = mgr->get_memory_space(Tier::GPU, 0);
    host = mgr->get_memory_space(Tier::HOST, 0);
  }
};

inline io_env& env()
{
  static io_env e;
  // Another suite in this process may have shut the global registry down since.
  sirius::converter_registry::initialize();
  return e;
}

inline void null_every_seventh(cudf::column& column, ::cuda::stream_ref stream)
{
  auto const rows = column.size();
  std::vector<std::uint32_t> words(
    cudf::bitmask_allocation_size_bytes(rows) / sizeof(std::uint32_t), 0);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (i % 7 == 0) {
      ++nulls;
    } else {
      words[static_cast<std::size_t>(i) / 32] |= 1u << (i % 32);
    }
  }
  rmm::device_buffer mask(words.data(), words.size() * sizeof(std::uint32_t), stream);
  column.set_null_mask(std::move(mask), nulls);
}

template <typename T>
inline std::unique_ptr<cudf::column> numeric_column(cudf::data_type type,
                                                    std::vector<T> const& values)
{
  auto column = cudf::make_fixed_width_column(type,
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              utils::default_stream(),
                                              utils::get_resource_ref(*env().gpu));
  cudaMemcpy(column->mutable_view().head(),
             values.data(),
             values.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  return column;
}

/// INT64s, nullable INT32s, DECIMAL(38,2) and strings.
inline std::unique_ptr<cudf::table> sample_table(std::size_t rows)
{
  auto stream = utils::default_stream();
  std::vector<std::int64_t> keys(rows);
  std::iota(keys.begin(), keys.end(), std::int64_t{1'000'000});
  std::vector<std::int32_t> small(rows);
  std::vector<__int128> amounts(rows);
  std::vector<std::string> flags(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    small[i]   = static_cast<std::int32_t>(i % 1000);
    amounts[i] = static_cast<__int128>(i % 50'000) * 100 + 25;
    flags[i]   = (i % 3 == 0) ? "RETURNED" : (i % 3 == 1 ? "" : "DELIVER IN PERSON");
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64}, keys));
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT32}, small));
  null_every_seventh(*columns.back(), stream);
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::DECIMAL128, -2}, amounts));
  columns.push_back(utils::make_string_column(flags, stream, utils::get_resource_ref(*env().gpu)));
  return std::make_unique<cudf::table>(std::move(columns));
}

inline std::shared_ptr<cucascade::data_batch> gpu_batch(std::unique_ptr<cudf::table> table)
{
  return sirius::make_data_batch(std::move(table),
                                 *env().gpu,
                                 utils::default_stream(),
                                 sirius::telemetry::batch_telemetry_info{});
}

struct table_digest {
  std::vector<std::uint32_t> row_hashes;
  std::vector<cudf::size_type> null_counts;
  bool operator==(table_digest const&) const = default;
};

inline table_digest digest(cudf::table_view const& view)
{
  auto stream = utils::default_stream();
  auto hashes = cudf::hashing::murmurhash3_x86_32(view, cudf::DEFAULT_HASH_SEED, stream);
  stream.sync();
  table_digest out;
  out.row_hashes = utils::copy_column_to_host<std::uint32_t>(hashes->view());
  for (auto const& column : view) {
    out.null_counts.push_back(column.null_count());
  }
  return out;
}

/// The digest of @p batch, brought to the GPU first.
inline table_digest digest(std::shared_ptr<cucascade::data_batch> const& batch)
{
  if (batch->to_read_only().get_current_tier() != Tier::GPU) {
    rmm::cuda_stream stream;
    sirius::convertible_data_batch convertible(batch);
    REQUIRE(convertible.convert({env().gpu}, stream.view(), *env().mgr, true).has_value());
    stream.synchronize();
  }
  auto ro = batch->to_read_only();
  if (cudaEvent_t writer = ro.get_writer_event()) { cudaEventSynchronize(writer); }
  return digest(ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
}

/// Reservations holding every byte the GPU space would still reserve.
inline std::vector<std::unique_ptr<cucascade::memory::reservation>> hold_gpu()
{
  std::vector<std::unique_ptr<cucascade::memory::reservation>> held;
  for (std::size_t chunk = std::size_t{1} << 30; chunk >= KiB;) {
    if (auto r = env().gpu->make_reservation_or_null(chunk)) {
      held.push_back(std::move(r));
    } else {
      chunk /= 2;
    }
  }
  return held;
}

}  // namespace sirius::test::exchange_utils

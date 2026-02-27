/*
 * Copyright 2025, Sirius Contributors.
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

#include "scan/test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <utils/utils.hpp>

#include <optional>
#include <type_traits>

namespace sirius::test::operator_utils {

using data_repository_mgr =
  cucascade::data_repository_manager<std::shared_ptr<cucascade::data_batch>>;
inline std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> initialize_memory_manager(
  std::size_t n_gpus = 1)
{
  // Reset converter registry to avoid cross-test leakage
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;

  // Configure a modest GPU space
  const size_t gpu_capacity  = 512ull << 20;  // 512MB
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;  // 1GB

  builder.set_number_of_gpus(n_gpus)
    .set_gpu_usage_limit(gpu_capacity / n_gpus)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity / n_gpus)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  // Initialize converters used by data representations
  sirius::converter_registry::initialize();
  return manager;
}

inline cucascade::memory::memory_space* get_default_gpu_space()
{
  static auto manager = initialize_memory_manager();
  return const_cast<cucascade::memory::memory_space*>(
    manager->get_memory_space(cucascade::memory::Tier::GPU, 0));
}
inline rmm::device_async_resource_ref get_resource_ref(cucascade::memory::memory_space& space)
{
  return rmm::to_device_async_resource_ref_checked(space.get_default_allocator());
}

inline rmm::cuda_stream_view default_stream() { return cudf::get_default_stream(); }

/**
 * @brief Horizontally concatenate multiple data_batch objects into a single data_batch.
 *
 * Takes multiple data_batch objects and combines their columns horizontally (side by side)
 * into a single data_batch with all columns.
 *
 * @param batches Vector of data_batch pointers to concatenate
 * @param space Memory space for the new batch
 * @return std::shared_ptr<cucascade::data_batch> New batch with all columns combined
 */
inline std::shared_ptr<cucascade::data_batch> concatenate_batches_horizontal(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  cucascade::memory::memory_space& space)
{
  if (batches.empty()) { throw std::runtime_error("Cannot concatenate empty batch list"); }

  auto mr     = get_resource_ref(space);
  auto stream = default_stream();

  // Collect all columns from all batches
  std::vector<std::unique_ptr<cudf::column>> all_columns;

  for (const auto& batch : batches) {
    auto& table     = batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
    auto table_view = table.view();

    // Release and collect each column from this table
    for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
      // We need to make a copy of each column since we can't move from the const table_view
      all_columns.push_back(std::make_unique<cudf::column>(table_view.column(i), stream, mr));
    }
  }

  // Create new table from all collected columns
  auto concatenated_table = std::make_unique<cudf::table>(std::move(all_columns));

  // Create and return new data_batch
  auto gpu_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(concatenated_table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(gpu_repr));
}

template <typename T>
inline std::vector<T> copy_column_to_host(const cudf::column_view& col)
{
  if constexpr (std::is_same_v<T, bool>) {
    std::vector<int8_t> tmp(col.size());
    if (col.size() > 0) {
      cudaMemcpy(
        tmp.data(), col.data<int8_t>(), sizeof(int8_t) * col.size(), cudaMemcpyDeviceToHost);
    }
    std::vector<bool> host(col.size());
    for (size_t i = 0; i < col.size(); ++i) {
      host[i] = tmp[i] != 0;
    }
    return host;
  } else if constexpr (std::is_same_v<T, std::string>) {
    std::vector<std::string> host;
    if (col.size() == 0) { return host; }
    cudf::strings_column_view str_col(col);
    std::vector<cudf::size_type> offsets(col.size() + 1);
    cudaMemcpy(offsets.data(),
               str_col.offsets().data<cudf::size_type>(),
               (col.size() + 1) * sizeof(cudf::size_type),
               cudaMemcpyDeviceToHost);
    std::vector<char> chars(offsets.back());
    if (!chars.empty()) {
      cudaMemcpy(chars.data(),
                 str_col.chars_begin(cudf::get_default_stream()),
                 offsets.back(),
                 cudaMemcpyDeviceToHost);
    }
    host.reserve(col.size());
    for (cudf::size_type i = 0; i < col.size(); ++i) {
      host.emplace_back(chars.data() + offsets[i], chars.data() + offsets[i + 1]);
    }
    return host;
  } else {
    std::vector<T> host(col.size());
    if (col.size() > 0) {
      cudaMemcpy(host.data(), col.data<T>(), sizeof(T) * col.size(), cudaMemcpyDeviceToHost);
    }
    return host;
  }
}

template <typename T>
inline std::shared_ptr<cucascade::data_batch> make_numeric_batch(
  cucascade::memory::memory_space& space, const std::vector<T>& values, cudf::type_id type_id)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_numeric_column(
    cudf::data_type{type_id}, size, cudf::mask_state::UNALLOCATED, stream, mr);
  if constexpr (std::is_same_v<T, bool>) {
    std::vector<int8_t> tmp(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      tmp[i] = static_cast<int8_t>(values[i]);
    }
    cudaMemcpy(col->mutable_view().data<int8_t>(),
               tmp.data(),
               sizeof(int8_t) * tmp.size(),
               cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(col->mutable_view().data<T>(),
               values.data(),
               sizeof(T) * values.size(),
               cudaMemcpyHostToDevice);
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(gpu_repr));
}

inline std::unique_ptr<cudf::column> make_string_column(const std::vector<std::string>& values,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto const strings_count = static_cast<cudf::size_type>(values.size());

  // Build offsets and chars on host
  std::vector<cudf::size_type> offsets(strings_count + 1, 0);
  cudf::size_type total_chars = 0;
  for (cudf::size_type i = 0; i < strings_count; ++i) {
    offsets[i + 1] = offsets[i] + static_cast<cudf::size_type>(values[i].size());
    total_chars    = offsets[i + 1];
  }
  std::vector<char> chars(static_cast<std::size_t>(total_chars));
  cudf::size_type cursor = 0;
  for (cudf::size_type i = 0; i < strings_count; ++i) {
    auto const& s = values[i];
    std::memcpy(chars.data() + cursor, s.data(), s.size());
    cursor += static_cast<cudf::size_type>(s.size());
  }

  // Offsets column
  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               static_cast<cudf::size_type>(offsets.size()),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
  cudaMemcpyAsync(offsets_col->mutable_view().data<cudf::size_type>(),
                  offsets.data(),
                  offsets.size() * sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Chars buffer
  rmm::device_buffer chars_buf(total_chars, stream, mr);
  if (total_chars > 0) {
    cudaMemcpyAsync(chars_buf.data(),
                    chars.data(),
                    chars.size() * sizeof(char),
                    cudaMemcpyHostToDevice,
                    stream.value());
  }

  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_col),
                                   std::move(chars_buf),
                                   0,
                                   rmm::device_buffer{0, stream, mr});
}

inline std::shared_ptr<cucascade::data_batch> make_string_batch(
  cucascade::memory::memory_space& space, const std::vector<std::string>& values)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_string_column(values, stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(gpu_repr));
}

inline std::shared_ptr<cucascade::data_batch> make_decimal64_batch(
  cucascade::memory::memory_space& space, const std::vector<int64_t>& values, int32_t scale)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, scale},
                                           size,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(col->mutable_view().data<int64_t>(),
             values.data(),
             sizeof(int64_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(gpu_repr));
}

template <typename T>
inline std::shared_ptr<cucascade::data_batch> make_timestamp_batch(
  cucascade::memory::memory_space& space, const std::vector<T>& values, cudf::type_id ts_type_id)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_timestamp_column(
    cudf::data_type{ts_type_id}, size, cudf::mask_state::UNALLOCATED, stream, mr);

  // TIMESTAMP_DAYS uses 32-bit underlying storage; others use 64-bit
  if (ts_type_id == cudf::type_id::TIMESTAMP_DAYS) {
    cudaMemcpy(col->mutable_view().data<int32_t>(),
               values.data(),
               sizeof(int32_t) * values.size(),
               cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(col->mutable_view().data<int64_t>(),
               values.data(),
               sizeof(int64_t) * values.size(),
               cudaMemcpyHostToDevice);
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(gpu_repr));
}

template <typename TFirst, typename TSecond>
inline std::shared_ptr<cucascade::data_batch> make_two_column_batch(
  cucascade::memory::memory_space& space,
  const std::vector<TFirst>& col0_values,
  const std::vector<TSecond>& col1_values,
  cudf::type_id col1_type_id,
  std::optional<int32_t> decimal_scale      = std::nullopt,
  std::optional<cudf::type_id> col0_type_id = std::nullopt)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();

  // Column 0: INT64 filter column
  auto col0 =
    cudf::make_numeric_column(cudf::data_type{col0_type_id.value_or(cudf::type_id::INT64)},
                              static_cast<cudf::size_type>(col0_values.size()),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  cudaMemcpy(col0->mutable_view().data<TFirst>(),
             col0_values.data(),
             sizeof(TFirst) * col0_values.size(),
             cudaMemcpyHostToDevice);

  std::unique_ptr<cudf::column> col1;
  if constexpr (std::is_same_v<TSecond, bool>) {
    std::vector<int8_t> tmp(col1_values.size());
    for (size_t i = 0; i < col1_values.size(); ++i) {
      tmp[i] = static_cast<int8_t>(col1_values[i]);
    }
    col1 = cudf::make_numeric_column(cudf::data_type{col1_type_id},
                                     static_cast<cudf::size_type>(col1_values.size()),
                                     cudf::mask_state::UNALLOCATED,
                                     stream,
                                     mr);
    cudaMemcpy(col1->mutable_view().data<int8_t>(),
               tmp.data(),
               sizeof(int8_t) * tmp.size(),
               cudaMemcpyHostToDevice);
  } else if constexpr (std::is_same_v<TSecond, std::string>) {
    col1 = make_string_column(col1_values, stream, mr);
  } else if (col1_type_id == cudf::type_id::TIMESTAMP_DAYS ||
             col1_type_id == cudf::type_id::TIMESTAMP_MICROSECONDS ||
             col1_type_id == cudf::type_id::TIMESTAMP_MILLISECONDS ||
             col1_type_id == cudf::type_id::TIMESTAMP_SECONDS ||
             col1_type_id == cudf::type_id::TIMESTAMP_NANOSECONDS) {
    col1 = cudf::make_timestamp_column(cudf::data_type{col1_type_id},
                                       static_cast<cudf::size_type>(col1_values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
    if (col1_type_id == cudf::type_id::TIMESTAMP_DAYS) {
      cudaMemcpy(col1->mutable_view().data<int32_t>(),
                 col1_values.data(),
                 sizeof(int32_t) * col1_values.size(),
                 cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(col1->mutable_view().data<int64_t>(),
                 col1_values.data(),
                 sizeof(int64_t) * col1_values.size(),
                 cudaMemcpyHostToDevice);
    }
  } else if (col1_type_id == cudf::type_id::DECIMAL32 || col1_type_id == cudf::type_id::DECIMAL64 ||
             col1_type_id == cudf::type_id::DECIMAL128) {
    auto data_type = cudf::data_type{col1_type_id, decimal_scale.value_or(0)};
    col1           = cudf::make_fixed_point_column(data_type,
                                         static_cast<cudf::size_type>(col1_values.size()),
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);
    cudaMemcpy(col1->mutable_view().data<int64_t>(),
               col1_values.data(),
               sizeof(int64_t) * col1_values.size(),
               cudaMemcpyHostToDevice);
  } else {
    col1 = cudf::make_numeric_column(cudf::data_type{col1_type_id},
                                     static_cast<cudf::size_type>(col1_values.size()),
                                     cudf::mask_state::UNALLOCATED,
                                     stream,
                                     mr);
    cudaMemcpy(col1->mutable_view().data<TSecond>(),
               col1_values.data(),
               sizeof(TSecond) * col1_values.size(),
               cudaMemcpyHostToDevice);
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col0));
  cols.push_back(std::move(col1));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(gpu_repr));
}

}  // namespace sirius::test::operator_utils

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

#include <cudf/table/table_view.hpp>

#include <data/data_batch.hpp>
#include <data/gpu_data_representation.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Global atomic counter for generating unique data batch IDs.
 *
 * This provides a simple way to generate unique IDs for data batches
 * without requiring a data_repository_manager instance.
 */
inline std::atomic<uint64_t> g_next_batch_id{0};

/**
 * @brief Generate a unique data batch ID.
 */
inline uint64_t get_next_batch_id() { return g_next_batch_id++; }

/**
 * @brief Get a cudf::table_view from a data_batch.
 *
 * Assumes the data_batch contains a gpu_table_representation.
 *
 * @param batch The data batch to extract the table view from.
 * @return cudf::table_view The underlying cudf table view.
 */
inline cudf::table_view get_cudf_table_view(const cucascade::data_batch& batch)
{
  auto* data = batch.get_data();
  if (data == nullptr) { throw std::runtime_error("data_batch has no data representation"); }
  return data->cast<cucascade::gpu_table_representation>().get_table();
}

/**
 * @brief Create a shared_ptr<data_batch> from a cudf::table.
 *
 * @param table The cudf table (will be moved from).
 * @param memory_space The memory space where the table resides.
 * @return std::shared_ptr<cucascade::data_batch> The new data batch.
 */
inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  cudf::table&& table, cucascade::memory::memory_space& memory_space)
{
  auto gpu_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(table), memory_space);
  return std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(gpu_repr));
}

/**
 * @brief Create a shared_ptr<data_batch> from a unique_ptr<cudf::table>.
 *
 * @param table The cudf table (will be moved from).
 * @param memory_space The memory space where the table resides.
 * @return std::shared_ptr<cucascade::data_batch> The new data batch.
 */
inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  std::unique_ptr<cudf::table> table, cucascade::memory::memory_space& memory_space)
{
  auto gpu_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(*table), memory_space);
  return std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(gpu_repr));
}

/**
 * @brief Acquire processing handles for a collection of data batches.
 *
 * This function attempts to acquire processing locks on all provided batches.
 * If any batch cannot be locked (e.g., it's being downgraded), the function
 * returns an empty optional and releases any previously acquired locks.
 *
 * Usage pattern:
 * @code
 *   auto handles = acquire_processing_handles(batches);
 *   if (!handles) {
 *     // Some batch is being downgraded, retry later
 *     return;
 *   }
 *   // Process batches safely - handles prevent downgrade
 *   do_processing(batches);
 *   // Handles automatically release when going out of scope
 * @endcode
 *
 * @param batches Vector of data batch shared pointers to lock.
 * @return std::optional<std::vector<cucascade::data_batch_processing_handle>>
 *         Vector of handles if all locks acquired, empty optional otherwise.
 */
inline std::optional<std::vector<cucascade::data_batch_processing_handle>>
acquire_processing_handles(const std::vector<std::shared_ptr<cucascade::data_batch>>& batches)
{
  std::vector<cucascade::data_batch_processing_handle> handles;
  handles.reserve(batches.size());

  for (const auto& batch : batches) {
    if (!batch->try_to_lock_for_processing()) {
      // Failed to acquire lock - handles vector will be destroyed,
      // releasing any locks we already acquired
      return std::nullopt;
    }
    handles.emplace_back(batch.get());
  }

  return handles;
}

/**
 * @brief Acquire a processing handle for a single data batch.
 *
 * @param batch The data batch to lock for processing.
 * @return std::optional<cucascade::data_batch_processing_handle>
 *         Handle if lock acquired, empty optional otherwise.
 */
inline std::optional<cucascade::data_batch_processing_handle> acquire_processing_handle(
  const std::shared_ptr<cucascade::data_batch>& batch)
{
  if (!batch->try_to_lock_for_processing()) { return std::nullopt; }
  return cucascade::data_batch_processing_handle(batch.get());
}

}  // namespace sirius

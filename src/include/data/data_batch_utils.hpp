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

#include "cudf/cudf_utils.hpp"
#include "log/logging.hpp"

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <duckdb/common/types.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
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
  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), memory_space);
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
    std::make_unique<cucascade::gpu_table_representation>(std::move(table), memory_space);
  return std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(gpu_repr));
}

/**
 * @brief Cast batch columns to match expected DuckDB LogicalTypes (e.g. decimal64 -> decimal128).
 *
 * If the batch has a different number of columns than expected_types, or all columns already
 * match their expected type, returns the original batch. Otherwise returns a new batch with
 * columns cast to their expected type.
 *
 * @param batch The data batch to cast (may be null).
 * @param expected_types Expected output types (one per column).
 * @param stream CUDA stream for allocations and casts.
 * @return The original batch if no casts needed or on error; otherwise a new batch.
 */
inline std::shared_ptr<cucascade::data_batch> cast_batch_to_expected_types(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const duckdb::vector<duckdb::LogicalType>& expected_types,
  rmm::cuda_stream_view stream)
{
  if (!batch || !batch->get_data()) { return batch; }
  cudf::table_view tbl = get_cudf_table_view(*batch);
  if (static_cast<size_t>(tbl.num_columns()) != expected_types.size()) {
    SIRIUS_LOG_ERROR(
      "cast_batch_to_expected_types: batch column count mismatch: got {}, expected {}",
      tbl.num_columns(),
      expected_types.size());
    return batch;
  }
  auto* space = batch->get_memory_space();
  if (!space) {
    SIRIUS_LOG_ERROR("cast_batch_to_expected_types: Could not get memory space from batch");
    return batch;
  }
  auto mr = space->get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(tbl.num_columns());
  bool any_cast = false;
  for (cudf::size_type c = 0; c < tbl.num_columns(); c++) {
    cudf::data_type expected = duckdb::GetCudfType(expected_types[c]);
    cudf::column_view col    = tbl.column(c);
    if (col.type() != expected) {
      out_cols.push_back(cudf::cast(col, expected, stream, mr));
      any_cast = true;
    } else {
      out_cols.push_back(std::make_unique<cudf::column>(col, stream, mr));
    }
  }
  if (!any_cast) { return batch; }
  auto out_table = std::make_unique<cudf::table>(std::move(out_cols), stream, mr);
  return make_data_batch(std::move(out_table), *space);
}

}  // namespace sirius

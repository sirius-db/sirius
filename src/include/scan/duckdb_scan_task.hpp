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

// sirius
#include <data/data_repository_manager.hpp>
#include <operator/gpu_physical_table_scan.hpp>
#include <parallel/task.hpp>
#include <scan/duckdb_scan_executor.hpp>

// duckdb
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/client_context.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sirius::parallel {

/**
 * @brief TODO
 */
class duckdb_scan_task_global_state : public itask_global_state, public duckdb::GlobalSourceState {
 public:
  /// TODO: add task_completion_message_queue, when available.
  duckdb_scan_task_global_state(uint64_t pipeline_id,
                                duckdb_scan_executor& scan_executor,
                                data_repository_manager& drm,
                                duckdb::ClientContext& client_ctx,
                                const duckdb::GPUPhysicalTableScan& op);

  uint64_t MaxThreads() const { return max_threads; }

  bool IsSourceDrained() const { return source_drained.load(std::memory_order_acquire); }

  void SetSourceDrained() { return source_drained.store(true, std::memory_order_release); }

  duckdb::optional_ptr<duckdb::TableFilterSet> GetTableFilters(
    const duckdb::GPUPhysicalTableScan& op) const
  {
    return table_filters ? table_filters.get() : op.fake_table_filters.get();
  }

  //===----------Fields----------===//
  std::atomic<bool> source_drained{false};  ///< Whether the table scan source is fully drained
  uint64_t pipeline_id;                     ///< The pipeline id to which this table scan belongs
  duckdb_scan_executor&
    scan_executor;  ///< For scheduling new scan tasks, if the source is not yet drained
  data_repository_manager& drm;  ///< The data repository manager to which to push data batches
  uint64_t
    max_threads;  ///< Maximum number of threads allowed for this scan (determined by scan executor)
  unique_ptr<duckdb::TableFilterSet>
    table_filters;  ///< Combined table filters, if there are dynamic filters
  unique_ptr<duckdb::GlobalTableFunctionState>
    global_tf_state;  ///< Global state for the table function
  duckdb::ClientContext&
    client_ctx;  ///< The duckdb client context, needed for allocation and table function execution
};

/**
 * @brief TODO
 *
 */
class duckdb_scan_task_local_state : public itask_local_state {
 public:
  duckdb_scan_task_local_state(const duckdb_scan_task_global_state& g_state,
                               duckdb::ExecutionContext& exec_ctx,
                               const duckdb::GPUPhysicalTableScan& op,
                               size_t approximate_batch_size);
  ~duckdb_scan_task_local_state() = default;

  //===----------Fields----------===//
  size_t approximate_batch_size;              ///< Approximate target batch size in bytes
  size_t num_columns;                         ///< Number of columns to be scanned
  vector<duckdb::LogicalType> scanned_types;  ///< Types of the scanned columns
  vector<size_t> column_sizes;                ///< Size of each DuckDB column in bytes
  size_t max_type_size = 0;                   ///< Maximum size of any single type in bytes

  duckdb::DataChunk chunk;        ///< DataChunk buffer
  vector<uint8_t*> data_ptrs;     ///< Pointers to each column's data buffer
  vector<uint8_t*> mask_ptrs;     ///< Pointers to each column's null mask buffer
  vector<uint64_t*> offset_ptrs;  ///< Pointers to CHAR/VARCHAR column offset buffers
  vector<size_t> byte_offsets;    ///< Current byte offsets in data buffers
  size_t row_offset        = 0;   ///< Current row offset in buffers
  size_t bytes_accumulated = 0;   ///< Total bytes accumulated so far by the scan

  unique_ptr<duckdb::LocalTableFunctionState>
    local_tf_state;                        ///< Local state for the table function.
  const duckdb::GPUPhysicalTableScan& op;  ///< Reference to the GPU physical table scan operator.
};

}  // namespace sirius::parallel
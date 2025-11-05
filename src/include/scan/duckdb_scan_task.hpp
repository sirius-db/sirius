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
#include <memory/memory_reservation.hpp>
#include <operator/gpu_physical_table_scan.hpp>
#include <parallel/task.hpp>
#include <scan/duckdb_scan_executor.hpp>
#include <scan/physical_table_scan_adapter.hpp>

// duckdb
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/client_context.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sirius::parallel {

//===----------------------------------------------------------------------===//
// DuckDB Scan Task Global State
//===----------------------------------------------------------------------===//

/**
 * @brief TODO
 */
class duckdb_scan_task_global_state : public itask_global_state, public duckdb::GlobalSourceState {
 public:
  duckdb_scan_task_global_state(uint64_t pipeline_id,
                                duckdb_scan_executor const& scan_executor,
                                duckdb::ClientContext& client_ctx,
                                duckdb::physical_table_scan_adapter const& ptsa);

  uint64_t MaxThreads() override { return max_threads; }

  bool IsSourceDrained() const { return source_drained.load(std::memory_order_acquire); }

  void SetSourceDrained() { source_drained.store(true, std::memory_order_release); }

  //===----------Fields----------===//
  std::atomic<bool> source_drained{false};  ///< Whether the table scan source is fully drained
  uint64_t pipeline_id;                     ///< The pipeline id to which this table scan belongs
  uint64_t max_threads;                     ///< Maximum number of threads for this scan task

  unique_ptr<duckdb::GlobalTableFunctionState>
    global_tf_state;                    ///< Global state for the table function
  const duckdb::PhysicalTableScan& op;  ///< The physical table scan operator adapter being executed
};

/**
 * @brief TODO
 *
 */
class duckdb_scan_task_local_state : public itask_local_state {
  static constexpr size_t DEFAULT_APPROXIMATE_BATCH_SIZE = 2ULL * 1024 * 1024 * 1024;  ///< 2 GB

  // The following VARCHAR size is selected to accomodate all CHAR/VARCHAR TPC-H columns.
  static constexpr size_t DEFAULT_VARCHAR_SIZE = 256ULL;  ///< 256 bytes

 public:
  //===----------Constructor & Destructor----------===//
  duckdb_scan_task_local_state(duckdb_scan_task_global_state& g_state,
                               memory::memory_reservation_manager& mem_res_mgr,
                               duckdb::ExecutionContext& exec_ctx,
                               size_t approximate_batch_size);
  ~duckdb_scan_task_local_state();

  //===----------Fields----------===//
  size_t num_columns;                         ///< Number of columns to be scanned
  vector<duckdb::LogicalType> scanned_types;  ///< Types of the scanned columns
  vector<size_t> column_sizes;                ///< Size of each DuckDB column in bytes
  vector<size_t> varchar_indices;             ///< Indices of VARCHAR columns

  memory::memory_reservation_manager&
    mem_res_mgr;                  ///< Memory reservation manager for requesting memory
  size_t approximate_batch_size;  ///< Approximate target batch size in bytes
  duckdb::DataChunk chunk;        ///< DataChunk buffer
  vector<unique_ptr<memory::reservation>>
    data_reservations;  ///< Reservations for each column's data buffer
  vector<unique_ptr<memory::reservation>>
    mask_reservations;  ///< Reservations for each column's null mask buffer
  vector<unique_ptr<memory::reservation>>
    offset_reservations;         ///< Reservations for CHAR/VARCHAR column offset buffers
  vector<size_t> byte_offsets;   ///< Current byte offsets in data buffers
  size_t row_offset        = 0;  ///< Current row offset in buffers
  size_t bytes_accumulated = 0;  ///< Total bytes accumulated so far by the scan

  unique_ptr<duckdb::LocalTableFunctionState>
    local_tf_state;                    ///< Local state for the table function.
  duckdb::ExecutionContext& exec_ctx;  ///< The duckdb execution context, needed for initializing
                                       ///< the local table function state

 private:
  void initialize_local_table_function_state(duckdb::PhysicalTableScan const& op,
                                             duckdb::ExecutionContext& exec_ctx,
                                             duckdb::GlobalTableFunctionState* global_tf_state);
  size_t initialize_batch_metadata(const duckdb::PhysicalTableScan& op);
  size_t estimate_rows_per_batch(size_t estimated_row_size);
  void make_memory_reservations(memory::memory_reservation_manager& mem_res_mgr,
                                size_t estimated_num_rows);
};

}  // namespace sirius::parallel
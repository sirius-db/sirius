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

// sirius
#include <helper/utils.hpp>
#include <memory/memory_reservation.hpp>
#include <scan/duckdb_scan_task.hpp>

// duckdb
#include <duckdb/function/table_function.hpp>

#include <cstddef>

namespace sirius::parallel {

//===----------------------------------------------------------------------===//
// DuckDB Scan Task Global State
//===----------------------------------------------------------------------===//
duckdb_scan_task_global_state::duckdb_scan_task_global_state(
  uint64_t pipeline_id,
  duckdb_scan_executor const& scan_executor,
  duckdb::ClientContext& client_ctx,
  duckdb::physical_table_scan_adapter const& ptsa)
  : pipeline_id(pipeline_id),
    max_threads(scan_executor.get_num_threads()) op(ptsa.physical_table_scan)
{
  // Initialize global table function state
  if (op.function.init_global) {
    duckdb::TableFunctionInitInput tf_input(
      op.bind_data.get(), op.column_ids, op.projection_ids, nullptr, op.extra_info.sample_options);
    global_tf_state = op.function.init_global(client_ctx, tf_input);
  }

  // We do not support in_out_functions
  if (op.function.in_out_function) {
    throw duckdb::NotImplementedException(
      "In-out table functions are not supported in sirius table scans.");
  }

  // For caching reasons, we do not push table filters into the scan
  if (op.dynamic_filters) {
    throw duckdb::NotImplementedException(
      "Dynamic table filters are not supported in sirius table scans.");
  }
}

//===----------------------------------------------------------------------===//
// DuckDB Scan Task Local State
//===----------------------------------------------------------------------===//
//===----------Constructor----------===//
duckdb_scan_task_local_state::duckdb_scan_task_local_state(
  duckdb_scan_task_global_state& g_state,
  memory::memory_reservation_manager& mem_res_mgr,
  duckdb::ExecutionContext& exec_ctx,
  size_t approximate_batch_size = DEFAULT_APPROXIMATE_BATCH_SIZE)
  : mem_res_mgr(mem_res_mgr), exec_ctx(exec_ctx), approximate_batch_size(approximate_batch_size)
{
  auto const& op = g_state.op;
  num_columns    = op.projection_ids.size();

  // Initialize local table function state
  initialize_local_table_function_state(op, exec_ctx, g_state.global_tf_state.get());

  // Initialize the batch metadata and get the estimated row size in bytes
  auto const estimated_row_size = initialize_batch_metadata(op);

  // Estimate number of rows per batch
  auto const estimated_num_rows = estimate_rows_per_batch(estimated_row_size);

  // Allocate data buffers
  make_memory_reservations(mem_res_mgr, estimated_num_rows);
}

//===----------Destructor----------===//
duckdb_scan_task_local_state::~duckdb_scan_task_local_state()
{
  for (size_t i = 0; i < num_columns; ++i) {
    if (data_reservations[i]) { mem_res_mgr.release_reservation(std::move(data_reservations[i])); }
    if (mask_reservations[i]) { mem_res_mgr.release_reservation(std::move(mask_reservations[i])); }
    if (offset_reservations[i]) {
      mem_res_mgr.release_reservation(std::move(offset_reservations[i]));
    }
  }
}

void duckdb_scan_task_local_state::initialize_local_table_function_state(
  duckdb::PhysicalTableScan const& op,
  duckdb::ExecutionContext& exec_ctx,
  duckdb::GlobalTableFunctionState* global_tf_state)
{
  if (op.function.init_local) {
    duckdb::TableFunctionInitInput tf_input(
      op.bind_data.get(), op.column_ids, op.projection_ids, nullptr, op.extra_info.sample_options);
    local_tf_state = op.function.init_local(exec_ctx, tf_input, global_tf_state);
  }
}

size_t duckdb_scan_task_local_state::initialize_batch_metadata(const duckdb::PhysicalTableScan& op)
{
  // Initialize projected types and column sizes
  scanned_types.resize(num_columns);
  column_sizes.resize(num_columns);
  size_t estimated_row_size = 0;
  for (size_t i = 0; i < num_columns; ++i) {
    scanned_types[i] = op.returned_types[op.column_ids[i].GetPrimaryIndex()];
    if (scanned_types[i] == duckdb::LogicalTypeId::VARCHAR) {
      varchar_indices.push_back(i);
      column_sizes[i] = DEFAULT_VARCHAR_SIZE;
    } else {
      column_sizes[i] = duckdb::GetTypeIdSize(scanned_types[i].InternalType());
    }
    byte_offsets[i] = 0;
    estimated_row_size += column_sizes[i];
  }
  return estimated_row_size;
}

void duckdb_scan_task_local_state::make_memory_reservations(
  memory::memory_reservation_manager& mem_res_mgr, size_t estimated_num_rows)
{
  data_reservations.resize(num_columns, nullptr);
  mask_reservations.resize(num_columns, nullptr);
  offset_reservations.resize(num_columns, nullptr);

  // HOST memory reservation request
  struct memory::any_memory_space_in_tier res_request(memory::Tier::HOST);

  for (size_t i = 0; i < num_columns; ++i) {
    // Allocate data buffer
    data_reservations[i] =
      mem_res_mgr.request_reservation(res_request, column_sizes[i] * estimated_num_rows);

    // Allocate null mask buffer
    mask_reservations[i] =
      mem_res_mgr.request_reservation(res_request, utils::ceil_div_8(estimated_num_rows));

    // Allocate offset buffer for VARCHAR columns
    if (scanned_types[i] == duckdb::LogicalTypeId::VARCHAR) {
      offset_reservations[i] =
        mem_res_mgr.request_reservation(res_request, sizeof(uint64_t) * (estimated_num_rows + 1));
    }
  }
}

size_t duckdb_scan_task_local_state::estimate_rows_per_batch(size_t estimated_row_size)
{
  return utils::ceil_div(estimated_row_size * CHAR_BIT + 1, approximate_batch_size * CHAR_BIT);
}

}  // namespace sirius::parallel
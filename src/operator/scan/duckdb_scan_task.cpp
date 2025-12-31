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
#include <memory/sirius_memory_manager.hpp>
#include <scan/duckdb_scan_task.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/function/table_function.hpp>

// standard library
#include <cstddef>

namespace sirius::parallel {

//===----------------------------------------------------------------------===//
// duckdb_scan_task_global_state
//===----------------------------------------------------------------------===//
duckdb_scan_task_global_state::duckdb_scan_task_global_state(
  uint64_t pipeline_id,
  duckdb_scan_executor& scan_exec,
  duckdb::ClientContext& client_ctx,
  duckdb::physical_table_scan_adapter const& ptsa)
  : pipeline_id(pipeline_id),
    max_threads(scan_exec.get_num_threads()),
    scan_executor(scan_exec),
    op(ptsa.get_physical_table_scan())
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
// duckdb_scan_task_local_state::column_builder
//===----------------------------------------------------------------------===//
duckdb_scan_task_local_state::column_builder::column_builder(duckdb::LogicalType t,
                                                             size_t default_varchar_size)
  : type(t)
{
  type_size = t.InternalType() == duckdb::PhysicalType::VARCHAR
                ? default_varchar_size
                : duckdb::GetTypeIdSize(t.InternalType());
}

void duckdb_scan_task_local_state::column_builder::initialize_accessors(
  size_t estimated_num_rows, size_t byte_offset, unique_ptr<multiple_blocks_allocation>& allocation)
{
  assert(allocation != nullptr);
  assert(!allocation->get_blocks().empty());

  if (type.InternalType() == duckdb::PhysicalType::VARCHAR) {
    // Initialize offset accessor
    offset_blocks_accessor.initialize(byte_offset, allocation);
    // Write the initial offset value of 0
    offset_blocks_accessor.set_current(0, allocation);
    // Initialize data accessor
    total_data_bytes_allocated = estimated_num_rows * type_size;
    size_t data_byte_offset    = byte_offset + (estimated_num_rows + 1) * sizeof(int64_t);
    data_blocks_accessor.initialize(data_byte_offset, allocation);
    // Initialize mask accessor
    size_t mask_byte_offset = data_byte_offset + total_data_bytes_allocated;
    mask_blocks_accessor.initialize(mask_byte_offset, allocation);
  } else {
    // Fixed-width column
    data_blocks_accessor.initialize(byte_offset, allocation);
    size_t mask_byte_offset = byte_offset + estimated_num_rows * type_size;
    mask_blocks_accessor.initialize(mask_byte_offset, allocation);
  }
}

// This method should be called only on variable-length data
bool duckdb_scan_task_local_state::column_builder::sufficient_space_for_column(
  duckdb::Vector& vec, duckdb::ValidityMask const& validity, size_t num_rows)
{
  size_t data_bytes = 0;
  if (type.InternalType() == duckdb::PhysicalType::VARCHAR) {
    auto const* str_data = reinterpret_cast<duckdb::string_t const*>(vec.GetData());
    for (size_t row = 0; row < num_rows; ++row) {
      if (validity.RowIsValid(row)) { data_bytes += str_data[row].GetSize(); }
    }
  } else {
    // Fixed-width column
    data_bytes = type_size * num_rows;
  }
  return data_bytes + total_data_bytes <= total_data_bytes_allocated;
}

void duckdb_scan_task_local_state::column_builder::process_mask_for_column(
  duckdb::ValidityMask const& validity,
  size_t num_rows,
  size_t row_offset,
  unique_ptr<multiple_blocks_allocation>& allocation)
{
  auto const* src_valid = reinterpret_cast<uint8_t const*>(validity.GetData());
  auto const cur_bit    = utils::mod_8(row_offset);  //< bit offset in current byte
  auto const num_bits   = num_rows;

  if (src_valid == nullptr) {
    // All valid case
    if (cur_bit == 0) {
      // Byte aligned case
      auto const full_bytes = utils::div_8(num_bits);
      auto const tail_bits  = utils::mod_8(num_bits);
      for (auto b = 0; b < full_bytes; ++b) {
        mask_blocks_accessor.set_current(FULL_MASK, allocation);
        mask_blocks_accessor.advance();
      }
      if (tail_bits > 0) {
        auto const tail_mask = utils::make_mask<uint8_t>(tail_bits);
        mask_blocks_accessor.set_current(tail_mask, allocation);
        mask_blocks_accessor.advance();
      }
    } else {
      // Byte unaligned case
      auto const bits_in_current_byte = std::min<uint32_t>(CHAR_BIT - cur_bit, num_bits);
      auto const remaining_bits =
        num_bits - bits_in_current_byte;  // Remaining bits after filling current byte
      auto const remaining_bytes = utils::div_8(remaining_bits);
      auto const tail_bits       = utils::mod_8(remaining_bits);

      // Set bits in the current byte
      auto const current_byte_mask =
        static_cast<uint8_t>(utils::make_mask<uint8_t>(bits_in_current_byte) << cur_bit);
      mask_blocks_accessor.set_current(
        (mask_blocks_accessor.get_current(allocation) & ~current_byte_mask) | current_byte_mask,
        allocation);
      mask_blocks_accessor.advance();

      // Set full bytes
      for (size_t b = 0; b < remaining_bytes; ++b) {
        mask_blocks_accessor.set_current(FULL_MASK, allocation);
        mask_blocks_accessor.advance();
      }

      // Set tail bits
      if (tail_bits > 0) {
        auto const tail_mask = utils::make_mask<uint8_t>(tail_bits);
        mask_blocks_accessor.set_current(tail_mask, allocation);
        mask_blocks_accessor.advance();
      }
    }
    return;
  }
  // condition: src_valid != nullptr
  has_nulls = true;
  if (cur_bit == 0) {
    // Byte aligned case
    auto const full_bytes = utils::div_8(num_bits);
    auto const tail_bits  = utils::mod_8(num_bits);
    mask_blocks_accessor.memcpy_from(src_valid, full_bytes, allocation);
    if (tail_bits > 0) {
      auto const tail_mask = utils::make_mask<uint8_t>(tail_bits);
      auto const tail      = src_valid[full_bytes] & tail_mask;
      mask_blocks_accessor.set_current(tail, allocation);
    }
  } else {
    // Byte unaligned case
    auto const num_bytes = utils::ceil_div_8(num_bits);
    for (size_t b = 0; b < num_bytes; ++b) {
      auto src_byte = src_valid[b];
      // The current number of bits we are copying from src_byte
      auto const cur_bits = std::min<uint32_t>(CHAR_BIT, num_bits - utils::mul_8(b));
      // Mask the source byte to only the bits we care about
      src_byte = src_byte & utils::make_mask<uint8_t>(cur_bits);
      // The number of bits that fit in the current destination byte (the lower bits from src_byte)
      auto const num_lower_bits = std::min<uint32_t>(CHAR_BIT - cur_bit, cur_bits);
      // The lower bits from the source byte to copy into the current destination byte, shifted into
      // position for copying into the destination byte
      auto const lower_bits =
        static_cast<uint8_t>((src_byte & utils::make_mask<uint8_t>(num_lower_bits)) << cur_bit);
      // The mask for the lower bits in the destination byte
      auto const lower_mask =
        static_cast<uint8_t>(utils::make_mask<uint8_t>(num_lower_bits) << cur_bit);
      // Set the lower bits in the current destination byte
      mask_blocks_accessor.set_current(
        (mask_blocks_accessor.get_current(allocation) & ~lower_mask) | lower_bits, allocation);
      mask_blocks_accessor.advance();

      // There may be leftover bits (the upper bits from src_byte) to propagate to the next byte
      auto const num_upper_bits = cur_bits - num_lower_bits;
      if (num_upper_bits > 0) {
        // The mask for the bits in the next destination byte
        auto const upper_mask = utils::make_mask<uint8_t>(num_upper_bits);
        // The upper bits from the source byte to copy into the next destination byte, shifted into
        // position for copying into the destination byte
        auto const upper_bits = static_cast<uint8_t>((src_byte >> num_lower_bits) & upper_mask);
        // Set the bits in the next destination byte
        mask_blocks_accessor.set_current(
          (mask_blocks_accessor.get_current(allocation) & ~upper_mask) | upper_bits, allocation);
      }
    }
  }
}

void duckdb_scan_task_local_state::column_builder::process_column(
  duckdb::Vector& vec,
  duckdb::ValidityMask const& validity,
  size_t num_rows,
  size_t row_offset,
  unique_ptr<multiple_blocks_allocation>& allocation)
{
  // PRECONDITION: Vector must be flattened
  if (type.InternalType() == duckdb::PhysicalType::VARCHAR) {
    size_t data_bytes    = 0;
    auto const* str_data = reinterpret_cast<duckdb::string_t const*>(vec.GetData());
    for (size_t row = 0; row < num_rows; ++row) {
      auto const prev_offset = offset_blocks_accessor.get_current(allocation);
      offset_blocks_accessor.advance();
      if (validity.RowIsValid(row)) {
        auto const& str = str_data[row];
        auto const len  = str.GetSize();
        offset_blocks_accessor.set_current(prev_offset + len, allocation);

        // Copy string data
        data_blocks_accessor.memcpy_from(str.GetData(), len, allocation);

        // Update data bytes
        data_bytes += len;
      } else {
        offset_blocks_accessor.set_current(prev_offset, allocation);
      }
    }
    total_data_bytes += data_bytes;
  } else {
    // Fixed-width column
    auto const data_bytes = type_size * num_rows;
    data_blocks_accessor.memcpy_from(vec.GetData(), data_bytes, allocation);
    total_data_bytes += data_bytes;
  }

  process_mask_for_column(validity, num_rows, row_offset, allocation);
}

//===----------------------------------------------------------------------===//
// duckdb_scan_task_local_state
//===----------------------------------------------------------------------===//
//===----------Constructor----------===//
duckdb_scan_task_local_state::duckdb_scan_task_local_state(
  duckdb_scan_task_global_state& g_state,
  duckdb::ExecutionContext& exec_ctx,
  size_t approximate_batch_size,
  size_t default_varchar_size,
  unique_ptr<duckdb::LocalTableFunctionState> existing_local_tf_state)
  : approximate_batch_size(approximate_batch_size),
    default_varchar_size(default_varchar_size),
    exec_ctx(exec_ctx),
    local_tf_state(std::move(existing_local_tf_state))  // Move the existing state if provided
{
  auto const& op = g_state.op;
  num_columns    = op.projection_ids.size();

  auto& mem_res_mgr = sirius::memory_manager::get();

  // Make the memory reservation request
  reservation = mem_res_mgr.request_reservation(res_request, approximate_batch_size);

  // Make the allocation
  auto const* mem_space =
    mem_res_mgr.get_memory_space(cucascade::memory::Tier::HOST, HOST_SPACE_DEVICE_ID);
  if (mem_space == nullptr) {
    throw std::runtime_error(
      "[duckdb_scan_task_local_state] Failed to get HOST memory space with device id " +
      std::to_string(HOST_SPACE_DEVICE_ID));
  }
  auto* allocator =
    mem_space->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (allocator == nullptr) {
    throw std::runtime_error(
      "[duckdb_scan_task_local_state] Failed to get fixed_size_host_memory_resource allocator for "
      "HOST memory space.");
  }
  allocation = allocator->allocate_multiple_blocks(approximate_batch_size);

  // Estimate number of rows per batch
  estimate_rows_per_batch(op);

  // Initialize the column builders
  initialize_builders();

  // Initialize local table function state (will skip if local_tf_state already set)
  initialize_local_table_function_state(op, exec_ctx, g_state.global_tf_state.get());
}

void duckdb_scan_task_local_state::estimate_rows_per_batch(const duckdb::PhysicalTableScan& op)
{
  assert(num_columns <= op.column_ids.size());

  size_t estimated_row_bytes = 0;
  column_builders.reserve(num_columns);
  for (size_t i = 0; i < num_columns; ++i) {
    auto const col_type = op.returned_types[op.column_ids[i].GetPrimaryIndex()];
    column_builders.emplace_back(col_type, default_varchar_size);
    if (col_type.InternalType() == duckdb::PhysicalType::VARCHAR) {
      varchar_indices.push_back(i);
      estimated_row_bytes += (sizeof(int64_t) + default_varchar_size);  // offset + data + mask
    } else {
      estimated_row_bytes += duckdb::GetTypeIdSize(col_type.InternalType());  // data + mask
    }
  }

  // We must make space for the mask bytes (1 bit per row, rounded up to bytes)
  // Add mask bytes to the estimated row size
  size_t mask_bytes_per_row = num_columns / CHAR_BIT + (num_columns % CHAR_BIT != 0 ? 1 : 0);
  estimated_row_bytes += mask_bytes_per_row;

  // For VARCHAR columns, add space for the extra offset at the end
  size_t extra_varchar_offset_bytes = varchar_indices.size() * sizeof(int64_t);

  // Calculate rows that fit in the batch
  estimated_rows_per_batch =
    (approximate_batch_size - extra_varchar_offset_bytes) / estimated_row_bytes;

  // Ensure at least 1 vector can fit, otherwise the task will be a no-op
  estimated_rows_per_batch = std::max<size_t>(estimated_rows_per_batch, STANDARD_VECTOR_SIZE);
}

void duckdb_scan_task_local_state::initialize_builders()
{
  size_t byte_offset = 0;
  for (size_t i = 0; i < num_columns; ++i) {
    column_builders[i].initialize_accessors(estimated_rows_per_batch, byte_offset, allocation);
    // Update byte_offset for next column
    if (column_builders[i].type.InternalType() == duckdb::PhysicalType::VARCHAR) {
      // VARCHAR column (offsets + data + mask)
      byte_offset += (estimated_rows_per_batch + 1) * sizeof(int64_t) +
                     estimated_rows_per_batch * default_varchar_size +
                     utils::ceil_div_8(estimated_rows_per_batch);
    } else {
      // Fixed-width column (data + mask)
      byte_offset += estimated_rows_per_batch * column_builders[i].type_size +
                     utils::ceil_div_8(estimated_rows_per_batch);
    }
  }
}

void duckdb_scan_task_local_state::initialize_local_table_function_state(
  duckdb::PhysicalTableScan const& op,
  duckdb::ExecutionContext& exec_ctx,
  duckdb::GlobalTableFunctionState* global_tf_state)
{
  // Note: local_tf_state might already be set if it was moved from a previous task
  // Only create a new one if it doesn't exist
  if (!local_tf_state && op.function.init_local) {
    duckdb::TableFunctionInitInput tf_input(
      op.bind_data.get(), op.column_ids, op.projection_ids, nullptr, op.extra_info.sample_options);
    local_tf_state = op.function.init_local(exec_ctx, tf_input, global_tf_state);
  }
}

//===----------------------------------------------------------------------===//
// DuckDB Scan Task
//===----------------------------------------------------------------------===//
bool duckdb_scan_task::get_next_chunk(duckdb_scan_task_local_state& l_state,
                                      duckdb_scan_task_global_state& g_state)
{
  // Reset the chunk before calling the table function to ensure it starts empty
  l_state.chunk.Reset();

  // Lock the global state during table function call to ensure thread-safe access
  {
    std::lock_guard<std::mutex> lock(g_state.scan_mutex);

    duckdb::TableFunctionInput tf_input(
      g_state.op.bind_data.get(), l_state.local_tf_state.get(), g_state.global_tf_state.get());
    g_state.op.function.function(l_state.exec_ctx.client, tf_input, l_state.chunk);
  }
  if (l_state.chunk.size() == 0) {
    g_state.set_source_drained();
    return false;
  }
  return true;
}

bool duckdb_scan_task::chunk_fits(duckdb_scan_task_local_state& l_state)
{
  // Loop over the VARCHAR columns and check if they fit in the allocated buffers
  for (auto varchar_idx : l_state.varchar_indices) {
    auto& vec = l_state.chunk.data[varchar_idx];
    vec.Flatten(l_state.chunk.size());
    auto const& validity = duckdb::FlatVector::Validity(l_state.chunk.data[varchar_idx]);
    if (!l_state.column_builders[varchar_idx].sufficient_space_for_column(
          vec, validity, l_state.chunk.size())) {
      return false;
    }
  }
  return true;
}

void duckdb_scan_task::process_chunk(duckdb_scan_task_local_state& l_state)
{
  for (size_t i = 0; i < l_state.num_columns; ++i) {
    auto& vec = l_state.chunk.data[i];
    vec.Flatten(l_state.chunk.size());
    auto const& validity = duckdb::FlatVector::Validity(vec);
    l_state.column_builders[i].process_column(
      vec, validity, l_state.chunk.size(), l_state.row_offset, l_state.allocation);
  }
  l_state.row_offset += l_state.chunk.size();
}

void duckdb_scan_task::execute()
{
  // Cast base task states to DuckDB scan task states
  auto& l_state = this->_local_state->cast<duckdb_scan_task_local_state>();
  auto& g_state = this->_global_state->cast<duckdb_scan_task_global_state>();

  // Initialize the data chunk
  l_state.chunk.Initialize(duckdb::Allocator::Get(l_state.exec_ctx.client),
                           g_state.op.returned_types);

  // Enter the scan loop to accumulate a data batch
  while (get_next_chunk(l_state, g_state)) {
    // We know a priori that the fixed-width columns and masks will fit in the allocated buffers.
    // For variable-length columns, we need to check that we have enough space.
    // If there isn't enough space, we just throw an exception for now.
    /// FUTURE WORK: push the current data batch into a new scan task.
    if (!chunk_fits(l_state)) {
      std::string err_msg =
        "[duckdb_scan_task]: current chunk does not fit in the allocated buffers.";
      throw std::runtime_error(err_msg);
    }

    // Process the chunk into the column builders
    process_chunk(l_state);

    // Termination condition
    if (STANDARD_VECTOR_SIZE + l_state.row_offset >= l_state.estimated_rows_per_batch) { break; }
  }

  // Add tasks back to the queue if the scan is not finished
  if (!g_state.is_source_drained()) {
    /// FUTURE WORK: we need the task_creator to get the next task id. For now, we just increment
    /// the task id by the number of workers.
    auto const new_task_id = this->task_id + g_state.max_threads;

    // Create a new local state, passing the existing local_tf_state to continue the scan
    // This ensures DuckDB continues scanning from the current position rather than starting over
    auto new_local_state =
      std::make_unique<duckdb_scan_task_local_state>(g_state,
                                                     l_state.exec_ctx,
                                                     l_state.approximate_batch_size,
                                                     l_state.default_varchar_size,
                                                     std::move(l_state.local_tf_state));

    // Create a new reference to the global state
    auto shared_global_state =
      std::static_pointer_cast<duckdb_scan_task_global_state>(this->_global_state);
    auto next_task = std::make_unique<duckdb_scan_task>(
      new_task_id, dr_mgr, std::move(new_local_state), shared_global_state);
    g_state.scan_executor.schedule(std::move(next_task));
  }

  /// FUTURE WORK: Create the data batch and push it to the data repository.

  /// FUTURE WORK: Notify task_creator of completion by sending a message to the message queue
}

}  // namespace sirius::parallel

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
#include "cucascade/memory/memory_space.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/cudf_utils.hpp>

#include <data/data_batch_utils.hpp>
#include <helper/utils.hpp>
#include <log/logging.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <op/scan/duckdb_scan_executor.hpp>
#include <op/scan/duckdb_scan_task.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/memory_reservation.hpp>

// duckdb
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/common/enums/vector_type.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/common/types/vector_buffer.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>

#include <chrono>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Scan instrumentation helpers
//===----------------------------------------------------------------------===//
namespace {

/// @brief Returns the name of a DuckDB VectorType as a string.
const char* vector_type_name(duckdb::VectorType vt)
{
  switch (vt) {
    case duckdb::VectorType::FLAT_VECTOR: return "FLAT";
    case duckdb::VectorType::CONSTANT_VECTOR: return "CONSTANT";
    case duckdb::VectorType::DICTIONARY_VECTOR: return "DICTIONARY";
    case duckdb::VectorType::FSST_VECTOR: return "FSST";
    case duckdb::VectorType::SEQUENCE_VECTOR: return "SEQUENCE";
    default: return "UNKNOWN";
  }
}

/// @brief Simple RAII timer that logs elapsed time in microseconds on destruction.
struct scoped_timer {
  using clock = std::chrono::steady_clock;
  clock::time_point start;
  double& target_us;
  scoped_timer(double& t) : start(clock::now()), target_us(t) {}
  ~scoped_timer()
  {
    target_us +=
      std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();
  }
};

/// @brief Try to estimate average varchar bytes per row for a column using DuckDB storage metadata.
/// Returns 0 if metadata is unavailable (caller should fall back to default).
size_t estimate_varchar_bytes_from_metadata(duckdb::FunctionData* bind_data_ptr,
                                            duckdb::StorageIndex storage_col_idx,
                                            size_t projected_col_index,
                                            duckdb::ClientContext& context)
{
  if (!bind_data_ptr) { return 0; }

  // Cast to TableScanBindData to access the underlying table
  duckdb::TableScanBindData* tsbd = nullptr;
  try {
    tsbd = &bind_data_ptr->Cast<duckdb::TableScanBindData>();
  } catch (...) {
    return 0;  // Not a seq_scan table function
  }

  try {
    auto& table   = tsbd->table.Cast<duckdb::DuckTableEntry>();
    auto& storage = table.GetStorage();
    auto total_rows = storage.GetTotalRows();
    if (total_rows == 0) { return 0; }

    // Read StringStats for this column using the storage column index
    auto stats = storage.GetStatistics(context, storage_col_idx);
    if (stats && duckdb::StringStats::HasMaxStringLength(*stats)) {
      auto max_len      = duckdb::StringStats::MaxStringLength(*stats);
      auto distinct_est = stats->GetDistinctCount();

      // Use max_string_length as the base estimate. For high-cardinality columns
      // (distinct > 10% of rows), the max is likely much larger than avg, so reduce.
      // GetStatistics and GetDistinctCount are cheap (in-memory), no segment walk needed.
      size_t estimated_avg = max_len;
      if (distinct_est > 0 && total_rows > 0) {
        double distinct_ratio = static_cast<double>(distinct_est) / total_rows;
        if (distinct_ratio > 0.1) {
          estimated_avg = std::max<size_t>(max_len * 3 / 4, 1);
        }
      }

      SIRIUS_LOG_INFO(
        "[duckdb_scan] metadata col {} (storage {}): max_string_length={}, distinct_est={}, "
        "total_rows={}, estimated_avg={}",
        projected_col_index, storage_col_idx.GetPrimaryIndex(), max_len, distinct_est,
        total_rows, estimated_avg);

      return estimated_avg;
    }
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG("[duckdb_scan] metadata access failed for col {}: {}", projected_col_index, e.what());
  }
  return 0;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// duckdb_scan_task_global_state
//===----------------------------------------------------------------------===//
duckdb_scan_task_global_state::duckdb_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  pipeline::pipeline_executor& pipeline_exec,
  duckdb::ClientContext& client_ctx,
  sirius_physical_duckdb_scan* scan_op)
  : sirius_pipeline_task_global_state(pipeline),
    _sirius_ctx(client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state").get()),
    _max_threads(pipeline_exec.get_scan_executor().get_num_threads()),
    _pipeline_executor(pipeline_exec),
    _op(*scan_op)
{
  // Initialize global table function state
  // Note: We pass nullptr for table_filters because filters are applied by the Sirius physical
  // table scan operator, not by the DuckDB table function. This ensures consistent filtering
  // behavior and allows GPU-accelerated filter execution.
  if (_op.function.init_global) {
    duckdb::TableFunctionInitInput tf_input(_op.bind_data.get(),
                                            _op.column_ids,
                                            _op.projection_ids,
                                            nullptr,  // Don't pass filters to DuckDB
                                            _op.extra_info.sample_options);
    _global_tf_state = _op.function.init_global(client_ctx, tf_input);
  }

  // We do not support in_out_functions
  if (_op.function.in_out_function) {
    throw duckdb::NotImplementedException(
      "In-out table functions are not supported in sirius table scans.");
  }

  // Dynamic filters (from joins) are not supported. Regular table_filters (from WHERE clauses) are
  // supported.
  if (_op.dynamic_filters) {
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
  size_t estimated_num_rows,
  size_t byte_offset,
  std::unique_ptr<multiple_blocks_allocation>& allocation)
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
    size_t data_byte_offset    = byte_offset + (estimated_num_rows + 1) * sizeof(int32_t);
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
  std::unique_ptr<multiple_blocks_allocation>& allocation)
{
  auto const* src_valid = reinterpret_cast<uint8_t const*>(validity.GetData());
  auto const cur_bit    = utils::mod_8(row_offset);  //< bit offset in current byte
  auto const num_bits   = num_rows;

  if (src_valid == nullptr) {
    //===----------All Valid----------===//
    // Set all bits in the mask to valid
    size_t full_bytes = 0;
    size_t tail_bits  = 0;
    if (cur_bit != 0) {
      //===----------Byte Unaligned Case----------===//
      auto const bits_in_current_byte = std::min<uint32_t>(CHAR_BIT - cur_bit, num_bits);
      auto const remaining_bits =
        num_bits - bits_in_current_byte;  // Remaining bits after filling current byte
      full_bytes = utils::div_8(remaining_bits);
      tail_bits  = utils::mod_8(remaining_bits);

      // Set bits in the current byte
      auto const current_byte_mask =
        static_cast<uint8_t>(utils::make_mask<uint8_t>(bits_in_current_byte) << cur_bit);
      auto const current_byte = mask_blocks_accessor.get_current(allocation);
      mask_blocks_accessor.set_current(current_byte | current_byte_mask, allocation);
      if (bits_in_current_byte + cur_bit == CHAR_BIT) { mask_blocks_accessor.advance(); }
    } else {
      //===----------Byte Aligned Case----------===//
      full_bytes = utils::div_8(num_bits);
      tail_bits  = utils::mod_8(num_bits);
    }
    if (full_bytes != 0) { mask_blocks_accessor.memset(FULL_MASK, full_bytes, allocation); }
    if (tail_bits != 0) {
      auto const tail_mask = utils::make_mask<uint8_t>(tail_bits);
      mask_blocks_accessor.set_current(tail_mask, allocation);
    }
    return;
  }
  // condition: src_valid != nullptr

  // Update the null count
  null_count += num_rows - validity.CountValid(num_rows);

  auto const full_bytes = utils::div_8(num_bits);
  auto const tail_bits  = utils::mod_8(num_bits);
  if (cur_bit == 0) {
    //===----------Byte Aligned Case----------===//
    mask_blocks_accessor.memcpy_from(src_valid, full_bytes, allocation);
    if (tail_bits > 0) {
      auto const tail_mask = utils::make_mask<uint8_t>(tail_bits);
      auto const tail      = src_valid[full_bytes] & tail_mask;
      mask_blocks_accessor.set_current(tail, allocation);
    }
  } else {
    //===----------Byte Unaligned Case----------===//
    auto const cur_shift  = static_cast<uint8_t>(cur_bit);
    auto const upper_mask = utils::make_mask<uint8_t>(cur_shift);
    auto const next_shift = static_cast<uint8_t>(CHAR_BIT - cur_bit);
    auto const lower_mask = utils::make_mask<uint8_t>(next_shift);
    auto current_byte     = mask_blocks_accessor.get_current(allocation);
    for (size_t b = 0; b < full_bytes; ++b) {
      auto const src_byte   = src_valid[b];
      auto const lower_bits = static_cast<uint8_t>((src_byte & lower_mask) << cur_shift);
      mask_blocks_accessor.set_current(current_byte | lower_bits, allocation);
      mask_blocks_accessor.advance();
      current_byte = static_cast<uint8_t>((src_byte >> next_shift) & upper_mask);
    }
    if (tail_bits != 0) {
      auto const tail_mask  = utils::make_mask<uint8_t>(tail_bits);
      auto const src_byte   = src_valid[full_bytes] & tail_mask;
      auto const lower_bits = static_cast<uint8_t>((src_byte & lower_mask) << cur_shift);
      mask_blocks_accessor.set_current(current_byte | lower_bits, allocation);
      if (tail_bits >= next_shift) {
        mask_blocks_accessor.advance();
        auto const upper_bits = static_cast<uint8_t>((src_byte >> next_shift) & upper_mask);
        mask_blocks_accessor.set_current(upper_bits, allocation);
      }
    }
  }
}

void duckdb_scan_task_local_state::column_builder::process_column(
  duckdb::Vector& vec,
  duckdb::ValidityMask const& validity,
  size_t num_rows,
  size_t row_offset,
  std::unique_ptr<multiple_blocks_allocation>& allocation)
{
  // PRECONDITION: Vector must be flattened (for non-dictionary path)
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

void duckdb_scan_task_local_state::column_builder::process_dictionary_varchar(
  duckdb::Vector& vec,
  size_t num_rows,
  size_t row_offset,
  std::unique_ptr<multiple_blocks_allocation>& allocation)
{
  // PRECONDITION: vec is DICTIONARY_VECTOR with VARCHAR child
  auto& sel_vec    = duckdb::DictionaryVector::SelVector(vec);
  auto& child      = duckdb::DictionaryVector::Child(vec);
  auto& child_validity = duckdb::FlatVector::Validity(child);

  // Step 1: Build dictionary lookup (small — fits in L1 cache)
  // The child vector contains the unique string values
  auto const* dict_data = duckdb::FlatVector::GetData<duckdb::string_t>(child);
  auto dict_size_hint   = duckdb::DictionaryVector::DictionarySize(vec);
  size_t dict_count     = dict_size_hint.IsValid() ? dict_size_hint.GetIndex() : 0;

  // If we don't know the dictionary size, fall back to finding the max index
  if (dict_count == 0) {
    for (size_t row = 0; row < num_rows; ++row) {
      auto idx = sel_vec.get_index(row);
      if (idx >= dict_count) { dict_count = idx + 1; }
    }
  }

  // Step 2: Pre-compute dictionary string pointers and lengths (small array, L1-friendly)
  // Using stack allocation for small dictionaries, heap for larger ones
  constexpr size_t STACK_DICT_LIMIT = 256;
  struct dict_entry {
    const char* data;
    uint32_t len;
  };
  dict_entry stack_dict[STACK_DICT_LIMIT];
  std::vector<dict_entry> heap_dict;
  dict_entry* dict = stack_dict;
  if (dict_count > STACK_DICT_LIMIT) {
    heap_dict.resize(dict_count);
    dict = heap_dict.data();
  }

  for (size_t k = 0; k < dict_count; ++k) {
    if (child_validity.RowIsValid(k)) {
      dict[k] = {dict_data[k].GetData(), static_cast<uint32_t>(dict_data[k].GetSize())};
    } else {
      dict[k] = {nullptr, 0};
    }
  }

  // Step 3: Build offsets and copy string data using dictionary lookup
  // The selection vector maps each row to a dictionary entry.
  // The dictionary is small (fits L1), so lookups are fast.
  // Validity for the output is determined by whether the selected dictionary entry is valid.
  size_t data_bytes = 0;
  for (size_t row = 0; row < num_rows; ++row) {
    auto const prev_offset = offset_blocks_accessor.get_current(allocation);
    offset_blocks_accessor.advance();

    auto const dict_idx = sel_vec.get_index(row);
    auto const& entry   = dict[dict_idx];

    if (entry.data != nullptr) {
      offset_blocks_accessor.set_current(prev_offset + entry.len, allocation);
      data_blocks_accessor.memcpy_from(entry.data, entry.len, allocation);
      data_bytes += entry.len;
    } else {
      offset_blocks_accessor.set_current(prev_offset, allocation);
    }
  }
  total_data_bytes += data_bytes;

  // For dictionary vectors, validity comes from the child (dictionary) mapped through selection.
  // We need to build the output validity mask from the dictionary validity + selection vector.
  // Construct a temporary validity mask for the output rows.
  if (!child_validity.AllValid()) {
    // Some dictionary entries are NULL — build per-row validity from selection
    duckdb::ValidityMask output_validity(num_rows);
    for (size_t row = 0; row < num_rows; ++row) {
      auto const dict_idx = sel_vec.get_index(row);
      if (!child_validity.RowIsValid(dict_idx)) {
        output_validity.SetInvalid(row);
        null_count++;
      }
    }
    process_mask_for_column(output_validity, num_rows, row_offset, allocation);
  } else {
    // All dictionary entries are valid — all output rows are valid
    duckdb::ValidityMask all_valid;  // nullptr = all valid
    process_mask_for_column(all_valid, num_rows, row_offset, allocation);
  }
}

cucascade::memory::column_metadata
duckdb_scan_task_local_state::column_builder::make_column_metadata(size_t num_rows) const
{
  using cucascade::memory::column_metadata;

  if (type.InternalType() == duckdb::PhysicalType::VARCHAR) {
    // VARCHAR column: data buffer + offsets child
    column_metadata offsets_child{};
    offsets_child.type_id          = cudf::type_id::INT32;
    offsets_child.num_rows         = static_cast<cudf::size_type>(num_rows + 1);
    offsets_child.null_count       = 0;
    offsets_child.scale            = 0;
    offsets_child.has_null_mask    = false;
    offsets_child.null_mask_offset = 0;
    offsets_child.null_mask_size   = 0;
    offsets_child.has_data         = true;
    offsets_child.data_offset      = offset_blocks_accessor.initial_byte_offset;
    offsets_child.data_size        = (num_rows + 1) * sizeof(int32_t);

    column_metadata col{};
    col.type_id          = cudf::type_id::STRING;
    col.num_rows         = static_cast<cudf::size_type>(num_rows);
    col.null_count       = static_cast<cudf::size_type>(null_count);
    col.scale            = 0;
    col.has_null_mask    = (null_count > 0);
    col.null_mask_offset = mask_blocks_accessor.initial_byte_offset;
    col.null_mask_size   = (null_count > 0) ? cudf::bitmask_allocation_size_bytes(num_rows) : 0;
    col.has_data         = true;
    col.data_offset      = data_blocks_accessor.initial_byte_offset;
    col.data_size        = total_data_bytes;
    col.children.push_back(std::move(offsets_child));
    return col;
  } else {
    // Fixed-width column
    auto cudf_type = duckdb::GetCudfType(type);

    column_metadata col{};
    col.type_id          = cudf_type.id();
    col.num_rows         = static_cast<cudf::size_type>(num_rows);
    col.null_count       = static_cast<cudf::size_type>(null_count);
    col.scale            = cudf_type.scale();
    col.has_null_mask    = (null_count > 0);
    col.null_mask_offset = mask_blocks_accessor.initial_byte_offset;
    col.null_mask_size   = (null_count > 0) ? cudf::bitmask_allocation_size_bytes(num_rows) : 0;
    col.has_data         = true;
    col.data_offset      = data_blocks_accessor.initial_byte_offset;
    col.data_size        = type_size * num_rows;
    return col;
  }
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
  std::unique_ptr<duckdb::LocalTableFunctionState> existing_local_tf_state)
  : _approximate_batch_size(approximate_batch_size),
    _default_varchar_size(default_varchar_size),
    _exec_ctx(exec_ctx)
{
  auto const& op = g_state._op;
  _num_columns   = op.scanned_types.size();

  if (existing_local_tf_state) {
    _local_tf_state = std::move(existing_local_tf_state);
  } else {
    g_state.increment_local_states();
  }

  // Make the memory reservation request
  auto& mem_res_mgr = g_state._sirius_ctx->get_memory_manager();
  _reservation      = mem_res_mgr.request_reservation(_res_request, approximate_batch_size);

  // Make the allocation
  auto& mem_space = _reservation->get_memory_space();
  _host_space     = const_cast<cucascade::memory::memory_space*>(&mem_space);
  auto* allocator =
    mem_space.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (allocator == nullptr) {
    throw std::runtime_error(
      "[duckdb_scan_task_local_state] Failed to get fixed_size_host_memory_resource allocator for "
      "HOST memory space with device id " +
      std::to_string(_reservation->device_id()) + ".");
  }
  _allocation = allocator->allocate_multiple_blocks(approximate_batch_size, _reservation.get());

  // Estimate number of rows per batch
  estimate_rows_per_batch(op);

  // Initialize the column builders
  initialize_builders();

  // Initialize local table function state (will skip if local_tf_state already set)
  initialize_local_table_function_state(op, exec_ctx, g_state._global_tf_state.get());
}

size_t duckdb_scan_task_local_state::get_tail_byte_offset() const
{
  auto const& last_builder = _column_builders.back();
  auto last_byte_offset    = last_builder.mask_blocks_accessor.get_current_global_byte_offset();
  if (utils::mod_8(_row_offset) != 0) {
    last_byte_offset++;  // Round up to next byte if partially filled
  }
  return std::min(last_byte_offset, _allocation->size_bytes());
}

void duckdb_scan_task_local_state::estimate_rows_per_batch(sirius_physical_duckdb_scan const& op)
{
  assert(_num_columns <= op.scanned_types.size());

  size_t estimated_row_bytes = 0;
  _column_builders.reserve(_num_columns);
  for (size_t i = 0; i < _num_columns; ++i) {
    auto const col_type = op.scanned_types[i];
    if (col_type.InternalType() == duckdb::PhysicalType::VARCHAR) {
      _varchar_indices.push_back(i);

      // Use metadata-based max_string_length instead of blind default when available
      auto storage_col_idx = duckdb::StorageIndex(op.column_ids[i].GetPrimaryIndex());
      size_t varchar_size  = estimate_varchar_bytes_from_metadata(
        op.bind_data.get(), storage_col_idx, i, _exec_ctx.client);
      if (varchar_size == 0) { varchar_size = _default_varchar_size; }

      _column_builders.emplace_back(col_type, varchar_size);
      _column_builders.back().has_metadata_bound = (varchar_size != _default_varchar_size);
      estimated_row_bytes += (sizeof(int32_t) + varchar_size);  // offset + data
    } else {
      _column_builders.emplace_back(col_type, _default_varchar_size);
      estimated_row_bytes += duckdb::GetTypeIdSize(col_type.InternalType());  // data
    }
  }

  // We must make space for the mask bytes (1 bit per row, rounded up to bytes)
  // Add mask bytes to the estimated row size
  size_t mask_bytes_per_row = utils::ceil_div_8(_num_columns);
  estimated_row_bytes += mask_bytes_per_row;

  // For VARCHAR columns, add space for the extra offset at the end
  size_t extra_varchar_offset_bytes = _varchar_indices.size() * sizeof(int32_t);

  // Calculate rows that fit in the batch
  _estimated_rows_per_batch =
    (_approximate_batch_size - extra_varchar_offset_bytes) / estimated_row_bytes;

  // Ensure at least 1 vector can fit, otherwise the task will be a no-op
  _estimated_rows_per_batch = std::max<size_t>(_estimated_rows_per_batch, STANDARD_VECTOR_SIZE);

  SIRIUS_LOG_INFO(
    "[duckdb_scan] estimation: {} cols ({} varchar), estimated_row_bytes={}, "
    "estimated_rows_per_batch={}, batch_size={}",
    _num_columns,
    _varchar_indices.size(),
    estimated_row_bytes,
    _estimated_rows_per_batch,
    _approximate_batch_size);
}

void duckdb_scan_task_local_state::initialize_builders()
{
  size_t byte_offset = 0;
  for (size_t i = 0; i < _num_columns; ++i) {
    // Align byte_offset to 8 bytes before each column to ensure proper alignment
    // for offset arrays (int32/int64) and data types
    byte_offset = (byte_offset + 7) & ~size_t{7};
    _column_builders[i].initialize_accessors(_estimated_rows_per_batch, byte_offset, _allocation);
    // Update byte_offset for next column
    if (_column_builders[i].type.InternalType() == duckdb::PhysicalType::VARCHAR) {
      // VARCHAR column (offsets + data + mask)
      // Use per-column type_size from metadata (or default if unavailable)
      byte_offset += (_estimated_rows_per_batch + 1) * sizeof(int32_t) +
                     _estimated_rows_per_batch * _column_builders[i].type_size +
                     utils::ceil_div_8(_estimated_rows_per_batch);
    } else {
      // Fixed-width column (data + mask)
      byte_offset += _estimated_rows_per_batch * _column_builders[i].type_size +
                     utils::ceil_div_8(_estimated_rows_per_batch);
    }
  }
}

void duckdb_scan_task_local_state::initialize_local_table_function_state(
  sirius_physical_duckdb_scan const& op,
  duckdb::ExecutionContext& exec_ctx,
  duckdb::GlobalTableFunctionState* global_tf_state)
{
  // Note: local_tf_state might already be set if it was moved from a previous task
  // Only create a new one if it doesn't exist
  // Don't pass filters to DuckDB - they're applied by Sirius physical table scan
  if (!_local_tf_state && op.function.init_local) {
    duckdb::TableFunctionInitInput tf_input(op.bind_data.get(),
                                            op.column_ids,
                                            op.projection_ids,
                                            nullptr,  // Don't pass filters to DuckDB
                                            op.extra_info.sample_options);
    _local_tf_state = op.function.init_local(exec_ctx, tf_input, global_tf_state);
  }
}

std::shared_ptr<cucascade::data_batch> duckdb_scan_task_local_state::make_data_batch()
{
  using data_batch               = cucascade::data_batch;
  using host_table_allocation    = cucascade::memory::host_table_allocation;
  using host_data_representation = cucascade::host_data_representation;

  // Create column metadata for each column
  std::vector<cucascade::memory::column_metadata> columns;
  columns.reserve(_num_columns);
  for (size_t ci = 0; ci < _column_builders.size(); ci++) {
    auto& builder = _column_builders[ci];
    columns.push_back(builder.make_column_metadata(_row_offset));
  }

  // Make the host table allocation
  auto const sz = get_tail_byte_offset();
  auto table_allocation =
    std::make_unique<host_table_allocation>(std::move(_allocation), std::move(columns), sz);

  // Make the host table representation
  auto table = std::make_unique<host_data_representation>(std::move(table_allocation), _host_space);

  // Create the data batch and return
  return std::make_shared<data_batch>(get_next_batch_id(), std::move(table));
}

//===----------------------------------------------------------------------===//
// DuckDB Scan Task
//===----------------------------------------------------------------------===//
duckdb_scan_task::~duckdb_scan_task()
{
  if (_global_state == nullptr ||
      _global_state->cast<duckdb_scan_task_global_state>().get_pipeline() == nullptr) {
    return;
  }
  _global_state->cast<duckdb_scan_task_global_state>().get_pipeline()->mark_task_completed();
}

bool duckdb_scan_task::get_next_chunk(duckdb_scan_task_local_state& l_state,
                                      duckdb_scan_task_global_state& g_state)
{
  // Reset the chunk before calling the table function to ensure it starts empty
  l_state._chunk.Reset();

  duckdb::TableFunctionInput tf_input(
    g_state._op.bind_data.get(), l_state._local_tf_state.get(), g_state._global_tf_state.get());
  tf_input.async_result = duckdb::AsyncResultType::IMPLICIT;

  g_state._op.function.function(l_state._exec_ctx.client, tf_input, l_state._chunk);

  if (l_state._chunk.size() == 0) {
    if (!l_state._local_state_drained) {
      l_state._local_state_drained = true;
      g_state.decrement_local_states();
    }
    return false;
  }
  return true;
}

bool duckdb_scan_task::chunk_fits(duckdb_scan_task_local_state& l_state)
{
  // Loop over the VARCHAR columns and check if they fit in the allocated buffers.
  // Skip columns where allocation used max_string_length from metadata — those are
  // guaranteed safe upper bounds, so Flatten + size check is redundant.
  for (auto varchar_idx : l_state._varchar_indices) {
    if (l_state._column_builders[varchar_idx].has_metadata_bound) { continue; }
    auto& vec = l_state._chunk.data[varchar_idx];
    vec.Flatten(l_state._chunk.size());
    auto const& validity = duckdb::FlatVector::Validity(l_state._chunk.data[varchar_idx]);
    if (!l_state._column_builders[varchar_idx].sufficient_space_for_column(
          vec, validity, l_state._chunk.size())) {
      return false;
    }
  }
  return true;
}

void duckdb_scan_task::process_chunk(duckdb_scan_task_local_state& l_state)
{
  for (size_t i = 0; i < l_state._num_columns; ++i) {
    auto& vec = l_state._chunk.data[i];
    vec.Flatten(l_state._chunk.size());
    auto const& validity = duckdb::FlatVector::Validity(vec);
    l_state._column_builders[i].process_column(
      vec, validity, l_state._chunk.size(), l_state._row_offset, l_state._allocation);
  }
  l_state._row_offset += l_state._chunk.size();
}

/// Instrumented version of process_chunk that reports Flatten vs copy time separately.
/// Uses dictionary-aware path for DICTIONARY_VECTOR VARCHAR columns.
void duckdb_scan_task::process_chunk_timed(duckdb_scan_task_local_state& l_state,
                                           double& us_flatten,
                                           double& us_copy)
{
  for (size_t i = 0; i < l_state._num_columns; ++i) {
    auto& vec    = l_state._chunk.data[i];
    auto const vt = vec.GetVectorType();
    bool const is_dict_varchar =
      (vt == duckdb::VectorType::DICTIONARY_VECTOR &&
       l_state._column_builders[i].type.InternalType() == duckdb::PhysicalType::VARCHAR);

    if (is_dict_varchar) {
      // Dictionary path: skip Flatten, process dictionary + indices directly
      scoped_timer t(us_copy);
      l_state._column_builders[i].process_dictionary_varchar(
        vec, l_state._chunk.size(), l_state._row_offset, l_state._allocation);
    } else {
      // Standard path: Flatten then copy
      {
        scoped_timer t(us_flatten);
        vec.Flatten(l_state._chunk.size());
      }
      auto const& validity = duckdb::FlatVector::Validity(vec);
      {
        scoped_timer t(us_copy);
        l_state._column_builders[i].process_column(
          vec, validity, l_state._chunk.size(), l_state._row_offset, l_state._allocation);
      }
    }
  }
  l_state._row_offset += l_state._chunk.size();
}

void duckdb_scan_task::execute(rmm::cuda_stream_view stream)
{
  auto estimated_bytes = get_estimated_reservation_size();

  // Record memory metrics for future reservation estimates.
  // Scan tasks don't have peak memory tracking, so use output size as proxy.
  if (auto output_data = compute_task(stream); output_data) {
    std::size_t output_bytes = 0;
    for (const auto& batch : output_data->get_data_batches()) {
      if (batch && batch->get_data()) { output_bytes += batch->get_data()->get_size_in_bytes(); }
    }
    auto& g_state = _global_state->cast<duckdb_scan_task_global_state>();
    // Use the raw task consumption basis from local state when recording history
    auto consumption_basis =
      this->_local_state->cast<duckdb_scan_task_local_state>().get_task_consumption_basis();
    g_state.get_memory_history().record({consumption_basis, output_bytes, output_bytes});

    publish_output(*output_data, stream);
  }
}

std::unique_ptr<op::operator_data> duckdb_scan_task::compute_task(rmm::cuda_stream_view stream)
{
  // Cast base task states to DuckDB scan task states
  auto& l_state = this->_local_state->cast<duckdb_scan_task_local_state>();
  auto& g_state = this->_global_state->cast<duckdb_scan_task_global_state>();

  // === Scan instrumentation counters ===
  double us_get_next_chunk = 0;
  double us_chunk_fits     = 0;
  double us_flatten        = 0;
  double us_copy           = 0;
  size_t num_chunks        = 0;
  // Per-VectorType counters: [FLAT, FSST, CONSTANT, DICTIONARY, SEQUENCE]
  std::array<size_t, 5> vtype_counts = {};
  // Track how many dict vectors we flatten (cost indicator)
  size_t dict_vectors_flattened = 0;
  size_t dict_varchar_vectors_flattened = 0;
  size_t chunk_fits_skipped = 0;

  // Initialize the data chunk with scanned_types (all projected columns, including ROW_ID).
  // This matches the column_ids and projection_ids passed to DuckDB's init functions.
  l_state._chunk.Initialize(duckdb::Allocator::Get(l_state._exec_ctx.client),
                            g_state._op.scanned_types);

  // Enter the scan loop to accumulate a data batch
  bool chunk_available = false;
  {
    scoped_timer t(us_get_next_chunk);
    chunk_available = get_next_chunk(l_state, g_state);
  }
  while (chunk_available) {
    ++num_chunks;

    // Record VectorType distribution before Flatten
    for (size_t i = 0; i < l_state._num_columns; ++i) {
      auto const vt = l_state._chunk.data[i].GetVectorType();
      auto const vt_idx = static_cast<size_t>(vt);
      if (vt_idx < vtype_counts.size()) { vtype_counts[vt_idx]++; }
      if (vt == duckdb::VectorType::DICTIONARY_VECTOR) {
        dict_vectors_flattened++;
        if (l_state._column_builders[i].type.InternalType() == duckdb::PhysicalType::VARCHAR) {
          dict_varchar_vectors_flattened++;
        }
      }
    }

    // For variable-length columns, check that we have enough space.
    // If ALL varchar columns have metadata-based bounds, skip entirely (guaranteed safe).
    // If there isn't enough space, flush the current batch and let the next task continue.
    bool all_have_bounds = true;
    for (auto vi : l_state._varchar_indices) {
      if (!l_state._column_builders[vi].has_metadata_bound) { all_have_bounds = false; break; }
    }
    bool fits = true;
    if (all_have_bounds) {
      chunk_fits_skipped++;
    } else {
      scoped_timer t(us_chunk_fits);
      fits = chunk_fits(l_state);
    }
    if (!fits) {
      if (l_state._row_offset > 0) {
        // We have accumulated data — flush it as a batch and let the next task handle
        // the remaining scan range. The current chunk is lost (a few thousand rows).
        // TODO: carry the unfitting chunk forward to the next task to avoid row loss.
        SIRIUS_LOG_WARN(
          "[duckdb_scan] task={}: varchar overflow after {} rows, flushing batch "
          "(chunk of {} rows does not fit, continuing in next task)",
          _task_id, l_state._row_offset, l_state._chunk.size());
        break;
      }
      // First chunk doesn't fit — allocation is too small even for one chunk.
      // This shouldn't happen with metadata-based estimation, but fall back to error.
      throw std::runtime_error(
        "[duckdb_scan_task]: first chunk does not fit in the allocated buffers. "
        "Consider increasing scan_task_batch_size.");
    }

    // Process the chunk into the column builders (timed: Flatten vs copy)
    process_chunk_timed(l_state, us_flatten, us_copy);

    // Termination condition
    if (STANDARD_VECTOR_SIZE + l_state._row_offset >= l_state._estimated_rows_per_batch) { break; }

    // Fetch next chunk
    {
      scoped_timer t(us_get_next_chunk);
      chunk_available = get_next_chunk(l_state, g_state);
    }
  }

  // === Log scan instrumentation summary ===
  {
    size_t actual_bytes = l_state._row_offset > 0 ? l_state.get_tail_byte_offset() : 0;
    size_t alloc_bytes  = l_state._allocation ? l_state._allocation->size_bytes() : 0;

    // Compute per-column varchar usage
    std::string varchar_detail;
    for (auto idx : l_state._varchar_indices) {
      auto const& cb = l_state._column_builders[idx];
      varchar_detail += fmt::format(" col{}={}/{} bytes", idx, cb.total_data_bytes,
                                    cb.total_data_bytes_allocated);
    }

    SIRIUS_LOG_INFO(
      "[duckdb_scan] task={} rows={}/{} est, chunks={}, alloc={}/{} bytes ({}% used), "
      "vtype=[FLAT={} FSST={} CONST={} DICT={} SEQ={}], "
      "dict_flatten=[total={} varchar={}], fits_skipped={}, "
      "time_us=[get_chunk={:.0f} chunk_fits={:.0f} flatten={:.0f} copy={:.0f} "
      "total_scan={:.0f}]{}",
      _task_id,
      l_state._row_offset,
      l_state._estimated_rows_per_batch,
      num_chunks,
      actual_bytes,
      alloc_bytes,
      alloc_bytes > 0 ? (actual_bytes * 100 / alloc_bytes) : 0,
      vtype_counts[0],
      vtype_counts[1],
      vtype_counts[2],
      vtype_counts[3],
      vtype_counts[4],
      dict_vectors_flattened,
      dict_varchar_vectors_flattened,
      chunk_fits_skipped,
      us_get_next_chunk,
      us_chunk_fits,
      us_flatten,
      us_copy,
      us_get_next_chunk + us_chunk_fits + us_flatten + us_copy,
      varchar_detail);
  }

  // Add tasks back to the queue if the local scan state is not finished
  if (!l_state._local_state_drained) {
    // Create a new local state, passing the existing local_tf_state to continue the scan
    // This ensures DuckDB continues scanning from the current position rather than starting over
    auto new_local_state =
      std::make_unique<duckdb_scan_task_local_state>(g_state,
                                                     l_state._exec_ctx,
                                                     l_state._approximate_batch_size,
                                                     l_state._default_varchar_size,
                                                     std::move(l_state._local_tf_state));

    // Create a new reference to the global state
    auto const new_task_id = g_state._sirius_ctx->get_task_creator().get_next_task_id();
    auto shared_global_state =
      std::static_pointer_cast<duckdb_scan_task_global_state>(this->_global_state);
    auto next_task = std::make_unique<duckdb_scan_task>(
      new_task_id, _data_repo, std::move(new_local_state), shared_global_state);
    g_state._pipeline_executor.schedule(std::move(next_task));
  }

  // Make data batch and push to repository
  if (l_state._row_offset > 0) {
    return std::make_unique<op::operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{l_state.make_data_batch()});
  }

  return std::make_unique<op::operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
}

void duckdb_scan_task::publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream)
{
  std::for_each(std::make_move_iterator(output_data.get_data_batches().begin()),
                std::make_move_iterator(output_data.get_data_batches().end()),
                [this](auto batch) { this->_data_repo->add_data_batch(std::move(batch)); });
}

std::size_t duckdb_scan_task::get_estimated_reservation_size() const
{
  auto current_estimate =
    this->_local_state->cast<duckdb_scan_task_local_state>().get_task_consumption_basis();
  auto& g_state = this->_global_state->cast<duckdb_scan_task_global_state>();
  auto refined  = g_state.get_memory_history().estimate_peak_memory(current_estimate);
  if (refined) { return *refined; }
  return current_estimate;
}

}  // namespace sirius::op::scan

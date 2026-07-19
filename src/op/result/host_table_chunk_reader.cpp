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
#include <helper/type_conversions.hpp>
#include <helper/utils.hpp>
#include <op/result/host_table_chunk_reader.hpp>

// cucascade
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

// duckdb
#include <duckdb/common/types/decimal.hpp>
#include <duckdb/common/vector_operations/vector_operations.hpp>
#include <duckdb/common/vector_size.hpp>
#include <duckdb/main/client_context.hpp>

// standard library
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace sirius::op::result {

host_table_chunk_reader::column_reader::column_reader(
  cucascade::memory::column_metadata const& col,
  std::shared_ptr<multiple_blocks_allocation> const& allocation)
{
  if (allocation == nullptr || allocation->block_size() == 0) {
    throw std::runtime_error(
      "[host_table_chunk_reader::column_reader::column_reader] Invalid allocation.");
  }
  size       = static_cast<size_t>(col.num_rows);
  null_count = static_cast<size_t>(col.null_count);
  // cucascade's column_metadata::type_id is now a generic int32_t tag (PR #150
  // decoupled the core from cudf); cast it back to the cudf::type_id it encodes.
  auto const cudf_type_id = static_cast<cudf::type_id>(col.type_id);
  cudf_col_type =
    col.scale != 0 ? cudf::data_type(cudf_type_id, col.scale) : cudf::data_type(cudf_type_id);
  if (!col.has_null_mask) { null_count = 0; }

  // Nested containers (STRUCT / LIST) carry no flat data buffer of their own.
  if (col.has_data) { data_accessor.initialize(col.data_offset, allocation); }

  if (null_count > 0) { mask_accessor.initialize(col.null_mask_offset, allocation); }

  if (cudf_type_id == cudf::type_id::STRING) {
    if (col.children.size() != 1) {
      throw std::runtime_error(
        "[host_table_chunk_reader::column_reader::initialize_accessors] STRING type must have one "
        "child node for offsets.");
    }
    use_int64_offsets =
      (static_cast<cudf::type_id>(col.children[0].type_id) == cudf::type_id::INT64);
    if (use_int64_offsets) {
      offset_accessor_64.initialize(col.children[0].data_offset, allocation);
    } else {
      offset_accessor_32.initialize(col.children[0].data_offset, allocation);
    }
  } else if (cudf_type_id == cudf::type_id::LIST) {
    // cuDF LIST layout: children[0] = offsets (size_type / INT32 or INT64), children[1] =
    // the value/elements child. A parquet MAP arrives here too, as a LIST whose value child
    // is a STRUCT<key, value>.
    if (col.children.size() != 2) {
      throw std::runtime_error(
        "[host_table_chunk_reader::column_reader::column_reader] LIST/ARRAY type must have two "
        "child nodes (offsets, values).");
    }
    // Offsets child drives the recursive read_into path (variable-length LIST / MAP).
    use_int64_offsets =
      (static_cast<cudf::type_id>(col.children[0].type_id) == cudf::type_id::INT64);
    if (use_int64_offsets) {
      offset_accessor_64.initialize(col.children[0].data_offset, allocation);
    } else {
      offset_accessor_32.initialize(col.children[0].data_offset, allocation);
    }
    // Value child: a recursive reader for read_into, plus data_accessor + null
    // mask when flat fixed-width (the copy_array ARRAY fast path). A nested value
    // child owns no flat buffer, so copy_array never fires for it.
    auto const& values_child = col.children[1];
    if (values_child.has_data) { data_accessor.initialize(values_child.data_offset, allocation); }
    child_null_count =
      values_child.has_null_mask ? static_cast<size_t>(values_child.null_count) : 0;
    if (child_null_count > 0) {
      child_mask_accessor.initialize(values_child.null_mask_offset, allocation);
    }
    children.emplace_back(values_child, allocation);
  } else if (cudf_type_id == cudf::type_id::STRUCT) {
    // cuDF STRUCT layout: one child per field, in order.
    children.reserve(col.children.size());
    for (auto const& field : col.children) {
      children.emplace_back(field, allocation);
    }
  }
}

void host_table_chunk_reader::column_reader::copy_validity_range(
  duckdb::ValidityMask& validity,
  size_t row_offset,
  size_t count,
  std::shared_ptr<multiple_blocks_allocation> const& allocation)
{
  if (row_offset + count > static_cast<size_t>(size)) {
    throw std::out_of_range("host_table_chunk_reader: out of range");
  }
  assert(utils::mod_8(row_offset) == 0);  // Must be byte-aligned start

  // Initialize to all-valid; a column with no copied null mask stays all-valid.
  validity.Initialize(count);
  if (null_count == 0) { return; }

  if (utils::mod_8(row_offset) == 0) {
    // Byte-aligned start: bulk-copy the packed validity bytes.
    mask_accessor.set_cursor(mask_accessor.initial_byte_offset + row_offset / 8);
    auto* validity_ptr       = reinterpret_cast<uint8_t*>(validity.GetData());
    auto const bytes_to_copy = utils::ceil_div_8(count);
    mask_accessor.memcpy_to(allocation, validity_ptr, bytes_to_copy);
  } else {
    // Misaligned start (LIST child slice): read bit-by-bit. Both cuDF and DuckDB
    // pack validity LSB-first, so bit (row_offset+i) maps to row i.
    for (size_t i = 0; i < count; ++i) {
      size_t const bit       = row_offset + i;
      uint8_t const src_byte = mask_accessor.get(bit / 8, allocation);
      if (((static_cast<unsigned>(src_byte) >> (bit % 8)) & 1U) == 0U) { validity.SetInvalid(i); }
    }
  }
}

void host_table_chunk_reader::column_reader::copy_fixed_width(
  duckdb::Vector& vector,
  size_t row_offset,
  size_t count,
  std::shared_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(vector.GetType().InternalType() != duckdb::PhysicalType::VARCHAR);
  if (row_offset + count > static_cast<size_t>(size)) {
    throw std::out_of_range("host_table_chunk_reader: out of range");
  }

  // We are copying into a flat vector
  vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);

  // Do the data copy — the vector's physical type must match the source data element size.
  // Type widening (when cudf type is narrower than DuckDB type) is handled in get_next_chunk()
  // by copying into a temp vector and using DuckDB's cast.
  auto const type_size =
    static_cast<size_t>(duckdb::GetTypeIdSize(vector.GetType().InternalType()));
  auto* dest_ptr = duckdb::FlatVector::GetData(vector);
  data_accessor.memcpy_to(allocation, dest_ptr, count * type_size);

  // Do the validity mask copy, if necessary
  if (null_count != 0) {
    auto& validity = duckdb::FlatVector::Validity(vector);
    copy_validity_range(validity, row_offset, count, allocation);
  }
}

namespace detail {
// Helper template function for constructing duckdb strings from offsets
template <bool HasNulls, typename OffsetType, typename AllocPtr>
void make_duckdb_strings(memory::multiple_blocks_allocation_accessor<OffsetType>& offset_accessor,
                         AllocPtr const& allocation,
                         duckdb::Vector& vector,
                         size_t count,
                         size_t start_offset,
                         size_t end_offset,
                         duckdb::data_ptr_t str_buffer_ptr)
{
  auto* strings = duckdb::FlatVector::GetData<duckdb::string_t>(vector);
  size_t start  = start_offset;
  offset_accessor.advance();
  size_t offset_counter = 0;
  while (offset_counter < count) {
    auto const offsets_in_block =
      std::min(count - offset_counter,
               (allocation->block_size() - offset_accessor.offset_in_block) / sizeof(OffsetType));
    auto* src = reinterpret_cast<OffsetType*>(
      allocation->get_blocks()[offset_accessor.block_index] + offset_accessor.offset_in_block);
    for (size_t i = 0; i < offsets_in_block; ++i) {
      auto const end = static_cast<size_t>(src[i]);
      if constexpr (HasNulls) {
        if (!duckdb::FlatVector::IsNull(vector, offset_counter + i)) {
          auto const d_ptr   = str_buffer_ptr + (start - start_offset);
          auto const str_len = end - start;
          strings[offset_counter + i] =
            duckdb::string_t(reinterpret_cast<char const*>(d_ptr), str_len);
        }
      } else {
        auto const d_ptr   = str_buffer_ptr + (start - start_offset);
        auto const str_len = end - start;
        strings[offset_counter + i] =
          duckdb::string_t(reinterpret_cast<char const*>(d_ptr), str_len);
      }
      start = end;
    }
    offset_counter += offsets_in_block;
    offset_accessor.offset_in_block += offsets_in_block * sizeof(OffsetType);
    if (offset_counter == count) { offset_accessor.offset_in_block -= sizeof(OffsetType); }
    if (offset_accessor.offset_in_block == allocation->block_size()) {
      offset_accessor.block_index++;
      offset_accessor.offset_in_block = 0;
    }
  }
}
}  // namespace detail

// Please see how duckdb converts arrow to duckdb strings for reference:
// https://github.com/duckdb/duckdb/blob/9612b5bea5a6df924daf5ce696d6992df2483bfe/src/function/table/arrow_conversion.cpp#L332
void host_table_chunk_reader::column_reader::copy_string(
  duckdb::Vector& vector,
  size_t row_offset,
  size_t count,
  std::shared_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(vector.GetType().InternalType() == duckdb::PhysicalType::VARCHAR);
  if (row_offset + count > static_cast<size_t>(size)) {
    throw std::out_of_range("host_table_chunk_reader: out of range");
  }

  // We are copying into a flat vector
  vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);

  if (use_int64_offsets) {
    // INT64 offsets (from cudf::pack after GPU roundtrip)
    auto start_offset           = offset_accessor_64.get_current(allocation);
    auto end_offset             = offset_accessor_64.get(row_offset + count, allocation);
    auto const total_data_bytes = end_offset - start_offset;
    auto str_buffer             = duckdb::make_buffer<duckdb::VectorBuffer>(total_data_bytes);
    auto str_buffer_ptr         = str_buffer->GetData();
    data_accessor.memcpy_to(allocation, str_buffer_ptr, total_data_bytes);

    if (null_count != 0) {
      auto& validity = duckdb::FlatVector::Validity(vector);
      copy_validity_range(validity, row_offset, count, allocation);
      detail::make_duckdb_strings<true, int64_t>(
        offset_accessor_64, allocation, vector, count, start_offset, end_offset, str_buffer_ptr);
      duckdb::StringVector::AddBuffer(vector, str_buffer);
    } else {
      detail::make_duckdb_strings<false, int64_t>(
        offset_accessor_64, allocation, vector, count, start_offset, end_offset, str_buffer_ptr);
      duckdb::StringVector::AddBuffer(vector, str_buffer);
    }
  } else {
    // INT32 offsets (from scan task)
    auto start_offset = static_cast<size_t>(offset_accessor_32.get_current(allocation));
    auto end_offset   = static_cast<size_t>(offset_accessor_32.get(row_offset + count, allocation));
    auto const total_data_bytes = end_offset - start_offset;
    auto str_buffer             = duckdb::make_buffer<duckdb::VectorBuffer>(total_data_bytes);
    auto str_buffer_ptr         = str_buffer->GetData();
    data_accessor.memcpy_to(allocation, str_buffer_ptr, total_data_bytes);

    if (null_count != 0) {
      auto& validity = duckdb::FlatVector::Validity(vector);
      copy_validity_range(validity, row_offset, count, allocation);
      detail::make_duckdb_strings<true, int32_t>(
        offset_accessor_32, allocation, vector, count, start_offset, end_offset, str_buffer_ptr);
      duckdb::StringVector::AddBuffer(vector, str_buffer);
    } else {
      detail::make_duckdb_strings<false, int32_t>(
        offset_accessor_32, allocation, vector, count, start_offset, end_offset, str_buffer_ptr);
      duckdb::StringVector::AddBuffer(vector, str_buffer);
    }
  }
}

void host_table_chunk_reader::column_reader::copy_array(
  duckdb::Vector& vector,
  size_t row_offset,
  size_t count,
  std::shared_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(vector.GetType().id() == duckdb::LogicalTypeId::ARRAY);
  if (row_offset + count > static_cast<size_t>(size)) {
    throw std::out_of_range("host_table_chunk_reader: out of range");
  }
  vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);

  auto const array_size = static_cast<size_t>(duckdb::ArrayType::GetSize(vector.GetType()));
  auto& child_vec       = duckdb::ArrayVector::GetEntry(vector);
  child_vec.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
  auto const child_width =
    static_cast<size_t>(duckdb::GetTypeIdSize(child_vec.GetType().InternalType()));
  auto* child_dest = duckdb::FlatVector::GetData(child_vec);
  data_accessor.memcpy_to(allocation, child_dest, count * array_size * child_width);

  // List-level validity
  if (null_count != 0) {
    auto& validity = duckdb::FlatVector::Validity(vector);
    copy_validity_range(validity, row_offset, count, allocation);
  }

  // Element-level validity: the values child has count*array_size elements
  // laid out row-major, so its null mask maps straight onto the array's
  // child vector validity.
  if (child_null_count != 0) {
    auto const child_count = count * array_size;
    assert(utils::mod_8(row_offset * array_size) == 0);  // byte-aligned start
    auto& child_validity = duckdb::FlatVector::Validity(child_vec);
    child_validity.Initialize(child_count);
    auto* child_validity_ptr = reinterpret_cast<uint8_t*>(child_validity.GetData());
    child_mask_accessor.memcpy_to(allocation, child_validity_ptr, utils::ceil_div_8(child_count));
  }
}

// Flat-only overload: the sirius->DuckDB conversion drops nested children.
host_table_chunk_reader::host_table_chunk_reader(
  duckdb::ClientContext& client_ctx,
  cucascade::host_data_representation const& host_table,
  duckdb::vector<sirius::logical_type> const& types_p)
  : host_table_chunk_reader(client_ctx, host_table, sirius::to_duckdb_vec(types_p))
{
}

host_table_chunk_reader::host_table_chunk_reader(
  duckdb::ClientContext& client_ctx,
  cucascade::host_data_representation const& host_table,
  duckdb::vector<duckdb::LogicalType> const& types_p)
  : _client_ctx(client_ctx), _allocation(host_table.get_host_table()->allocation), _types(types_p)
{
  if (!host_table.get_host_table().get()) {
    throw std::runtime_error(
      "[host_table_chunk_reader] get_host_table() is null (unique_ptr not set)");
  }
  if (!_allocation) {
    throw std::runtime_error(
      "[host_table_chunk_reader] host_table allocation is null (cannot read column data)");
  }
  // Access column metadata directly
  auto const& columns = host_table.get_host_table()->columns;
  if (columns.size() != _types.size()) {
    throw std::runtime_error(
      "[host_table_chunk_reader] Metadata column count does not match expected column count.");
  }
  if (_allocation->size_bytes() == 0) {
    if (columns[0].num_rows == 0) {
      // Empty result host table, return without any column readers (creating them would fail).
      // Because _row_offset and _total_rows are 0 by default, get_next_chunk() will immediately
      // return false.
      return;
    } else {
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] host_table has rows but a zero-sized allocation");
    }
  }
  // Initialize column readers
  _column_readers.reserve(columns.size());
  for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx) {
    if (col_idx == 0) {
      if (columns[col_idx].num_rows < 0) {
        throw std::runtime_error("[host_table_chunk_reader] Negative total rows in first column.");
      }
      _total_rows = static_cast<size_t>(columns[col_idx].num_rows);
    } else if (static_cast<size_t>(columns[col_idx].num_rows) != _total_rows) {
      throw std::runtime_error(
        "[host_table_chunk_reader] Metadata column size mismatch across columns.");
    }

    _column_readers.emplace_back(columns[col_idx], _allocation);
  }
}

/// Map a cudf data_type to the DuckDB LogicalType with the same physical storage size.
/// Used to create temp vectors for type-widening casts.
static duckdb::LogicalType cudf_type_to_duckdb(cudf::data_type type)
{
  int const scale   = type.scale();
  int abs_scale_int = scale < 0 ? (scale == INT_MIN ? 255 : -scale) : 0;
  uint8_t abs_scale = static_cast<uint8_t>(std::min(abs_scale_int, 255));
  switch (type.id()) {
    case cudf::type_id::INT8: return duckdb::LogicalType::TINYINT;
    case cudf::type_id::INT16: return duckdb::LogicalType::SMALLINT;
    case cudf::type_id::INT32: return duckdb::LogicalType::INTEGER;
    case cudf::type_id::INT64: return duckdb::LogicalType::BIGINT;
    case cudf::type_id::UINT8: return duckdb::LogicalType::UTINYINT;
    case cudf::type_id::UINT16: return duckdb::LogicalType::USMALLINT;
    case cudf::type_id::UINT32: return duckdb::LogicalType::UINTEGER;
    case cudf::type_id::UINT64: return duckdb::LogicalType::UBIGINT;
    case cudf::type_id::FLOAT32: return duckdb::LogicalType::FLOAT;
    case cudf::type_id::FLOAT64: return duckdb::LogicalType::DOUBLE;
    case cudf::type_id::DECIMAL32:
      return duckdb::LogicalType::DECIMAL(duckdb::Decimal::MAX_WIDTH_INT32, abs_scale);
    case cudf::type_id::DECIMAL64:
      return duckdb::LogicalType::DECIMAL(duckdb::Decimal::MAX_WIDTH_INT64, abs_scale);
    case cudf::type_id::DECIMAL128:
      return duckdb::LogicalType::DECIMAL(duckdb::Decimal::MAX_WIDTH_INT128, abs_scale);
    default: return duckdb::LogicalType::SQLNULL;
  }
}

void host_table_chunk_reader::column_reader::read_into(
  duckdb::ClientContext& client_ctx,
  duckdb::Vector& vector,
  size_t row_offset,
  size_t count,
  std::shared_ptr<multiple_blocks_allocation> const& allocation)
{
  switch (cudf_col_type.id()) {
    case cudf::type_id::STRUCT: {
      // One child vector per field over the same range, then the struct's own
      // validity (null struct != struct with null fields).
      vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
      auto& entries = duckdb::StructVector::GetEntries(vector);
      assert(entries.size() == children.size());
      for (size_t f = 0; f < children.size(); ++f) {
        children[f].read_into(client_ctx, *entries[f], row_offset, count, allocation);
      }
      copy_validity_range(duckdb::FlatVector::Validity(vector), row_offset, count, allocation);
      break;
    }
    case cudf::type_id::LIST: {
      // cuDF stores N+1 offsets; lists in [row_offset, row_offset+count) cover
      // child elements [offsets[row_offset], offsets[row_offset+count]).
      // list_entry_t offsets are relative to the fresh child vector — subtract
      // the slice base. MAP is physically LIST<STRUCT<key,value>>: same path.
      vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
      auto const offset_at = [this, &allocation](size_t idx) -> int64_t {
        return use_int64_offsets ? offset_accessor_64.get(idx, allocation)
                                 : static_cast<int64_t>(offset_accessor_32.get(idx, allocation));
      };
      int64_t const base = offset_at(row_offset);
      auto* list_entries = duckdb::ListVector::GetData(vector);
      for (size_t i = 0; i < count; ++i) {
        int64_t const lo       = offset_at(row_offset + i);
        int64_t const hi       = offset_at(row_offset + i + 1);
        list_entries[i].offset = static_cast<uint64_t>(lo - base);
        list_entries[i].length = static_cast<uint64_t>(hi - lo);
      }
      copy_validity_range(duckdb::FlatVector::Validity(vector), row_offset, count, allocation);

      auto const child_count = static_cast<size_t>(offset_at(row_offset + count) - base);
      duckdb::ListVector::Reserve(vector, child_count);
      duckdb::ListVector::SetListSize(vector, child_count);
      if (child_count > 0) {
        children[0].read_into(client_ctx,
                              duckdb::ListVector::GetEntry(vector),
                              static_cast<size_t>(base),
                              child_count,
                              allocation);
      }
      break;
    }
    case cudf::type_id::STRING: {
      copy_string(vector, row_offset, count, allocation);
      break;
    }
    default: {
      // Fixed-width leaf, possibly type-widened — mirrors get_next_chunk().
      auto const src_type = cudf_type_to_duckdb(cudf_col_type);
      if (src_type.id() == duckdb::LogicalTypeId::SQLNULL ||
          src_type.InternalType() == vector.GetType().InternalType()) {
        copy_fixed_width(vector, row_offset, count, allocation);
      } else {
        duckdb::Vector temp_vec(src_type);
        copy_fixed_width(temp_vec, row_offset, count, allocation);
        duckdb::VectorOperations::Cast(client_ctx, temp_vec, vector, count);
      }
      break;
    }
  }
}

bool host_table_chunk_reader::get_next_chunk(duckdb::DataChunk& chunk)
{
  if (_row_offset >= _total_rows) {
    chunk.SetCardinality(0);
    return false;
  }

  // Initialize the chunk
  auto const remaining = _total_rows - _row_offset;
  auto const count     = std::min(remaining, static_cast<size_t>(STANDARD_VECTOR_SIZE));
  chunk.Initialize(_client_ctx, _types, count);

  // Copy each column into the chunk. Dispatch on actual stored type (metadata), not expected
  // type (_types), so that we never call copy_string on non-string data (e.g. when projection
  // output column order doesn't match the plan and we have int where plan expects string).
  for (size_t col_idx = 0; col_idx < _column_readers.size(); ++col_idx) {
    auto& vec            = chunk.data[col_idx];
    auto const actual_id = _column_readers[col_idx].cudf_col_type.id();
    if (actual_id == cudf::type_id::STRING) {
      // Stored data is string: read with copy_string into a VARCHAR vector, then cast if needed.
      if (vec.GetType().InternalType() == duckdb::PhysicalType::VARCHAR) {
        _column_readers[col_idx].copy_string(vec, _row_offset, count, _allocation);
      } else {
        duckdb::Vector temp_vec(duckdb::LogicalType::VARCHAR);
        _column_readers[col_idx].copy_string(temp_vec, _row_offset, count, _allocation);
        duckdb::VectorOperations::Cast(_client_ctx, temp_vec, vec, count);
      }
    } else if (actual_id == cudf::type_id::STRUCT) {
      // Nested STRUCT top-level column: materialize recursively into the pre-typed vector.
      _column_readers[col_idx].read_into(_client_ctx, vec, _row_offset, count, _allocation);
    } else if (actual_id == cudf::type_id::LIST) {
      // cuDF LIST -> DuckDB fixed-size ARRAY (fast path) or, when variable-length,
      // LIST/MAP materialized recursively.
      if (vec.GetType().id() == duckdb::LogicalTypeId::ARRAY) {
        _column_readers[col_idx].copy_array(vec, _row_offset, count, _allocation);
      } else {
        _column_readers[col_idx].read_into(_client_ctx, vec, _row_offset, count, _allocation);
      }
    } else {
      // Stored data is fixed-width: read with copy_fixed_width, then cast if needed.
      auto src_duckdb_type = cudf_type_to_duckdb(_column_readers[col_idx].cudf_col_type);
      if (src_duckdb_type.id() == duckdb::LogicalTypeId::SQLNULL ||
          src_duckdb_type.InternalType() == vec.GetType().InternalType()) {
        _column_readers[col_idx].copy_fixed_width(vec, _row_offset, count, _allocation);
      } else {
        duckdb::Vector temp_vec(src_duckdb_type);
        _column_readers[col_idx].copy_fixed_width(temp_vec, _row_offset, count, _allocation);
        duckdb::VectorOperations::Cast(_client_ctx, temp_vec, vec, count);
      }
    }
  }

  chunk.SetCardinality(static_cast<std::size_t>(count));
  _row_offset += count;

  return true;
}
}  // namespace sirius::op::result

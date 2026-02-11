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
#include <memory/host_table_utils.hpp>
#include <op/result/host_table_chunk_reader.hpp>

// cucascade
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

// duckdb
#include <duckdb/common/vector_size.hpp>
#include <duckdb/main/client_context.hpp>

// standard library
#include <algorithm>
#include <cstring>
#include <vector>

namespace sirius::op::result {

host_table_chunk_reader::column_reader::column_reader(
  metadata_node const& node, std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  if (allocation == nullptr || allocation->block_size() == 0) {
    throw std::runtime_error(
      "[host_table_chunk_reader::column_reader::column_reader] Invalid allocation.");
  }
  size       = static_cast<size_t>(node.size);
  null_count = static_cast<size_t>(node.null_count);
  cudf_type  = node.type.id();
  if (node.null_mask_offset < 0) { null_count = 0; }

  data_accessor.initialize(static_cast<size_t>(node.data_offset), allocation);

  if (null_count > 0) {
    mask_accessor.initialize(static_cast<size_t>(node.null_mask_offset), allocation);
  }

  if (node.type.id() == cudf::type_id::STRING) {
    if (node.children.size() != 1) {
      throw std::runtime_error(
        "[host_table_chunk_reader::column_reader::initialize_accessors] STRING type must have one "
        "child node for offsets.");
    }
    use_int64_offsets = (node.children[0].type.id() == cudf::type_id::INT64);
    if (use_int64_offsets) {
      offset_accessor_64.initialize(node.children[0].data_offset, allocation);
    } else {
      offset_accessor_32.initialize(node.children[0].data_offset, allocation);
    }
  }
}

void host_table_chunk_reader::column_reader::copy_mask_to_validity(
  duckdb::ValidityMask& validity,
  size_t row_offset,
  size_t count,
  std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(row_offset + count <= static_cast<size_t>(size));
  assert(utils::mod_8(row_offset) == 0);  // Must be byte-aligned start

  // Initialize validity mask
  validity.Initialize(count);

  auto* validity_ptr       = reinterpret_cast<uint8_t*>(validity.GetData());
  auto const bytes_to_copy = utils::ceil_div_8(count);
  mask_accessor.memcpy_to(allocation, validity_ptr, bytes_to_copy);
}

void host_table_chunk_reader::column_reader::copy_fixed_width(
  duckdb::Vector& vector,
  size_t row_offset,
  size_t count,
  std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(vector.GetType().InternalType() != duckdb::PhysicalType::VARCHAR);
  assert(row_offset + count <= static_cast<size_t>(size));

  // We are copying into a flat vector
  vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);

  // Do the data copy
  auto const type_size =
    static_cast<size_t>(duckdb::GetTypeIdSize(vector.GetType().InternalType()));

  // Determine the element size of the source cudf column
  size_t cudf_elem_size = 0;
  switch (cudf_type) {
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8: cudf_elem_size = 1; break;
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16: cudf_elem_size = 2; break;
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::DECIMAL32: cudf_elem_size = 4; break;
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::DECIMAL64: cudf_elem_size = 8; break;
    case cudf::type_id::DECIMAL128: cudf_elem_size = 16; break;
    default: cudf_elem_size = type_size; break;
  }

  auto* dest_ptr = duckdb::FlatVector::GetData(vector);

  if (cudf_elem_size == type_size) {
    // Fast path: sizes match, bulk memcpy
    data_accessor.memcpy_to(allocation, dest_ptr, count * type_size);
  } else if (cudf_elem_size < type_size) {
    // Widening path: cudf produced a narrower type than DuckDB expects
    // (e.g. cudf COUNT → INT32, DuckDB COUNT → BIGINT/INT64,
    //  or cudf SUM → DECIMAL64, DuckDB SUM → DECIMAL(38,x)/INT128)
    // Read source data at native cudf size, then sign-extend each element.
    // This works on little-endian: low bytes are first, high bytes are extended.
    bool is_signed = (cudf_type != cudf::type_id::UINT8 && cudf_type != cudf::type_id::UINT16 &&
                      cudf_type != cudf::type_id::UINT32 && cudf_type != cudf::type_id::UINT64);

    std::vector<uint8_t> temp(count * cudf_elem_size);
    data_accessor.memcpy_to(allocation, temp.data(), count * cudf_elem_size);

    // Zero-fill destination, then copy each element with optional sign extension
    std::memset(dest_ptr, 0, count * type_size);
    for (size_t i = 0; i < count; ++i) {
      auto* src_elem = temp.data() + i * cudf_elem_size;
      auto* dst_elem = dest_ptr + i * type_size;
      std::memcpy(dst_elem, src_elem, cudf_elem_size);
      if (is_signed && (src_elem[cudf_elem_size - 1] & 0x80)) {
        // Sign-extend: fill high bytes with 0xFF for negative values
        std::memset(dst_elem + cudf_elem_size, 0xFF, type_size - cudf_elem_size);
      }
    }
  } else {
    // Narrowing: cudf type is wider than DuckDB type — should not normally happen.
    // Truncate each element to the destination size (little-endian: keep low bytes).
    std::vector<uint8_t> temp(count * cudf_elem_size);
    data_accessor.memcpy_to(allocation, temp.data(), count * cudf_elem_size);
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(dest_ptr + i * type_size, temp.data() + i * cudf_elem_size, type_size);
    }
  }

  // Do the validity mask copy, if necessary
  if (null_count != 0) {
    auto& validity = duckdb::FlatVector::Validity(vector);
    copy_mask_to_validity(validity, row_offset, count, allocation);
  }
}

namespace detail {
// Helper template function for constructing duckdb strings from offsets
template <bool HasNulls, typename OffsetType>
void make_duckdb_strings(
  memory::multiple_blocks_allocation_accessor<OffsetType>& offset_accessor,
  std::unique_ptr<
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation> const&
    allocation,
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
  std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(vector.GetType().InternalType() == duckdb::PhysicalType::VARCHAR);
  assert(row_offset + count <= static_cast<size_t>(size));

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
      copy_mask_to_validity(validity, row_offset, count, allocation);
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
      copy_mask_to_validity(validity, row_offset, count, allocation);
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

host_table_chunk_reader::host_table_chunk_reader(
  duckdb::ClientContext& client_ctx,
  cucascade::host_table_representation const& host_table,
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
  // Unpack metadata
  auto metadata_nodes = sirius::unpack_metadata_to_nodes(host_table.get_host_table()->metadata);
  if (metadata_nodes.size() != _types.size()) {
    throw std::runtime_error(
      "[host_table_chunk_reader] Metadata column count does not match expected column count.");
  }
  if (_allocation->size_bytes() == 0) {
    if (metadata_nodes[0].size == 0) {
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
  _column_readers.reserve(metadata_nodes.size());
  for (size_t col_idx = 0; col_idx < metadata_nodes.size(); ++col_idx) {
    if (col_idx == 0) {
      _total_rows = static_cast<size_t>(metadata_nodes[col_idx].size);
      if (_total_rows < 0) {
        throw std::runtime_error("[host_table_chunk_reader] Negative total rows in first column.");
      }
    } else if (metadata_nodes[col_idx].size != _total_rows) {
      throw std::runtime_error(
        "[host_table_chunk_reader] Metadata column size mismatch across columns.");
    }

    // For the time being, we do not handle HUGEINT, as cudf does not support it
    if (_types[col_idx] == duckdb::LogicalType::HUGEINT) {
      throw std::runtime_error(
        "[host_table_chunk_reader] HUGEINT type is not currently supported.");
    }
    _column_readers.emplace_back(metadata_nodes[col_idx], _allocation);
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

  // Copy each column into the chunk
  for (size_t col_idx = 0; col_idx < _column_readers.size(); ++col_idx) {
    auto& vec = chunk.data[col_idx];
    if (vec.GetType().InternalType() == duckdb::PhysicalType::VARCHAR) {
      _column_readers[col_idx].copy_string(vec, _row_offset, count, _allocation);
    } else {
      _column_readers[col_idx].copy_fixed_width(vec, _row_offset, count, _allocation);
    }
  }

  chunk.SetCardinality(static_cast<duckdb::idx_t>(count));
  _row_offset += count;

  return true;
}
}  // namespace sirius::op::result

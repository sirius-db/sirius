/*
 * Copyright 2026, Sirius Contributors.
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

#include "helper/duckdb_chunk_staging.hpp"

#include "cudf/cudf_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius {

namespace {

rmm::device_buffer to_device(const void* host_data,
                             std::size_t bytes,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  rmm::device_buffer buf(bytes, stream, mr);
  if (bytes > 0) {
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(buf.data(), host_data, bytes, cudaMemcpyHostToDevice, stream.value()));
  }
  return buf;
}

}  // namespace

duckdb_chunk_staging::duckdb_chunk_staging(duckdb::vector<sirius::logical_type> types,
                                           cudf::size_type reserve_rows)
  : _types(std::move(types)), _columns(_types.size())
{
  for (std::size_t c = 0; c < _types.size(); c++) {
    auto& s      = _columns[c];
    s.is_varchar = _types[c].is_varchar();
    if (s.is_varchar) {
      s.offsets.reserve(static_cast<std::size_t>(reserve_rows) + 1);
      s.offsets.push_back(0);
    } else if (reserve_rows > 0) {
      s.fixed_data.reserve(static_cast<std::size_t>(reserve_rows) *
                           _types[c].fixed_width_byte_size());
    }
  }
}

void duckdb_chunk_staging::reset()
{
  for (auto& s : _columns) {
    s.fixed_data.clear();
    s.chars.clear();
    s.offsets.clear();
    if (s.is_varchar) { s.offsets.push_back(0); }
    s.mask_words.clear();
    s.null_count = 0;
  }
  _num_rows = 0;
}

cudf::size_type duckdb_chunk_staging::null_count(std::size_t column) const
{
  if (column >= _columns.size()) {
    throw std::runtime_error("[duckdb_chunk_staging] null_count: column index out of range");
  }
  return _columns[column].null_count;
}

std::optional<std::size_t> duckdb_chunk_staging::first_column_with_nulls() const
{
  for (std::size_t c = 0; c < _columns.size(); c++) {
    if (_columns[c].null_count > 0) { return c; }
  }
  return std::nullopt;
}

void duckdb_chunk_staging::append(duckdb::DataChunk& chunk) { append(chunk, 0, chunk.size()); }

void duckdb_chunk_staging::append(duckdb::DataChunk& chunk,
                                  duckdb::idx_t offset,
                                  duckdb::idx_t count)
{
  if (chunk.ColumnCount() != _columns.size()) {
    throw std::runtime_error("[duckdb_chunk_staging] chunk has " +
                             std::to_string(chunk.ColumnCount()) + " columns, staging has " +
                             std::to_string(_columns.size()));
  }
  if (offset + count > chunk.size()) {
    throw std::runtime_error("[duckdb_chunk_staging] append range exceeds the chunk's row count");
  }
  if (count == 0) { return; }
  // Flatten the whole chunk once, up front: a dictionary or constant vector materializes only
  // the rows a Flatten is asked for, so flattening a slice and then a later slice of the SAME
  // chunk would leave the second slice reading rows that were never written.
  chunk.Flatten();
  if (static_cast<std::uint64_t>(_num_rows) + count >
      static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error("[duckdb_chunk_staging] staged row count exceeds cudf size_type");
  }
  for (std::size_t c = 0; c < _columns.size(); c++) {
    stage_column(_columns[c], chunk.data[c], offset, count, _types[c]);
  }
  _num_rows += static_cast<cudf::size_type>(count);
}

void duckdb_chunk_staging::stage_column(column_staging& s,
                                        duckdb::Vector& vec,
                                        duckdb::idx_t offset,
                                        duckdb::idx_t count,
                                        const sirius::logical_type& type)
{
  auto const& validity = duckdb::FlatVector::Validity(vec);

  // The bitmask grows with the staged rows rather than being sized up front: a writer does not
  // know how many rows a COPY will produce, and an under-sized mask is a silently wrong null.
  auto const needed_words =
    (static_cast<std::size_t>(_num_rows) + count + 31) / 32;  // NOLINT(readability-magic-numbers)
  if (s.mask_words.size() < needed_words) { s.mask_words.resize(needed_words, 0); }
  auto const set_valid = [&s](cudf::size_type row) {
    s.mask_words[row / 32] |= (cudf::bitmask_type{1} << (row % 32));
  };

  if (s.is_varchar) {
    auto* string_data = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
    for (duckdb::idx_t i = 0; i < count; i++) {
      auto const r = offset + i;
      if (validity.RowIsValid(r)) {
        auto const& str = string_data[r];
        if (s.chars.size() + str.GetSize() >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
          throw std::runtime_error(
            "[duckdb_chunk_staging] string column exceeds cudf int32 offset limit");
        }
        s.chars.insert(s.chars.end(), str.GetData(), str.GetData() + str.GetSize());
        set_valid(_num_rows + static_cast<cudf::size_type>(i));
      } else {
        s.null_count++;
      }
      s.offsets.push_back(static_cast<std::int32_t>(s.chars.size()));
    }
    return;
  }

  auto const width = type.fixed_width_byte_size();
  // Untyped GetData: the templated accessor type-checks the vector against T, and no single T
  // matches every fixed-width type staged through here.
  auto const* src = reinterpret_cast<const std::uint8_t*>(duckdb::FlatVector::GetData(vec));
  for (duckdb::idx_t i = 0; i < count; i++) {
    auto const r = offset + i;
    if (validity.RowIsValid(r)) {
      s.fixed_data.insert(s.fixed_data.end(), src + r * width, src + (r + 1) * width);
      set_valid(_num_rows + static_cast<cudf::size_type>(i));
    } else {
      // DuckDB does not zero the backing storage of invalid rows; append zeros instead of the
      // uninitialized bytes so nothing uninitialized ever reaches the GPU (keeps ASAN/MSAN and
      // compute-sanitizer clean).
      s.fixed_data.insert(s.fixed_data.end(), width, std::uint8_t{0});
      s.null_count++;
    }
  }
}

void duckdb_chunk_staging::append_null_row()
{
  for (std::size_t c = 0; c < _columns.size(); c++) {
    auto& s = _columns[c];
    if (s.is_varchar) {
      s.offsets.push_back(static_cast<std::int32_t>(s.chars.size()));
    } else {
      s.fixed_data.insert(s.fixed_data.end(), _types[c].fixed_width_byte_size(), std::uint8_t{0});
    }
    s.null_count++;
  }
  _num_rows++;
}

std::unique_ptr<cudf::table> duckdb_chunk_staging::build(rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(_columns.size());
  for (std::size_t c = 0; c < _columns.size(); c++) {
    auto& s = _columns[c];

    rmm::device_buffer null_mask{};
    if (s.null_count > 0) {
      // cudf reads the mask a word at a time, so it must be allocation-sized rather than only
      // large enough for the rows that happen to carry a null.
      s.mask_words.resize(
        cudf::bitmask_allocation_size_bytes(_num_rows) / sizeof(cudf::bitmask_type), 0);
      null_mask = to_device(
        s.mask_words.data(), s.mask_words.size() * sizeof(cudf::bitmask_type), stream, mr);
    }

    if (s.is_varchar) {
      auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        _num_rows + 1,
        to_device(s.offsets.data(), s.offsets.size() * sizeof(std::int32_t), stream, mr),
        rmm::device_buffer{0, stream, mr},
        0);
      columns.push_back(
        cudf::make_strings_column(_num_rows,
                                  std::move(offsets_col),
                                  to_device(s.chars.data(), s.chars.size(), stream, mr),
                                  s.null_count,
                                  std::move(null_mask)));
      continue;
    }

    columns.push_back(std::make_unique<cudf::column>(
      sirius::get_cudf_type(_types[c]),
      _num_rows,
      to_device(s.fixed_data.data(), s.fixed_data.size(), stream, mr),
      std::move(null_mask),
      s.null_count));
  }
  // The copies above are async against host staging the caller may reset or destroy as soon as
  // this returns.
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius

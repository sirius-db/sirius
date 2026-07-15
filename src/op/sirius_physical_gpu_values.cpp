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

#include "op/sirius_physical_gpu_values.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sirius::op {

namespace {

//===----------------------------------------------------------------------===//
// Host-side column staging
//===----------------------------------------------------------------------===//
/// Host-side staging for one output column, accumulated across DataChunks
/// before a single H2D upload. VALUES/materialized-subquery data is small,
/// so pageable host staging + cudaMemcpyAsync is deliberate — no pinned
/// allocation or IO machinery is warranted here.
struct column_staging {
  std::vector<uint8_t> fixed_data;             // fixed-width payload
  std::vector<int32_t> offsets;                // varchar: num_rows + 1 entries
  std::vector<char> chars;                     // varchar payload
  std::vector<cudf::bitmask_type> mask_words;  // cudf validity bitmask (1 = valid)
  cudf::size_type null_count = 0;
  bool is_varchar            = false;
};

void init_staging(column_staging& s, const sirius::logical_type& type, cudf::size_type total_rows)
{
  s.is_varchar = type.is_varchar();
  s.mask_words.assign(cudf::bitmask_allocation_size_bytes(total_rows) / sizeof(cudf::bitmask_type),
                      0);
  if (s.is_varchar) {
    s.offsets.reserve(static_cast<size_t>(total_rows) + 1);
    s.offsets.push_back(0);
  } else {
    s.fixed_data.reserve(static_cast<size_t>(total_rows) * type.fixed_width_byte_size());
  }
}

void set_row_valid(column_staging& s, cudf::size_type row)
{
  s.mask_words[row / 32] |= (cudf::bitmask_type{1} << (row % 32));
}

/// Append one DataChunk's worth of rows for column @p c into its staging.
/// @p row_base is the number of rows already staged from earlier chunks.
void stage_chunk_column(column_staging& s,
                        duckdb::Vector& vec,
                        duckdb::idx_t chunk_rows,
                        const sirius::logical_type& type,
                        cudf::size_type row_base)
{
  vec.Flatten(chunk_rows);
  auto const& validity = duckdb::FlatVector::Validity(vec);

  if (s.is_varchar) {
    auto* string_data = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
    for (duckdb::idx_t r = 0; r < chunk_rows; r++) {
      if (validity.RowIsValid(r)) {
        auto const& str = string_data[r];
        if (s.chars.size() + str.GetSize() >
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
          throw std::runtime_error(
            "[sirius_physical_gpu_values] string column exceeds cudf int32 offset limit");
        }
        s.chars.insert(s.chars.end(), str.GetData(), str.GetData() + str.GetSize());
        set_row_valid(s, row_base + static_cast<cudf::size_type>(r));
      } else {
        s.null_count++;
      }
      s.offsets.push_back(static_cast<int32_t>(s.chars.size()));
    }
    return;
  }

  auto const width = type.fixed_width_byte_size();
  // Untyped GetData: the templated accessor type-checks the vector against T,
  // and no single T matches every fixed-width type staged through here.
  auto const* src = reinterpret_cast<const uint8_t*>(duckdb::FlatVector::GetData(vec));
  for (duckdb::idx_t r = 0; r < chunk_rows; r++) {
    if (validity.RowIsValid(r)) {
      s.fixed_data.insert(s.fixed_data.end(), src + r * width, src + (r + 1) * width);
      set_row_valid(s, row_base + static_cast<cudf::size_type>(r));
    } else {
      // DuckDB does not zero the backing storage of invalid rows; append
      // zeros instead of the uninitialized bytes so nothing uninitialized
      // ever reaches the GPU (keeps ASAN/MSAN and compute-sanitizer clean).
      s.fixed_data.insert(s.fixed_data.end(), width, uint8_t{0});
      s.null_count++;
    }
  }
}

/// Stage one all-null row for every column (DUMMY_SCAN behavior).
void stage_null_row(column_staging& s, const sirius::logical_type& type)
{
  if (s.is_varchar) {
    s.offsets.push_back(static_cast<int32_t>(s.chars.size()));
  } else {
    s.fixed_data.insert(s.fixed_data.end(), type.fixed_width_byte_size(), uint8_t{0});
  }
  s.null_count++;
}

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

std::unique_ptr<cudf::column> make_device_column(const column_staging& s,
                                                 const sirius::logical_type& type,
                                                 cudf::size_type num_rows,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  rmm::device_buffer null_mask{};
  if (s.null_count > 0) {
    null_mask =
      to_device(s.mask_words.data(), s.mask_words.size() * sizeof(cudf::bitmask_type), stream, mr);
  }

  if (s.is_varchar) {
    auto offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      num_rows + 1,
      to_device(s.offsets.data(), s.offsets.size() * sizeof(int32_t), stream, mr),
      rmm::device_buffer{0, stream, mr},
      0);
    return cudf::make_strings_column(num_rows,
                                     std::move(offsets_col),
                                     to_device(s.chars.data(), s.chars.size(), stream, mr),
                                     s.null_count,
                                     std::move(null_mask));
  }

  return std::make_unique<cudf::column>(
    sirius::get_cudf_type(type),
    num_rows,
    to_device(s.fixed_data.data(), s.fixed_data.size(), stream, mr),
    std::move(null_mask),
    s.null_count);
}

std::unique_ptr<cudf::table> staging_to_table(const std::vector<column_staging>& staging,
                                              const duckdb::vector<sirius::logical_type>& types,
                                              cudf::size_type num_rows,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(staging.size());
  for (size_t c = 0; c < staging.size(); c++) {
    columns.push_back(make_device_column(staging[c], types[c], num_rows, stream, mr));
  }
  // The host staging vectors are destroyed when this call chain returns;
  // the async H2D copies above must complete before then.
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> make_empty_table(const duckdb::vector<sirius::logical_type>& types)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(types.size());
  for (auto const& t : types) {
    columns.push_back(cudf::make_empty_column(sirius::get_cudf_type(t)));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/// cuDF derives table cardinality from its columns, so a positive-row,
/// zero-column DuckDB source needs a private sentinel column. Downstream
/// operators use only num_rows() and never expose the sentinel in their output.
std::unique_ptr<cudf::table> make_row_count_sentinel_table(cudf::size_type num_rows,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT8}, num_rows, cudf::mask_state::ALL_NULL, stream, mr));
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace

//===----------------------------------------------------------------------===//
// sirius_physical_gpu_values
//===----------------------------------------------------------------------===//
sirius_physical_gpu_values::sirius_physical_gpu_values(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_VALUES, std::move(types), estimated_cardinality),
    _collection(std::move(collection))
{
}

sirius_physical_gpu_values::sirius_physical_gpu_values(duckdb::vector<sirius::logical_type> types,
                                                       duckdb::idx_t estimated_cardinality,
                                                       bool produce_single_row)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_VALUES, std::move(types), estimated_cardinality),
    _produce_single_row(produce_single_row)
{
}

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_physical_gpu_values::get_next_task_hint()
{
  // Single-shot: one task converts the whole source. Latch so concurrent
  // scheduling requests from the task creator cannot both see READY.
  bool expected = false;
  if (!_task_scheduled.compare_exchange_strong(expected, true)) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_gpu_values::all_ports_empty()
{
  return _input_handed_out.load(std::memory_order_acquire);
}

std::unique_ptr<operator_data> sirius_physical_gpu_values::get_next_task_input_data()
{
  if (_input_handed_out.exchange(true, std::memory_order_acq_rel)) { return nullptr; }
  return std::make_unique<gpu_values_input>(estimated_source_bytes());
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_physical_gpu_values::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_gpu_values::execute"};

  auto const* values_input = dynamic_cast<const gpu_values_input*>(&input_data);
  if (!values_input) {
    throw std::runtime_error(
      "[sirius_physical_gpu_values::execute] expected input of type gpu_values_input; got " +
      std::string(typeid(input_data).name()));
  }
  auto* mem_space = values_input->get_gpu_memory_space();
  if (!mem_space) {
    throw std::runtime_error(
      "[sirius_physical_gpu_values::execute] no memory space set on task input; "
      "prepare_for_processing was not called");
  }
  auto mr = mem_space->get_default_allocator();

  std::unique_ptr<cudf::table> output_table;
  if (_collection && _collection->Count() > 0) {
    auto const total_rows_idx = _collection->Count();
    if (total_rows_idx > static_cast<duckdb::idx_t>(std::numeric_limits<cudf::size_type>::max())) {
      throw std::runtime_error(
        "[sirius_physical_gpu_values] collection row count exceeds cudf size_type");
    }
    auto const total_rows = static_cast<cudf::size_type>(total_rows_idx);

    if (types.empty()) {
      // Projection pruning can leave a positive-row ColumnDataCollection with
      // zero output columns (for example COUNT(*) over VALUES). Preserve its
      // cardinality with the same private sentinel used by a zero-column
      // DUMMY_SCAN.
      output_table = make_row_count_sentinel_table(total_rows, stream, mr);
    } else {
      std::vector<column_staging> staging(types.size());
      for (size_t c = 0; c < types.size(); c++) {
        init_staging(staging[c], types[c], total_rows);
      }

      duckdb::ColumnDataScanState scan_state;
      _collection->InitializeScan(scan_state);
      duckdb::DataChunk chunk;
      chunk.Initialize(duckdb::Allocator::DefaultAllocator(), _collection->Types());
      cudf::size_type row_base = 0;
      while (_collection->Scan(scan_state, chunk)) {
        if (chunk.ColumnCount() != types.size()) {
          throw std::runtime_error(
            "[sirius_physical_gpu_values] collection chunk column count does not match operator "
            "output types");
        }
        for (size_t c = 0; c < types.size(); c++) {
          stage_chunk_column(staging[c], chunk.data[c], chunk.size(), types[c], row_base);
        }
        row_base += static_cast<cudf::size_type>(chunk.size());
        chunk.Reset();
      }
      if (row_base != total_rows) {
        throw std::runtime_error(
          "[sirius_physical_gpu_values] collection scan row count does not match Count()");
      }
      output_table = staging_to_table(staging, types, total_rows, stream, mr);
    }
  } else if (_produce_single_row) {
    // DUMMY_SCAN: one all-null row. A 0-output-column DUMMY_SCAN (e.g.
    // SELECT 42) gets a TINYINT sentinel column because cudf derives table
    // row count from columns; downstream constant projections read
    // num_rows() and never the sentinel values, and projection output
    // contains only the evaluated expression columns.
    if (types.empty()) {
      output_table = make_row_count_sentinel_table(1, stream, mr);
    } else {
      std::vector<column_staging> staging(types.size());
      for (size_t c = 0; c < types.size(); c++) {
        init_staging(staging[c], types[c], 1);
        stage_null_row(staging[c], types[c]);
      }
      output_table = staging_to_table(staging, types, 1, stream, mr);
    }
  } else {
    // EMPTY_RESULT (or an empty collection): 0-row table with the declared
    // schema, so downstream operators see a real (empty) input — the same
    // shape a filter-everything pipeline produces.
    output_table = make_empty_table(types);
  }

  auto batch =
    sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

//===----------------------------------------------------------------------===//
// Memory estimation
//===----------------------------------------------------------------------===//
std::size_t sirius_physical_gpu_values::estimated_source_bytes() const
{
  if (_collection) { return _collection->SizeInBytes(); }
  // DUMMY_SCAN / EMPTY_RESULT: nominal nonzero basis so the reservation
  // request is well-formed.
  return 64;
}

std::size_t sirius_physical_gpu_values::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // The device table is about the size of the host-side source; 2x covers
  // cudf allocation rounding and the transient offsets/mask buffers. Floor
  // at 1 MiB so tiny sources still get a workable reservation.
  return std::max<std::size_t>(stats.bytes * 2, std::size_t{1} << 20);
}

//===----------------------------------------------------------------------===//
// Plan-time viability gate
//===----------------------------------------------------------------------===//
void sirius_physical_gpu_values::throw_if_unsupported_types(
  const duckdb::vector<sirius::logical_type>& types)
{
  for (auto const& t : types) {
    if (t.id() == sirius::type_id::HUGEINT || t.id() == sirius::type_id::UHUGEINT) {
      throw std::runtime_error(
        "[sirius_physical_gpu_values] HUGEINT/UHUGEINT VALUES data cannot be represented in "
        "cuDF; falling back to DuckDB CPU");
    }
    if (!t.is_varchar()) {
      // Both throw for unsupported types → CPU fallback.
      static_cast<void>(sirius::get_cudf_type(t));
      static_cast<void>(t.fixed_width_byte_size());
    }
  }
}

void sirius_physical_gpu_values::throw_if_collection_too_large(
  const duckdb::ColumnDataCollection& collection, std::size_t max_source_bytes)
{
  auto const rows = collection.Count();
  if (rows > static_cast<duckdb::idx_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error(
      "[sirius_physical_gpu_values] collection row count exceeds cudf size_type; "
      "falling back to DuckDB CPU");
  }

  // A zero-column collection can report no payload bytes even though its
  // cardinality sentinel needs one INT8 value (plus validity) per row on the
  // GPU. Include that private representation in the source-size gate.
  auto const bytes = collection.Types().empty()
                       ? std::max<std::size_t>(collection.SizeInBytes(), rows)
                       : collection.SizeInBytes();
  if (bytes > max_source_bytes) {
    throw std::runtime_error(
      "[sirius_physical_gpu_values] collection size (" + std::to_string(bytes) +
      " bytes) exceeds the single-task GPU_VALUES limit (" + std::to_string(max_source_bytes) +
      " bytes); falling back to DuckDB CPU");
  }
}

}  // namespace sirius::op

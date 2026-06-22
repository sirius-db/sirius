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

#include "op/scan/cpu_source_task.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "helper/utils.hpp"
#include "log/logging.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/host_table.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/common/vector_size.hpp>

#include <cstring>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// cpu_source_task_global_state
//===----------------------------------------------------------------------===//
cpu_source_task_global_state::cpu_source_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline, sirius_physical_cpu_source* source_op)
  : sirius_pipeline_task_global_state(std::move(pipeline)), _op(*source_op)
{
}

//===----------------------------------------------------------------------===//
// cpu_source_task
//===----------------------------------------------------------------------===//
cpu_source_task::cpu_source_task(uint64_t task_id,
                                 cucascade::shared_data_repository* data_repo,
                                 std::unique_ptr<cpu_source_task_local_state> local_state,
                                 std::shared_ptr<cpu_source_task_global_state> global_state)
  : sirius_pipeline_itask(task_id, std::move(local_state), std::move(global_state)),
    _data_repo(data_repo)
{
  if (auto* pipeline = _global_state->cast<cpu_source_task_global_state>().get_pipeline()) {
    pipeline->mark_task_created();
  }
}

cpu_source_task::~cpu_source_task()
{
  if (_global_state == nullptr ||
      _global_state->cast<cpu_source_task_global_state>().get_pipeline() == nullptr) {
    return;
  }
  auto& g_state = _global_state->cast<cpu_source_task_global_state>();
  // If compute_task didn't reach the exhausted=true assignment (threw, was
  // cancelled, etc.), release the scheduling gate so the task creator can
  // produce a replacement. On success the gate stays latched together with
  // exhausted=true, preventing a redundant second task.
  auto& source = g_state.get_source_op();
  if (!source.exhausted.load(std::memory_order_acquire)) {
    source.task_scheduled.store(false, std::memory_order_release);
  }
  g_state.get_pipeline()->mark_task_completed();
}

/// Build a data_batch from a DuckDB DataChunk by copying fixed-width column data
/// and validity masks into a pinned host allocation.
///
/// Layout per column (8-byte aligned):
///   [data: type_size * num_rows] [mask: ceil(num_rows/8) bytes]
///   For VARCHAR: [offsets: (num_rows+1)*4] [data: total_string_bytes] [mask: ceil(num_rows/8)]
static std::shared_ptr<cucascade::data_batch> chunk_to_data_batch(
  duckdb::DataChunk& chunk,
  const duckdb::vector<sirius::logical_type>& types,
  cucascade::memory::memory_space& mem_space,
  cucascade::memory::reservation* reservation)
{
  using host_table_allocation    = cucascade::memory::host_table_allocation;
  using host_data_representation = cucascade::host_data_representation;

  auto num_rows = static_cast<cudf::size_type>(chunk.size());
  auto num_cols = chunk.ColumnCount();

  // Empty-table fast path: zero columns (DUMMY_SCAN) or zero rows produce no
  // host allocation. allocate_multiple_blocks(0, ...) is not guaranteed to
  // return an allocation with a usable block, so skip it entirely.
  if (num_rows == 0 || num_cols == 0) {
    return cucascade::data_batch::make(
      get_next_batch_id(),
      std::make_unique<host_data_representation>(
        host_table_allocation::create(
          nullptr, std::vector<cucascade::memory::column_metadata>{}, 0),
        &mem_space));
  }

  // First pass: calculate total allocation size and flatten vectors. Dispatch
  // on sirius::logical_type predicates rather than re-deriving DuckDB types
  // from chunk.data[c].GetType(); types[] is the canonical shape and avoids
  // a sirius → DuckDB → sirius hop.
  size_t total_size = 0;
  for (size_t c = 0; c < num_cols; c++) {
    chunk.data[c].Flatten(chunk.size());
    total_size = (total_size + 7) & ~size_t{7};  // align to 8 bytes

    if (types[c].is_varchar()) {
      // Offsets array
      total_size += static_cast<size_t>(num_rows + 1) * sizeof(int32_t);
      // Calculate total string data size
      auto* string_data    = duckdb::FlatVector::GetData<duckdb::string_t>(chunk.data[c]);
      auto const& validity = duckdb::FlatVector::Validity(chunk.data[c]);
      size_t string_bytes  = 0;
      for (duckdb::idx_t r = 0; r < chunk.size(); r++) {
        if (validity.RowIsValid(r)) { string_bytes += string_data[r].GetSize(); }
      }
      total_size += string_bytes;
    } else {
      total_size += static_cast<size_t>(num_rows) * types[c].fixed_width_byte_size();
    }
    // Validity mask
    total_size += utils::ceil_div_8(static_cast<size_t>(num_rows));
  }

  if (total_size == 0) {
    return cucascade::data_batch::make(
      get_next_batch_id(),
      std::make_unique<host_data_representation>(
        host_table_allocation::create(
          nullptr, std::vector<cucascade::memory::column_metadata>{}, 0),
        &mem_space));
  }

  // Allocate pinned host memory
  auto* allocator =
    mem_space.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (!allocator) {
    throw std::runtime_error("[cpu_source_task] Failed to get host memory allocator");
  }
  auto allocation = allocator->allocate_multiple_blocks(total_size, reservation);
  auto* base_ptr  = reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]);

  // Second pass: copy data and build metadata
  std::vector<cucascade::memory::column_metadata> columns;
  columns.reserve(num_cols);
  size_t offset = 0;

  for (size_t c = 0; c < num_cols; c++) {
    offset = (offset + 7) & ~size_t{7};  // align

    auto& vec             = chunk.data[c];
    auto const& sirius_t  = types[c];
    auto const& validity  = duckdb::FlatVector::Validity(vec);
    cudf::size_type nulls = 0;

    cucascade::memory::column_metadata col{};
    col.type_id  = sirius::get_cudf_type(sirius_t).id();
    col.num_rows = num_rows;
    col.scale    = 0;
    if (sirius_t.is_decimal()) { col.scale = static_cast<int32_t>(sirius_t.decimal_scale()); }

    if (sirius_t.is_varchar()) {
      // STRING column: offsets child + data
      auto* string_data = duckdb::FlatVector::GetData<duckdb::string_t>(vec);

      // Write offsets
      size_t offsets_offset = offset;
      size_t offsets_size   = static_cast<size_t>(num_rows + 1) * sizeof(int32_t);
      auto* offsets_ptr     = reinterpret_cast<int32_t*>(base_ptr + offsets_offset);
      offset += offsets_size;

      // Write string data
      size_t data_offset = offset;
      int32_t str_pos    = 0;
      for (duckdb::idx_t r = 0; r < chunk.size(); r++) {
        offsets_ptr[r] = str_pos;
        if (validity.RowIsValid(r)) {
          auto str_size = static_cast<int32_t>(string_data[r].GetSize());
          std::memcpy(base_ptr + data_offset + str_pos, string_data[r].GetData(), str_size);
          str_pos += str_size;
        } else {
          nulls++;
        }
      }
      offsets_ptr[num_rows] = str_pos;
      size_t data_size      = static_cast<size_t>(str_pos);
      offset += data_size;

      // Write mask
      size_t mask_offset = offset;
      size_t mask_size   = utils::ceil_div_8(static_cast<size_t>(num_rows));
      auto* mask_ptr     = base_ptr + mask_offset;
      std::memset(mask_ptr, 0xFF, mask_size);
      for (duckdb::idx_t r = 0; r < chunk.size(); r++) {
        if (!validity.RowIsValid(r)) { mask_ptr[r / 8] &= ~(1 << (r % 8)); }
      }
      offset += mask_size;

      col.has_data         = true;
      col.data_offset      = data_offset;
      col.data_size        = data_size;
      col.null_count       = nulls;
      col.has_null_mask    = (nulls > 0);
      col.null_mask_offset = mask_offset;
      col.null_mask_size   = mask_size;

      // Child: offsets
      cucascade::memory::column_metadata offsets_child{};
      offsets_child.type_id       = cudf::type_id::INT32;
      offsets_child.num_rows      = num_rows + 1;
      offsets_child.null_count    = 0;
      offsets_child.has_null_mask = false;
      offsets_child.has_data      = true;
      offsets_child.data_offset   = offsets_offset;
      offsets_child.data_size     = offsets_size;
      col.children.push_back(std::move(offsets_child));

    } else {
      // Fixed-width column
      size_t type_size = sirius_t.fixed_width_byte_size();

      // Write data
      size_t data_offset = offset;
      size_t data_size   = static_cast<size_t>(num_rows) * type_size;
      std::memcpy(base_ptr + data_offset, duckdb::FlatVector::GetData(vec), data_size);
      offset += data_size;

      // Write mask
      size_t mask_offset = offset;
      size_t mask_size   = utils::ceil_div_8(static_cast<size_t>(num_rows));
      auto* mask_ptr     = base_ptr + mask_offset;
      std::memset(mask_ptr, 0xFF, mask_size);
      for (duckdb::idx_t r = 0; r < chunk.size(); r++) {
        if (!validity.RowIsValid(r)) {
          mask_ptr[r / 8] &= ~(1 << (r % 8));
          nulls++;
        }
      }
      offset += mask_size;

      col.has_data         = true;
      col.data_offset      = data_offset;
      col.data_size        = data_size;
      col.null_count       = nulls;
      col.has_null_mask    = (nulls > 0);
      col.null_mask_offset = mask_offset;
      col.null_mask_size   = mask_size;
    }

    columns.push_back(std::move(col));
  }

  auto table_allocation =
    host_table_allocation::create(std::move(allocation), std::move(columns), offset);
  auto table = std::make_unique<host_data_representation>(std::move(table_allocation), &mem_space);
  return cucascade::data_batch::make(get_next_batch_id(), std::move(table));
}

pipeline::reservation_size_info cpu_source_task::get_estimated_reservation_size_info() const
{
  constexpr std::size_t kMinBytes = 64ULL * 1024;
  auto& g_state                   = _global_state->cast<cpu_source_task_global_state>();
  auto& source                    = g_state.get_source_op();
  std::size_t reservation_size    = kMinBytes;
  if (source.collection) {
    auto count = source.collection->Count();
    if (count > 0) {
      reservation_size = std::max(kMinBytes, static_cast<std::size_t>(count) * std::size_t{256});
    }
  }
  pipeline::reservation_size_info info;
  info.reservation_size = reservation_size;
  return info;
}

std::unique_ptr<op::operator_data> cpu_source_task::compute_task(rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"cpu_source_task::compute_task"};

  auto& g_state = _global_state->cast<cpu_source_task_global_state>();
  auto& source  = g_state.get_source_op();

  // The reservation was acquired by duckdb_scan_executor::manager_loop before
  // this task was dispatched. Batches produced below reference memory backed
  // by this reservation, so it must stay alive on the local state until the
  // batches are consumed downstream.
  auto* local = dynamic_cast<cpu_source_task_local_state*>(local_state());
  if (!local) { throw std::runtime_error("[cpu_source_task] Unexpected local state type"); }
  auto* reservation_ptr = local->reservation();
  if (!reservation_ptr) { throw std::runtime_error("[cpu_source_task] No memory reservation set"); }
  auto& mem_space =
    const_cast<cucascade::memory::memory_space&>(reservation_ptr->get_memory_space());

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;

  if (source.collection && source.collection->Count() > 0) {
    // Scan the ColumnDataCollection chunk by chunk. source.types is the
    // operator's sirius-type view of the same schema as
    // source.collection->Types() — we use it directly to avoid a per-batch
    // sirius::from_duckdb_vec hop.
    duckdb::ColumnDataScanState scan_state;
    source.collection->InitializeScan(scan_state);

    duckdb::DataChunk chunk;
    chunk.Initialize(duckdb::Allocator::DefaultAllocator(), source.collection->Types());

    while (source.collection->Scan(scan_state, chunk)) {
      if (chunk.size() == 0) break;
      batches.push_back(chunk_to_data_batch(chunk, source.types, mem_space, reservation_ptr));
      chunk.Reset();
    }
  } else if (source.produce_single_row) {
    // DUMMY_SCAN: produce one row with all-null values. DuckDB's
    // DataChunk::Initialize allocates but does NOT zero the backing storage,
    // and chunk_to_data_batch memcpys that backing into the pinned host
    // allocation. Mark the validity invalid AND zero each fixed-width
    // vector's data bytes so the produced batch never carries uninitialized
    // bytes out to downstream operators (which would trip ASAN/MSAN even
    // though the null mask makes it semantically safe). VARCHAR vectors are
    // handled safely in chunk_to_data_batch: the loop checks validity before
    // reading the string_t payload.
    duckdb::DataChunk chunk;
    // source.types is sirius::logical_type post-#643. DataChunk still speaks
    // DuckDB types — to_duckdb_vec is the unavoidable boundary conversion.
    chunk.Initialize(duckdb::Allocator::DefaultAllocator(), sirius::to_duckdb_vec(source.types));
    chunk.SetCardinality(1);
    for (size_t c = 0; c < chunk.data.size(); c++) {
      auto& vec      = chunk.data[c];
      auto& validity = duckdb::FlatVector::Validity(vec);
      validity.SetAllInvalid(1);
      if (!source.types[c].is_varchar()) {
        auto type_size = source.types[c].fixed_width_byte_size();
        if (type_size > 0) { std::memset(duckdb::FlatVector::GetData(vec), 0, type_size); }
      }
    }
    batches.push_back(chunk_to_data_batch(chunk, source.types, mem_space, reservation_ptr));
  }
  // else: EMPTY_RESULT — produce no data batches

  source.exhausted.store(true);

  SIRIUS_LOG_DEBUG("[cpu_source_task] Produced {} data batches from {} source",
                   batches.size(),
                   source.collection ? "ColumnDataCollection"
                                     : (source.produce_single_row ? "DUMMY_SCAN" : "EMPTY_RESULT"));

  return std::make_unique<op::pipelineable_operator_data>(std::move(batches));
}

void cpu_source_task::publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream)
{
  // DUMMY_SCAN batches are legitimately 0-byte (0 columns, 1 row of "nothing"),
  // so publish every valid batch rather than filtering on size_in_bytes.
  // Downstream operators still need to see the batch to know a row existed.
  auto& pipelineable_output = dynamic_cast<op::pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.get_data_batches()) {
    if (!batch) { continue; }
    {
      auto ro = batch->get_read_only();
      if (!ro->get_data()) { continue; }
    }
    _data_repo->add_data_batch(batch);
  }
}

std::vector<op::sirius_physical_operator*> cpu_source_task::get_output_consumers()
{
  return _global_state->cast<cpu_source_task_global_state>().get_output_consumers();
}

}  // namespace sirius::op::scan

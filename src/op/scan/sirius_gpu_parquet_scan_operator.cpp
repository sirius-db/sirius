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

#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"

#include "log/logging.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>

#include <cucascade/data/data_batch.hpp>

#include <mutex>
#include <stdexcept>

namespace sirius::op::scan {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
sirius_gpu_parquet_scan_operator::sirius_gpu_parquet_scan_operator(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::idx_t estimated_cardinality,
  size_t batch_data_size)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_PARQUET_SCAN, std::move(types), estimated_cardinality),
    _batch_data_size(batch_data_size)
{
}

// ---------------------------------------------------------------------------
// Sink interface (pipeline 1)
// ---------------------------------------------------------------------------
void sirius_gpu_parquet_scan_operator::sink(const operator_data& input_data,
                                            rmm::cuda_stream_view /*stream*/)
{
  auto const* metadata = dynamic_cast<const partitioned_parquet_metadata*>(&input_data);
  if (!metadata) {
    SIRIUS_LOG_WARN(
      "[sirius_gpu_parquet_scan_operator] sink() received unexpected operator_data type "
      "(typeid: {}); ignoring.",
      typeid(input_data).name());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_metadata_mutex);
    _accumulated_metadata.push_back(*metadata);
  }

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_parquet_scan_operator] Received partitioned_parquet_metadata with {} partitions",
    metadata->row_group_partitions.size());
}

// ---------------------------------------------------------------------------
// Pipeline 1 completion signal
// ---------------------------------------------------------------------------
void sirius_gpu_parquet_scan_operator::mark_pipeline1_finished()
{
  _pipeline1_finished.store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Partition list construction
//
// Uses double-checked locking with std::atomic<bool> (_partitions_built) and
// acquire/release semantics to ensure that:
//   1. _pre_built_batches is fully written before _partitions_built is set to true.
//   2. Any thread that observes _partitions_built == true (via acquire load) sees
//      the complete _pre_built_batches vector.
//   3. The partition list is only built after pipeline 1 has finished (all sink()
//      calls completed), preventing premature freezing on a partial metadata set.
// ---------------------------------------------------------------------------
void sirius_gpu_parquet_scan_operator::build_partition_list()
{
  // Do not build until pipeline 1 signals it is done.
  if (!_pipeline1_finished.load(std::memory_order_acquire)) { return; }
  // Fast path: already built.
  if (_partitions_built.load(std::memory_order_acquire)) { return; }

  std::lock_guard<std::mutex> lock(_metadata_mutex);
  // Re-check under lock (double-checked locking).
  if (_partitions_built.load(std::memory_order_relaxed)) { return; }

  _pre_built_batches.clear();

  for (auto const& meta : _accumulated_metadata) {
    size_t batch_uncompressed = 0;
    std::vector<row_group_range> batch_ranges;

    for (auto const& rg : meta.row_group_partitions) {
      // Start a new batch when adding this range would exceed the target size
      // (but always include at least one range per batch so we make progress).
      if (!batch_ranges.empty() &&
          batch_uncompressed + rg.reserved_uncompressed_bytes > _batch_data_size) {
        _pre_built_batches.push_back(pre_built_batch{&meta, std::move(batch_ranges)});
        batch_ranges.clear();
        batch_uncompressed = 0;
      }
      batch_ranges.push_back(rg);
      batch_uncompressed += rg.reserved_uncompressed_bytes;
    }
    if (!batch_ranges.empty()) {
      _pre_built_batches.push_back(pre_built_batch{&meta, std::move(batch_ranges)});
    }
  }

  // Release store: all writes to _pre_built_batches happen-before this store.
  _partitions_built.store(true, std::memory_order_release);

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_parquet_scan_operator] Built {} pre-built scan batches from {} metadata objects",
    _pre_built_batches.size(),
    _accumulated_metadata.size());
}

// ---------------------------------------------------------------------------
// Source interface (pipeline 2)
// ---------------------------------------------------------------------------
size_t sirius_gpu_parquet_scan_operator::get_total_partitions() const
{
  if (_partitions_built.load(std::memory_order_acquire)) { return _pre_built_batches.size(); }
  std::lock_guard<std::mutex> lock(_metadata_mutex);
  size_t total = 0;
  for (auto const& meta : _accumulated_metadata) {
    total += meta.row_group_partitions.size();
  }
  return total;
}

std::optional<task_creation_hint> sirius_gpu_parquet_scan_operator::get_next_task_hint()
{
  build_partition_list();
  if (!_partitions_built.load(std::memory_order_acquire)) { return std::nullopt; }
  if (_next_batch_idx.load(std::memory_order_relaxed) < _pre_built_batches.size()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return std::nullopt;
}

bool sirius_gpu_parquet_scan_operator::all_ports_empty()
{
  build_partition_list();
  if (!_partitions_built.load(std::memory_order_acquire)) { return false; }
  return _next_batch_idx.load(std::memory_order_relaxed) >= _pre_built_batches.size();
}

std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::get_next_task_input_data()
{
  build_partition_list();
  if (!_partitions_built.load(std::memory_order_acquire)) { return nullptr; }

  auto idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _pre_built_batches.size()) { return nullptr; }

  auto const& batch = _pre_built_batches[idx];

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_parquet_scan_operator] Creating parquet_scan_data batch {} with {} row-group "
    "ranges",
    idx,
    batch.scan_ranges.size());

  return std::make_unique<parquet_scan_data>(*batch.metadata, batch.scan_ranges);
}

// ---------------------------------------------------------------------------
// execute() — byte-range preloading
//
// This method contains the execution logic that was previously spread across
// parquet_scan_task::compute_task() and parquet_scan_task_local_state.
//
// NOTE: The method currently throws std::runtime_error because byte-range
// preloading requires a cucascade host-memory reservation which is only
// available inside a dedicated task type (parquet_scan_task or a new
// parquet_gpu_scan_task). Full integration will be added in a follow-up.
// ---------------------------------------------------------------------------
std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::execute(
  const operator_data& input_data, rmm::cuda_stream_view /*stream*/)
{
  auto const* scan = dynamic_cast<const parquet_scan_data*>(&input_data);
  if (!scan) {
    SIRIUS_LOG_ERROR(
      "[sirius_gpu_parquet_scan_operator] execute() called with unexpected operator_data type "
      "(typeid: {}); expected parquet_scan_data.",
      typeid(input_data).name());
    throw std::runtime_error(
      "[sirius_gpu_parquet_scan_operator] execute() called with unexpected operator_data type; "
      "expected parquet_scan_data.");
  }

  // TODO: Implement full byte-range preloading here.
  //
  // The required logic (currently in parquet_scan_task::compute_task()) is:
  //  1. Create a cudf::io::datasource for the file.
  //  2. Build a hybrid_scan_reader from the cached FileMetaData + reader_options.
  //  3. Allocate a cucascade multiple_blocks_allocation for compressed bytes.
  //  4. Read all byte ranges (PAR1 header, column chunks, footer+trailer) into
  //     the allocation asynchronously.
  //  5. Construct a host_parquet_representation and wrap it in a data_batch.
  //
  // This requires a host-memory reservation provided by the enclosing task
  // (analogous to parquet_scan_task_local_state::make_allocation()).  A new
  // task type (parquet_gpu_scan_task) will supply that context.

  SIRIUS_LOG_ERROR(
    "[sirius_gpu_parquet_scan_operator] execute() is not yet fully integrated. "
    "A dedicated parquet_gpu_scan_task type must provide the host-memory reservation context.");
  throw std::runtime_error(
    "[sirius_gpu_parquet_scan_operator] execute() is not yet fully integrated. "
    "A dedicated parquet_gpu_scan_task type must provide the host-memory reservation context.");
}

}  // namespace sirius::op::scan

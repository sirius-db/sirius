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

#pragma once

#include "exec/invocable.hpp"
#include "op/scan/duckdb_insert_delta.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "scan_manager/mvcc_chunk_mask.hpp"
#include "sirius_config.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
class DataTable;
class SingleFileBlockManager;
}  // namespace duckdb

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::exec {
class scoped_dispatcher;
}  // namespace sirius::exec

namespace sirius::io {
class sirius_datasource;
}  // namespace sirius::io

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace sirius::scan_manager {

/**
 * One batch-sized unit of the insert delta: a run of the plan's row groups,
 * their visibility mask, and the staging the transient copies landed in.
 * Per-operator splits are cut from these and share the same storage.
 */
struct insert_delta_bundle {
  std::vector<std::size_t> rg_indices;  ///< indexes into the request plan's row_groups
  std::size_t total_rows{0};
  /// Empty when every covered row is visible; the split then skips the mask
  /// upload and apply entirely.
  mvcc_chunk_mask mask;
  /// Owns the staging bytes the descriptors point into, pinned or pageable.
  std::shared_ptr<void> staging;
  std::uint8_t* slab_base{nullptr};        ///< contiguous staging base (null if none)
  std::vector<std::size_t> rg_slab_base;   ///< parallel to rg_indices
  std::vector<std::size_t> rg_bit_offset;  ///< parallel to rg_indices (mask bits)
  int preferred_device{-1};                ///< round-robin GPU for this bundle
};

/**
 * One pending insert-delta computation for a pinned entry, recorded by the
 * cache-match pass and run before serving starts. Deduped per entry: a
 * self-join queues one request holding the union of the operators' columns.
 * Queued unconditionally even when the entry currently has no delta.
 */
struct insert_delta_job_request {
  duckdb::DataTable* storage{nullptr};
  duckdb::ClientContext* context{nullptr};
  std::size_t n_cache{0};
  /// Union of the requesting operators' storage columns; each operator's
  /// splits select their subset at cut time.
  std::vector<duckdb::storage_t> union_cols;
  std::vector<sirius::logical_type> union_types;  ///< parallel to union_cols
  std::size_t approximate_batch_size{sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE};
  std::string entry_name;

  /// Filled by the job. The plan owns the segment refs the bundles index into.
  op::scan::insert_delta_plan plan;
  std::vector<insert_delta_bundle> bundles;
};

/// Per-bundle fill bookkeeping. Held behind unique_ptr so tasks can keep
/// stable pointers to it.
struct insert_delta_work {
  insert_delta_job_request* request{nullptr};
  std::size_t bundle_index{0};
  std::size_t visible{0};  ///< written by the bundle's single fill task
};

/**
 * Output of prepare_insert_delta_tasks: per-bundle works and the fill tasks
 * ready to dispatch. One task per bundle, so each mask has a single writer
 * and bit offsets need no word alignment.
 */
struct insert_delta_workset {
  std::vector<std::unique_ptr<insert_delta_work>> works;
  std::vector<sirius::exec::invocable<void()>> fill_tasks;
};

/**
 * Build the workset for every pending request: capture, plan bundles, carve
 * staging and masks, build fill tasks. The capture is two-phase — a serial
 * prepare does the ClientContext-touching work (transaction, buffer manager,
 * row-group skeletons), then the column segment-tree walks fan out over
 * @p dispatcher in metadata_parse_chunk()-sized row-group ranges and merge
 * back in row-group order. The fill tasks themselves are not dispatched here.
 *
 * A bundle closes when the next row group would exceed the
 * approximate_batch_size byte budget, push a varchar column past the cudf
 * int32 chars threshold, or leave the cumulative row count off a byte
 * boundary. Staged validity needs byte-aligned row offsets, so a row group
 * with row_count % 8 != 0 always ends its bundle.
 *
 * Each bundle gets one pinned staging block on its GPU's NUMA node when the
 * bytes fit, else pageable memory. Mask words start zeroed; fill tasks set
 * visible bits only.
 *
 * @throws std::runtime_error on capture, validation, or reservation failure;
 *         capture-range worker exceptions propagate through fan_out_and_join.
 */
[[nodiscard]] insert_delta_workset prepare_insert_delta_tasks(
  std::span<insert_delta_job_request> requests,
  exec::scoped_dispatcher& dispatcher,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  sirius::memory::topology_index const& topology,
  std::span<int const> gpu_ids);

/**
 * Drop all-invisible bundles and clear all-visible bundles' masks so they
 * serve unmasked. Call only after every fill task has run.
 */
void finalize_insert_delta_jobs(insert_delta_workset& workset);

/**
 * Compute every pending request's delta bundles: prepare_insert_delta_tasks
 * (which fans the capture's column walks out over the dispatcher itself),
 * fan_out_and_join over the fill tasks, then finalize_insert_delta_jobs.
 * Blocks so serving starts with finished staging and masks.
 *
 * @throws std::runtime_error on failure; the plan-time CPU fallback gate has
 *         already passed, as in run_mvcc_mask_jobs.
 */
void run_insert_delta_jobs(std::span<insert_delta_job_request> requests,
                           exec::scoped_dispatcher& dispatcher,
                           cucascade::memory::memory_reservation_manager& reservation_manager,
                           sirius::memory::topology_index const& topology,
                           std::span<int const> gpu_ids);

/// One per-operator split cut from a bundle: the scan info plus the mask and
/// placement to attach to the outgoing scan_operator_input.
struct insert_delta_split {
  std::unique_ptr<op::scan::duckdb_native_scan_info> info;
  mvcc_chunk_mask mask;
  int preferred_device{-1};
};

/**
 * Cut one operator's splits from @p request's finished bundles.
 *
 * Columns are emitted in @p op_projected_cols order, resolved into the
 * request's union by storage index. Rowid projections throw; the pin declines
 * those at plan time. Splits share the bundles' staging and mask storage.
 * @p datasource and @p block_manager serve the persistent segments' file
 * reads.
 */
std::vector<insert_delta_split> cut_delta_splits_for_op(
  insert_delta_job_request const& request,
  std::span<op::scan::projected_column const> op_projected_cols,
  std::shared_ptr<sirius::io::sirius_datasource> datasource,
  duckdb::SingleFileBlockManager const* block_manager);

}  // namespace sirius::scan_manager

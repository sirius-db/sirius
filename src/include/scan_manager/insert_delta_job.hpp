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

#include "op/scan/duckdb_insert_delta.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "scan_manager/mvcc_chunk_mask.hpp"
#include "sirius_config.hpp"

#include <absl/functional/any_invocable.h>

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
 * @brief One finished, batch-sized unit of the insert delta: a run of the
 *        plan's row groups, its visibility mask, and the pinned staging the
 *        transient copies landed in. Per-operator splits are cut from these
 *        (sharing the staging and mask storage) at provider handoff.
 */
struct insert_delta_bundle {
  std::vector<std::size_t> rg_indices;  ///< indexes into the request plan's row_groups
  std::size_t total_rows{0};
  /// Empty (default) when every covered row is visible — the split then skips
  /// the mask upload + apply entirely.
  mvcc_chunk_mask mask;
  /// Owner of the transient staging bytes descriptors point into (pinned
  /// {reservation, blocks} bundle or the pageable fallback).
  std::shared_ptr<void> staging;
  std::uint8_t* slab_base{nullptr};        ///< contiguous staging base (null if none)
  std::vector<std::size_t> rg_slab_base;   ///< parallel to rg_indices
  std::vector<std::size_t> rg_bit_offset;  ///< parallel to rg_indices (mask bits)
  int preferred_device{-1};                ///< round-robin GPU for this bundle
};

/**
 * @brief One pending per-pinned-entry insert-delta computation, recorded by
 *        the cache-match pass (deduped by entry — a self-join queues ONE
 *        request, accumulating the union of the operators' columns) and
 *        executed before serving starts.
 */
struct insert_delta_job_request {
  duckdb::DataTable* storage{nullptr};
  duckdb::ClientContext* context{nullptr};
  std::size_t n_cache{0};
  /// Union of the requesting operators' storage columns (superset staging;
  /// per-operator splits select their subset at cut time).
  std::vector<duckdb::storage_t> union_cols;
  std::vector<sirius::logical_type> union_types;  ///< parallel to union_cols
  std::size_t approximate_batch_size{sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE};
  std::string entry_name;

  /// OUT — filled by the job: the capture (owns segment refs the bundles
  /// index into) and the finished bundles.
  op::scan::insert_delta_plan plan;
  std::vector<insert_delta_bundle> bundles;
};

/// Internal per-bundle fill bookkeeping; address-stable behind unique_ptr so
/// tasks can hold pointers.
struct insert_delta_work {
  insert_delta_job_request* request{nullptr};
  std::size_t bundle_index{0};
  std::size_t visible{0};  ///< written by the bundle's single fill task
};

/**
 * @brief Everything prepare stages and the later steps consume: per-bundle
 *        works and the ready-to-dispatch fill tasks (one per bundle — a
 *        bundle's mask is single-writer, so its bit offsets need no word
 *        alignment).
 */
struct insert_delta_workset {
  std::vector<std::unique_ptr<insert_delta_work>> works;
  std::vector<absl::AnyInvocable<void()>> fill_tasks;
};

/**
 * @brief Stage every pending request's delta work, dispatch-free: serial
 *        captures (prepare thread — ClientContext discipline), bundle
 *        planning, staging + mask carve, and fill-task construction.
 *
 * Bundle planning walks the captured row groups in order and closes a bundle
 * when adding the next row group would (a) exceed approximate_batch_size by
 * decoded-byte budget, (b) push any varchar column past the cudf int32 chars
 * threshold, or (c) leave the bundle's cumulative row count off a byte
 * boundary (row_count % 8 != 0) — staged validity runs require byte-aligned
 * row offsets, so such a row group always ends its bundle.
 *
 * Staging + mask storage per bundle: one pinned staging block when the
 * combined bytes fit (reservation-first, on the bundle's preferred GPU's NUMA
 * node), else a pageable fallback (only the async-DMA benefit is lost). Mask
 * words are zero-initialized; fill tasks set visible bits only.
 *
 * @throws std::runtime_error on capture/validation or reservation failures.
 */
[[nodiscard]] insert_delta_workset prepare_insert_delta_tasks(
  std::span<insert_delta_job_request> requests,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  sirius::memory::topology_index const& topology,
  std::span<int const> gpu_ids);

/**
 * @brief Drop all-invisible bundles and reset all-visible bundles' masks to
 *        the default (unmasked fast path). Call only after every fill task
 *        ran.
 */
void finalize_insert_delta_jobs(insert_delta_workset& workset);

/**
 * @brief Compute every pending request's delta bundles; blocks in prepare so
 *        serving starts with finished staging and masks. The composition:
 *        @ref prepare_insert_delta_tasks → fan_out_and_join →
 *        @ref finalize_insert_delta_jobs.
 *
 * @throws std::runtime_error — loud by design (past the plan-time CPU
 *         fallback gate; see run_mvcc_mask_jobs for the rationale).
 */
void run_insert_delta_jobs(std::span<insert_delta_job_request> requests,
                           exec::scoped_dispatcher& dispatcher,
                           cucascade::memory::memory_reservation_manager& reservation_manager,
                           sirius::memory::topology_index const& topology,
                           std::span<int const> gpu_ids);

/// One per-operator split cut from a bundle: the scan_info (metadata-flavor,
/// decoded through the existing file/host lanes) plus the mask and placement
/// the drain attaches to the outgoing scan_operator_input.
struct insert_delta_split {
  std::unique_ptr<op::scan::duckdb_native_scan_info> info;
  mvcc_chunk_mask mask;
  int preferred_device{-1};
};

/**
 * @brief Cut one operator's splits from @p request's finished bundles.
 *
 * Columns are emitted in @p op_projected_cols order (each resolved into the
 * request's union by storage index — a rowid projection throws, the pin
 * declines those at plan time). Splits share the bundles' staging and mask
 * storage via their owning pointers; @p datasource / @p block_manager feed
 * the persistent segments' file reads and prefetch hints.
 */
std::vector<insert_delta_split> cut_delta_splits_for_op(
  insert_delta_job_request const& request,
  std::span<op::scan::projected_column const> op_projected_cols,
  std::shared_ptr<sirius::io::sirius_datasource> datasource,
  duckdb::SingleFileBlockManager const* block_manager);

}  // namespace sirius::scan_manager

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
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "scan_manager/duckdb_mvcc_metadata.hpp"
#include "scan_manager/mvcc_chunk_mask.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace duckdb {
class ClientContext;
class DataTable;
}  // namespace duckdb

namespace cucascade::memory {
class memory_reservation_manager;
class memory_space;
}  // namespace cucascade::memory

namespace sirius::exec {
class scoped_dispatcher;
}  // namespace sirius::exec

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace sirius::scan_manager {

/**
 * @brief Run @p tasks on @p dispatcher and block until every one has run.
 *        Generic drop-safe fan-out/join for prepare-time work units.
 *
 * Each task's completion_controller slot is acquired BEFORE its enqueue and
 * moved into the task lambda: a stopping dispatcher silently drops enqueues
 * (and skips already-queued lambdas), destroying the lambda and releasing the
 * slot — the join still fires and the completed-count check turns the drop
 * into a loud error instead of a deadlock. Task exceptions are captured (the
 * dispatcher swallows them) and the FIRST one is rethrown after the join.
 *
 * @throws whatever the first failing task threw; std::runtime_error when
 *         fewer than tasks.size() tasks ran (dispatcher stopped mid-fan-out).
 */
void fan_out_and_join(exec::scoped_dispatcher& dispatcher,
                      std::vector<sirius::exec::invocable<void()>> tasks,
                      std::string_view label);

/**
 * @brief One pending per-pinned-entry mask computation, recorded by the
 *        cache-match pass on a duckdb+mvcc cache hit (deduped by entry — a
 *        self-join queues ONE request) and executed by run_mvcc_mask_jobs
 *        before serving starts.
 *
 * Holds COPIES (metadata) and manager-owned pointers only: memory spaces are
 * owned by the memory manager and outlive pinned entries, so @ref
 * chunk_spaces adds no unpin-lifetime exposure; @ref storage / @ref context
 * come from the query-side ingestible and live for the query.
 */
struct mvcc_mask_job_request {
  /// Owned by the request while the job fills it in place; slot i masks
  /// chunk i. After run_mvcc_mask_jobs each cached provider of this entry
  /// takes its own copy (word storage is shared via the masks' owners).
  mvcc_chunk_mask_set masks;
  duckdb_mvcc_metadata metadata;  ///< copy of the entry's v_base + per-chunk counts
  duckdb::DataTable* storage{nullptr};
  duckdb::ClientContext* context{nullptr};
  /// Per-chunk memory space (GPU tier: the chunk's device space; HOST tier:
  /// the entry's host space) — used only to derive each chunk's NUMA
  /// preference for the consolidated pinned-mask reservations.
  std::vector<cucascade::memory::memory_space*> chunk_spaces;
  /// The pinned-entry key: dedups requests within a query and routes each
  /// completed set back to its entry's providers (also names diagnostics).
  std::string entry_name;
};

/**
 * @brief One dirty chunk's fill bookkeeping: the set slot whose carved mask
 *        the fill tasks write (prepare installed the words there), the
 *        row-group slices that fill it, and the all-visible gate finalize
 *        reads.
 *
 * Address-stable — fill tasks hold pointers — so the workset stores these
 * behind unique_ptr (the atomic makes the type immovable anyway).
 */
struct mvcc_mask_work {
  mvcc_chunk_mask_set* masks{nullptr};  ///< the request's set; slot chunk_index holds the words
  std::size_t chunk_index{0};
  std::span<op::scan::mvcc_row_group_slice const> slices;  ///< into the workset's plans
  duckdb::TransactionData transaction{duckdb::TransactionData::Committed()};
  std::atomic<bool> any_deleted{false};  ///< set by fill tasks; finalize resets clean slots
};

/**
 * @brief Everything prepare_mvcc_mask_tasks stages and the later steps
 *        consume: the captured visibility plans (which OWN the row-group
 *        slices the works reference), one work per dirty chunk, and the
 *        ready-to-dispatch fill tasks. Empty @ref fill_tasks = nothing dirty
 *        (every mask slot stays default / all rows visible).
 */
struct mvcc_mask_workset {
  std::vector<op::scan::mvcc_visibility_plan> plans;
  std::vector<std::unique_ptr<mvcc_mask_work>> works;
  std::vector<sirius::exec::invocable<void()>> fill_tasks;
};

/**
 * @brief Stage every pending request's mask work, dispatch-free: serial
 *        captures (prepare thread — ClientContext discipline) with the
 *        all-clean short-circuit, reservation-first pinned storage, and fill
 *        task construction. Each dirty chunk's carved (still-unfilled) mask
 *        is installed into its request's set here; clean chunks keep their
 *        default slots.
 *
 * Pinned mask storage is acquired like the decoder's staging path: dirty
 * chunks group by host NUMA node, one consolidated reservation + multi-block
 * allocation per node, per-chunk words carved within block boundaries and
 * installed through the aliasing ctor — each mask's words pointer IS the
 * {reservation, blocks} bundle owner (a mask larger than one block falls
 * back to a pageable vector — blocks are not virtually contiguous — losing
 * only the async-DMA benefit). One fill task per <= metadata_parse_chunk()
 * row groups; a task never spans chunks (the lock-free bit-write invariant).
 *
 * @throws std::runtime_error on capture/validation or reservation failures.
 */
[[nodiscard]] mvcc_mask_workset prepare_mvcc_mask_tasks(
  std::span<mvcc_mask_job_request> requests,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  sirius::memory::topology_index const& topology);

/**
 * @brief Reset every work whose fill dropped no rows back to the default
 *        (all-visible) slot, releasing that mask's storage ref; masks whose
 *        fill dropped rows are already in place. Call only after every fill
 *        task ran.
 */
void finalize_mvcc_masks(mvcc_mask_workset& workset);

/**
 * @brief Compute every pending request's keep-masks; blocks in prepare so
 *        serving starts with finished plain buffers.
 *
 * The composition the scan manager calls: @ref prepare_mvcc_mask_tasks →
 * @ref fan_out_and_join → @ref finalize_mvcc_masks. The stages are exposed
 * above so tests (and gdb) can drive and observe them separately.
 *
 * @throws std::runtime_error on capture/validation failures, reservation
 *         failure, or dropped tasks — loud by design: this runs after the
 *         plan-time CPU-fallback gate, where the alternatives are all
 *         silent-wrong-data. A throw can leave carved-but-unfilled masks in
 *         the request sets; nothing may serve from them — prepare_for_query
 *         sequences provider handoff after this returns and drops the
 *         pending requests on error.
 */
void run_mvcc_mask_jobs(std::span<mvcc_mask_job_request> requests,
                        exec::scoped_dispatcher& dispatcher,
                        cucascade::memory::memory_reservation_manager& reservation_manager,
                        sirius::memory::topology_index const& topology);

}  // namespace sirius::scan_manager

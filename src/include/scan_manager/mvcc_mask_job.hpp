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

#include "scan_manager/duckdb_mvcc_metadata.hpp"
#include "scan_manager/mvcc_chunk_mask.hpp"

#include <absl/functional/any_invocable.h>

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
                      std::vector<absl::AnyInvocable<void()>> tasks,
                      std::string_view label);

/**
 * @brief One pending per-(scan-op, pinned-entry) mask computation, recorded
 *        by try_assign_cached_entries on a duckdb+mvcc cache hit and executed
 *        by run_mvcc_mask_jobs before serving starts.
 *
 * Holds COPIES (metadata) and manager-owned pointers only: memory spaces are
 * owned by the memory manager and outlive pinned entries, so @ref
 * chunk_spaces adds no unpin-lifetime exposure; @ref storage / @ref context
 * come from the query-side ingestible and live for the query.
 */
struct mvcc_mask_job_request {
  /// Shared with the cached provider (as const); slot i masks chunk i.
  std::shared_ptr<mvcc_chunk_mask_set> masks;
  duckdb_mvcc_metadata metadata;  ///< copy of the entry's v_base + per-chunk counts
  duckdb::DataTable* storage{nullptr};
  duckdb::ClientContext* context{nullptr};
  /// Per-chunk memory space (GPU tier: the chunk's device space; HOST tier:
  /// the entry's host space) — used only to derive each chunk's NUMA
  /// preference for the consolidated pinned-mask reservations.
  std::vector<cucascade::memory::memory_space*> chunk_spaces;
  std::string entry_name;  ///< diagnostics only
};

/**
 * @brief Compute every pending request's keep-masks; blocks in prepare so
 *        serving starts with finished plain buffers.
 *
 * Serial capture per request (prepare thread — ClientContext discipline);
 * zero version state anywhere returns immediately with every slot null.
 * Pinned mask storage is acquired reservation-first like the decoder's
 * staging path: dirty chunks group by host NUMA node, one consolidated
 * reservation + multi-block allocation per node, per-chunk word spans carved
 * within block boundaries, the {reservation, blocks} bundle retained by the
 * published masks (a mask larger than one block falls back to pageable
 * memory — blocks are not virtually contiguous — losing only the async-DMA
 * benefit). Fill fans out one task per <= metadata_parse_chunk() row groups
 * (a task never spans chunks — the lock-free bit-write invariant); chunks
 * that dropped rows publish their mask, all-visible chunks stay null.
 *
 * @throws std::runtime_error on capture/validation failures, reservation
 *         failure, or dropped tasks — loud by design: this runs after the
 *         plan-time CPU-fallback gate, where the alternatives are all
 *         silent-wrong-data.
 */
void run_mvcc_mask_jobs(std::span<mvcc_mask_job_request> requests,
                        exec::scoped_dispatcher& dispatcher,
                        cucascade::memory::memory_reservation_manager& reservation_manager,
                        sirius::memory::topology_index const& topology);

}  // namespace sirius::scan_manager

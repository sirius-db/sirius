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
 * @brief Run @p tasks on @p dispatcher and block until every one has run —
 *        the drop-safe fan-out/join the prepare-time MVCC jobs use (#819;
 *        generic so the PR4 insert-delta job reuses it with a different
 *        payload).
 *
 * Each task's completion_controller slot is acquired BEFORE its enqueue and
 * moved into the task lambda: `scoped_dispatcher::enqueue` after
 * `request_stop` is a silent no-op that destroys the lambda, which releases
 * the slot — so the join still fires and the completed-count check below
 * turns the drop into a loud error instead of a deadlock (the same applies to
 * lambdas the dispatcher skips after a stop). Task exceptions are captured
 * (the dispatcher itself swallows them) and the FIRST one is rethrown here
 * after the join.
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
 * @brief Compute every pending request's keep-masks; block-in-prepare (#819).
 *
 * (1) SERIAL capture per request (prepare thread — ClientContext
 * discipline); requests with zero version state anywhere return immediately
 * (the clean-table common case costs one non-loading row-group scan, no
 * reservation, no tasks). (2) Pinned mask acquisition, reservation-first
 * exactly like the decoder's staging path: dirty chunks group by host NUMA
 * node (chunk device id + topology), ONE consolidated
 * request_reservation(any_memory_space_in_tier_with_preference{HOST, node})
 * + allocate_multiple_blocks per node, per-chunk word spans carved 64-byte
 * aligned within block boundaries; the {reservation, blocks} bundle becomes
 * the masks' shared retention. (3) fan_out_and_join over one task per
 * ≤ metadata_parse_chunk() row groups (a task never spans chunks — the
 * bit-packed masks' lock-free write invariant). (4) Publish: chunks that
 * dropped rows get their mask slot set; all-visible chunks stay null (served
 * unmasked) and the bundle frees once the last published mask releases.
 *
 * A mask larger than one staging block (blocks are not virtually contiguous)
 * falls back to plain pageable host memory for that chunk — correctness is
 * identical; only the true-async-DMA benefit is lost, and a log line records
 * it.
 *
 * @throws std::runtime_error on capture/validation failures, reservation
 *         failure, or dropped tasks — loud by design: this runs after the
 *         plan-time CPU-fallback gate, so the alternatives are all
 *         silent-wrong-data.
 */
void run_mvcc_mask_jobs(std::span<mvcc_mask_job_request> requests,
                        exec::scoped_dispatcher& dispatcher,
                        cucascade::memory::memory_reservation_manager& reservation_manager,
                        sirius::memory::topology_index const& topology);

}  // namespace sirius::scan_manager

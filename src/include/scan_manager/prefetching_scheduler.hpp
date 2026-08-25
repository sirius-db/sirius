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

#include "io/cache/types.hpp"
#include "planner/query_index.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::scan_manager {

// ---------------------------------------------------------------------------
// prefetching_scheduler — the cursor over a query's prefetching order
// ---------------------------------------------------------------------------
//
// @ref planner::query_index::prefetching_orders gives the scans in the order
// downstream operators will ask for them, each tagged with the barrier that
// gates it (@ref planner::scheduling_mode) and how much of it is worth reading
// ahead.  This class turns that static list into a moving cursor: it answers
// "which scan should be prefetched next" and advances as scans report progress.
//
// ---- Grouping --------------------------------------------------------------
//
// The order is first cut into consecutive GROUPS.  A group is served
// round-robin and is only left once every member is depleted; groups themselves
// run strictly in order.  What starts a new group depends on the mode:
//
//   barrier_all     always its own group.  The scan feeds a FULL port, so
//                   nothing downstream can start until it is finished — there
//                   is no one to interleave with.
//
//   barrier_serial  joins the previous step only if that step is also
//                   barrier_serial AND carries the same branch id.  Two scans
//                   feeding the same barrier are both blocking the same
//                   consumer, so they are advanced together, a quantum each:
//
//                     (A, serial, br 12, 5), (B, serial, br 12, 3)
//                       -> 5xA, 3xB, 5xA, 3xB, ...   (one group)
//
//                     (A, serial, br 12, 5), (B, serial, br 14, 3)
//                       -> A until depleted, then B  (two groups)
//
//   pipeline        joins the previous step if that step is also pipeline —
//                   branch id is NOT considered.  Nothing gates these scans, so
//                   there is no barrier to group them by; they simply take
//                   turns, one split each.
//
// ---- Depletion -------------------------------------------------------------
//
// A member leaves the rotation when it is depleted, which this class defines as
// having reported @c scan_stage::disposed.  It does NOT decide when an operator
// as a whole is finished: an operator emits many splits and each reports
// independently, so the owner (@ref readahead_scan_manager, which holds the
// live task list) is responsible for only reporting @c disposed once the
// operator really is done.  Everything else is recorded and otherwise ignored.
//
// ---- Threading -------------------------------------------------------------
//
// Not internally synchronised.  It is owned by @ref readahead_scan_manager and
// called under that class's mutex; keeping the lock outside makes the cursor
// directly unit-testable and avoids locking twice on every scan update.

class prefetching_scheduler {
 public:
  prefetching_scheduler() = default;

  /// Rebuild the cursor from @p order (as returned by
  /// @c query_index::prefetching_orders).  Steps whose scan is null or carries
  /// no operator id are skipped.
  void reset(std::span<const planner::prefetch_step> order);

  /// Drop the order and the cursor.
  void clear();

  /// Record @p stage for @p operator_id and re-position the cursor.  Unknown
  /// operator ids are ignored.  Reporting @c scan_stage::disposed retires the
  /// operator from the rotation.
  void update(std::size_t operator_id, io::cache::scan_stage stage);

  /// The scan that should be prefetched next, or nullptr once every step is
  /// depleted.  Each call consumes one unit of the current member's quantum, so
  /// repeated calls walk the rotation described above.
  [[nodiscard]] op::sirius_physical_operator* get_next_prefetching_operator();

  /// The operator id @ref get_next_prefetching_operator would return, or nullopt
  /// when the order is exhausted.  Does not consume a quantum.
  [[nodiscard]] std::optional<std::size_t> peek_next_operator_id() const;

  /// The live members of the current rotation group, in rotation order starting
  /// at the cursor.  Members of one group are concurrently schedulable peers, so
  /// a caller whose head member has nothing ready to prefetch may serve a later
  /// peer without breaking the order — only crossing into the NEXT group would
  /// do that, and this never reports one.  Empty once the order is exhausted.
  [[nodiscard]] std::vector<std::size_t> peek_group_operator_ids() const;

  /// Live members of every group AFTER the current one, in order.  Offered so a
  /// caller with capacity the current group cannot use can look ahead instead of
  /// idling; never to be served ahead of the current group.
  [[nodiscard]] std::vector<std::size_t> peek_lookahead_operator_ids() const;

  /// Park the cursor on @p operator_id when it is a live member of the current
  /// group, so the next @ref get_next_prefetching_operator serves it.  Returns
  /// false (leaving the cursor alone) for any other operator.  Moving the cursor
  /// restarts that member's quantum, exactly as the rotation itself does.
  bool focus_member(std::size_t operator_id);

  [[nodiscard]] bool empty() const noexcept { return _entries.empty(); }

  /// Number of round-robin groups the order was cut into.  Exposed for tests.
  [[nodiscard]] std::size_t group_count() const noexcept { return _groups.size(); }

  /// True once every step has been depleted.
  [[nodiscard]] bool exhausted() const noexcept { return _group >= _groups.size(); }

  /// Last stage reported for @p operator_id, or @c scan_stage::none.
  [[nodiscard]] io::cache::scan_stage stage_of(std::size_t operator_id) const;

 private:
  struct entry {
    op::sirius_physical_operator* scan{nullptr};
    std::size_t operator_id{0};
    std::size_t branch_id{0};
    planner::scheduling_mode mode{planner::scheduling_mode::pipeline};
    /// Consecutive picks before the rotation moves on.  SIZE_MAX for
    /// barrier_all (hold until depleted), 1 for pipeline, the step's count for
    /// barrier_serial.
    std::size_t quantum{1};
    io::cache::scan_stage stage{io::cache::scan_stage::none};
    bool depleted{false};
  };

  /// Indices into @ref _entries, served round-robin.
  using group = std::vector<std::size_t>;

  /// Move the cursor onto a member that can still be served: past members whose
  /// quantum is spent or which are depleted, and past groups with nothing live
  /// left.  Leaves @ref _group == _groups.size() when the order is exhausted.
  void advance();

  /// True when @p g has no member left to serve.
  [[nodiscard]] bool group_depleted(const group& g) const;

  std::vector<entry> _entries;
  std::vector<group> _groups;
  std::unordered_map<std::size_t, std::size_t> _by_operator;  // operator id -> index into _entries

  std::size_t _group{0};    // current group
  std::size_t _member{0};   // index within the current group
  std::size_t _emitted{0};  // picks already served from the current member
};

}  // namespace sirius::scan_manager

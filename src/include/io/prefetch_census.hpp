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

// Per-scan prefetch census.  Answers, for one query: of the scans we ran, how
// many did the readahead actually get in front of, how many did it merely
// half-cover (so a reader had to wait on it), and how many did it never reach.
//
// Counted per SCAN, not per read: each datasource classifies itself exactly
// once, on its first device read, by the state its prefetch was in at that
// moment.  A scan therefore lands in exactly one of the read-side buckets.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sirius::io {

struct prefetch_census {
  // ---- issue side (readahead worker) --------------------------------------
  std::atomic<std::uint64_t> scans_registered{0};   ///< splits emitted
  std::atomic<std::uint64_t> prefetch_issued{0};    ///< splits the worker issued IO for
  std::atomic<std::uint64_t> skipped_no_ranges{0};  ///< nothing to read (host-backed / cached)
  std::atomic<std::uint64_t> declined_reading{0};   ///< refused: the reader had already started

  // ---- read side (one classification per scan, on its first read) ---------
  std::atomic<std::uint64_t> read_no_handle{0};   ///< scan never entered the cache at all
  std::atomic<std::uint64_t> read_ready{0};       ///< prefetch finished before the read: a win
  std::atomic<std::uint64_t> read_waited{0};      ///< prefetch in flight; the read blocked on it
  std::atomic<std::uint64_t> read_not_started{0};  ///< prefetch had not begun; read did its own IO

  // ---- why the worker stopped collecting ----------------------------------
  // Each collect pass ends for exactly one reason.  Distinguishes "we ran out
  // of budget" (throttled) from "the head of the order had nothing to give"
  // (head-of-line blocked), which need opposite fixes.
  std::atomic<std::uint64_t> collect_passes{0};
  std::atomic<std::uint64_t> stop_budget_full{0};       ///< ongoing >= budget
  std::atomic<std::uint64_t> stop_order_exhausted{0};   ///< scheduler had nothing left
  std::atomic<std::uint64_t> stop_operator_unknown{0};  ///< head operator has emitted no splits
  std::atomic<std::uint64_t> stop_no_split_ready{0};    ///< head operator's splits all taken

  // ---- predicted vs actual order ------------------------------------------
  // Operator ids in the order the scheduler handed them out, against the order
  // their splits were first read.  If the readahead is picking the right scans
  // but too late, these agree; if it is mispredicting, they diverge.
  mutable std::mutex order_mtx;
  std::vector<std::size_t> prefetch_order;  ///< scheduler's static order
  std::vector<std::size_t> issue_order;     ///< operators actually issued, in order
  std::vector<std::size_t> read_order;      ///< operators first reaching `reading`

  void note_issue(std::size_t op_id)
  {
    std::lock_guard g(order_mtx);
    if (issue_order.empty() || issue_order.back() != op_id) { issue_order.push_back(op_id); }
  }

  void note_read(std::size_t op_id)
  {
    std::lock_guard g(order_mtx);
    for (auto id : read_order) {
      if (id == op_id) { return; }
    }
    read_order.push_back(op_id);
  }

  static prefetch_census& instance() noexcept
  {
    static prefetch_census c;
    return c;
  }

  [[nodiscard]] std::string to_string() const
  {
    auto const reg     = scans_registered.load();
    auto const issued  = prefetch_issued.load();
    auto const skipped = skipped_no_ranges.load();
    auto const decl    = declined_reading.load();
    auto const nh      = read_no_handle.load();
    auto const ready   = read_ready.load();
    auto const waited  = read_waited.load();
    auto const late    = read_not_started.load();

    return "=== prefetch census ===\n"
           "  scans registered          : " + std::to_string(reg) +
           "\n  prefetch issued           : " + std::to_string(issued) +
           "\n  skipped (no ranges)       : " + std::to_string(skipped) +
           "\n  declined (already reading): " + std::to_string(decl) +
           "\n  -- at first read --" +
           "\n  prefetched, ready in time : " + std::to_string(ready) +
           "\n  prefetched, had to wait   : " + std::to_string(waited) +
           "\n  prefetch not started yet  : " + std::to_string(late) +
           "\n  no prefetch handle        : " + std::to_string(nh) +
           "\n  -- why the worker stopped collecting --" +
           "\n  collect passes            : " + std::to_string(collect_passes.load()) +
           "\n    budget full             : " + std::to_string(stop_budget_full.load()) +
           "\n    order exhausted         : " + std::to_string(stop_order_exhausted.load()) +
           "\n    head operator unknown   : " + std::to_string(stop_operator_unknown.load()) +
           "\n    head operator no split  : " + std::to_string(stop_no_split_ready.load()) +
           "\n  -- operator order --" +
           "\n  scheduler order : " + join(prefetch_order) +
           "\n  issued order    : " + join(issue_order) +
           "\n  first-read order: " + join(read_order) + "\n";
  }

  static std::string join(std::vector<std::size_t> const& v)
  {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
      s += (i ? ", " : "") + std::to_string(v[i]);
    }
    return s.empty() ? "(none)" : s;
  }

  void reset() noexcept
  {
    scans_registered  = 0;
    prefetch_issued   = 0;
    skipped_no_ranges = 0;
    declined_reading  = 0;
    read_no_handle    = 0;
    read_ready        = 0;
    read_waited       = 0;
    read_not_started  = 0;

    collect_passes        = 0;
    stop_budget_full      = 0;
    stop_order_exhausted  = 0;
    stop_operator_unknown = 0;
    stop_no_split_ready   = 0;

    std::lock_guard g(order_mtx);
    prefetch_order.clear();
    issue_order.clear();
    read_order.clear();
  }
};

}  // namespace sirius::io

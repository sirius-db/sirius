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
#include <chrono>
#include <cstdint>
#include <cstdio>
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
  /// Issued before anything attached buffers: every chunk was still
  /// empty/queued, so the claim loop found nothing to read and the request was
  /// retired `ready` having done NO IO.  A reader then sees a "ready" prefetch
  /// that never fetched anything, and pays for the whole split itself.
  std::atomic<std::uint64_t> prefetch_unprepared{0};

  /// Prefetch requests with IO currently in flight.  Live gauge, not a total:
  /// the read side consults it to tell "nothing was prefetched for me" from
  /// "nothing was prefetched for me WHILE the link was busy with someone else".
  std::atomic<std::int64_t> inflight_prefetches{0};

  // ---- scan occupancy -----------------------------------------------------
  // Time-weighted account of how many scans were doing IO at once, against the
  // budget the backend published.  A scan counts while its prefetch is in
  // flight, or while it is being read having never been prefetched -- a
  // prefetched split already in the executor is reading from cache and is doing
  // no IO, so it does not count.
  //
  // A mean well below the budget means the readahead is not saturated, and
  // reordering which splits it picks cannot help until that is fixed.
  std::atomic<std::uint64_t> active_weighted_ns{0};  ///< sum of count * duration
  std::atomic<std::uint64_t> active_total_ns{0};     ///< duration observed
  std::atomic<std::uint64_t> active_at_max_ns{0};    ///< duration spent at >= budget
  std::atomic<std::uint64_t> active_budget{0};       ///< the budget in force

  // ---- prefetch order vs execution order ----------------------------------
  // Splits are ranked in the order the readahead issues their prefetch, and
  // again in the order the executor first reads them.  If the two agree, every
  // read finds either a finished prefetch or one already running.  Where they
  // disagree, a prefetch for a split the executor did not want yet is sitting
  // in front of the split it did -- the single request queue turns that
  // disagreement directly into wait.
  std::atomic<std::uint64_t> order_reads{0};           ///< reads with both ranks known
  std::atomic<std::uint64_t> order_inversions{0};      ///< read out of prefetch order
  std::atomic<std::uint64_t> order_displacement{0};    ///< sum |read_rank - prefetch_rank|
  std::atomic<std::uint64_t> read_before_prefetch{0};  ///< read before its prefetch was issued
  /// Sum over reads of how many prefetches had been issued after this split's
  /// own -- how far ahead of the read cursor the readahead was running.
  std::atomic<std::uint64_t> order_lead{0};

  // ---- ordering failures --------------------------------------------------
  // The IO stack has one global request queue, so a prefetch issued for a split
  // the executor is not about to run puts its requests ahead of the split the
  // executor IS about to run.  These count the three ways that goes wrong; all
  // of them mean the readahead's order and the executor's order disagreed.

  /// A scan began reading with nothing prefetched for it while OTHER prefetches
  /// were in flight -- its reads queued behind work for a split nobody wanted yet.
  std::atomic<std::uint64_t> read_cold_while_prefetching{0};
  /// A scan waited on its own in-flight prefetch while an already-ready split of
  /// the same operator was sitting there -- the ready one should have gone first.
  std::atomic<std::uint64_t> read_loading_while_ready_available{0};
  /// The worker was asked to prefetch, could not, and still had unprefetched
  /// splits to work on -- capacity went unused with work available.
  std::atomic<std::uint64_t> prefetch_declined_with_work{0};

  /// When splits reached the readahead, measured from the census reset at the
  /// start of the query.  A readahead that reports "nothing to prefetch" is
  /// either genuinely out of work or waiting on a producer that trickles; the
  /// spread between the first and last registration is what tells them apart.
  std::chrono::steady_clock::time_point window_start{std::chrono::steady_clock::now()};
  std::atomic<std::uint64_t> first_registration_ms{0};
  std::atomic<std::uint64_t> last_registration_ms{0};

  void note_registration() noexcept
  {
    auto const ms =
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - window_start)
                                   .count());
    std::uint64_t expected = 0;
    first_registration_ms.compare_exchange_strong(expected, ms, std::memory_order_relaxed);
    auto prev = last_registration_ms.load(std::memory_order_relaxed);
    while (ms > prev &&
           !last_registration_ms.compare_exchange_weak(prev, ms, std::memory_order_relaxed)) {}
  }

  // ---- read side (one classification per scan, on its first read) ---------
  std::atomic<std::uint64_t> read_no_handle{0};  ///< scan never entered the cache at all
  std::atomic<std::uint64_t> read_ready{0};      ///< prefetch finished before the read: a win
  std::atomic<std::uint64_t> read_waited{0};     ///< prefetch in flight; the read blocked on it
  /// Prefetch was submitted but had not reached the wire: parked behind other
  /// splits' requests.  Picked in time, delivered late.
  std::atomic<std::uint64_t> read_queued_behind{0};
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
  /// Opportunistic only: budget was free and splits were waiting, but the
  /// strategy had no credits left -- the executor had not invited another
  /// prefetch.  The one stop reason that means "throttled on purpose" rather
  /// than "nothing to do", so it reads as healthy until you separate it out.
  std::atomic<std::uint64_t> stop_no_credits{0};

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

  // -- byte accounting -------------------------------------------------------
  // Decomposes what actually reaches the device against what the reader asked
  // for.  bytes_logical is the sum of the callers' ranges; the rest are disk
  // bytes broken out by which path fetched them.
  std::atomic<std::uint64_t> bytes_logical{0};   ///< bytes the reader requested
  std::atomic<std::uint64_t> bytes_hit{0};       ///< served from cache (no disk)
  std::atomic<std::uint64_t> bytes_h2d{0};       ///< disk -> cache buffer, then device
  std::atomic<std::uint64_t> bytes_miss{0};      ///< disk -> bounce, not cached
  std::atomic<std::uint64_t> bytes_prefetch{0};  ///< disk, issued by the readahead

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
           "  scans registered          : " +
           std::to_string(reg) + "\n  prefetch issued           : " + std::to_string(issued) +
           "\n  skipped (no ranges)       : " + std::to_string(skipped) +
           "\n  issued before prepared    : " + std::to_string(prefetch_unprepared.load()) +
           "\n  -- ordering failures --" + "\n  cold read while prefetching: " +
           std::to_string(read_cold_while_prefetching.load()) +
           "\n  waited while ready avail   : " +
           std::to_string(read_loading_while_ready_available.load()) +
           "\n  prefetch declined w/ work  : " +
           std::to_string(prefetch_declined_with_work.load()) + "\n  -- scan occupancy --" +
           occupancy_line() + order_line() +
           "\n  splits registered over    : " + std::to_string(first_registration_ms.load()) +
           " ms .. " + std::to_string(last_registration_ms.load()) + " ms" +
           "\n  declined (already reading): " + std::to_string(decl) + "\n  -- bytes --" +
           "\n  logical (requested)       : " + std::to_string(bytes_logical.load()) +
           "\n  disk: cache fill (h2d)    : " + std::to_string(bytes_h2d.load()) +
           "\n  disk: bounce (uncached)   : " + std::to_string(bytes_miss.load()) +
           "\n  disk: prefetch            : " + std::to_string(bytes_prefetch.load()) +
           "\n  served from cache         : " + std::to_string(bytes_hit.load()) +
           "\n  -- at first read --" + "\n  prefetched, ready in time : " + std::to_string(ready) +
           "\n  prefetched, had to wait   : " + std::to_string(waited) +
           "\n  prefetched, queued behind : " + std::to_string(read_queued_behind.load()) +
           "\n  prefetch not started yet  : " + std::to_string(late) +
           "\n  no prefetch handle        : " + std::to_string(nh) +
           "\n  -- why the worker stopped collecting --" +
           "\n  collect passes            : " + std::to_string(collect_passes.load()) +
           "\n    budget full             : " + std::to_string(stop_budget_full.load()) +
           "\n    order exhausted         : " + std::to_string(stop_order_exhausted.load()) +
           "\n    head operator unknown   : " + std::to_string(stop_operator_unknown.load()) +
           "\n    head operator no split  : " + std::to_string(stop_no_split_ready.load()) +
           "\n    out of credits          : " + std::to_string(stop_no_credits.load()) +
           "\n  -- operator order --" + "\n  scheduler order : " + join(prefetch_order) +
           "\n  issued order    : " + join(issue_order) +
           "\n  first-read order: " + join(read_order) + "\n";
  }

  /// Mean concurrent scans doing IO, and the share of time spent at the budget.
  [[nodiscard]] std::string occupancy_line() const
  {
    auto const total  = active_total_ns.load();
    auto const budget = active_budget.load();
    if (total == 0) { return "\n  active scans              : (not observed)"; }
    double const mean = static_cast<double>(active_weighted_ns.load()) / static_cast<double>(total);
    double const at_max_pct =
      100.0 * static_cast<double>(active_at_max_ns.load()) / static_cast<double>(total);
    char buf[160];
    std::snprintf(buf,
                  sizeof(buf),
                  "\n  active scans              : mean %.2f of budget %llu, at budget %.1f%% of "
                  "the time",
                  mean,
                  static_cast<unsigned long long>(budget),
                  at_max_pct);
    return buf;
  }

  /// How far the order splits were prefetched in drifted from the order they
  /// were read in.
  [[nodiscard]] std::string order_line() const
  {
    auto const n = order_reads.load();
    if (n == 0) { return "\n  prefetch vs read order    : (not observed)"; }
    char buf[200];
    std::snprintf(buf,
                  sizeof(buf),
                  "\n  prefetch vs read order    : %llu/%llu inverted (%.0f%%), mean displacement "
                  "%.1f, mean lead %.1f, %llu read before prefetch",
                  static_cast<unsigned long long>(order_inversions.load()),
                  static_cast<unsigned long long>(n),
                  100.0 * static_cast<double>(order_inversions.load()) / static_cast<double>(n),
                  static_cast<double>(order_displacement.load()) / static_cast<double>(n),
                  static_cast<double>(order_lead.load()) / static_cast<double>(n),
                  static_cast<unsigned long long>(read_before_prefetch.load()));
    return buf;
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
    bytes_logical       = 0;
    bytes_hit           = 0;
    bytes_h2d           = 0;
    bytes_miss          = 0;
    bytes_prefetch      = 0;
    scans_registered    = 0;
    prefetch_issued     = 0;
    skipped_no_ranges   = 0;
    declined_reading    = 0;
    prefetch_unprepared = 0;

    read_cold_while_prefetching        = 0;
    read_loading_while_ready_available = 0;
    prefetch_declined_with_work        = 0;
    inflight_prefetches                = 0;
    active_weighted_ns                 = 0;
    active_total_ns                    = 0;
    active_at_max_ns                   = 0;
    active_budget                      = 0;
    order_reads                        = 0;
    order_inversions                   = 0;
    order_displacement                 = 0;
    read_before_prefetch               = 0;
    order_lead                         = 0;

    window_start          = std::chrono::steady_clock::now();
    first_registration_ms = 0;
    last_registration_ms  = 0;
    read_no_handle        = 0;
    read_ready            = 0;
    read_waited           = 0;
    read_queued_behind    = 0;
    read_not_started      = 0;

    collect_passes        = 0;
    stop_budget_full      = 0;
    stop_order_exhausted  = 0;
    stop_operator_unknown = 0;
    stop_no_split_ready   = 0;
    stop_no_credits       = 0;

    std::lock_guard g(order_mtx);
    prefetch_order.clear();
    issue_order.clear();
    read_order.clear();
  }
};

}  // namespace sirius::io

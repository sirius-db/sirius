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

#include "scan_manager/split_connector.hpp"

#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

namespace {

/// Bit set in a @c _split_kinds entry when the split has IO a prefetch could fetch.
constexpr std::uint8_t kIoCandidate = 1U << 0;
/// Bit set in a @c _split_kinds entry when the split is a resident batch an early upload could
/// promote.
constexpr std::uint8_t kMemoryCandidate = 1U << 1;

/// Structural facts about a split, computed once at push time on the producer thread.
struct split_class {
  /// kIoCandidate | kMemoryCandidate.
  std::uint8_t kinds{0};
  /// datasource_count(): exactly how many folds a prefetch_state() of this split would perform.
  std::uint32_t fold_cost{0};
};

/// Classify a split for the incremental @c n_prefetchable counts and the consumer's fold budget.
/// Structural, so it is evaluated exactly once per split (at push time) and read back from
/// @c _split_kinds / @c _split_fold_costs on the consumer path -- which runs under the mutex on
/// the critical path and must re-run neither this @c dynamic_cast nor the datasource walk.
[[nodiscard]] split_class classify(const op::operator_data& split) noexcept
{
  auto const* input = dynamic_cast<const op::scan::scan_operator_input*>(&split);
  if (input == nullptr) { return {}; }
  split_class result;
  // One datasource walk, here on the producer thread, outside _mutex and outside the pipeline
  // status lock. Its count is kept so the consumer never has to repeat it.
  auto const n_datasources = input->datasource_count();
  result.fold_cost         = static_cast<std::uint32_t>(
    std::min<std::size_t>(n_datasources, std::numeric_limits<std::uint32_t>::max()));
  // Structural on both axes: "has datasources" and "is a resident batch" are fixed for the
  // split's lifetime, so the counts never need a rescan. The dynamic tests live in prefetch_if.
  if (n_datasources > 0) { result.kinds |= kIoCandidate; }
  if (input->is_resident()) { result.kinds |= kMemoryCandidate; }
  return result;
}

/// A split's advisory prefetch progress, or @c empty for anything that is not a scan split.
[[nodiscard]] io::cache::prefetch_progress progress_of(const op::operator_data& split) noexcept
{
  auto const* input = dynamic_cast<const op::scan::scan_operator_input*>(&split);
  return input == nullptr ? io::cache::prefetch_progress::empty : input->prefetch_state();
}

}  // namespace

split_connector::split_connector()  = default;
split_connector::~split_connector() = default;

void split_connector::push_split(std::unique_ptr<op::operator_data> split)
{
  // A hard check, not an assert: asserts are compiled out in release, and both classify() and
  // fill_progress_window() dereference every queued split -- the latter under _mutex (L2) while
  // the consumer holds sirius_pipeline::_status_mutex (L1), on the path every task completion on
  // the pipeline blocks behind. A null entry that used to flow through harmlessly would be a null
  // deref there, in the worst place in the engine to take one.
  if (split == nullptr) {
    throw std::invalid_argument("[split_connector::push_split] split must be non-null");
  }
  // Classified outside the lock: one dynamic_cast plus one datasource walk, none of which touch
  // connector state.
  auto const cls = classify(*split);

  // The one datasource walk that answers the arming question, taken here rather than under the
  // lock: it is the walk the whole fold budget exists to keep out of the critical section, and it
  // happens once per connector. _arming_checked is producer-private -- the consumer never reads
  // it -- so testing it unlocked races with nothing.
  //
  // Only an io candidate can answer the question meaningfully: a resident split has no datasource
  // and no backend, so letting one latch the flag would report "not armed" for a connector that
  // later carries delta splits.
  std::optional<bool> arming_answer;
  if (!_arming_checked && (cls.kinds & kIoCandidate) != 0) {
    auto const* input = dynamic_cast<const op::scan::scan_operator_input*>(split.get());
    if (input != nullptr) { arming_answer = input->can_land_while_queued(); }
  }

  {
    std::lock_guard<std::mutex> lock(_mutex);
    assert(!_closed && "push_split after close() is forbidden");
    // _splits, _split_kinds and _split_fold_costs are index-parallel. Grow the two side channels
    // first and undo them if a later push fails, so an allocation failure can never leave them
    // misaligned.
    _split_kinds.push_back(cls.kinds);
    try {
      _split_fold_costs.push_back(cls.fold_cost);
    } catch (...) {
      _split_kinds.pop_back();
      throw;
    }
    try {
      _splits.push_back(std::move(split));
    } catch (...) {
      _split_fold_costs.pop_back();
      _split_kinds.pop_back();
      throw;
    }
    if ((cls.kinds & kIoCandidate) != 0) { ++_n_io_prefetchable; }
    if ((cls.kinds & kMemoryCandidate) != 0) { ++_n_memory_prefetchable; }
    // Latched under the lock, because get_next_split reads it under the lock: the answer was
    // computed off the lock but must not be *stored* off it, or the store would race the read.
    // After the pushes, so a throwing push does not leave the connector armed by a split that
    // never joined the queue.
    if (arming_answer.has_value() && !_arming_checked) {
      _arming_checked  = true;
      _selection_armed = *arming_answer;
    }
  }
  _cv.notify_one();
}

void split_connector::close(std::exception_ptr const& exception)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _closed = true;
    // First non-null exception wins. Subsequent close() calls (idempotent)
    // do not overwrite an already-recorded error so the consumer always
    // sees the original cause of the producer's failure.
    if (exception && !_exception) { _exception = exception; }
  }
  _cv.notify_all();
}

std::optional<std::unique_ptr<op::operator_data>> split_connector::get_next_split()
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cv.wait(lock, [this] { return !_splits.empty() || _closed; });
  // if there is an exception, propagate it to the consumer instead of returning more splits
  if (_exception) { std::rethrow_exception(_exception); }
  if (_splits.empty()) { return std::nullopt; }

  // The gate. No backend on this connector activates a prefetch before the dequeue, so nothing
  // queued can report `cached` or `loading` and the selection walk provably cannot beat the queue
  // front. Skip it entirely: this is the path every shipped configuration takes, and it is the
  // same O(1) move-and-pop_front the connector had before the policy existed -- no state reads,
  // no virtual calls, no datasource walks.
  std::size_t index = 0;
  if (_selection_armed) {
    // Everything in here is bounded by kSelectionWindow splits AND kSelectionFoldBudget folds,
    // never by _splits.size() or by a split's datasource count: this runs while the consumer
    // holds sirius_pipeline::_status_mutex, behind which every task completing on the pipeline
    // blocks. There must be no loop over the whole deque, and none over a whole split, here.
    std::array<io::cache::prefetch_progress, kSelectionWindow> window{};
    auto const filled = fill_progress_window(window);
    index =
      select_split_index(std::span<const io::cache::prefetch_progress>{window.data(), filled});
  }

  // Always hands out what it selected -- select_split_index is total and never refuses. Waiting
  // for a better candidate would be a permanent hang: a split leaving prefetch_progress::loading
  // notifies io::cache::entry_state's atomic, not _cv.
  auto split = std::move(_splits[index]);
  drop_at(index);
  return std::optional<std::unique_ptr<op::operator_data>>{std::move(split)};
}

bool split_connector::is_closed() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _closed && _splits.empty();
}

[[nodiscard]] bool split_connector::has_more_splits() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return !_splits.empty();
}

std::size_t split_connector::prefetch_if(std::size_t upto_n,
                                         prefetch_kind kind,
                                         const std::function<bool(const op::operator_data&)>& pred)
{
  // Not noexcept, on purpose: pred is caller-supplied and may throw. The lock_guard releases the
  // mutex on the way out, so the connector is left consistent either way.
  std::lock_guard<std::mutex> lock(_mutex);
  std::size_t const window = std::min(upto_n, _splits.size());
  std::size_t hinted       = 0;

  for (std::size_t i = 0; i < window; ++i) {
    auto* input = dynamic_cast<op::scan::scan_operator_input*>(_splits[i].get());
    if (input == nullptr) { continue; }
    if (pred && !pred(*input)) { continue; }

    if (kind == prefetch_kind::io) {
      if (!input->is_io_prefetchable()) { continue; }
      // Counted only when the split's ladder actually advanced. This walk restarts from the queue
      // front on every invocation and the dequeue fires task_queued again afterwards, so counting
      // inspections instead would grow without bound over a static queue.
      if (!input->prefetch(io::cache::prefetching_stage::task_queued)) { continue; }
    } else {
      // nullopt means the batch is exclusively locked right now; assume it still needs the
      // upload, because over-scheduling one is cheap and skipping a needed one stalls the task.
      if (!input->is_memory_prefetchable().value_or(true)) { continue; }
      // Counters only for now (D-N3): the resident path never calls prefetch(), and the only
      // place an early GPU upload can run is prepare_for_processing on the executor thread.
      // Scheduling one needs a dispatcher task and a memory reservation -- a separate design.
      // The return value therefore reports candidates, not work done, for this kind.
    }
    ++hinted;
  }
  return hinted;
}

std::size_t split_connector::n_prefetchable(prefetch_kind kind) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return kind == prefetch_kind::io ? _n_io_prefetchable : _n_memory_prefetchable;
}

// IMPLEMENTATION NOTE: rule 3 (fall back to index 0) is a liveness requirement, not a tie-break.
// A split leaving prefetch_progress::loading notifies io::cache::entry_state's atomic, never this
// connector's condition variable, so refusing to hand out a loading split is a permanent hang.
//
// This function is also what the positional cached-serving regression case in
// test/cpp/scan_manager/test_cached_serving_hardening.cpp ("drain_cached_provider forwards the
// mvcc keep-mask and filter flag onto each split") depends on, and any change to the policy must
// be checked against it. Note the connector that case builds is NOT resident-only in general --
// drain_cached_provider feeds resident batches and delta metadata splits into one connector -- so
// "resident splits carry no handle" is not on its own a reordering argument. The guarantee comes
// from get_next_split's arming gate instead: a delta split's datasources are subject to the same
// activation-stage test, no shipped backend passes it, the connector never arms, and the dequeue
// is a literal pop_front that cannot reorder anything.
std::size_t split_connector::select_split_index(
  std::span<const io::cache::prefetch_progress> states) noexcept
{
  // One front-to-back pass covers all three rules: rule 1 wants the FIRST landed split, so the
  // first `cached` seen is the answer and nothing past it can change it; rule 2's candidate is
  // recorded on the way there and only used if the pass finds no landed split at all.
  std::size_t first_not_loading = 0;
  bool have_not_loading         = false;

  for (std::size_t i = 0; i < states.size(); ++i) {
    if (states[i] == io::cache::prefetch_progress::cached) { return i; }  // rule 1
    if (!have_not_loading && states[i] != io::cache::prefetch_progress::loading) {
      first_not_loading = i;  // rule 2 candidate
      have_not_loading  = true;
    }
  }

  // Rule 2 when some split is not in flight, rule 3 (the queue front, i.e. FIFO) otherwise. Note
  // an empty span also lands on 0; the caller documents states as non-empty, and returning a
  // sentinel or trapping here would only trade a contract violation for a worse one.
  return have_not_loading ? first_not_loading : 0;
}

std::size_t split_connector::fill_progress_window(
  std::array<io::cache::prefetch_progress, kSelectionWindow>& out) const noexcept
{
  // Bounded by kSelectionWindow, not by the queue length -- see get_next_split.
  std::size_t const n = std::min(_splits.size(), kSelectionWindow);
  std::size_t filled  = 0;
  std::size_t spent   = 0;
  while (filled < n) {
    // Second axis of the bound. The per-split fold cost was measured once by the producer, so
    // this decision costs a deque index rather than the datasource walk it is protecting against.
    // The front split is always inspected whatever it costs, so selection is never skipped
    // outright and rule 3 of select_split_index always has a candidate.
    if (filled > 0 && spent + _split_fold_costs[filled] > kSelectionFoldBudget) { break; }
    spent += _split_fold_costs[filled];
    out[filled] = progress_of(*_splits[filled]);
    ++filled;
    // Rule 1 of select_split_index takes the FIRST landed split, so nothing past it can change
    // the answer. This is what keeps the best case as cheap as the plain pop_front it replaced.
    if (out[filled - 1] == io::cache::prefetch_progress::cached) { break; }
  }
  return filled;
}

void split_connector::drop_at(std::size_t index) noexcept
{
  assert(index < _splits.size() && index < _split_kinds.size() &&
         index < _split_fold_costs.size() &&
         "drop_at requires an index inside _splits, _split_kinds and _split_fold_costs");

  auto const mask = _split_kinds[index];
  if ((mask & kIoCandidate) != 0 && _n_io_prefetchable > 0) { --_n_io_prefetchable; }
  if ((mask & kMemoryCandidate) != 0 && _n_memory_prefetchable > 0) { --_n_memory_prefetchable; }

  // index < kSelectionWindow, and std::deque::erase relocates the shorter side, so this moves at
  // most kSelectionWindow-1 pointers; index == 0 is a plain pop_front.
  auto const offset =
    static_cast<std::deque<std::unique_ptr<op::operator_data>>::difference_type>(index);
  _splits.erase(std::next(_splits.begin(), offset));
  _split_kinds.erase(
    std::next(_split_kinds.begin(), static_cast<std::deque<std::uint8_t>::difference_type>(index)));
  _split_fold_costs.erase(std::next(
    _split_fold_costs.begin(), static_cast<std::deque<std::uint32_t>::difference_type>(index)));
}

}  // namespace sirius::scan_manager

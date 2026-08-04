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
#include <iterator>
#include <utility>

namespace sirius::scan_manager {

namespace {

/// Bit set in a @c _split_kinds entry when the split has IO a prefetch could fetch.
constexpr std::uint8_t kIoCandidate = 1U << 0;
/// Bit set in a @c _split_kinds entry when the split is a resident batch an early upload could
/// promote.
constexpr std::uint8_t kMemoryCandidate = 1U << 1;

/// Classify a split for the incremental @c n_prefetchable counts. Structural, so it is evaluated
/// exactly once per split (at push time) and read back from @c _split_kinds on the removal path --
/// which runs under the mutex on the critical path and must not re-run this @c dynamic_cast.
[[nodiscard]] std::uint8_t classify(const op::operator_data& split) noexcept
{
  auto const* input = dynamic_cast<const op::scan::scan_operator_input*>(&split);
  if (input == nullptr) { return 0; }
  std::uint8_t mask = 0;
  // Structural on both axes: "has datasources" and "is a resident batch" are fixed for the
  // split's lifetime, so the counts never need a rescan. The dynamic tests live in prefetch_if.
  if (input->is_io_prefetchable()) { mask |= kIoCandidate; }
  if (input->is_resident()) { mask |= kMemoryCandidate; }
  return mask;
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
  assert(split != nullptr && "push_split requires a non-null split");
  // Classified outside the lock: one dynamic_cast plus two lock-free predicates, none of which
  // touch connector state.
  auto const mask = classify(*split);
  {
    std::lock_guard<std::mutex> lock(_mutex);
    assert(!_closed && "push_split after close() is forbidden");
    // _splits and _split_kinds are index-parallel. Grow the kinds side first and undo it if the
    // split push fails, so an allocation failure can never leave them misaligned.
    _split_kinds.push_back(mask);
    try {
      _splits.push_back(std::move(split));
    } catch (...) {
      _split_kinds.pop_back();
      throw;
    }
    if ((mask & kIoCandidate) != 0) { ++_n_io_prefetchable; }
    if ((mask & kMemoryCandidate) != 0) { ++_n_memory_prefetchable; }
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

  // Everything below is bounded by kSelectionWindow, never by _splits.size(): this runs while the
  // consumer holds sirius_pipeline::_status_mutex, behind which every task completing on the
  // pipeline blocks. There must be no loop over the whole deque here.
  std::array<io::cache::prefetch_progress, kSelectionWindow> window{};
  auto const filled = fill_progress_window(window);
  auto const index =
    select_split_index(std::span<const io::cache::prefetch_progress>{window.data(), filled});

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
      input->prefetch(io::cache::prefetching_stage::task_queued);
    } else {
      // nullopt means the batch is exclusively locked right now; assume it still needs the
      // upload, because over-scheduling one is cheap and skipping a needed one stalls the task.
      if (!input->is_memory_prefetchable().value_or(true)) { continue; }
      // Counters only for now (D-N3): the resident path never calls prefetch(), and the only
      // place an early GPU upload can run is prepare_for_processing on the executor thread.
      // Scheduling one needs a dispatcher task and a memory reservation -- a separate design.
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
// Rule 3 is also what keeps test/cpp/scan_manager/test_cached_serving_hardening.cpp:376-410
// passing unmodified: resident splits carry no prefetch handle, so every state reads `empty`,
// selection falls through to index 0, and the cached-serving dequeue stays exactly FIFO. Any
// change to this policy must be checked against that positional test.
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
  while (filled < n) {
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
         "drop_at requires an index inside both _splits and _split_kinds");

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
}

}  // namespace sirius::scan_manager

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

#include <cassert>
#include <chrono>
#include <utility>

namespace sirius::scan_manager {

split_connector::split_connector()  = default;
split_connector::~split_connector() = default;

void split_connector::push_split(std::unique_ptr<op::operator_data> split)
{
  assert(split != nullptr && "push_split requires a non-null split");
  {
    std::lock_guard<std::mutex> lock(_mutex);
    assert(!_closed && "push_split after close() is forbidden");
    // Accumulate before the move: this is the one choke point every split passes through,
    // so it is where the scan's total input basis is tallied for the data size estimator.
    _discovered_bytes += split->get_estimated_size_in_bytes();
    _splits.push_back(std::move(split));
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
  if (!_splits.empty()) {
    auto split = std::move(_splits.front());
    _splits.pop_front();
    _last_pop_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count(),
                       std::memory_order_relaxed);
    return std::optional<std::unique_ptr<op::operator_data>>{std::move(split)};
  }
  return std::nullopt;
}

bool split_connector::is_draining(std::size_t quiet_ms) const
{
  const auto last = _last_pop_ms.load(std::memory_order_relaxed);
  if (last == 0) { return false; }
  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  return now_ms - last < static_cast<std::int64_t>(quiet_ms);
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

std::vector<std::shared_ptr<::cucascade::data_batch>> split_connector::peek_resident_batches() const
{
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  std::lock_guard<std::mutex> lock(_mutex);
  batches.reserve(_splits.size());
  for (const auto& split : _splits) {
    auto* scan_input = dynamic_cast<const op::scan::scan_operator_input*>(split.get());
    if (scan_input != nullptr && scan_input->is_resident()) {
      batches.push_back(scan_input->get_cached_batch());
    }
  }
  return batches;
}

bool split_connector::is_discovery_complete() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _closed;
}

std::size_t split_connector::discovered_bytes() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _discovered_bytes;
}

}  // namespace sirius::scan_manager

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

#include "op/sirius_physical_operator.hpp"

#include <cassert>
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
    return std::optional<std::unique_ptr<op::operator_data>>{std::move(split)};
  }
  return std::nullopt;
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

}  // namespace sirius::scan_manager

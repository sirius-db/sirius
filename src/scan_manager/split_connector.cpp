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

#include <utility>

namespace sirius::scan_manager {

split_connector::split_connector()  = default;
split_connector::~split_connector() = default;

void split_connector::push_split(std::unique_ptr<op::operator_data> split)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _splits.push_back(std::move(split));
}

void split_connector::close()
{
  std::lock_guard<std::mutex> lock(_mutex);
  _closed = true;
}

std::optional<std::unique_ptr<op::operator_data>> split_connector::get_next_split()
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_splits.empty()) {
    auto split = std::move(_splits.front());
    _splits.pop_front();
    return std::optional<std::unique_ptr<op::operator_data>>{std::move(split)};
  }
  if (_closed) { return std::nullopt; }
  return std::optional<std::unique_ptr<op::operator_data>>{nullptr};
}

bool split_connector::is_closed() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _closed && _splits.empty();
}

bool split_connector::has_pending_split() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return !_splits.empty();
}

}  // namespace sirius::scan_manager

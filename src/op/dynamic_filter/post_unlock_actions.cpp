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

#include "op/dynamic_filter/post_unlock_actions.hpp"

#include "log/logging.hpp"

#include <duckdb/common/assert.hpp>

#include <iterator>
#include <utility>
#include <vector>

namespace sirius::op {

post_unlock_actions::post_unlock_actions(std::vector<dynamic_filter_key_line> key_lines) noexcept
  : _key_lines(std::move(key_lines))
{
}

post_unlock_actions::~post_unlock_actions() noexcept { D_ASSERT(empty()); }

void post_unlock_actions::append(post_unlock_actions&& other)
{
  if (other.empty()) { return; }
  if (empty()) {
    _key_lines.swap(other._key_lines);
    return;
  }
  _key_lines.insert(_key_lines.end(),
                    std::make_move_iterator(other._key_lines.begin()),
                    std::make_move_iterator(other._key_lines.end()));
  other._key_lines.clear();
}

void post_unlock_actions::dispatch() && noexcept
{
  for (auto const& line : _key_lines) {
    sirius::log::log_noexcept([&] { SIRIUS_LOG_DEBUG("{}", line); });
  }
  _key_lines.clear();
}

}  // namespace sirius::op

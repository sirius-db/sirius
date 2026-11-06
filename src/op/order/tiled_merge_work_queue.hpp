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

#include <deque>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sirius::op {
// Only the tiled operator's finite stage/final-publication lists use this queue.
// Claimed work stays owned by its task, not by a repeatedly scanned stage list.
template <typename Work>
struct tiled_merge_work_queue {
  std::deque<std::shared_ptr<Work>> pending;
  std::size_t outstanding{0};
  std::size_t inspections{0};

  void add(std::shared_ptr<Work> item)
  {
    pending.push_back(std::move(item));
    ++outstanding;
  }
  // Prepare task-owned metadata before committing the claim. Allocation failure
  // leaves both the pending item and its outstanding count intact.
  template <typename Prepare>
  auto claim(Prepare&& prepare)
  {
    using result_type = std::invoke_result_t<Prepare, std::shared_ptr<Work> const&>;
    if (pending.empty()) { return result_type{}; }
    auto result = std::forward<Prepare>(prepare)(pending.front());
    pending.pop_front();
    ++inspections;
    return result;
  }
  std::shared_ptr<Work> claim()
  {
    return claim([](auto const& item) { return item; });
  }
  void complete()
  {
    if (!outstanding) { throw std::logic_error("Sort work count underflow"); }
    --outstanding;
  }
  bool complete_stage() const { return outstanding == 0; }
};
}  // namespace sirius::op

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

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

#include "memory/pool_peer_access.hpp"

#include <cucascade/memory/memory_space.hpp>

#include <vector>

namespace sirius::op {

std::optional<std::pair<int, int>> find_pair_without_pool_access(
  std::span<dynamic_filter_replica_space const> replicas)
{
  std::vector<cucascade::memory::memory_space const*> spaces;
  spaces.reserve(replicas.size());
  for (auto const& replica : replicas) {
    spaces.push_back(&replica.get_gpu_space());
  }
  return sirius::memory::find_pair_without_pool_access(spaces);
}

}  // namespace sirius::op

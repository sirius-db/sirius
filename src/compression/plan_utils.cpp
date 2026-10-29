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

#include "plan_utils.hpp"

#include <api/simpatico_codegen.hpp>

namespace sirius::compression {

std::optional<std::string> select_plan_blocks(const std::string& full_plan_dsl,
                                              const std::vector<std::size_t>& column_indices)
{
  auto blocks = simpatico::split_plan_dsl(full_plan_dsl);
  std::string out;
  for (std::size_t k = 0; k < column_indices.size(); ++k) {
    std::size_t const idx = column_indices[k];
    if (idx >= blocks.size()) { return std::nullopt; }
    if (k != 0) { out += "\n---\n"; }
    out += blocks[idx];
  }
  return out;
}

}  // namespace sirius::compression

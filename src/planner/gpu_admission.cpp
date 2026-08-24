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

#include "planner/gpu_admission.hpp"

#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_result_collector.hpp"

namespace sirius::planner {

void collect_gpu_scans(const op::sirius_physical_operator& node,
                       std::vector<const op::sirius_physical_operator*>& out)
{
  if (node.type == op::SiriusPhysicalOperatorType::GPU_SCAN) { out.push_back(&node); }

  for (auto& child : node.children) {
    if (child) { collect_gpu_scans(*child, out); }
  }

  if (node.type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      node.type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = node.Cast<op::sirius_physical_delim_join>();
    if (delim.join) { collect_gpu_scans(*delim.join, out); }
    if (delim.distinct_root) { collect_gpu_scans(*delim.distinct_root, out); }
  }
  if (node.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    auto& rc = node.Cast<op::sirius_physical_result_collector>();
    collect_gpu_scans(rc.plan, out);
  }
}

}  // namespace sirius::planner

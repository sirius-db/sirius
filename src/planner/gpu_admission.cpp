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

void for_each_gpu_scan(const op::sirius_physical_operator& node,
                       absl::FunctionRef<void(const op::sirius_physical_operator&)> visit)
{
  if (node.type == op::SiriusPhysicalOperatorType::GPU_SCAN) { visit(node); }

  for (auto& child : node.children) {
    if (child) { for_each_gpu_scan(*child, visit); }
  }

  if (node.type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      node.type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = node.Cast<op::sirius_physical_delim_join>();
    if (delim.join) { for_each_gpu_scan(*delim.join, visit); }
    if (delim.distinct_root) { for_each_gpu_scan(*delim.distinct_root, visit); }
  }
  if (node.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    auto& rc = node.Cast<op::sirius_physical_result_collector>();
    for_each_gpu_scan(rc.plan, visit);
  }
}

void collect_gpu_scans(const op::sirius_physical_operator& node,
                       std::vector<const op::sirius_physical_operator*>& out)
{
  for_each_gpu_scan(node,
                    [&out](const op::sirius_physical_operator& scan) { out.push_back(&scan); });
}

}  // namespace sirius::planner

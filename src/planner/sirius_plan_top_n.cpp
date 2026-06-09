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

#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_vss.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "vss/vss_pattern.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalTopN& op)
{
  D_ASSERT(op.children.size() == 1);

  // bypass the projection and plan its child as the VSS source
  if (auto pattern = sirius::vss::match_vss_top_n(op)) {
    auto& proj     = op.children[0]->Cast<duckdb::LogicalProjection>();
    auto vss_child = create_plan(*proj.children[0]);
    auto vss       = duckdb::make_uniq<sirius::op::sirius_physical_vss>(
      sirius::from_duckdb_vec(op.types),
      std::move(*pattern),
      duckdb::NumericCast<std::size_t>(op.limit),
      duckdb::NumericCast<std::size_t>(op.offset),
      op.estimated_cardinality);
    vss->children.push_back(std::move(vss_child));
    return std::move(vss);
  }

  auto plan = create_plan(*op.children[0]);

  auto top_n = duckdb::make_uniq<sirius::op::sirius_physical_top_n>(
    sirius::from_duckdb_vec(op.types),
    std::move(op.orders),
    duckdb::NumericCast<std::size_t>(op.limit),
    duckdb::NumericCast<std::size_t>(op.offset),
    std::move(op.dynamic_filter),
    op.estimated_cardinality);

  top_n->children.push_back(std::move(plan));
  return std::move(top_n);
}

}  // namespace sirius::planner

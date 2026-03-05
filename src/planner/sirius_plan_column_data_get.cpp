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

#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalColumnDataGet& op)
{
  D_ASSERT(op.children.size() == 0);
  D_ASSERT(op.collection);

  return duckdb::make_uniq<sirius::op::sirius_physical_column_data_scan>(
    op.types,
    sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN,
    op.estimated_cardinality,
    std::move(op.collection));
}

}  // namespace sirius::planner

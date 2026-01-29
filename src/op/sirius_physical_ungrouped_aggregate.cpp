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

#include "op/sirius_physical_ungrouped_aggregate.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"

namespace sirius {
namespace op {

sirius_physical_ungrouped_aggregate::sirius_physical_ungrouped_aggregate(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_operator(
      duckdb::PhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
    aggregates(std::move(expressions))
{
  distinct_collection_info = duckdb::DistinctAggregateCollectionInfo::Create(aggregates);
  // aggregation_result       = duckdb::make_shared_ptr<GPUIntermediateRelation>(aggregates.size());
  if (!distinct_collection_info) { return; }
  distinct_data =
    duckdb::make_uniq<duckdb::DistinctAggregateData>(*distinct_collection_info, distinct_validity);
}

}  // namespace op
}  // namespace sirius

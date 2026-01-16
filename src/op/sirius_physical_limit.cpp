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

#include "op/sirius_physical_limit.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"

namespace sirius {
namespace op {

sirius_physical_streaming_limit::sirius_physical_streaming_limit(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::BoundLimitNode limit_val_p,
  duckdb::BoundLimitNode offset_val_p,
  duckdb::idx_t estimated_cardinality,
  bool parallel)
  : sirius_physical_operator(
      duckdb::PhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
    limit_val(std::move(limit_val_p)),
    offset_val(std::move(offset_val_p)),
    parallel(parallel)
{
}

}  // namespace op
}  // namespace sirius

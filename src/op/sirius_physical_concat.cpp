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

#include "op/sirius_physical_concat.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_top_n.hpp"

namespace sirius {
namespace op {

sirius_physical_concat::sirius_physical_concat(duckdb::vector<duckdb::LogicalType> types,
                                               duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      duckdb::PhysicalOperatorType::INVALID, std::move(types), estimated_cardinality)
{
  _num_partitions = (estimated_cardinality + PARTITION_SIZE - 1) / PARTITION_SIZE;
}

std::string sirius_physical_concat::get_name() const { return "CONCAT"; }

bool sirius_physical_concat::is_source() const { return true; }

bool sirius_physical_concat::is_sink() const { return true; }

}  // namespace op
}  // namespace sirius

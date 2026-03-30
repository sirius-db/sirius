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

#pragma once

#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/logical_operator.hpp>

namespace sirius::transparent {

/// \brief Post-optimization hook that captures the optimized logical plan for GPU execution.
///
/// When transparent execution is enabled and SiriusContext is initialized, this hook:
/// 1. Checks whether the query contains only GPU-acceleratable operators
/// 2. Creates a copy of the optimized logical plan via serialize/deserialize
/// 3. Stores the copy in SiriusContext for later use by OnFinalizePrepare
///
/// The original logical plan is NOT modified — DuckDB continues with its normal
/// physical plan generation. The stored copy is consumed in OnFinalizePrepare to
/// generate a Sirius physical plan.
void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

/// \brief Quick check whether a logical plan tree contains only operators that Sirius supports.
///
/// Walks the tree recursively. Returns false if any node is an operator type that
/// sirius_physical_plan_generator::create_plan() would throw NotImplementedException for.
bool is_acceleratable_query(const duckdb::LogicalOperator& root);

}  // namespace sirius::transparent

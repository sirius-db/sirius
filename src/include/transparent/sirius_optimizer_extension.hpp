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

/// \brief Pre-optimization hook that disables DuckDB optimizers incompatible with Sirius.
///
/// Disables IN_CLAUSE and COMPRESSED_MATERIALIZATION optimizers before they run,
/// so the logical plan remains in a form Sirius can process.
void sirius_pre_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

/// \brief Post-optimization hook that captures the optimized logical plan for GPU execution.
///
/// Re-enables the disabled optimizers (so they don't leak to non-GPU queries), then
/// copies the optimized logical plan and stores it in SiriusContext for OnFinalizePrepare.
void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

/// \brief Quick check whether a logical plan tree contains only operators that Sirius supports.
///
/// Walks the tree recursively. Returns false if any node is an operator type that
/// sirius_physical_plan_generator::create_plan() would throw NotImplementedException for.
bool is_acceleratable_query(const duckdb::LogicalOperator& root);

}  // namespace sirius::transparent

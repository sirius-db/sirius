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
/// Copies the optimized logical plan into the connection's per-connection state,
/// stamped with the current planning generation. (The optimizer mask that a
/// pre-hook used to write per query is published once at extension load —
/// publish_transparent_optimizer_mask in sirius_extension.cpp.) OnFinalizePrepare calls
/// create_plan() on this copy — that is the single source of truth for whether Sirius supports the
/// query. If create_plan() throws, we silently fall back to CPU.
void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

/// \brief Copy a logical plan for Sirius's transparent execution path.
///
/// Wraps \c duckdb::LogicalOperator::Copy. Serialization omits DuckDB join-filter metadata, but
/// Sirius discovers targets from plan structure and does not consume that metadata. The original
/// plan remains unchanged for CPU fallback.
///
/// \param plan     The plan to copy. Not consumed.
/// \param context  DuckDB client context for \c LogicalOperator::Copy.
[[nodiscard]] duckdb::unique_ptr<duckdb::LogicalOperator> copy_logical_plan(
  duckdb::LogicalOperator const& plan, duckdb::ClientContext& context);

}  // namespace sirius::transparent

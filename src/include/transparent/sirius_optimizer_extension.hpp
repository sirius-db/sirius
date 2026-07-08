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

#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>

namespace sirius::transparent {

/// \brief Pre-optimization hook that disables DuckDB optimizers incompatible with Sirius.
///
/// Disables IN_CLAUSE and COMPRESSED_MATERIALIZATION optimizers before they run,
/// so the logical plan remains in a form Sirius can process.
void sirius_pre_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

/// \brief Post-optimization hook that captures the optimized logical plan for GPU execution.
///
/// Re-enables the disabled optimizers, then copies the optimized logical plan and stores
/// it in SiriusContext. OnFinalizePrepare calls create_plan() on this copy — that is the
/// single source of truth for whether Sirius supports the query. If create_plan() throws,
/// we silently fall back to CPU.
void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

namespace detail {

/// @brief Count of nodes that received metadata during preserve_dynamic_filter_metadata.
/// Surfaced for diagnostic logging and unit-test assertions.
struct preserved_counts {
  std::size_t joins{0};
  std::size_t gets{0};
};

/// @brief Deep-copy a @c JoinFilterPushdownInfo, sharing the @c DynamicTableFilterSet shared_ptrs
/// (preserving the route-key pointer identity that pairs joins with their downstream scans).
/// Omits @c min_max_aggregates — Sirius does not consume them and cloning @c Expression trees
/// is non-trivial.
[[nodiscard]] duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> clone_filter_pushdown_info(
  duckdb::JoinFilterPushdownInfo const& src);

/// @brief Walk @p original and @p copy in parallel and re-attach
/// @c LogicalComparisonJoin::filter_pushdown / @c LogicalGet::dynamic_filters from original onto
/// copy. @c LogicalOperator::Copy round-trips through serialize/deserialize, and these fields
/// are absent from both operators' serialization schemas — without this fixup the copy has them
/// null. Bails defensively at any structural mismatch.
void preserve_dynamic_filter_metadata(duckdb::LogicalOperator& original,
                                      duckdb::LogicalOperator& copy,
                                      preserved_counts& counts);

}  // namespace detail

/// @brief Copy a logical plan for Sirius's transparent execution path. Wraps
/// @c duckdb::LogicalOperator::Copy and additionally preserves the dynamic-filter metadata from
/// the source plan onto the copy: @c LogicalComparisonJoin::filter_pushdown and
/// @c LogicalGet::dynamic_filters. Neither field is in DuckDB's serialization schema, so a plain
/// @c Copy (serialize → deserialize) strips them, and the downstream Sirius wiring would never
/// see the optimizer's pushdown work. Use this at every site that copies a captured logical plan
/// for GPU execution; see @ref detail::preserve_dynamic_filter_metadata for the parallel-walk
/// mechanics.
///
/// @param plan      The plan to copy. Not consumed; @c filter_pushdown / @c dynamic_filters are
///                  left on the original so DuckDB's CPU fallback path can still see them.
/// @param context   DuckDB client context for @c LogicalOperator::Copy.
/// @param counts    Optional out-param recording how many joins/gets received metadata.
[[nodiscard]] duckdb::unique_ptr<duckdb::LogicalOperator> copy_logical_plan(
  duckdb::LogicalOperator& plan,
  duckdb::ClientContext& context,
  detail::preserved_counts* counts = nullptr);

}  // namespace sirius::transparent

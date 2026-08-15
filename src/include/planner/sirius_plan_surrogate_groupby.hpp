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

#include "duckdb/main/client_context.hpp"
#include "op/sirius_physical_operator.hpp"
#include "sirius_config.hpp"

namespace sirius::planner {

/**
 * @brief Surrogate-key group-by pass (late string materialization).
 *
 * For each HASH_GROUP_BY whose STRING group keys are all pure pass-throughs (projection
 * references / INNER-join payload columns) of one side of a single upstream INNER hash join,
 * rewrite the plan so the strings are never carried between that join and the group-by:
 * the join emits a BIGINT rowid at the first deferred slot of each side and constant TINYINT
 * dummies at the rest (column count and positions unchanged everywhere), the group-by
 * aggregates on the numeric carriers, and MERGE_GROUP_BY materializes the strings from the
 * join-retained source batches and restores the original schema. See
 * op/groupby_surrogate_deferral.hpp for the correctness argument.
 *
 * Runs after the compressed-schema passes (physical sidecars at rewritten slots are patched)
 * and before insert_gpu_pipeline_operators (the CONCAT/PARTITION/MERGE wrappers copy the
 * rewritten schemas). Gated by the `groupby_surrogate_keys` operator param.
 */
void apply_groupby_surrogate_keys(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan,
                                  duckdb::ClientContext& context);

/// Params-level entry (also the unit-test seam): applies the knob gate and the tree walk, but
/// none of the context-derived gates (single-GPU). The context overload delegates here.
void apply_groupby_surrogate_keys(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan,
                                  const sirius::operator_params& op_params);

}  // namespace sirius::planner

/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/sirius_physical_operator.hpp"

namespace sirius::planner {

/**
 * @file
 * @brief Compressed-materialization physical-schema passes over the Sirius physical plan.
 *
 * Each operator carries an optional physical sidecar
 * (`sirius_physical_operator::set_physical_types`) describing narrower cuDF carriers for its output
 * columns; an empty sidecar means every column uses its native carrier. The passes here derive,
 * restore, and prune those sidecars after `sirius_plan_get` installs the residency-derived scan
 * sidecars. `sirius_physical_plan_generator::create_plan` invokes them through
 * `apply_compressed_schema_passes` when `enable_compressed_materialization` is on; the individual
 * passes are exposed so planner contract tests can drive each one over a hand-built operator tree.
 *
 * Every pass takes the owning `duckdb::unique_ptr` slot by reference because restoration may
 * replace the slot's operator with a projection wrapping it.
 */

/**
 * @brief Derive each operator's physical output sidecar bottom-up from its children.
 *
 * FILTER, PROJECTION, LIMIT, and HASH_JOIN forward child carriers through their output-column maps;
 * a HASH_JOIN first restores its key columns to native on both inputs. A TABLE_SCAN keeps the
 * sidecar installed by `sirius_plan_get`; when the scan is wired to runtime dynamic-filter
 * producers, each producer's planned target columns are forced native in that sidecar (published
 * filter literals use the native key carrier) while unrelated payload columns keep their carriers,
 * and a producer that declared no targets forces the whole sidecar native. HASH_GROUP_BY keeps
 * bare-reference group keys narrow through the aggregation (grouping is equality-only) while
 * restoring only value-sensitive aggregate inputs on its child. COUNT inputs and columns unused by
 * the aggregate do not constrain their value carriers. Shapes outside its preconditions (multiple
 * grouping sets, grouping functions, AVG or COUNT(DISTINCT) partial layouts) fall through to the
 * native boundary. Every other operator type restores all of its children to native and clears its
 * own sidecar; the `join` and `distinct_root` sub-trees of a DELIM_JOIN are likewise forced native.
 *
 * This pass runs on the pre-wrap tree only: pipeline wrapper operators (PARTITION, CONCAT,
 * MERGE_*, SORT_PARTITION, SORT_SAMPLE, GPU_SCAN, DYNAMIC_FILTER) are inserted afterwards by
 * `sirius_physical_plan_generator::insert_gpu_pipeline_operators`, which copies the finished
 * sidecars onto the wrappers.
 */
void propagate_compressed_schema(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot);

/**
 * @brief Restore @p slot 's output to the native schema, inserting a cast projection when needed.
 *
 * A no-op when the operator has no physical overrides. Otherwise wraps it in a projection that
 * casts every narrowed column back to its logical type and forwards the rest, so the parent
 * observes native carriers.
 */
void restore_native_schema(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot);

/**
 * @brief Remove scan-time narrowing that a restore projection undoes before any batch is
 * materialized narrow.
 *
 * A pruned narrowing is one whose restore sits directly above the scan (join keys,
 * aggregate/ordering inputs, root restores) or is separated from it only by zero-copy
 * pure-reference projections. Such a column pays exact range verification plus a narrowing cast at
 * the scan and a widening cast at the restore without a single narrow batch write in between, so
 * the round trip cannot pay for itself. Pin-time narrowing is unaffected: a pruned (native) sidecar
 * restores resident narrow chunks during scan normalization instead of at the restore projection.
 * Columns whose carrier survives a materializing operator (e.g. scan -> filter -> restore) keep
 * their narrowing, and so does a column another output of the restore projection forwards as a bare
 * reference, so the pruned tree stays indistinguishable from one `propagate_compressed_schema`
 * could have produced.
 */
void prune_immediate_scan_restores(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot);

/**
 * @brief Run the compressed-materialization pass pipeline over a complete plan.
 *
 * A tree with no physical sidecars returns immediately. Otherwise, when every logical type in the
 * tree (including DELIM_JOIN sub-trees) maps to a cuDF carrier, runs `propagate_compressed_schema`,
 * `restore_native_schema`, and `prune_immediate_scan_restores` in that order; an unmappable tree
 * instead clears every sidecar so the plan is entirely native.
 */
void apply_compressed_schema_passes(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan);

}  // namespace sirius::planner

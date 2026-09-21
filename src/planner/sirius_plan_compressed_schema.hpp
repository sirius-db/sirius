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

#include <cudf/types.hpp>

#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @file
 * @brief Compressed-materialization physical-schema passes over the Sirius physical plan.
 *
 * Each operator carries an optional physical sidecar
 * (`sirius_physical_operator::set_physical_types`) describing narrower cuDF carriers for its output
 * columns; an empty sidecar means every column uses its native carrier. The passes here decide,
 * derive, restore, and prune those sidecars after `sirius_plan_get` installs the residency-derived
 * scan sidecars, in the order `apply_tier_narrowing_policy`, `propagate_compressed_schema`,
 * `restore_native_schema`, `prune_immediate_scan_restores`.
 * `sirius_physical_plan_generator::create_plan` invokes them through
 * `apply_compressed_schema_passes` when `enable_compressed_materialization` is on; the individual
 * passes are exposed so planner contract tests can drive each one over a hand-built operator tree.
 *
 * Every restoring pass takes the owning `duckdb::unique_ptr` slot by reference because restoration
 * may replace the slot's operator with a projection wrapping it; `apply_tier_narrowing_policy`
 * takes the operator directly because it only edits scan sidecars in place.
 */

// Implementation details shared between the compressed-schema passes and the tier narrowing
// policy; not part of the pass contract.

/**
 * @brief The native cuDF carrier of each of @p op 's logical output columns, or an empty vector
 * when any column has no cuDF mapping. An empty result is indistinguishable from a legitimate
 * zero-column schema, so callers guard by comparing sizes against the schema they pair it with;
 * `apply_compressed_schema_passes` additionally rejects unmappable trees up front.
 */
[[nodiscard]] std::vector<cudf::data_type> native_physical_schema(
  sirius::op::sirius_physical_operator const& op);

/**
 * @brief Install @p schema as @p op 's physical sidecar, normalizing an all-native schema to the
 * empty sidecar so a nonempty sidecar always describes at least one narrowed column.
 */
void install_physical_schema(sirius::op::sirius_physical_operator& op,
                             std::vector<cudf::data_type> schema);

/**
 * @brief Retract narrow scan targets that show no plan benefit for a GPU-resident pin.
 *
 * The residency gate decides what can be narrow from the carriers stored in the pinned cache; this
 * pass decides what should stay narrow for the query. A GPU-tier serve pays no host-to-GPU upload,
 * so a narrow column must earn its keep inside the plan: the pass walks the tree bottom-up with the
 * same operator column maps `propagate_compressed_schema` uses (including DELIM_JOIN sub-trees) and
 * classifies every use of each candidate column of a scan marked `sidecar_from_gpu_tier_pin`. A
 * column stays narrow iff it survives into a hash-join payload output or an eligible
 * grouped-aggregate key output (transport benefit), or it engages a narrow-domain
 * comparison/BETWEEN (the evaluator's `narrow_domain_carrier` shape, probed against the planned
 * carrier) while no use meets a boundary restore projection. Every other use — an evaluator
 * restore inside an expression, a join key, a value-sensitive aggregate input, an unmodeled
 * operator, or survival to the plan root — costs a restoration, so the column's sidecar entry flips
 * back to native and its pinned-narrow chunks instead widen once per batch during scan
 * normalization. A column with no uses at all stays narrow. Host-tier-backed and sidecar-less scans
 * are not visited because their narrow carriers reduce the host-to-GPU upload. Returns the number
 * of retracted targets.
 */
std::size_t apply_tier_narrowing_policy(sirius::op::sirius_physical_operator& plan);

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
 * A restore qualifies when it sits directly above the scan or is separated from it only by
 * zero-copy pure-reference projections. On a cache hit, pruning moves the required widening of a
 * resident narrow chunk into scan normalization. On a stale plan that reads the source, it avoids a
 * verified downcast followed by an immediate widening cast. Pin-time storage is unchanged. Columns
 * whose carrier survives a materializing operator (for example, scan → filter → restore) keep their
 * target, as does a scan column that another output of the restore projection forwards as a bare
 * reference.
 */
void prune_immediate_scan_restores(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot);

/**
 * @brief Run the compressed-materialization pass pipeline over a complete plan.
 *
 * A tree with no physical sidecars returns immediately. Otherwise, when every logical type in the
 * tree (including DELIM_JOIN sub-trees) maps to a cuDF carrier, runs `apply_tier_narrowing_policy`,
 * `propagate_compressed_schema`, `restore_native_schema`, and `prune_immediate_scan_restores` in
 * that order; an unmappable tree instead clears every sidecar so the plan is entirely native.
 * Returns the number of narrow targets the tier policy retracted, for the caller to record.
 */
std::size_t apply_compressed_schema_passes(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan);

}  // namespace sirius::planner

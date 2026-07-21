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

/**
 * @file dynamic_filter_key_admission.hpp
 * @brief Pure plan-time admission of hash-join build keys for dynamic filtering
 *
 * `sirius_plan_comparison_join` calls these helpers at the **producer-key admission boundary**:
 * the plan-time boundary where DuckDB join conditions and optimizer hints become the Sirius-owned
 * metadata that `dynamic_filter_publish_plan` carries and the runtime publisher consumes. Runtime
 * publication receives the finished plan and does not reinterpret DuckDB metadata.
 *
 * Admission translates four persisted coordinates, which are never interchangeable:
 *
 *  - **condition index** -- position in the planner's original (pre-`wrap_join_conditions`,
 *    pre-reorder) condition vector; `classify_join_key_shapes` results and
 *    `admitted_key::condition_index` live in this space;
 *  - **admitted-key index** -- position in `key_admission_result::admitted_keys`;
 *    `key_binding::admitted_key_index` lives in this space;
 *  - **build-key ordinal** -- position of the materialized key column in the runtime build table;
 *    `admitted_key::build_key_ordinal` lives in this space;
 *  - **channel push ordinal** -- a coordinate in one target channel's push space (see
 *    @ref sirius::op::dynamic_filter_route_class); `dynamic_filter_scan_target_input` push
 *    ordinals and `key_binding::channel_push_ordinal` live in this space.
 *
 * Two vector indexes align the inputs but are not key-coordinate spaces:
 *
 *  - **target index** selects corresponding entries in `scan_targets`,
 *    `probe_targets`, and `per_target_key_bindings`;
 *  - **DuckDB filter ordinal** temporarily zips `hinted_condition_indexes[f]` to
 *    `scan_targets[t].channel_push_ordinals[f]` and
 *    `scan_targets[t].probe_storage_types[f]`. It is not persisted in the result.
 *
 * For example, hints `[2, 0]` and one target's push ordinals `[12, 7]` pair condition 2 with
 * channel ordinal 12 and condition 0 with channel ordinal 7. If condition 0 is rejected, condition
 * 2 keeps ordinal 12 even though the admitted-key array is compacted. For a scan route, the channel
 * later remaps that `column_ids`-space ordinal once into the consumer's output-position space;
 * that downstream output position is not an admission coordinate.
 *
 * Every function here is a pure function of its arguments: no globals, no mutation of inputs.
 * Channel creation and producer registration are side effects and stay in
 * `sirius_physical_plan_generator`; the admission helpers receive already-resolved target
 * descriptors.
 */

#pragma once

#include "duckdb/common/enums/join_type.hpp"
#include "expression/join_condition.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace duckdb {
class Expression;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Classify one join-condition side as a direct reference, a cast of one, or computed
 *
 * The trichotomy matches the physical hash join's key extraction: `BOUND_REF` is direct,
 * `BOUND_CAST` wrapping a `BOUND_REF` is cast, and anything else -- including a cast of a computed
 * expression -- is computed.
 *
 * @param[in] key_side One side of a join condition, before computed-key materialization
 * @return The side's shape
 */
[[nodiscard]] op::dynamic_filter_key_shape classify_key_side(duckdb::Expression const& key_side);

/**
 * @brief Classify both sides of every join condition before computed-key materialization
 *
 * Must be called before `materialize_expression_join_keys` rewrites computed equality sides into
 * plain bound references -- afterwards the shapes are unrecoverable from the conditions alone.
 *
 * @param[in] conditions The join's conditions in original planner order
 * @return One shape per condition, index-aligned with `conditions`
 */
[[nodiscard]] std::vector<op::dynamic_filter_condition_shape> classify_join_key_shapes(
  duckdb::vector<duckdb::JoinCondition> const& conditions);

/**
 * @brief One resolved scan endpoint, described in its channel's push-coordinate space
 *
 * Built by the planner from one probe target that DuckDB's join-filter pushdown metadata named,
 * after that target's channel has been minted. Both vectors are indexed by the temporary DuckDB
 * filter ordinal. At filter ordinal `f`, `hinted_condition_indexes[f]` names the original
 * condition while this target's entries at `f` provide that condition's channel push ordinal and
 * probe storage type. The filter ordinal aligns inputs; it is not itself a channel push ordinal.
 */
struct dynamic_filter_scan_target_input {
  /**
   * @brief One key's landing site in this scan target
   */
  struct scan_target_column {
    /// Where the key lands in this scan's `column_ids` space
    std::size_t channel_push_ordinal = 0;
    /**
     * @brief Probe-side storage type at that coordinate
     *
     * `cudf::type_id::EMPTY` when the DuckDB type has no cuDF representation, which suppresses
     * zone maps for the binding built from this column.
     */
    cudf::data_type probe_storage_type{cudf::type_id::EMPTY};
  };

  /// Indexed by the filter ordinal DuckDB recorded, so a push ordinal cannot become separated
  /// from the probe type it belongs with.
  std::vector<scan_target_column> columns;
};

/**
 * @brief The admission decision for one producing hash join
 */
struct key_admission_result {
  /**
   * @brief Statically admitted build keys, in admitted order
   *
   * Positions in this vector are the admitted-key index space.
   */
  std::vector<op::dynamic_filter_publish_plan::admitted_key> admitted_keys;
  /**
   * @brief Sparse key bindings, aligned one-to-one with the scan-target inputs
   */
  std::vector<std::vector<op::dynamic_filter_publish_plan::key_binding>> per_target_key_bindings;
};

/**
 * @brief Admit build keys and bind them onto resolved scan targets
 *
 * Scan-route legality preserves the runtime publisher's behavior from before producer-key
 * admission moved to this plan-time boundary: a key is admitted when its condition compares with
 * `sirius::comparison_type::equal` (null-equal comparison is outside the supported scope), neither
 * carried side shape is a cast, its build side is a plain bound reference after materialization,
 * and its build type has a cudf representation. Computed (materialized) keys are admitted -- their
 * columns hold the computed values -- and key types are not narrowed.
 *
 * The build ordinal is converted from the DuckDB reference index at exactly one checked point; see
 * the implementation. Works identically when DuckDB's optimizer attached no join-filter pushdown
 * metadata to this join: pass an empty optional and no targets, and the admitted-key array is
 * computed from the conditions alone.
 *
 * @throw std::invalid_argument if `condition_shapes` is not aligned with `conditions`, if
 * `scan_targets` is non-empty without `hinted_condition_indexes`, or if a hinted index or target
 * arity is inconsistent with `conditions`
 *
 * @param[in] conditions The wrapped join conditions in original planner order
 * (post-materialization; wrapping preserves that order)
 * @param[in] condition_shapes Carried pre-materialization classification, aligned with `conditions`
 * @param[in] hinted_condition_indexes The filter-ordinal to condition-index mapping DuckDB
 * recorded, when its optimizer attached join-filter pushdown metadata to this join; admission is
 * then restricted to those conditions so the publisher constructs exactly the filters constructed
 * by the runtime publisher before producer-key admission moved to plan time. Empty optional when
 * DuckDB attached no such metadata: every legality-passing condition is admitted.
 * @param[in] scan_targets Resolved scan endpoints, in the order DuckDB recorded its probe targets;
 * this target index aligns with the returned `per_target_key_bindings` entry, while each target's
 * inner vectors align by DuckDB filter ordinal
 * @param[in] condition_domain_cardinalities Per condition index, the build key's domain
 * cardinality (0 = unknown); empty when no domain evidence exists. Recorded onto each admitted
 * key, so the result carries no parallel array.
 * @return The admitted keys and their per-target bindings. `per_target_key_bindings` always has
 * exactly one entry per element of `scan_targets`.
 */
[[nodiscard]] key_admission_result admit_dynamic_filter_keys(
  duckdb::vector<sirius::join_condition> const& conditions,
  std::vector<op::dynamic_filter_condition_shape> const& condition_shapes,
  std::optional<std::span<std::size_t const>> hinted_condition_indexes,
  std::vector<dynamic_filter_scan_target_input> const& scan_targets,
  std::vector<std::size_t> const& condition_domain_cardinalities);

/**
 * @brief Static legality of one admitted key for a join-edge (direct) endpoint
 *
 * The join-edge endpoint installed at the producing join's probe boundary (issue #1010; its
 * documentation unit delivers the design as
 * `docs/super-sirius/issue-1010-dynamic-filter-sip-design-v2.md`) accepts a strictly narrower scope
 * than the scan route: the producing join must be INNER or left SEMI, the comparison `equal`, both
 * carried side shapes direct -- a computed key is rejected here even though the scan route admits
 * it -- and the key type a matching INT32 or INT64 on both sides. Consumed by the join-edge
 * placement helper; exercised by unit tests until that helper lands.
 *
 * @param[in] join_type The producing join's type
 * @param[in] comparison The condition's comparison operator
 * @param[in] shape The condition's carried pre-materialization side shapes
 * @param[in] probe_storage_type The probe-side key storage type
 * @param[in] build_storage_type The build-side key storage type
 * @return True when the key may be applied at a join-edge endpoint
 */
[[nodiscard]] bool direct_route_admissible(duckdb::JoinType join_type,
                                           sirius::comparison_type comparison,
                                           op::dynamic_filter_condition_shape shape,
                                           cudf::data_type probe_storage_type,
                                           cudf::data_type build_storage_type) noexcept;

}  // namespace sirius::planner

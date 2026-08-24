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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace duckdb {
class LogicalComparisonJoin;
}  // namespace duckdb

namespace sirius::planner {

/// Why a DELIM join was not lowered to a direct semi/anti join. Every reason is terminal for the
/// candidate: the caller keeps the regular delim lowering. The reasons are load-bearing (unit
/// tests pin them per plan shape), so add new ones instead of repurposing existing ones.
enum class delim_direct_refusal : uint8_t {
  none = 0,  ///< Eligible — no refusal.
  /// The delim join's type is not SEMI/ANTI/RIGHT_SEMI/RIGHT_ANTI. This is the reason that
  /// refuses the scalar-aggregate correlation shapes (TPC-H q2/q20 plan as LEFT, q17 as RIGHT):
  /// their delim join must *extend* outer rows with a correlated aggregate, which a membership
  /// join cannot express.
  unsupported_join_type,
  /// The join type is in the semi/anti family but the duplicate-eliminated side is not the
  /// membership-output side (e.g. a flipped delim with a left-SEMI type). The collapse argument
  /// requires the dedup keys to come from the side the join emits.
  orientation_mismatch,
  /// The delim-scan side is not the canonical dedup sandwich:
  /// [column-reference-only PROJECTION ->] INNER comparison join with a bare DELIM_GET child.
  sandwich_shape,
  /// The inner relation kept by the rewrite still contains a DELIM_GET, i.e. the delim data has
  /// a consumer beyond the membership sandwich (nested correlation). Collapsing would orphan it.
  residual_delim_consumer,
  /// The correlated inner join carries a condition outside the equality family (=, IS NOT
  /// DISTINCT FROM). TPC-H q21's `<>` correlations land here. The rewrite itself would still be
  /// sound, but this pass only claims the pure-equality EXISTS / NOT EXISTS shapes.
  non_equality_correlation,
  /// A correlated inner-join condition's dedup-key side is not a plain column reference into the
  /// DELIM_GET (e.g. a cast), so the key cannot be traced to a duplicate-eliminated column.
  inner_condition_shape,
  /// The delim join's own conditions are not the canonical join-back — each pairing one
  /// duplicate-eliminated source column with the same dedup key it produced.
  join_back_shape,
  /// Some duplicate-eliminated column is not pinned by any join-back condition, so an outer row
  /// is not provably matched against its own correlation key.
  delim_column_mismatch,
  /// A DELIM_GET column's type differs from the duplicate-eliminated column that produced it, so
  /// the correlated condition was typed against a different value than the one the rewrite
  /// substitutes. Also the type half of the DELIM_GET-ownership proof (see nested_delim_context).
  delim_column_type_mismatch,
  /// The candidate sits inside another DELIM join that kept its delim lowering, so the matched
  /// DELIM_GET's owner is not locally provable -- it could be the enclosing join's. Supplied by
  /// sirius_physical_plan_generator, which is the only place that knows what is still being
  /// planned above the cursor; classify_delim_direct_lowering, being a pure per-join analysis,
  /// never returns it.
  nested_delim_context,
  /// A plain `=` join-back drops NULL-keyed outer rows in ways the direct join cannot
  /// reproduce: either it is paired with a null-safe correlated condition (the direct join would
  /// let the dropped rows match), or its key column has no correlated condition at all (the
  /// rewrite deletes the join-back and carries no condition on the column).
  null_safety,
  /// The delim join or the inner join carries a residual ON-clause predicate the rewrite does
  /// not model.
  residual_predicate,
};

/// Human-readable, log-stable name for a refusal reason.
const char* to_string(delim_direct_refusal refusal);

/// Result of classifying a DELIM join for direct semi/anti lowering. When `refusal == none` the
/// remaining fields describe the proven match and can be fed to apply_delim_direct_lowering.
struct delim_direct_analysis {
  delim_direct_refusal refusal = delim_direct_refusal::unsupported_join_type;

  /// Index of the delim-join child holding the dedup sandwich (the DELIM_GET side).
  std::size_t sandwich_index = 0;
  /// Index of the inner join's child that is the DELIM_GET.
  std::size_t delim_get_index = 0;
  /// The sandwich's correlated INNER join (borrowed from inside op's sandwich child).
  duckdb::LogicalComparisonJoin* inner_join = nullptr;
  /// Per inner-join condition: the duplicate-eliminated column its dedup-key side references.
  std::vector<std::size_t> dedup_column_of_condition;

  [[nodiscard]] bool eligible() const { return refusal == delim_direct_refusal::none; }
};

/// Classify @p op (a LOGICAL_DELIM_JOIN) for lowering to a direct semi/anti comparison join.
/// Pure analysis: collects the candidate sandwich, matches its shape, and proves the collapse is
/// semantics-preserving (sole-consumer, join-back coverage, NULL-key safety). Never mutates the
/// plan. The result borrows into op's subtree and is valid only while op is unmodified.
[[nodiscard]] delim_direct_analysis classify_delim_direct_lowering(
  duckdb::LogicalComparisonJoin& op);

/// Rewrite @p op in place into the direct comparison join described by @p analysis, always in
/// right-family form: probe = the inner relation (children[0]), build = the outer relation
/// (children[1]), join type RIGHT_SEMI / RIGHT_ANTI. The conditions become the correlated join's
/// conditions with the dedup key substituted by its outer source column. Consumes the analysis
/// (one-shot): it must be eligible and @p op must be the same, unmutated operator it was produced
/// from. After the call, op is a plain LOGICAL_COMPARISON_JOIN ready for plan_comparison_join,
/// whose native dynamic-filter discovery re-derives any probe-scan membership filter (the pass
/// itself carries no filter metadata).
void apply_delim_direct_lowering(duckdb::LogicalComparisonJoin& op,
                                 delim_direct_analysis&& analysis);

}  // namespace sirius::planner

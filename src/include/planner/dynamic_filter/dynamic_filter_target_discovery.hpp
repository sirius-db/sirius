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
 * @file dynamic_filter_target_discovery.hpp
 * @brief Discovers one producing join's dynamic-filter targets in its built probe subtree
 *
 * `trace_probe_key()` follows one admitted probe key through the physical probe subtree.
 * `TABLE_SCAN` terminals are scan-binding sites; other terminals can be wrapped by
 * `place_endpoint()`. Both operations use @ref descent_steps. Each accepted step remaps the key
 * ordinal into its child's output space.
 */

#pragma once

// duckdb
#include <duckdb/common/enums/join_type.hpp>
#include <duckdb/common/unique_ptr.hpp>

// cudf
#include <cudf/types.hpp>

// sirius
#include <op/dynamic_filter/top_n_dynamic_filter_publish_plan.hpp>

// stdlib
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sirius::ast {
struct node;
}  // namespace sirius::ast

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::planner {

/**
 * @brief Controls which hops a dynamic-filter trace may take
 */
struct descent_policy {
  /// Whether the trace may enter an intervening join's build block
  bool descend_build_blocks = false;

  /**
   * @brief Restrict hops to the Top-N self-trace set
   *
   * Accepts plain-reference projections, `FILTER` pass-through/gather, and existing endpoints;
   * refuses joins, aggregates, set operations, and every other operator, so each Top-N trace
   * yields exactly one terminal. Positional `UNION` stays a fan-out hop for the join policy but
   * is refused here: Sirius rejects set operations at planning, so the fan-out branch and the
   * multi-branch guards it needs are unreachable and untestable. Restoring it is part of
   * supporting set operations, together with those guards and a runnable test. Stage 7 widens
   * this set per proven hop, not by flipping the bit off. Default false, so join callers keep
   * their existing hop set exactly.
   *
   * The aggregate refusal is keyed to this bit, not to the producer kind: the Stage-5 group-key
   * producer needs its own admission path with inclusive-only predicates, not a relaxation of
   * this one.
   */
  bool top_n_self_trace = false;
};

/**
 * @brief One accepted trace step into an operator child
 */
struct descent_step {
  std::size_t child_index   = 0;  ///< Index of the child to enter
  std::size_t child_ordinal = 0;  ///< Traced key ordinal in the child's output space
};

/**
 * @brief Where one branch of a key's trace bottomed out
 */
struct route_terminal {
  sirius::op::sirius_physical_operator* node = nullptr;  ///< Borrowed from the walked subtree
  std::size_t ordinal                        = 0;        ///< In @c node's output space
};

/**
 * @brief Result of splicing endpoints into a subtree
 *
 * `subtree` owns the complete rewrapped subtree. `site_ordinals` holds one entry per spliced site,
 * in the same deterministic pre-order (ascending child index) the walk visits terminals, so a
 * caller minting one channel per `endpoint_factory` invocation can zip its mints with these
 * ordinals. Every plan without a physical set operation has exactly one site.
 */
struct endpoint_placement {
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>
    subtree;  ///< Rewrapped subtree containing the endpoint(s)
  std::vector<std::size_t>
    site_ordinals;  ///< Traced ordinal per site, in the sited operator's output space
};

/**
 * @brief Resolve the input column forwarded by a projection expression
 *
 * @param[in] expression Projection expression for the traced output column
 * @return The referenced input column, or `std::nullopt` when `expression` is not a plain reference
 */
[[nodiscard]] std::optional<std::size_t> projection_reference_input(
  sirius::ast::node const& expression);

/**
 * @brief Resolve the input column forwarded by a group-by output
 *
 * @param[in] group_idx Input column for each grouping-key output
 * @param[in] grouping_set_count Number of grouping sets
 * @param[in] output_ordinal The traced ordinal in the aggregate's output space
 * @return The input column, or `std::nullopt` when the ordinal is not a key of a single grouping
 * set
 */
[[nodiscard]] std::optional<std::size_t> group_by_key_input(std::vector<int> const& group_idx,
                                                            std::size_t grouping_set_count,
                                                            std::size_t output_ordinal);

/**
 * @brief Resolve the join child that supplies one output column
 *
 * The probe block is pure mechanism; the build block is additionally gated on
 * `descent_policy::descend_build_blocks`, because a build-block hop is the SIP capability rather
 * than a property of the join's output layout.
 *
 * @param[in] join_type Join type whose output is being traced
 * @param[in] probe_block_output_columns Probe-child columns emitted by the join
 * @param[in] build_block_output_columns Build-child columns emitted by the join
 * @param[in] output_ordinal The traced ordinal in the join's output space
 * @param[in] policy Which hops the walking producer may take
 * @return The child and its corresponding ordinal, or `std::nullopt` when the join output cannot be
 * traced safely
 */
[[nodiscard]] std::optional<descent_step> join_block_descent(
  duckdb::JoinType join_type,
  std::vector<cudf::size_type> const& probe_block_output_columns,
  std::vector<cudf::size_type> const& build_block_output_columns,
  std::size_t output_ordinal,
  descent_policy policy);

/**
 * @brief Resolve the trace steps through one physical operator
 *
 * Zero steps mean @p node is a terminal for the traced key; one step is a pass-through; more than
 * one is a set-operation fan-out (a physical UNION's output is positionally aligned with every
 * child, so each step carries the same ordinal). Steps are ordered by ascending child index.
 *
 * @param[in] node The operator to descend through
 * @param[in] output_ordinal The traced ordinal in @p node's output space
 * @param[in] policy Which hops the walking producer may take
 * @return The accepted child steps; empty when the trace terminates at @p node
 */
[[nodiscard]] std::vector<descent_step> descent_steps(
  sirius::op::sirius_physical_operator const& node,
  std::size_t output_ordinal,
  descent_policy policy);

/**
 * @brief Pure classification of where one key's trace bottoms out
 *
 * Follows `descent_steps()` pre-order with ascending child index and returns one terminal per
 * reached branch; no plan without a physical set operation produces more than one. The walk
 * mutates nothing and transfers no ownership; terminal node pointers are borrowed from the walked
 * subtree.
 *
 * @param[in] root The probe subtree to trace
 * @param[in] a0 The traced key ordinal in @p root's output space (the producing join's probe-child
 * entry ordinal)
 * @param[in] policy Which hops this producer may take
 * @return The terminal node and exit ordinal of every reached branch
 */
[[nodiscard]] std::vector<route_terminal> trace_probe_key(
  sirius::op::sirius_physical_operator& root, std::size_t a0, descent_policy policy);

/**
 * @brief Where one branch of an all-keys trace bottomed out
 *
 * The multi-key counterpart of @ref route_terminal: every traced ordinal survives to @p node,
 * remapped independently, with key zero's ordinal first.
 */
struct multi_key_route_terminal {
  sirius::op::sirius_physical_operator* node = nullptr;  ///< Borrowed from the walked subtree
  std::vector<std::size_t> ordinals;  ///< In @c node's output space, primary (key zero) first
  std::size_t material_hops = 0;      ///< Accepted hops over per-row work between the Top-N input
                                      ///< and @c node (see @ref hop_is_material)
};

/**
 * @brief Whether an accepted hop crosses per-row work a target below it would save
 *
 * Material today means `FILTER`: a target below it prunes rows the filter would otherwise
 * evaluate. Pass-through projections, `UNION` fan-out, and existing endpoints move no work, so a
 * site separated from the Top-N input by those alone saves nothing -- the sink prefilter already
 * applies the same predicate over the same rows.
 *
 * Deliberate under-approximation: an accepted PROJECTION hop may still compute non-traced
 * expressions per row, and an existing endpoint applies masks per row; both count as immaterial
 * today, so the rule can only under-site -- skipping is always sound. Treating computing
 * projections as material (with a plan-shape test to pin it) is recorded follow-up work.
 */
[[nodiscard]] bool hop_is_material(sirius::op::sirius_physical_operator const& node) noexcept;

/**
 * @brief Trace the full ordinal set toward the deepest schema where every key coexists
 *
 * The all-keys counterpart of @ref trace_probe_key for the LEX layer: a hop is accepted only when
 * every ordinal in @p key_ordinals survives it, each remapped independently by the same
 * @ref descent_steps rules. The terminal carries all remapped ordinals, primary (key zero)
 * first, and the count of material hops above it. Terminates at worst at @p root itself, which
 * always exists. A fan-out step yields one terminal per reached branch.
 *
 * @param[in] root The Top-N child subtree to trace
 * @param[in] key_ordinals Every ORDER BY key's ordinal in @p root's output space, key zero first
 * @param[in] policy Which hops this producer may take
 * @return The terminal node, remapped ordinals, and material-hop count of every reached branch
 */
[[nodiscard]] std::vector<multi_key_route_terminal> trace_top_n_all_keys(
  sirius::op::sirius_physical_operator& root,
  std::span<std::size_t const> key_ordinals,
  descent_policy policy);

/**
 * @brief Whether table function @p function_name builds the Parquet reader
 *
 * The single name list shared by @ref target_skips_reads and the plan generator's scan-leaf
 * dispatch (`wrap_table_scan_source`), so the siting decision and the reader construction cannot
 * disagree about which functions carry a read-time filter path.
 */
[[nodiscard]] bool is_parquet_reader_function(std::string const& function_name) noexcept;

/**
 * @brief Whether a predicate published at @p node reduces the data @p node reads
 *
 * The consumer half of the siting rule, asked of the target rather than inferred from its format:
 * a scan whose reader evaluates dynamic-filter predicates while reading turns the predicate into
 * row groups it never fetches or decodes, which is a saving nothing downstream has to justify.
 * Every other target -- a sited endpoint, and any scan whose only filter path runs after decode --
 * answers false, because its rows are already read by the time the predicate can be applied.
 *
 * The answer follows the reader a scan will be given, which the table function it carries decides:
 * `parquet_scan` / `read_parquet` / `sirius_read_parquet` build the Parquet reader, whose
 * `reader_options::set_filter` prunes by row-group statistics; `seq_scan` builds the DuckDB-native
 * reader, which has no read-time filter hook at all. `wrap_table_scan_source` asks the same
 * question to decide whether the scan's dynamic-filter wrapper must re-evaluate AST-capable
 * filters after decode, so the two decisions cannot drift apart. When a backend gains a read-time
 * filter path, this function is the only place that changes -- neither the siting rule nor its
 * callers mention a format.
 *
 * The answer is a plan-time property of the reader the scan would be given; serving can falsify
 * it at runtime -- a pinned-cache hit runs no reader -- in which case consumption flips to the
 * post-decode wrapper at prepare time via `read_time_filter_bypass` while the siting decision
 * stands (main doc, "Pinned-cache-served scans").
 */
[[nodiscard]] bool target_skips_reads(sirius::op::sirius_physical_operator const& node) noexcept;

/**
 * @brief How one Top-N trace terminal should be consumed
 */
enum class top_n_target_kind {
  SCAN_BIND,              ///< Bind into the scan's reader AST
  ENDPOINT_SITE,          ///< Splice a sited endpoint above the terminal
  SKIPPED_NO_WORK_SAVED,  ///< Saves no work the sink prefilter does not already save
  SUBSUMED_BY_LEX,        ///< A LEX target already covers this site, which implies this bound
  LEX_ENDPOINT_DEFERRED   ///< Worth siting, but multi-ordinal placement lands in Stage 7
};

/**
 * @brief Classify one Top-N trace terminal into its target kind
 *
 * A terminal is worth siting only when it saves work nothing else already saves (main doc,
 * "Siting rule: a target must save work it would not otherwise save"). At least one of these must
 * hold:
 *
 *  - @p consumer_skips_reads -- the target turns the predicate into data it never reads, so the
 *    saving is upstream of any pass and nothing downstream is required to justify it.
 *  - @p material_hops_above is non-zero -- per-row work sits between the site and the sink, so the
 *    site's own compaction pass buys more than it costs on every rejected row.
 *
 * A terminal meeting neither is @c SKIPPED_NO_WORK_SAVED: sink self-consumption already applies
 * the same predicate over the same rows, so the site would buy one O(rows) pass to save an
 * identical one. This test governs scan binds exactly as it governs endpoints -- a scan was
 * previously exempt on the premise that binding a scan always avoids reads, which holds for a
 * reader with a read-time filter path and fails for one whose only path is post-decode.
 *
 * A terminal that passes the test is a scan bind when it is a supported scan (for the all-keys
 * trace, every component must land on a supported decoded scan column -- the caller checks that
 * before calling) and a sited endpoint otherwise. A `FIRST_KEY` terminal coinciding with a planned
 * `LEX` target's site is marked subsumed by the caller, which alone knows the LEX sites; pass
 * @p coincides_with_lex_site to have it reported here.
 *
 * A `LEX` terminal that is not a scan reports @c LEX_ENDPOINT_DEFERRED rather than
 * @c ENDPOINT_SITE: siting it needs multi-ordinal placement, which lands in Stage 7. Keeping it
 * distinct from @c SKIPPED_NO_WORK_SAVED is deliberate -- the skip counter must mean "the cost
 * gate said no", never "this stage cannot express it".
 *
 * @param[in] terminal Where the trace bottomed out
 * @param[in] layer Which layer this terminal would carry
 * @param[in] material_hops_above Accepted hops over per-row work between the site and the sink
 * @param[in] consumer_skips_reads Whether the target reads less because of the predicate, from
 * @ref target_skips_reads
 * @param[in] coincides_with_lex_site Whether a planned LEX target sits at the same node
 */
[[nodiscard]] top_n_target_kind classify_top_n_terminal(route_terminal const& terminal,
                                                        sirius::op::top_n_filter_layer layer,
                                                        std::size_t material_hops_above,
                                                        bool consumer_skips_reads,
                                                        bool coincides_with_lex_site = false);

/**
 * @brief Whether a boundary key may be published against a site column of @p site_column_type
 *
 * A key is compared as its raw representation everywhere downstream -- the kernel widens it,
 * `exact_host_scalar::compare` widens it, and neither consults the type. For a fixed-point key
 * that is correct only when both sides carry the same scale, so any mismatch refuses the site
 * rather than rescaling: rescaling is where precision is silently lost. The check is stated for
 * every key type rather than only the scaled ones, because every traced hop is type-preserving and
 * so the equality holds regardless -- asserting it is free and leaves no type-specific carve-out
 * for a later widening to overlook.
 *
 * @param[in] key_storage_type The key's admitted cuDF storage type, from the planner's allowlist
 * @param[in] site_column_type The cuDF type of the column this target would bind at
 */
[[nodiscard]] bool boundary_key_matches_site_type(cudf::data_type key_storage_type,
                                                  cudf::data_type site_column_type) noexcept;

/**
 * @brief Whether a producing join of type @p t may bind a key INTO a probe-side scan
 *
 * Mirrors the early-return gate of DuckDB's `JoinFilterPushdownOptimizer::GenerateJoinFilters`,
 * which generates scan filters only for INNER, RIGHT, and SEMI producers. The other types preserve
 * or negate unmatched probe rows, so pre-filtering the probe by build-key membership would change
 * results. RIGHT is sound even though it is not direct-route admissible: a RIGHT join keeps every
 * BUILD row and drops probe rows that match nothing, so removing probe rows whose key is outside
 * the build-key set removes only rows the join was going to drop -- the NULL-padded rows a RIGHT
 * join fabricates come from unmatched BUILD rows, which a probe-side filter cannot touch.
 */
[[nodiscard]] constexpr bool scan_route_join_type_admissible(duckdb::JoinType t) noexcept
{
  return t == duckdb::JoinType::INNER || t == duckdb::JoinType::RIGHT ||
         t == duckdb::JoinType::SEMI;
}

/**
 * @brief Factory for a childless endpoint that matches the operator it will wrap
 *
 * `place_endpoint()` passes the selected operator to the factory, then installs that operator as
 * the returned endpoint's child. The factory is invoked once per spliced site, in the same
 * deterministic order `trace_probe_key()` reports terminals.
 */
using endpoint_factory = std::function<duckdb::unique_ptr<sirius::op::sirius_physical_operator>(
  sirius::op::sirius_physical_operator const& site_node)>;

/**
 * @brief Place an endpoint at the deepest safe site of every reached branch for one probe key
 *
 * Starting with @p a0 in @p subtree's output space, the function follows `descent_steps()` until
 * an operator refuses the trace, then inserts an endpoint immediately above that operator. If the
 * root refuses, the endpoint wraps the root. A fan-out step splices one endpoint per reached
 * branch. The factory has no scan knowledge, so a caller that scan-binds any branch of a key must
 * not call this for that key. On mixed branches, unbound branches therefore receive no direct
 * endpoint.
 *
 * Existing `sirius_physical_dynamic_filter` operators are transparent to the trace, so placing one
 * key does not prevent later keys from reaching the same depth.
 *
 * @pre The producing join compares the key with `sirius::comparison_type::equal`;
 * `direct_route_admissible()` enforces this before placement.
 * @pre @p subtree is not null.
 *
 * @param[in] subtree Probe subtree whose ownership transfers to the result
 * @param[in] a0 The traced key ordinal in @p subtree's output space
 * @param[in] policy Which hops this producer may take
 * @param[in] make_endpoint Factory for the endpoint operator(s)
 * @return The rewrapped subtree and the traced ordinal at each spliced site
 */
[[nodiscard]] endpoint_placement place_endpoint(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree,
  std::size_t a0,
  descent_policy policy,
  endpoint_factory const& make_endpoint);

}  // namespace sirius::planner

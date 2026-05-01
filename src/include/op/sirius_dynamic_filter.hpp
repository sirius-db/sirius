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

// cudf
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sirius::op {

/**
 * @brief Kind tag for @ref sirius_dynamic_filter subtypes.
 *
 * Extend this enum as new filter kinds (bloom, IN-list, etc.) are added in later phases.
 * See @c docs/super-sirius/dynamic-filters.md for the staged rollout.
 */
enum class sirius_dynamic_filter_kind { ZONE_MAP };

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Polymorphic runtime-computed filter produced by one operator and consumed by another.
 *
 * Produced by some upstream operator (e.g., a hash-join build) and delivered to a downstream
 * consumer (e.g., a parquet scan) through a @ref sirius_dynamic_filter_set.
 *
 * Concrete filters expose their consumer-side capabilities by inheriting from one or more
 * capability mixins (today: @ref sirius_ast_lowerable; future: a runtime-apply mixin). The base
 * class carries only the kind tag because not every filter kind supports every lowering path —
 * for example, a bloom filter cannot produce a cuDF AST fragment. Consumers @c dynamic_cast a
 * @ref sirius_dynamic_filter pointer to the capability they need; a failed cast means the filter
 * does not support that path.
 */
class sirius_dynamic_filter {
 public:
  virtual ~sirius_dynamic_filter() = default;

  /// The kind of this filter.
  [[nodiscard]] virtual sirius_dynamic_filter_kind kind() const = 0;
};

//===----------------------------------------------------------------------===//
// AST mix-in
//===----------------------------------------------------------------------===//
/**
 * @brief Capability mixin: filter can lower itself to a cuDF AST fragment.
 *
 * Filters inheriting from this interface support consumer paths that build a @c cudf::ast::tree
 * (e.g., parquet reader @c set_filter, expression executor).
 */
class sirius_ast_lowerable {
 public:
  virtual ~sirius_ast_lowerable() = default;

  /**
   * @brief Lower the filter to a cuDF AST fragment rooted at a BOOL expression.
   *
   * The fragment, when AND-ed into the consumer's filter tree, rejects rows that this filter
   * excludes. All nodes emitted are owned by @p tree; any device scalars referenced by literals
   * are owned by the implementing filter and must outlive @p tree.
   *
   * @param tree        The consumer's AST tree; new nodes are emplaced into it.
   * @param column_ref  An AST expression naming or referencing the column this filter applies to,
   *                    already emplaced by the caller into @p tree.
   * @return            Reference to the fragment's root expression (BOOL), owned by @p tree.
   */
  [[nodiscard]] virtual cudf::ast::expression const& to_ast(
    cudf::ast::tree& tree, cudf::ast::expression const& column_ref) const = 0;

  /**
   * @brief Construct a fresh AST tree containing only this filter's fragment.
   *
   * Convenience wrapper around @ref to_ast for callers that don't have an existing tree to merge
   * into. The factory emplaces whatever column reference the consumer needs (typically a
   * @c cudf::ast::column_reference or @c cudf::ast::column_name_reference) into the fresh tree
   * and returns it; the filter's fragment is then built on top.
   *
   * @param column_ref_factory Callback that emplaces and returns the column expression in the
   *                           fresh tree. Invoked exactly once.
   * @return                   A new AST tree owning the filter's fragment. The fragment root is
   *                           @c tree.back(). Ownership transfers to the caller.
   */
  [[nodiscard]] cudf::ast::tree to_standalone_ast(
    std::function<cudf::ast::expression const&(cudf::ast::tree&)> const& column_ref_factory) const;
};

/// One zone of a zone-map filter: bounds on the column over a contiguous range of rows.
struct zone_map_entry {
  /// Lower bound scalar on device. Owned by the enclosing filter.
  std::unique_ptr<cudf::scalar> min;
  /// Upper bound scalar on device. Owned by the enclosing filter.
  std::unique_ptr<cudf::scalar> max;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_zone_map_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Multi-zone range filter that keeps rows where any zone's @c [min, max] contains them.
 *
 * Lowers to @c OR_i ( min_i ≤ col AND col ≤ max_i ) (or strict variants based on @c inclusive_*).
 *
 * @note A degenerate single-zone (N=1) filter is the simplest case and is equivalent to a
 *       global min/max range. Multi-zone (N>1) filters retain per-block bounds and prune
 *       more aggressively when build values cluster.
 *
 * @pre  At least one zone must be supplied; every zone's @c min and @c max must be non-null
 *       and share the same @ref cudf::data_type as the column being filtered.
 *
 * @throws std::invalid_argument if zones is empty or any zone has a null bound.
 */
class sirius_dynamic_zone_map_filter final : public sirius_dynamic_filter,
                                             public sirius_ast_lowerable {
 public:
  /**
   * @brief Construct a zone-map filter.
   *
   * @param zones          One entry per zone (ownership transferred).
   * @param inclusive_min  If true, the lower comparison is @c GREATER_EQUAL; else @c GREATER.
   * @param inclusive_max  If true, the upper comparison is @c LESS_EQUAL; else @c LESS.
   */
  explicit sirius_dynamic_zone_map_filter(std::vector<zone_map_entry> zones,
                                          bool inclusive_min = true,
                                          bool inclusive_max = true);

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  { return sirius_dynamic_filter_kind::ZONE_MAP; }

  [[nodiscard]] cudf::ast::expression const& to_ast(
    cudf::ast::tree& tree, cudf::ast::expression const& column_ref) const override;

  [[nodiscard]] std::size_t num_zones() const noexcept { return _zones.size(); }
  [[nodiscard]] std::vector<zone_map_entry> const& zones() const noexcept { return _zones; }
  [[nodiscard]] bool inclusive_min() const noexcept { return _inclusive_min; }
  [[nodiscard]] bool inclusive_max() const noexcept { return _inclusive_max; }

 private:
  std::vector<zone_map_entry> _zones;
  bool _inclusive_min;
  bool _inclusive_max;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter_set
//===----------------------------------------------------------------------===//
/**
 * @brief Thread-safe append-only channel connecting one or more producer operators to a consumer.
 *
 * Filters are keyed by the column index in the consumer's output schema. Multiple producers may
 * push filters for the same column. The set is append-only — once pushed, filters cannot be
 * removed.
 *
 * @note A producer pushes filters for the consumer's column index — i.e. the column index in the
 *       downstream operator's output schema, not the producer's. The plan-gen layer is
 *       responsible for translating between the two when wiring producers and consumers.
 */
class sirius_dynamic_filter_set {
 public:
  /**
   * @brief Register a filter for column @p col_idx. No-op if @p f is null.
   *
   * Thread-safe; may be called concurrently from multiple producer operators. The same
   * @p f may be pushed into multiple channels and/or columns to fan-out a filter without
   * cloning it; the channels co-own the filter.
   */
  void push_filter(std::size_t col_idx, std::shared_ptr<sirius_dynamic_filter const> f);

  /**
   * @brief Snapshot of filters for @p col_idx, in insertion order.
   *
   * The returned snapshots own a share of each filter; they remain valid even if the set
   * itself is later destroyed. Subsequent @ref push_filter calls do not invalidate
   * previously-returned snapshots.
   */
  [[nodiscard]] std::vector<std::shared_ptr<sirius_dynamic_filter const>> filters_for_column(
    std::size_t col_idx) const;

  /// Column indices with at least one registered filter. Order is unspecified.
  [[nodiscard]] std::vector<std::size_t> filtered_columns() const;

  /// True iff no filters have been pushed for any column.
  [[nodiscard]] bool empty() const;

 private:
  mutable std::mutex _mu;
  std::unordered_map<std::size_t, std::vector<std::shared_ptr<sirius_dynamic_filter const>>>
    _filters;
};

/// Resolves a consumer column index to an AST expression already emplaced in the tree (typically
/// a @c cudf::ast::column_reference or @c cudf::ast::column_name_reference).
using column_ref_resolver_fn = std::function<cudf::ast::expression const&(std::size_t col_idx)>;

/**
 * @brief AND-conjoin @p set's AST-lowerable filters with @p existing_root in @p tree.
 *
 * For each column with filters, all AST-capable filters are AND-conjoined (multiple producers);
 * across columns, the per-column conjunctions are AND-conjoined; the final dynamic root is then
 * AND-ed with @p existing_root and that AND-node is returned.
 *
 * If @p set is empty, or every filter declines the AST capability, the function emplaces nothing
 * and returns @p existing_root unchanged.
 *
 * @param tree                The AST tree the consumer is constructing. @p existing_root must
 *                            already be emplaced in @p tree.
 * @param existing_root       The root expression to AND with the dynamic filters' fragments.
 *                            Returned unchanged if no filter contributes.
 * @param set                 The dynamic filter channel.
 * @param column_ref_resolver Callback that returns the AST column expression for a column index.
 *                            Invoked at most once per column with at least one AST-capable filter.
 *                            Must remain valid for the duration of this call.
 * @return Reference to the new root, owned by @p tree. Stable across further @c emplace calls.
 */
[[nodiscard]] cudf::ast::expression const& merge_ast_dynamic_filters_into_tree(
  cudf::ast::tree& tree,
  cudf::ast::expression const& existing_root,
  sirius_dynamic_filter_set const& set,
  column_ref_resolver_fn const& column_ref_resolver);

}  // namespace sirius::op

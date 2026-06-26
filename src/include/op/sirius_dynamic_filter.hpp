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
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op {

/**
 * @brief Kind tag for @ref sirius_dynamic_filter subtypes.
 *
 * Extend this enum as new filter kinds (bloom, IN-list, etc.) are added in later phases.
 * See @c docs/super-sirius/dynamic-filters.md for the staged rollout.
 */
enum class sirius_dynamic_filter_kind { ZONE_MAP, IN_LIST, BLOOM };

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
  {
    return sirius_dynamic_filter_kind::ZONE_MAP;
  }

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
// Runtime-apply mix-in
//===----------------------------------------------------------------------===//
/**
 * @brief Capability mixin: filter can compute a per-row BOOL keep-mask over a probe column.
 *
 * For filter kinds whose predicate cannot be (cheaply) expressed as a cuDF AST — notably set
 * membership (IN-list, bloom) — the consumer applies them at runtime: it calls @ref compute_mask
 * with the materialized probe column and drops rows where the result is false via
 * @c cudf::apply_boolean_mask. This is distinct from @ref sirius_ast_lowerable (which feeds the
 * parquet reader's @c set_filter and row-group stats pruning); a filter may implement either, both,
 * or — for membership — only this one.
 */
class sirius_mask_applicable {
 public:
  virtual ~sirius_mask_applicable() = default;

  /**
   * @brief Compute a BOOL8 keep-mask: @c true where @p probe's value passes this filter.
   *
   * @param probe  The materialized probe column to test (size == output rows).
   * @param stream Stream the work and result are ordered on.
   * @param mr     Allocator for the result.
   * @return A BOOL8 column of size @c probe.size() (true == keep the row), or @c nullptr if the
   *         filter cannot apply to this column (e.g. a type mismatch) — the caller skips it.
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_in_list_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Exact set-membership filter: keeps rows whose key appears on the build side.
 *
 * Built from a hash-join build-key column. Where the
 * zone-map only captures the build keys' @c [min,max] range — useless when the keys are scattered
 * across the domain (e.g. a selective dimension subset) — this tests exact membership, so it prunes
 * exactly the rows the join would discard. It cannot be expressed as a cheap AST or row-group stats
 * predicate, so it rides the @ref sirius_mask_applicable path: a row-level persistent-set probe
 * applied post-decode. Intended for small-to-moderate key sets; larger builds use a bloom filter.
 *
 * @note A probe key equal to the key type's minimum (INT32_MIN / INT64_MIN) is treated as an
 *       empty-slot sentinel by the underlying @c cuco::static_set and will not match, even if that
 *       value was inserted into the build set. This is safe — the join remains authoritative, so a
 *       missed singleton only costs a pruning opportunity, never a false negative — but it requires
 *       that build keys never legitimately use that minimum (a precondition satisfied by
 * TPC-H-style surrogate-key domains).
 */
class sirius_dynamic_in_list_filter final : public sirius_dynamic_filter,
                                            public sirius_mask_applicable {
 public:
  /// @param keys   The build keys. The view only needs to remain valid for the constructor; the
  ///               filter eagerly builds its own persistent set and does not retain the view.
  /// @param stream Stream to build the persistent probe structure on (the producer's stream; the
  ///               publish path synchronizes it before fan-out, so consumer streams never observe
  ///               a partially built set).
  /// @param mr     Device memory resource backing the structure.
  ///
  /// For supported keys (INT32 or INT64 with no nulls, the join-key common case) the constructor
  /// builds a persistent @c cuco::static_set ONCE; every @ref compute_mask is then a single
  /// read-only probe kernel.
  sirius_dynamic_in_list_filter(cudf::column_view const& keys,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr);

  ~sirius_dynamic_in_list_filter() override;

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::IN_LIST;
  }

  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override;

  /// Number of build keys backing the set.
  [[nodiscard]] std::size_t size() const noexcept;

  /// True when the persistent probe structure was built — exposed for tests.
  [[nodiscard]] bool has_persistent_set() const noexcept;

  /// Whether @p keys can back the persistent exact membership set.
  [[nodiscard]] static bool supports(cudf::column_view const& keys) noexcept;

  /// Estimated device footprint (bytes) of the @c cuco::static_set built over an IN-list of
  /// @p num_keys keys of @p key_type — the structure that must stay L2-resident for the per-row
  /// membership probe to run at cache bandwidth (capacity ≈ num_keys / load_factor slots, each
  /// @c sizeof(key)). Consumed by the producer's L2-fit filter-kind policy.
  [[nodiscard]] static std::size_t estimated_set_bytes(std::size_t num_keys,
                                                       cudf::data_type key_type) noexcept;

 private:
  cudf::data_type _key_type{cudf::type_id::EMPTY};
  std::size_t _num_keys = 0;

  /// Persistent cuco::static_set over INT64 keys; PIMPL'd so cuCollections device code stays in
  /// the .cu translation unit.
  struct set_impl;
  std::unique_ptr<set_impl> _set;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_bloom_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Probabilistic set-membership filter backed by a GPU blocked Bloom filter (cuCollections).
 *
 * The scale-up of @ref sirius_dynamic_in_list_filter for *large* selective builds (e.g. a
 * date-filtered @c orders, millions of keys) whose exact key set is too big to store and probe
 * cheaply. A Bloom filter is a few bits per key and answers membership with no false negatives —
 * false *positives* only let through a few extra rows, which is harmless because the join is
 * authoritative. Applied row-level (post-decode) via the @ref sirius_mask_applicable path.
 *
 * Implementation (the @c cuco::bloom_filter and its kernels) is hidden behind a PIMPL so this
 * header stays compilable by the host toolchain; the definitions live in a @c .cu translation unit.
 */
class sirius_dynamic_bloom_filter final : public sirius_dynamic_filter,
                                          public sirius_mask_applicable {
 public:
  /// Build a Bloom filter over the build's join keys. @p keys must be of a @ref supports type.
  /// @throws std::invalid_argument if @c keys.type() is unsupported.
  sirius_dynamic_bloom_filter(cudf::column_view const& keys,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);
  ~sirius_dynamic_bloom_filter() override;

  sirius_dynamic_bloom_filter(sirius_dynamic_bloom_filter const&)            = delete;
  sirius_dynamic_bloom_filter& operator=(sirius_dynamic_bloom_filter const&) = delete;

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::BLOOM;
  }

  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override;

  /// Whether a key/probe column of type @p t can back a Bloom filter (INT32 or INT64).
  [[nodiscard]] static bool supports(cudf::data_type t) noexcept;

  /// Estimated device footprint (bytes) of the Bloom bit array for @p num_keys keys at this
  /// filter's fixed bits-per-key budget. Consumed by the producer's L2-fit filter-kind policy.
  [[nodiscard]] static std::size_t estimated_bytes(std::size_t num_keys) noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter_set
//===----------------------------------------------------------------------===//
/**
 * @brief Thread-safe append-only channel connecting one or more producer operators to a consumer.
 *
 * Filters are keyed by the column index in the consumer's output schema. Multiple producers may
 * push filters for the same column. The set is append-only — once pushed, filters cannot be
 * removed. A consumer may close the channel when its scan pipeline drains; later producer pushes
 * are ignored because no future split can use them.
 *
 * @note A producer pushes filters for the consumer's column index — i.e. the column index in the
 *       downstream operator's output schema, not the producer's. The plan-gen layer is
 *       responsible for translating between the two when wiring producers and consumers.
 *
 * ## Filter availability
 *
 * In Sirius the consumer (probe scan) executes *concurrently* with the producer (hash-join build):
 * injected partition/concat pipelines decouple the probe scan from the join's build dependency, so
 * a scan task may run before, during, or after the producing build finalizes. Consumption is
 * opportunistic: a consumer cheaply tests whether any filter has arrived yet (@ref has_filters) and
 * applies whatever is present when each split runs. A filter that publishes after a split was read
 * simply does not prune it — correctness is unaffected because the join remains authoritative.
 */
class sirius_dynamic_filter_set {
 public:
  /**
   * @brief Register a filter for column @p col_idx. No-op if @p f is null or the channel is
   * closed.
   *
   * @p col_idx is the producer's column reference in the consumer's @b column_ids space — DuckDB
   * hands it over as a @c LogicalGet binding's @c column_index (see @c JoinFilterPushdownColumn).
   * When @ref set_consumer_column_remap has installed a translation, @p col_idx is mapped to the
   * consumer's output-column position before it is stored, so every consumer-side lookup
   * (@ref filters_for_column, @ref filtered_columns, the AST merge, the post-decode apply) keys by
   * output position. A @p col_idx that the remap maps to no output column (pure-filter / pruned /
   * partition) is rejected. With no remap installed the index is stored as-is (the identity case,
   * e.g. tests and scans whose output already matches column_ids order).
   *
   * Thread-safe; may be called concurrently from multiple producer operators. The same
   * @p f may be pushed into multiple channels and/or columns to fan-out a filter without
   * cloning it; the channels co-own the filter.
   *
   * @return true iff the filter was accepted by this channel.
   */
  bool push_filter(std::size_t col_idx, std::shared_ptr<sirius_dynamic_filter const> f);

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

  /// Drop filters targeting these consumer columns — @ref push_filter becomes a no-op for them.
  /// A scan marks its hive-partition columns here: those are pruned at the file level and the
  /// values aren't in the decoded data, so a dynamic filter on them must never reach the consumer.
  /// Indices are in the consumer's output-column space (matching what @ref push_filter stores after
  /// remapping). Wiring-time setup — call before any producer publishes (not synchronized against
  /// push_filter).
  void ignore_columns(std::vector<std::size_t> const& cols);

  /// Install the consumer's column_ids → output-position translation applied by @ref push_filter.
  /// @p remap is indexed by column_ids position and yields the output-column position, or
  /// @c scan_plan::no_output_position for column_ids entries that produce no output. Typically the
  /// scan's @c scan_plan::output_position_by_column_id. Wiring-time setup — call before any
  /// producer publishes. An empty @p remap (the default) means identity: indices are stored
  /// unchanged.
  void set_consumer_column_remap(std::vector<std::size_t> remap);

  /// Mark that a producer has been wired to this channel at plan time.
  void register_producer();

  /// True iff at least one producer can publish into this channel.
  [[nodiscard]] bool has_producers() const noexcept
  {
    return _producer_count.load(std::memory_order_acquire) > 0;
  }

  /// Close the channel once the consumer scan has drained; future pushes cannot prune anything.
  void close_for_new_filters();

  /// True while at least one future consumer split may still observe newly-pushed filters.
  [[nodiscard]] bool accepting_filters() const noexcept
  {
    return _accepting_filters.load(std::memory_order_acquire);
  }

  //===--------------------------------------------------------------------===//
  // Filter availability
  //===--------------------------------------------------------------------===//

  /// Lock-free fast path: true iff at least one filter has been pushed. Cheaper than
  /// @ref empty (no lock) — the consumer's per-task hot check.
  [[nodiscard]] bool has_filters() const noexcept
  {
    return _filter_count.load(std::memory_order_acquire) > 0;
  }

  /// Lock-free count of filters pushed so far, across all columns. Monotonically
  /// non-decreasing; @ref dynamic_filter_gate uses it to detect filters that published
  /// after a disable decision and re-arm itself.
  [[nodiscard]] std::size_t filter_count() const noexcept
  {
    return _filter_count.load(std::memory_order_acquire);
  }

 private:
  mutable std::mutex _mu;
  std::unordered_map<std::size_t, std::vector<std::shared_ptr<sirius_dynamic_filter const>>>
    _filters;
  /// Consumer columns whose filters are dropped on push (e.g. hive partitions); see @ref
  /// ignore_columns.
  std::unordered_set<std::size_t> _ignored_columns;

  /// column_ids → output-position translation applied on push; empty means identity. See
  /// @ref set_consumer_column_remap.
  std::vector<std::size_t> _consumer_col_remap;

  /// Total filters pushed across all columns; backs the lock-free @ref has_filters.
  std::atomic<std::size_t> _filter_count{0};

  /// Number of plan-time producers wired to this channel; zero means the consumer can be elided.
  std::atomic<std::size_t> _producer_count{0};

  /// False once the consumer has drained. Producers use this to skip late filter construction.
  std::atomic<bool> _accepting_filters{true};
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

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

#include "op/dynamic_filter_replica_space.hpp"

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
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op {

/**
 * @brief Kind tag for @ref sirius_dynamic_filter subtypes.
 *
 * Keep this enum in sync with the concrete zone-map, IN-list, and Bloom implementations.
 * See @c docs/super-sirius/dynamic-filters.md for their consumer capabilities.
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
 * capability mixins: @ref sirius_ast_lowerable for reader/AST predicates and @ref
 * sirius_mask_applicable for post-decode masks. The base class carries only the kind tag because
 * not every filter kind supports every application path — for example, a Bloom filter cannot
 * produce a cuDF AST fragment. Consumers @c dynamic_cast a @ref sirius_dynamic_filter pointer to
 * the capability they need; a failed cast means the filter does not support that path.
 */
class sirius_dynamic_filter {
 public:
  virtual ~sirius_dynamic_filter() = default;

  /// The kind of this filter.
  [[nodiscard]] virtual sirius_dynamic_filter_kind kind() const = 0;

  /// True when this filter has a replica local to @p device_id. Consumers use this cheap guard
  /// before lowering an AST; runtime-mask filters also validate the device in @ref compute_mask.
  [[nodiscard]] virtual bool is_available_on_device(int /*device_id*/) const noexcept
  {
    return true;
  }
};

/**
 * @brief Producer-side capability for filters whose device-local representations must be
 * materialized before publication.
 *
 * Replication is deliberately separate from @ref sirius_dynamic_filter: scan consumers only need
 * filter semantics and availability, while the publisher alone owns this construction concern.
 */
class sirius_device_replicable {
 public:
  virtual ~sirius_device_replicable() = default;

  /**
   * @brief Materialize replicas in the supplied GPU memory spaces before publication.
   *
   * Each implementation borrows a stream and allocator from the same target space. The caller
   * retains placement ownership; the completed replica retains only the allocator/stream views
   * whose lifetime is governed by @ref dynamic_filter_replica_space.
   */
  virtual void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) = 0;
};

//===----------------------------------------------------------------------===//
// AST mix-in
//===----------------------------------------------------------------------===//
/**
 * @brief Capability mixin: filter can lower itself to a cuDF AST fragment.
 *
 * Filters inheriting from this interface support consumer paths that build a @c cudf::ast::tree
 * (e.g., parquet reader @c set_filter, expression evaluator).
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
   * @param device_id   Device owning the consumer AST/scalars, or -1 for the current device.
   * @return            Reference to the fragment's root expression (BOOL), owned by @p tree.
   */
  [[nodiscard]] virtual cudf::ast::expression const& to_ast(cudf::ast::tree& tree,
                                                            cudf::ast::expression const& column_ref,
                                                            int device_id = -1) const = 0;

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
 * @note A degenerate single-zone (N=1) filter is the simplest case and is equivalent to a global
 *       min/max range. The representation can retain multiple independently supplied ranges, but
 *       the current hash-join publisher supplies one global zone per key.
 *
 * @pre  At least one zone must be supplied; every zone's @c min and @c max must be non-null
 *       and share the same @ref cudf::data_type as the column being filtered.
 *
 * @throws std::invalid_argument if zones is empty or any zone has a null bound.
 */
class sirius_dynamic_zone_map_filter final : public sirius_dynamic_filter,
                                             public sirius_ast_lowerable,
                                             public sirius_device_replicable {
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

  ~sirius_dynamic_zone_map_filter() noexcept override;

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::ZONE_MAP;
  }

  [[nodiscard]] cudf::ast::expression const& to_ast(cudf::ast::tree& tree,
                                                    cudf::ast::expression const& column_ref,
                                                    int device_id = -1) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] std::size_t num_zones() const noexcept { return _zones.size(); }
  [[nodiscard]] std::vector<zone_map_entry> const& zones() const noexcept { return _zones; }
  [[nodiscard]] bool inclusive_min() const noexcept { return _inclusive_min; }
  [[nodiscard]] bool inclusive_max() const noexcept { return _inclusive_max; }

 private:
  std::vector<zone_map_entry> _zones;
  bool _inclusive_min;
  bool _inclusive_max;
  int _source_device = -1;

  struct device_zones;
  std::vector<std::unique_ptr<device_zones>> _replicas;
};

//===----------------------------------------------------------------------===//
// Runtime-apply mix-in
//===----------------------------------------------------------------------===//
/**
 * @brief Capability mixin: filter can compute a per-row BOOL keep-mask over a probe column.
 *
 * For filter kinds that call @ref compute_mask with the materialized probe column and drops rows
 * where the result is false via @c cudf::apply_boolean_mask. This is distinct from @ref
 * sirius_ast_lowerable (which feeds the parquet reader's @c set_filter and row-group stats
 * pruning); a filter may implement either, both, or — for membership — only this one.
 */
class sirius_mask_applicable {
 public:
  virtual ~sirius_mask_applicable() = default;

  /**
   * @brief Compute a BOOL8 keep-mask: @c true where @p probe's value passes this filter.
   *
   * @param probe  The materialized probe column to test (size == output rows).
   * @param device_id Device owning @p probe and the selected filter replica.
   * @param stream Stream the work and result are ordered on.
   * @param mr     Allocator for the result.
   * @return A BOOL8 column of size @c probe.size() (true == keep the row), or @c nullptr if the
   *         filter cannot apply to this column (e.g. a type mismatch) — the caller skips it.
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_in_list_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Exact set-membership filter: keeps rows whose key appears on the build side.
 *
 * @note Exact for every key except the backing static_set's empty-slot sentinel
 *       (numeric_limits<KeyT>::min()), which the set never stores. A probe key equal to that
 *       sentinel is always kept (never pruned), so a build key equal to it can never be a false
 *       negative -- the authoritative join still filters it. This costs at most a lost pruning
 *       opportunity for that single value. We save an extra kernel pass that would have to scan the
 *       probe set for sentinel values.
 */
class sirius_dynamic_in_list_filter final : public sirius_dynamic_filter,
                                            public sirius_mask_applicable,
                                            public sirius_device_replicable {
 public:
  /// @param keys   The build keys. The view only needs to remain valid for the constructor; the
  ///               filter eagerly builds its own persistent set and does not retain the view.
  /// @param stream Stream to build the persistent probe structure on (the producer's stream; the
  ///               publish path synchronizes it before fan-out, so consumer streams never observe
  ///               a partially built set).
  /// @param mr     Device memory resource backing the structure.
  ///
  /// For supported keys (INT32 or INT64 with no nulls, the join-key common case) the constructor
  /// builds a persistent @c cuco::static_set ; every @ref compute_mask is then a single read-only
  /// probe kernel.
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
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  /// Number of ready device-local replicas (exposed for focused multi-GPU tests/telemetry).
  [[nodiscard]] std::size_t replica_count() const noexcept;

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

  /// Persistent cuco::static_set over INT32 or INT64 keys; PIMPL'd so cuCollections device code
  /// stays in the .cu translation unit.
  struct set_impl;
  std::unique_ptr<set_impl> _set;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_bloom_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Probabilistic set-membership filter backed by a GPU blocked Bloom filter (cuCollections).
 *
 * The scale-up of @ref sirius_dynamic_in_list_filter for *large* selective builds.
 *
 * Implementation (the @c cuco::bloom_filter and its kernels) is hidden behind a PIMPL so this
 * header stays compilable by the host toolchain; the definitions live in a @c .cu translation unit.
 */
class sirius_dynamic_bloom_filter final : public sirius_dynamic_filter,
                                          public sirius_mask_applicable,
                                          public sirius_device_replicable {
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
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  /// Number of ready device-local replicas (exposed for focused multi-GPU tests/telemetry).
  [[nodiscard]] std::size_t replica_count() const noexcept;

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
 * In the normal @c BUILD_PROBE path, build-side @c CONCAT synchronously delivers the complete build
 * batch to the join's publication hook. Construction, device replication, and channel fan-out all
 * complete before that push returns, and downstream task creation reaches the probe data scan only
 * afterwards. Metadata preparation and prefetch may occur earlier, but probe read/decode does not
 * race this normal build-port publication.
 *
 * Consumption is nevertheless opportunistic: there is no readiness wait in this channel API. A
 * consumer snapshots the filters that exist, selects device-local representations, and safely
 * passes data through when publication intentionally emitted nothing or no applicable local filter
 * exists. The append-only/multi-producer behavior also keeps the channel useful outside that normal
 * ordered path; it must not be read as evidence that normal probe scans precede publication.
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

  /// Install the consumer's column_ids -> output-position translation applied by @ref push_filter.
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

  /// Lock-free count of filters pushed so far, across all columns. Monotonically non-decreasing;
  /// @ref dynamic_filter_gate uses it to detect that a generic append-only channel grew beyond the
  /// snapshot on which a disable decision was based.
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

  /// False once the consumer has drained. Producers use this to skip construction that no future
  /// consumer can use.
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

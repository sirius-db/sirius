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

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

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
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op {

/**
 * @brief Kind tag for @ref sirius_dynamic_filter subtypes
 *
 * See `docs/super-sirius/dynamic-filters.md` for each kind's consumer capabilities.
 */
enum class sirius_dynamic_filter_kind { ZONE_MAP, IN_LIST, BLOOM };

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Polymorphic runtime-computed filter produced by one operator and consumed by another
 *
 * Delivered producer-to-consumer through a @ref sirius_dynamic_filter_set. Concrete filters expose
 * consumer capabilities through @ref sirius_ast_lowerable and @ref sirius_mask_applicable;
 * consumers `dynamic_cast` to the required capability, and a failed cast marks that path as
 * unsupported.
 */
class sirius_dynamic_filter {
 public:
  virtual ~sirius_dynamic_filter() = default;

  /**
   * @brief The kind of this filter
   */
  [[nodiscard]] virtual sirius_dynamic_filter_kind kind() const = 0;

  /**
   * @brief Whether this filter has a replica on the requested device
   *
   * Consumers use this cheap guard before lowering an AST; runtime-mask filters also validate the
   * device in @ref compute_mask.
   */
  [[nodiscard]] virtual bool is_available_on_device(int /*device_id*/) const noexcept
  {
    return true;
  }
};

/**
 * @brief Producer-side capability for filters whose device-local representations must be
 * materialized before publication
 */
class sirius_device_replicable {
 public:
  virtual ~sirius_device_replicable() = default;

  /**
   * @brief Materialize replicas in the supplied GPU memory spaces before publication
   *
   * Implementations construct each replica with the target space's stream and allocator; replicas
   * retain only allocator/stream views whose lifetime is governed by
   * @ref dynamic_filter_replica_space. A replica that fails to materialize leaves the filter
   * unavailable on that device.
   *
   * @param[in] spaces Planned GPU and host-staging placements
   */
  virtual void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) = 0;
};

//===----------------------------------------------------------------------===//
// AST mix-in
//===----------------------------------------------------------------------===//
/**
 * @brief Capability mixin: filter can lower itself to a cuDF AST fragment
 *
 * Filters inheriting from this interface support consumer paths that build a `cudf::ast::tree`
 * (e.g., parquet reader `set_filter`, expression evaluator).
 */
class sirius_ast_lowerable {
 public:
  virtual ~sirius_ast_lowerable() = default;

  /**
   * @brief Lower the filter to a cuDF AST fragment rooted at a BOOL expression
   *
   * The fragment, when AND-ed into the consumer's filter tree, rejects rows that this filter
   * excludes. All nodes emitted are owned by @p tree; any device scalars referenced by literals are
   * owned by the implementing filter and must outlive @p tree.
   *
   * @param[in,out] tree The consumer's AST tree; new nodes are emplaced into it.
   * @param[in] column_ref An AST expression naming or referencing the column this filter applies
   * to, already emplaced by the caller into @p tree.
   * @param[in] device_id Device owning the consumer AST/scalars, or -1 for the current device.
   * @return Reference to the fragment's root expression (BOOL), owned by @p tree.
   */
  [[nodiscard]] virtual cudf::ast::expression const& to_ast(cudf::ast::tree& tree,
                                                            cudf::ast::expression const& column_ref,
                                                            int device_id = -1) const = 0;

  /**
   * @brief Construct a fresh AST tree containing only this filter's fragment
   *
   * Convenience wrapper around @ref to_ast for callers that don't have an existing tree to merge
   * into. The factory emplaces whatever column reference the consumer needs (typically a
   * `cudf::ast::column_reference` or `cudf::ast::column_name_reference`) into the fresh tree and
   * returns it; the filter's fragment is then built on top.
   *
   * @param[in] column_ref_factory Callback that emplaces and returns the column expression in the
   * fresh tree. Invoked exactly once.
   * @return A new AST tree owning the filter's fragment. The fragment root is `tree.back()`.
   * Ownership transfers to the caller.
   */
  [[nodiscard]] cudf::ast::tree to_standalone_ast(
    std::function<cudf::ast::expression const&(cudf::ast::tree&)> const& column_ref_factory) const;
};

/**
 * @brief One zone of a zone-map filter: bounds on the column over a contiguous range of rows
 *
 * Both bound scalars live on device and are owned by the enclosing filter.
 */
struct zone_map_entry {
  std::unique_ptr<cudf::scalar> min;  ///< Lower bound on the column
  std::unique_ptr<cudf::scalar> max;  ///< Upper bound on the column
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_zone_map_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Multi-zone range filter that keeps rows where any zone's `[min, max]` contains them
 *
 * Lowers to `OR_i ( min_i <= col AND col <= max_i )` (or strict variants based on `inclusive_*`).
 *
 * @pre Every bound has the same @ref cudf::data_type as the consumer column.
 */
class sirius_dynamic_zone_map_filter final : public sirius_dynamic_filter,
                                             public sirius_ast_lowerable,
                                             public sirius_device_replicable {
 public:
  /**
   * @brief Construct a zone-map filter
   *
   * @throw std::invalid_argument if @p zones is empty or any zone has a null bound
   * @throw std::runtime_error if the current CUDA device cannot be identified
   *
   * @param[in] zones Non-empty zone vector with non-null bounds; ownership transfers to the filter
   * @param[in] inclusive_min Whether the lower comparison includes the bound
   * @param[in] inclusive_max Whether the upper comparison includes the bound
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
 * @brief Capability mixin: filter can compute a per-row BOOL keep-mask over a probe column
 *
 * Distinct from @ref sirius_ast_lowerable; a filter may implement either capability or both.
 */
class sirius_mask_applicable {
 public:
  virtual ~sirius_mask_applicable() = default;

  /**
   * @brief Compute a BOOL8 keep-mask: `true` where @p probe's value passes this filter
   *
   * @param[in] probe The materialized probe column to test (size == output rows).
   * @param[in] device_id Device owning @p probe and the selected filter replica.
   * @param[in] stream Stream the work and result are ordered on.
   * @param[in] mr Allocator for the result.
   * @return A BOOL8 column of size `probe.size()` (true == keep the row), or `nullptr` if the
   * filter cannot apply to this column (e.g. a type mismatch) -- the caller skips it.
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
 * @brief Exact set-membership filter: keeps rows whose key appears on the build side
 *
 * @note Exact for every key except the backing static_set's empty-slot sentinel
 * (`numeric_limits<KeyT>::min()`), which the set never stores. A probe key equal to that sentinel
 * is always kept (never pruned), so a build key equal to it can never be a false negative -- the
 * authoritative join still filters it.
 */
class sirius_dynamic_in_list_filter final : public sirius_dynamic_filter,
                                            public sirius_mask_applicable,
                                            public sirius_device_replicable {
 public:
  /**
   * @brief Construct an exact set-membership filter over the build keys
   *
   * The constructor builds a persistent `cuco::static_set`; each @ref compute_mask is a read-only
   * probe.
   *
   * @pre The backing storage for @p keys remains valid until work enqueued on @p stream completes.
   *
   * @throw std::invalid_argument if @p keys is not a null-free INT32 or INT64 column
   * @throw std::runtime_error if the current CUDA device cannot be identified
   * @throw std::logic_error if the validated key type changes during construction
   *
   * @param[in] keys Null-free INT32 or INT64 build keys
   * @param[in] stream Stream used to build the persistent set
   * @param[in] mr Device memory resource backing the set
   */
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

  /**
   * @brief Number of ready device-local replicas (exposed for focused multi-GPU tests/telemetry)
   */
  [[nodiscard]] std::size_t replica_count() const noexcept;

  /**
   * @brief Number of build keys backing the set
   */
  [[nodiscard]] std::size_t size() const noexcept;

  /**
   * @brief True when the persistent probe structure was built -- exposed for tests
   */
  [[nodiscard]] bool has_persistent_set() const noexcept;

  /**
   * @brief Whether @p keys can back the persistent exact membership set
   */
  [[nodiscard]] static bool supports(cudf::column_view const& keys) noexcept;

  /**
   * @brief Estimated device footprint in bytes of the `cuco::static_set` built over @p num_keys
   * keys of @p key_type
   */
  [[nodiscard]] static std::size_t estimated_set_bytes(std::size_t num_keys,
                                                       cudf::data_type key_type) noexcept;

 private:
  cudf::data_type _key_type{cudf::type_id::EMPTY};
  std::size_t _num_keys = 0;

  /**
   * @brief Persistent `cuco::static_set` over INT32 or INT64 keys; PIMPL'd so cuCollections device
   * code stays in the `.cu` translation unit
   */
  struct set_impl;
  std::unique_ptr<set_impl> _set;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_small_in_list_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Exact linear-scan membership filter for at most @ref k_max_keys build keys
 *
 * Each device replica owns a raw snapshot of the build values. `compute_mask()` compares each probe
 * value with every snapshot value and therefore reserves no sentinel value.
 */
class sirius_dynamic_small_in_list_filter final : public sirius_dynamic_filter,
                                                  public sirius_mask_applicable,
                                                  public sirius_device_replicable {
 public:
  /**
   * @brief Maximum number of build keys this filter accepts
   */
  static constexpr std::size_t k_max_keys = 12;

  /**
   * @brief Construct an exact set-membership filter over a small set of build keys
   *
   * @pre The backing device storage for @p keys remains valid until the snapshot copy enqueued on
   * @p stream completes.
   *
   * @throw std::invalid_argument if @p keys is unsupported
   * @throw std::runtime_error if the current CUDA device cannot be identified
   *
   * @param[in] keys Build keys accepted by @ref supports; the column-view object need only remain
   * valid for the constructor call.
   * @param[in] stream Stream the source snapshot is enqueued on.
   * @param[in] mr Device memory resource backing the source snapshot.
   */
  sirius_dynamic_small_in_list_filter(cudf::column_view const& keys,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

  ~sirius_dynamic_small_in_list_filter() override;

  sirius_dynamic_small_in_list_filter(sirius_dynamic_small_in_list_filter const&) = delete;
  sirius_dynamic_small_in_list_filter& operator=(sirius_dynamic_small_in_list_filter const&) =
    delete;
  sirius_dynamic_small_in_list_filter(sirius_dynamic_small_in_list_filter&&)            = delete;
  sirius_dynamic_small_in_list_filter& operator=(sirius_dynamic_small_in_list_filter&&) = delete;

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

  /**
   * @brief Number of ready device-local replicas (exposed for focused multi-GPU tests/telemetry)
   */
  [[nodiscard]] std::size_t replica_count() const noexcept;

  /**
   * @brief Number of build keys (needles) backing the scan
   */
  [[nodiscard]] std::size_t size() const noexcept { return _num_keys; }

  /**
   * @brief Whether @p keys can back the small-list scan: 1..k_max_keys INT32/INT64 keys with no
   * nulls
   */
  [[nodiscard]] static bool supports(cudf::column_view const& keys) noexcept;

 private:
  cudf::data_type _key_type{cudf::type_id::EMPTY};
  std::size_t _num_keys = 0;

  /**
   * @brief Per-device raw needle snapshots; PIMPL'd so rmm/device buffers stay in the `.cu`
   * translation unit
   */
  struct needle_store;
  std::unique_ptr<needle_store> _store;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_bloom_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Probabilistic set-membership filter backed by a GPU blocked Bloom filter
 *
 * False positives pass extra rows to the authoritative join; the filter must not produce false
 * negatives. The `cuco::bloom_filter` implementation is hidden in the CUDA translation unit.
 */
class sirius_dynamic_bloom_filter final : public sirius_dynamic_filter,
                                          public sirius_mask_applicable,
                                          public sirius_device_replicable {
 public:
  /**
   * @brief Build a Bloom filter over the build's join keys
   *
   * @pre @p keys has a type accepted by @ref supports and its backing storage remains valid until
   * work enqueued on @p stream completes.
   *
   * @throw std::invalid_argument if `keys.type()` is unsupported
   * @throw std::runtime_error if the current CUDA device cannot be identified
   * @throw std::logic_error if the validated key type changes during construction
   *
   * @param[in] keys INT32 or INT64 build keys; may be nullable — null slots carry no key value
   * and are excluded from the set (the filter is built from the valid rows only)
   * @param[in] stream Stream used to build the Bloom filter
   * @param[in] mr Device memory resource backing the filter
   */
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

  /**
   * @brief Number of ready device-local replicas (exposed for focused multi-GPU tests/telemetry)
   */
  [[nodiscard]] std::size_t replica_count() const noexcept;

  /**
   * @brief Whether a key/probe column of type @p t can back a Bloom filter (INT32 or INT64)
   */
  [[nodiscard]] static bool supports(cudf::data_type t) noexcept;

  /**
   * @brief Estimated device footprint in bytes of the Bloom bit array for @p num_keys keys at this
   * filter's fixed bits-per-key budget. With nullable build keys the constructed filter is sized
   * from the valid rows, so an estimate from the total row count is an upper bound.
   */
  [[nodiscard]] static std::size_t estimated_bytes(std::size_t num_keys) noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter_set
//===----------------------------------------------------------------------===//
/**
 * @brief Thread-safe append-only channel connecting one or more producer operators to a consumer
 *
 * Push, storage, and lookup share one coordinate: the consumer operator's output ordinal. No
 * translation happens inside the channel; @ref dynamic_filter_route_class documents how each
 * route derives the ordinal.
 *
 * Filters remain immutable and co-owned by the channel and any snapshots returned to consumers.
 * Appends are individually visible; publishing several filters or targets is not atomic. A consumer
 * never waits for readiness and may observe no filters, a prefix of a fan-out, or the completed
 * fan-out.
 *
 * `close_for_new_filters()` ends the append lifecycle after the consumer drains. Later pushes are
 * rejected. Missing filters and missing device-local replicas pass data through; the producing join
 * still checks correctness.
 */
class sirius_dynamic_filter_set {
 public:
  /**
   * @brief Register a filter for consumer output column @p col_idx
   *
   * A null filter, a closed channel, or an ignored output column is rejected without modifying the
   * channel.
   *
   * The function is thread-safe. The same immutable filter may be pushed into multiple channels or
   * columns.
   *
   * @return True when the channel accepted the filter
   */
  bool push_filter(std::size_t col_idx, std::shared_ptr<sirius_dynamic_filter const> f);

  /**
   * @brief Snapshot of filters for @p col_idx, in insertion order
   *
   * The returned snapshots own a share of each filter; they remain valid even if the set itself is
   * later destroyed. Subsequent @ref push_filter calls do not invalidate previously-returned
   * snapshots.
   */
  [[nodiscard]] std::vector<std::shared_ptr<sirius_dynamic_filter const>> filters_for_column(
    std::size_t col_idx) const;

  /**
   * @brief Column indices with at least one registered filter
   *
   * Order is unspecified.
   */
  [[nodiscard]] std::vector<std::size_t> filtered_columns() const;

  /**
   * @brief True iff no filters have been pushed for any column
   */
  [[nodiscard]] bool empty() const;

  /**
   * @brief Drop filters targeting these consumer columns -- @ref push_filter becomes a no-op for
   * them
   *
   * A scan marks its hive-partition columns here: those are pruned at the file level and the values
   * are not in the decoded data, so a dynamic filter on them must never reach the consumer. Indices
   * are in the consumer's output-column space, the same space @ref push_filter stores. Call before
   * publication; the function blocks future pushes for these columns but does not remove filters
   * already stored.
   */
  void ignore_columns(std::vector<std::size_t> const& cols);

  /**
   * @brief Mark that a producer has been wired to this channel at plan time, together with the
   * columns it plans to publish filters for
   *
   * @p planned_target_columns is in the channel's push-coordinate space (the same space @ref
   * push_filter receives). An empty vector registers an unscoped producer: consumers that need
   * the target set must then assume every column is a potential target.
   */
  void register_producer(std::vector<std::size_t> planned_target_columns);

  /**
   * @brief True iff at least one producer can publish into this channel
   */
  [[nodiscard]] bool has_producers() const noexcept
  {
    return _producer_count.load(std::memory_order_acquire) > 0;
  }

  /**
   * @brief Union of all producers' planned target columns, sorted, in the channel's
   * push-coordinate space
   *
   * Meaningful only when has_producers() && !has_unscoped_producer().
   */
  [[nodiscard]] std::vector<std::size_t> planned_target_columns() const;

  /**
   * @brief True iff any producer registered without declaring its target columns
   */
  [[nodiscard]] bool has_unscoped_producer() const noexcept
  {
    return _has_unscoped_producer.load(std::memory_order_acquire);
  }

  /**
   * @brief Close the channel once the consumer scan has drained; future pushes cannot prune
   * anything
   */
  void close_for_new_filters();

  /**
   * @brief True while at least one future consumer split may still observe newly-pushed filters
   */
  [[nodiscard]] bool accepting_filters() const noexcept
  {
    return _accepting_filters.load(std::memory_order_acquire);
  }

  //===--------------------------------------------------------------------===//
  // Filter availability
  //===--------------------------------------------------------------------===//

  /**
   * @brief Lock-free fast path (cheaper than @ref empty): true iff at least one filter has been
   * pushed
   */
  [[nodiscard]] bool has_filters() const noexcept
  {
    return _filter_count.load(std::memory_order_acquire) > 0;
  }

  /**
   * @brief Lock-free count of filters pushed so far, across all columns
   *
   * Monotonically non-decreasing, so a caller can detect that the channel grew past a snapshot.
   */
  [[nodiscard]] std::size_t filter_count() const noexcept
  {
    return _filter_count.load(std::memory_order_acquire);
  }

 private:
  mutable std::mutex _mu;
  std::unordered_map<std::size_t, std::vector<std::shared_ptr<sirius_dynamic_filter const>>>
    _filters;
  /**
   * @brief Consumer columns whose filters are dropped on push (e.g. hive partitions); see @ref
   * ignore_columns
   */
  std::unordered_set<std::size_t> _ignored_columns;

  /**
   * @brief Union of the plan-time producers' planned target columns (channel push-coordinate
   * space); see @ref register_producer
   */
  std::set<std::size_t> _planned_target_columns;

  /**
   * @brief Total filters pushed across all columns; backs the lock-free @ref has_filters
   */
  std::atomic<std::size_t> _filter_count{0};

  /**
   * @brief Number of plan-time producers wired to this channel; backs the lock-free
   * @ref has_producers
   */
  std::atomic<std::size_t> _producer_count{0};

  /**
   * @brief True once any producer registered without declaring its target columns; backs the
   * lock-free @ref has_unscoped_producer
   */
  std::atomic<bool> _has_unscoped_producer{false};

  /**
   * @brief False once the consumer has drained; see @ref close_for_new_filters
   */
  std::atomic<bool> _accepting_filters{true};
};

/**
 * @brief Resolves a consumer column index to an AST expression already emplaced in the tree
 *
 * The expression is typically a `cudf::ast::column_reference` or a
 * `cudf::ast::column_name_reference`.
 */
using column_ref_resolver_fn = std::function<cudf::ast::expression const&(std::size_t col_idx)>;

/**
 * @brief AND-conjoin @p set's AST-lowerable filters with @p existing_root in @p tree
 *
 * For each column with filters, all AST-capable filters are AND-conjoined (multiple producers);
 * across columns, the per-column conjunctions are AND-conjoined; the final dynamic root is then
 * AND-ed with @p existing_root and that AND-node is returned.
 *
 * If @p set is empty, or every filter declines the AST capability, the function emplaces nothing
 * and returns @p existing_root unchanged.
 *
 * @param[in,out] tree The AST tree the consumer is constructing. @p existing_root must already be
 * emplaced in @p tree.
 * @param[in] existing_root The root expression to AND with the dynamic filters' fragments. Returned
 * unchanged if no filter contributes.
 * @param[in] set The dynamic filter channel.
 * @param[in] column_ref_resolver Callback that returns the AST column expression for a column
 * index. Invoked at most once per column with at least one AST-capable filter. Must remain valid
 * for the duration of this call.
 * @return Reference to the new root, owned by @p tree. Stable across further `emplace` calls.
 */
[[nodiscard]] cudf::ast::expression const& merge_ast_dynamic_filters_into_tree(
  cudf::ast::tree& tree,
  cudf::ast::expression const& existing_root,
  sirius_dynamic_filter_set const& set,
  column_ref_resolver_fn const& column_ref_resolver);

}  // namespace sirius::op

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
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op {

/**
 * @brief Kind tag for @ref sirius_dynamic_filter subtypes
 *
 * Keep this enum in sync with the concrete zone-map, IN-list, and Bloom implementations. See
 * `docs/super-sirius/dynamic-filters.md` for their consumer capabilities.
 */
enum class sirius_dynamic_filter_kind { ZONE_MAP, IN_LIST, BLOOM };

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter
//===----------------------------------------------------------------------===//
/**
 * @brief Polymorphic runtime-computed filter produced by one operator and consumed by another
 *
 * Produced by some upstream operator (e.g., a hash-join build) and delivered to a downstream
 * consumer (e.g., a parquet scan) through a @ref sirius_dynamic_filter_set.
 *
 * The base class supplies the kind tag. Concrete filters expose consumer capabilities through
 * @ref sirius_ast_lowerable and @ref sirius_mask_applicable. Consumers `dynamic_cast` a
 * @ref sirius_dynamic_filter pointer to the required capability; a failed cast marks that path as
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
 *
 * Replication is deliberately separate from @ref sirius_dynamic_filter: scan consumers only need
 * filter semantics and availability, while the publisher alone owns this construction concern.
 */
class sirius_device_replicable {
 public:
  virtual ~sirius_device_replicable() = default;

  /**
   * @brief Materialize replicas in the supplied GPU memory spaces before publication
   *
   * Each implementation constructs the replica with the target space's stream and allocator.
   * Membership filters reserve destination capacity for construction; after unused capacity is
   * returned, their completed allocations remain accounted. Replicas retain only allocator/stream
   * views whose lifetime is governed by @ref dynamic_filter_replica_space.
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
 * `dynamic_filter_publisher` supplies one global range per key. The representation also accepts
 * multiple ranges and combines them with OR.
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
 * The consumer calls @ref compute_mask with the materialized probe column and drops rows where the
 * result is false via `cudf::apply_boolean_mask`. This is distinct from @ref sirius_ast_lowerable
 * (which feeds the parquet reader's `set_filter` and row-group stats pruning); a filter may
 * implement either, both, or -- for membership -- only this one.
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
 * authoritative join still filters it. This costs at most a lost pruning opportunity for that
 * single value.
 */
class sirius_dynamic_in_list_filter final : public sirius_dynamic_filter,
                                            public sirius_mask_applicable,
                                            public sirius_device_replicable {
 public:
  /**
   * @brief Construct an exact set-membership filter over the build keys
   *
   * For supported keys (INT32 or INT64 with no nulls, the join-key common case) the constructor
   * builds a persistent `cuco::static_set`; every @ref compute_mask is then a single read-only
   * probe kernel.
   *
   * @pre The backing storage for @p keys remains valid until work enqueued on @p stream completes.
   * `dynamic_filter_publisher` satisfies this by retaining the build batch and synchronizing before
   * replication.
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
   * @brief Estimated device footprint (bytes) of the `cuco::static_set` built over an IN-list of @p
   * num_keys keys of @p key_type
   *
   * The producer uses this estimate for its L2-fit representation policy. Capacity is approximately
   * `num_keys / load_factor` slots.
   *
   * @param[in] num_keys Number of keys in the set
   * @param[in] key_type Key storage type
   * @return Estimated set allocation size in bytes
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
 *
 * The constructor enqueues the source snapshot. `replicate_to_devices()` copies that completed
 * snapshot to each planned probe device before publication; a failed target replica remains
 * unavailable on that device.
 */
class sirius_dynamic_small_in_list_filter final : public sirius_dynamic_filter,
                                                  public sirius_mask_applicable,
                                                  public sirius_device_replicable {
 public:
  /**
   * @brief Maximum number of build keys this filter accepts
   *
   * Above this the producer prefers the cuco IN-list set or a Bloom filter instead (see the
   * hash-join membership policy).
   */
  static constexpr std::size_t k_max_keys = 12;

  /**
   * @brief Construct an exact set-membership filter over a small set of build keys
   *
   * @pre The backing device storage for @p keys remains valid until the snapshot copy enqueued on
   * @p stream completes. The publisher satisfies this precondition by pinning the build
   * representation and synchronizing @p stream before fan-out.
   *
   * @throw std::invalid_argument if @p keys is unsupported
   * @throw std::runtime_error if the current CUDA device cannot be identified
   *
   * @param[in] keys The build keys (INT32/INT64, no nulls; the producer restricts the count to <=
   * @ref k_max_keys). The column-view object need only remain valid for the constructor call.
   * @param[in] stream Producer stream the snapshot is enqueued on; the publish path synchronizes it
   * before fan-out, so consumer streams never observe a partial copy.
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
   *
   * Mirrors @ref sirius_dynamic_in_list_filter's replica store.
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
   * work enqueued on @p stream completes. `dynamic_filter_publisher` retains the build batch and
   * synchronizes before replication.
   *
   * @throw std::invalid_argument if `keys.type()` is unsupported
   * @throw std::runtime_error if the current CUDA device cannot be identified
   * @throw std::logic_error if the validated key type changes during construction
   *
   * @param[in] keys INT32 or INT64 build keys
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
   * @brief Estimated device footprint (bytes) of the Bloom bit array for @p num_keys keys at this
   * filter's fixed bits-per-key budget
   *
   * Consumed by the producer's L2-fit filter-kind policy.
   *
   * @param[in] num_keys Number of keys represented by the filter
   * @return Estimated bit-array size in bytes
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
 * Push, storage, and lookup share one coordinate: the consumer operator's output ordinal. On every
 * route the producing join's discovery walk supplies it as the trace's exit ordinal -- the bound
 * scan's output ordinal for a scan route, the sited operator's output ordinal for a join-edge
 * endpoint -- so no translation happens inside the channel.
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
   * @brief Mark that a producer has been wired to this channel at plan time
   */
  void register_producer();

  /**
   * @brief True iff at least one producer can publish into this channel
   */
  [[nodiscard]] bool has_producers() const noexcept
  {
    return _producer_count.load(std::memory_order_acquire) > 0;
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
   * @brief Lock-free fast path: true iff at least one filter has been pushed
   *
   * Cheaper than @ref empty (no lock) -- the consumer's per-task hot check.
   */
  [[nodiscard]] bool has_filters() const noexcept
  {
    return _filter_count.load(std::memory_order_acquire) > 0;
  }

  /**
   * @brief Lock-free count of filters pushed so far, across all columns
   *
   * Monotonically non-decreasing; @ref dynamic_filter_gate uses it to detect that a generic
   * append-only channel grew beyond the snapshot on which a disable decision was based.
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
   * @brief Total filters pushed across all columns; backs the lock-free @ref has_filters
   */
  std::atomic<std::size_t> _filter_count{0};

  /**
   * @brief Number of plan-time producers wired to this channel; zero means the consumer can be
   * elided
   */
  std::atomic<std::size_t> _producer_count{0};

  /**
   * @brief False once the consumer has drained
   *
   * Producers use this to skip construction that no future consumer can use.
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

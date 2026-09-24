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

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

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

enum class sirius_dynamic_filter_kind { ZONE_MAP, IN_LIST, BLOOM };

/**
 * @brief Runtime filter exposed through AST and/or row-mask capability interfaces
 */
class sirius_dynamic_filter {
 public:
  virtual ~sirius_dynamic_filter() = default;

  [[nodiscard]] virtual sirius_dynamic_filter_kind kind() const = 0;

  [[nodiscard]] virtual bool is_available_on_device(int /*device_id*/) const noexcept
  {
    return true;
  }
};

/**
 * @brief Capability for filters that materialize device-local replicas before publication
 */
class sirius_device_replicable {
 public:
  virtual ~sirius_device_replicable() = default;

  // A failed target remains unavailable; successful replicas must be ready before publication.
  virtual void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) = 0;
};

/**
 * @brief Capability for lowering a filter to a cuDF AST
 */
class sirius_ast_lowerable {
 public:
  virtual ~sirius_ast_lowerable() = default;

  /**
   * @brief Appends a BOOL filter expression to @p tree
   *
   * @p column_ref must already belong to @p tree. The tree owns emitted nodes; filter-owned
   * scalars must outlive it. A negative device ID selects the current device.
   */
  [[nodiscard]] virtual cudf::ast::expression const& to_ast(cudf::ast::tree& tree,
                                                            cudf::ast::expression const& column_ref,
                                                            int device_id = -1) const = 0;

  /// Calls the factory once; `tree.back()` is the returned filter root.
  [[nodiscard]] cudf::ast::tree to_standalone_ast(
    std::function<cudf::ast::expression const&(cudf::ast::tree&)> const& column_ref_factory) const;
};

struct zone_map_entry {
  std::unique_ptr<cudf::scalar> min;
  std::unique_ptr<cudf::scalar> max;
};

/**
 * @brief Keeps rows contained by any configured zone
 */
class sirius_dynamic_zone_map_filter final : public sirius_dynamic_filter,
                                             public sirius_ast_lowerable,
                                             public sirius_device_replicable {
 public:
  /**
   * @brief Builds zones for one consumer column
   *
   * @pre Every bound matches the consumer column type.
   * @throw std::invalid_argument if the zone list is empty, a bound is missing, a zone's bound
   * types differ, or supports() rejects the bound type
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

  /**
   * @brief True when zone maps can be built, replicated, and lowered for @p t
   *
   * Floating-point keys are excluded: the lowered AST bounds compare with IEEE semantics under
   * which NaN fails both, while the authoritative join matches NaN keys to each other (DuckDB
   * total order), so a range filter could drop matching rows. cudf::reduce min/max also has no
   * NaN-exclusion contract.
   */
  [[nodiscard]] static bool supports(cudf::data_type t) noexcept;

 private:
  std::vector<zone_map_entry> _zones;
  bool _inclusive_min;
  bool _inclusive_max;
  int _source_device = -1;

  struct device_zones;
  std::vector<std::unique_ptr<device_zones>> _replicas;
};

/**
 * @brief Capability for computing a per-row keep mask
 */
class sirius_mask_applicable {
 public:
  virtual ~sirius_mask_applicable() = default;

  /**
   * @brief Returns `probe.size()` BOOL8 values (`true` keeps), or null for an incompatible probe
   *
   * A probe whose carrier is a narrowed form of the filter's key type is restored first rather
   * than declined -- a pinned chunk may store the key narrower than the type the filter was
   * published with (@ref sirius::op::detail::restore_probe_to).
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    int device_id,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief Exact hash membership filter
 *
 * The backing set cannot store `numeric_limits<KeyT>::min()`; probes with that value are kept to
 * avoid false negatives.
 */
class sirius_dynamic_in_list_filter final : public sirius_dynamic_filter,
                                            public sirius_mask_applicable,
                                            public sirius_device_replicable {
 public:
  /**
   * @brief Builds a persistent set from null-free INT32 or INT64 keys
   *
   * @pre The backing storage for @p keys remains valid until work enqueued on @p stream completes.
   * @throw std::invalid_argument if @p keys is unsupported
   * @throw std::runtime_error if the current CUDA device cannot be identified
   * @throw std::logic_error if the validated key type changes during construction
   */
  sirius_dynamic_in_list_filter(cudf::column_view const& keys,
                                ::cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr);

  ~sirius_dynamic_in_list_filter() override;

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::IN_LIST;
  }

  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    int device_id,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] std::size_t replica_count() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool has_persistent_set() const noexcept;
  [[nodiscard]] static bool supports(cudf::column_view const& keys) noexcept;
  [[nodiscard]] static std::size_t estimated_set_bytes(std::size_t num_keys,
                                                       cudf::data_type key_type) noexcept;

 private:
  cudf::data_type _key_type{cudf::type_id::EMPTY};
  std::size_t _num_keys = 0;

  struct set_impl;
  std::unique_ptr<set_impl> _set;
};

/**
 * @brief Exact linear membership over a small, null-free INT32 or INT64 set
 */
class sirius_dynamic_small_in_list_filter final : public sirius_dynamic_filter,
                                                  public sirius_mask_applicable,
                                                  public sirius_device_replicable {
 public:
  static constexpr std::size_t k_max_keys = 12;

  /**
   * @brief Copies a small build-key set into device-local storage
   *
   * @pre The backing storage for @p keys remains valid until the copy on @p stream completes.
   * @throw std::invalid_argument if @p keys is unsupported
   * @throw std::runtime_error if the current CUDA device cannot be identified
   */
  sirius_dynamic_small_in_list_filter(cudf::column_view const& keys,
                                      ::cuda::stream_ref stream,
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
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] std::size_t replica_count() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return _num_keys; }
  [[nodiscard]] static bool supports(cudf::column_view const& keys) noexcept;

 private:
  cudf::data_type _key_type{cudf::type_id::EMPTY};
  std::size_t _num_keys = 0;

  struct needle_store;
  std::unique_ptr<needle_store> _store;
};

/**
 * @brief Probabilistic membership filter with no false negatives
 *
 * False positives pass extra rows to the authoritative join.
 */
class sirius_dynamic_bloom_filter final : public sirius_dynamic_filter,
                                          public sirius_mask_applicable,
                                          public sirius_device_replicable {
 public:
  /**
   * @brief Builds a Bloom filter from INT32 or INT64 keys, excluding nulls
   *
   * @pre Key storage remains valid until work enqueued on @p stream completes.
   * @throw std::invalid_argument if @p keys is unsupported
   * @throw std::runtime_error if the current CUDA device cannot be identified
   * @throw std::logic_error if the validated key type changes during construction
   */
  sirius_dynamic_bloom_filter(cudf::column_view const& keys,
                              ::cuda::stream_ref stream,
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
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] std::size_t replica_count() const noexcept;
  [[nodiscard]] static bool supports(cudf::data_type t) noexcept;
  [[nodiscard]] static std::size_t estimated_bytes(std::size_t num_keys) noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

class sirius_dynamic_filter_set;

/**
 * @brief An owning observation of one endpoint's immutable filters and producer completion
 */
class dynamic_filter_snapshot final {
 public:
  /** @brief One filter and its target column in the consumer's output schema */
  struct entry {
    std::size_t column_index;
    std::shared_ptr<sirius_dynamic_filter const> filter;
  };

  [[nodiscard]] std::span<entry const> entries() const noexcept { return _entries; }
  /**
   * @brief The generation of the snapshot (measured by the number of filters added to the endpoint)
   */
  [[nodiscard]] std::size_t generation() const noexcept { return _entries.size(); }
  /**
   * @brief Indicates if the snapshot represents a terminal state (no more filters will be added to
   * the endpoint)
   */
  [[nodiscard]] bool terminal() const noexcept { return _terminal; }
  [[nodiscard]] bool empty() const noexcept { return _entries.empty(); }

 private:
  friend class sirius_dynamic_filter_set;
  std::vector<entry> _entries;
  bool _terminal = false;
};

/**
 * @brief Thread-safe append-only endpoint with identified publication owners
 *
 * Register every producer before freeze_registration(). Only a producer can append filters, and its
 * terminal transition follows its last possible push. Completion never removes an already-visible
 * filter. Snapshots distinguish pending and terminal channels independently of whether filters
 * exist.
 */
class sirius_dynamic_filter_set {
  struct state;

 public:
  enum class completion { published, skipped, failed, cancelled };

  /**
   * @brief Move-only publication right retaining its channel's state
   *
   * `producer` bundles 2 things:
   *  - The right to append filters (push_filter())
   *  - The responsibility to declare that this producer will not append more filters (finish())
   *
   * Destruction resolves an unfinished producer as skipped. finish() is allocation-free and
   * idempotent; subsequent pushes are rejected. Moving or destroying the right requires exclusive
   * ownership, while push_filter() and finish() may race safely.
   */
  class producer final {
   public:
    producer(producer const&)            = delete;
    producer& operator=(producer const&) = delete;
    producer(producer&&) noexcept;
    producer& operator=(producer&&) noexcept;
    ~producer();

    /**
     * @brief Appends a filter to the producer's channel
     *
     * @param col_idx The column index to append the filter to in the target consumer's output
     *                schema
     * @param filter The filter to append
     * @return true if the filter was successfully appended, false otherwise
     */
    [[nodiscard]] bool push_filter(std::size_t col_idx,
                                   std::shared_ptr<sirius_dynamic_filter const> filter) const;

    /**
     * @brief Declares that the producer will not append more filters
     *
     * @param result The completion result of the producer
     */
    void finish(completion result = completion::skipped) const noexcept;

   private:
    friend class sirius_dynamic_filter_set;
    producer(std::shared_ptr<state> channel, std::size_t index) noexcept;
    std::shared_ptr<state> _channel;
    std::size_t _index = 0;
  };

  sirius_dynamic_filter_set();
  sirius_dynamic_filter_set(sirius_dynamic_filter_set const&)            = delete;
  sirius_dynamic_filter_set& operator=(sirius_dynamic_filter_set const&) = delete;

  /**
   * @brief Returns a snapshot of the current state of the dynamic filter set.
   *
   * @return A snapshot of the current state of the dynamic filter set, including all filters that
   *         have been added so far.
   * @note The snapshot is valid even if new filters are added or the filter set is destroyed after
   *       the snapshot is taken (the snapshot owns its entries).
   */
  [[nodiscard]] dynamic_filter_snapshot snapshot() const;

  /**
   * @brief Returns an insertion-order owning snapshot valid after later pushes or destruction
   */
  [[nodiscard]] std::vector<std::shared_ptr<sirius_dynamic_filter const>> filters_for_column(
    std::size_t col_idx) const;

  /**
   * @brief Returns the columns that have filters
   *
   * @note Meaningful only with producers and no unscoped producer.
   */
  [[nodiscard]] std::vector<std::size_t> filtered_columns() const;
  [[nodiscard]] bool empty() const;

  /**
   * @brief Rejects future pushes for these output columns; existing filters remain
   *
   * Call before publication.
   */
  void ignore_columns(std::vector<std::size_t> const& cols);

  /**
   * @brief Registers one producer's target output columns
   *
   * An empty vector is unscoped, so consumers must treat every column as a possible target.
   * @return A move-only right to append filters and declare completion for the producer. The
   *         producer handle also holds a shared_ptr to the channel state, so destroying the outer
   *         channel object doesn't invalidate the survivor producer handle.
   */
  [[nodiscard]] producer register_producer(std::vector<std::size_t> planned_target_columns);

  /**
   * @brief Seals the producer set before execution or a manual plan's first observation
   *
   * Idempotent. Before this boundary snapshots are always pending, including an empty channel.
   */
  void freeze_registration() noexcept;

  [[nodiscard]] bool has_producers() const noexcept;

  /**
   * @brief Get sorted consumer-output ordinals.
   *
   * @note Meaningful only with producers and no unscoped producer.
   */
  [[nodiscard]] std::vector<std::size_t> planned_target_columns() const;

  [[nodiscard]] bool has_unscoped_producer() const noexcept;

  void close_for_new_filters();

  [[nodiscard]] bool accepting_filters() const noexcept;

  [[nodiscard]] bool has_filters() const noexcept;

  /**
   * @brief The number of filters in the dynamic filter set
   *
   * This is a monotonic counter, allowing consumers to detect growth past a snapshot.
   */
  [[nodiscard]] std::size_t filter_count() const noexcept;

 private:
  std::shared_ptr<state> _state;
};

// Resolver results must already belong to the destination AST tree.
using column_ref_resolver_fn = std::function<cudf::ast::expression const&(std::size_t col_idx)>;

/**
 * @brief ANDs all AST-capable filters into @p tree
 *
 * Returns @p existing_root unchanged when none apply. Resolver output and the returned root belong
 * to @p tree; filter-owned scalars must outlive it.
 */
[[nodiscard]] cudf::ast::expression const& merge_ast_dynamic_filters_into_tree(
  cudf::ast::tree& tree,
  cudf::ast::expression const& existing_root,
  dynamic_filter_snapshot const& filters,
  column_ref_resolver_fn const& column_ref_resolver);

}  // namespace sirius::op

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

#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

// libcudf's AST header uses std::variant without including <variant>.
// clang-format off
#include <variant>
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
// clang-format on
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
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
   * The membership implementations accept any integer carrier of the key's signedness
   * (INT8..INT64 for signed keys, UINT8..UINT64 for unsigned) and, for decimal keys, any
   * fixed-point width at the key's scale (DECIMAL32/64/128), converting per element in-kernel: a
   * pinned chunk may store the key narrower than the type the filter was published with, and no
   * consumer should have to materialize a widened copy to probe it. A DATE key accepts
   * TIMESTAMP_DAYS or its INT8/INT16/INT32 storage carriers; a sub-day timestamp key accepts only
   * its own unit. String keys accept a STRING probe, fingerprinted in-kernel with the hash the
   * build side used. `membership_probe_compatible` is the host-side mirror of what a filter
   * accepts.
   *
   * The result is never nullable. A null probe row is written as `false`: admission never routes
   * a null-safe comparison to a dynamic filter and the authoritative join runs with
   * `null_equality::UNEQUAL`, so a null key is a definite non-member on either side.
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;

  /**
   * @brief Prior-mask-aware variant: rows the prior keep-mask already killed skip the probe
   *
   * @p prior_mask_words is packed 1 bit/row over @p probe's rows (bit `row % 32` of word
   * `row / 32`, 1 = keep), or null for no restriction. A pruning hint only: ignoring it is sound
   * because every caller ANDs the result with that same mask.
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    std::uint32_t const* prior_mask_words,
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    (void)prior_mask_words;
    return compute_mask(probe, device_id, stream, mr);
  }
};

/**
 * @brief Hash membership filter: exact for integer keys, no false negatives for string keys
 *
 * String keys are stored as 64-bit XXHash_64 fingerprints (see `membership_key_domain`), so two
 * distinct strings sharing a fingerprint pass a probe the authoritative join then drops. The
 * backing set reserves one sentinel value it cannot store (`numeric_limits::min()` for signed
 * reps, `::max()` for unsigned and string fingerprints); probes equal to it are kept to avoid
 * false negatives. Null build keys are compacted out (they match nothing under the join's
 * `null_equality::UNEQUAL`).
 */
class sirius_dynamic_in_list_filter final : public sirius_dynamic_filter,
                                            public sirius_mask_applicable,
                                            public sirius_device_replicable {
 public:
  /**
   * @brief Builds a persistent set from keys of a supported type (see
   * `membership_key_supported`), excluding nulls
   *
   * The set is typed at the key's rep: a build column arriving at a narrowed carrier (INT8/INT16,
   * UINT8/UINT16, DECIMAL32 for a DECIMAL64 key) widens per element into a 32-bit set; a temporal
   * column is read through its integer storage (int32 epoch days, int64 ticks); a DECIMAL128
   * column narrows into the int64 set once `membership_build_fits_rep` has verified it; a STRING
   * build column is hashed once into a UINT64 fingerprint set. `size()` reports the valid keys
   * stored.
   *
   * @pre The backing storage for @p keys remains valid until work enqueued on @p stream completes.
   * @throw std::invalid_argument if @p keys is unsupported or its values do not fit the key rep
   * @throw std::runtime_error if the current CUDA device cannot be identified
   * @throw std::logic_error if the validated key type changes during construction
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

  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    std::uint32_t const* prior_mask_words,
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] std::size_t replica_count() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool has_persistent_set() const noexcept;
  [[nodiscard]] membership_key_domain const& domain() const noexcept { return _domain; }
  [[nodiscard]] static bool supports(cudf::column_view const& keys) noexcept;
  /// Baseline footprint of a set over @p num_keys keys of @p key_type, sized at the key's rep.
  [[nodiscard]] static std::size_t estimated_set_bytes(std::size_t num_keys,
                                                       cudf::data_type key_type) noexcept;

 private:
  membership_key_domain _domain{};
  std::size_t _num_keys = 0;

  struct set_impl;
  std::unique_ptr<set_impl> _set;
};

/**
 * @brief Linear membership over a small key set of a supported type
 *
 * Needles are stored at the key's rep (see `membership_key_domain`): integer, temporal, and
 * decimal needles compare exactly; string needles are 64-bit fingerprints compared against the
 * probe's in-kernel fingerprint, so the filter has no false negatives rather than being exact.
 * Null build keys are compacted out; `supports()` and `size()` count the valid keys.
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
   * @throw std::invalid_argument if @p keys is unsupported or its values do not fit the key rep
   * @throw std::runtime_error if the current CUDA device cannot be identified
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

  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    std::uint32_t const* prior_mask_words,
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] std::size_t replica_count() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return _num_keys; }
  [[nodiscard]] membership_key_domain const& domain() const noexcept { return _domain; }
  [[nodiscard]] static bool supports(cudf::column_view const& keys) noexcept;

 private:
  membership_key_domain _domain{};
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
   * @brief Builds a Bloom filter from keys of a supported type (see `membership_key_supported`),
   * excluding nulls
   *
   * @pre Key storage remains valid until work enqueued on @p stream completes.
   * @throw std::invalid_argument if @p keys is unsupported or its values do not fit the key rep
   * @throw std::runtime_error if the current CUDA device cannot be identified
   * @throw std::logic_error if the validated key type changes during construction
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

  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    std::uint32_t const* prior_mask_words,
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override;

  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] std::size_t replica_count() const noexcept;
  [[nodiscard]] membership_key_domain const& domain() const noexcept { return _domain; }
  [[nodiscard]] static bool supports(cudf::data_type t) noexcept;
  [[nodiscard]] static std::size_t estimated_bytes(std::size_t num_keys) noexcept;

 private:
  membership_key_domain _domain{};
  struct impl;
  std::unique_ptr<impl> _impl;
};

/**
 * @brief Thread-safe append-only channel keyed by consumer output ordinal
 *
 * Snapshots co-own immutable filters and may observe any published prefix. Closing rejects future
 * pushes.
 */
class sirius_dynamic_filter_set {
 public:
  /// Returns false for a null filter or a closed or ignored column; otherwise appends it.
  bool push_filter(std::size_t col_idx, std::shared_ptr<sirius_dynamic_filter const> f);

  /**
   * @brief Returns an insertion-order owning snapshot valid after later pushes or destruction
   */
  [[nodiscard]] std::vector<std::shared_ptr<sirius_dynamic_filter const>> filters_for_column(
    std::size_t col_idx) const;

  /// Returns filtered columns in unspecified order.
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
   */
  void register_producer(std::vector<std::size_t> planned_target_columns);

  [[nodiscard]] bool has_producers() const noexcept
  {
    return _producer_count.load(std::memory_order_acquire) > 0;
  }

  // Sorted consumer-output ordinals; meaningful only with producers and no unscoped producer.
  [[nodiscard]] std::vector<std::size_t> planned_target_columns() const;

  [[nodiscard]] bool has_unscoped_producer() const noexcept
  {
    return _has_unscoped_producer.load(std::memory_order_acquire);
  }

  void close_for_new_filters();

  [[nodiscard]] bool accepting_filters() const noexcept
  {
    return _accepting_filters.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool has_filters() const noexcept
  {
    return _filter_count.load(std::memory_order_acquire) > 0;
  }

  // Monotonic, allowing consumers to detect growth past a snapshot.
  [[nodiscard]] std::size_t filter_count() const noexcept
  {
    return _filter_count.load(std::memory_order_acquire);
  }

 private:
  mutable std::mutex _mu;
  std::unordered_map<std::size_t, std::vector<std::shared_ptr<sirius_dynamic_filter const>>>
    _filters;
  std::unordered_set<std::size_t> _ignored_columns;
  std::set<std::size_t> _planned_target_columns;
  std::atomic<std::size_t> _filter_count{0};
  std::atomic<std::size_t> _producer_count{0};
  std::atomic<bool> _has_unscoped_producer{false};
  std::atomic<bool> _accepting_filters{true};
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
  sirius_dynamic_filter_set const& set,
  column_ref_resolver_fn const& column_ref_resolver);

}  // namespace sirius::op

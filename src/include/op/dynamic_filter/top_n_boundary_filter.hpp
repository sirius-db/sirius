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

#pragma once

#include "op/dynamic_filter/exact_host_scalar.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace sirius::op::detail {

/**
 * @brief Launch-parameter form of one boundary: POD, passed by value, no device state
 *
 * Components hold the exact value widened to `__int128_t` (sign-extension is exact for the signed
 * allowlist; `TIMESTAMP_DAYS` widens its int32 tick count), an engaged flag (a disengaged
 * component is a null boundary value, ordered by the same null placement flag), and the key's
 * direction, output-order null placement, and physical width, now including width 16 for
 * `DECIMAL128`. `strict` selects the row producer's predicate; inclusive covers the degraded
 * first-key form and boundaries clamped to `k_max_components` (an inclusive prefix predicate is
 * sound standalone: prefix-worse rows are worse, prefix-ties are kept).
 */
struct boundary_filter_params {
  static constexpr std::size_t k_max_components = 8;

  struct component {
    __int128_t value   = 0;      ///< Widened exact boundary value; meaningful only when engaged
    bool engaged       = false;  ///< False: the boundary row's key is null here
    bool descending    = false;  ///< Key direction
    bool nulls_first   = false;  ///< Null placement in the final output order
    std::uint8_t width = 0;      ///< Physical representation width in bytes (1/2/4/8/16)
  };

  component components[k_max_components]{};
  std::uint32_t count = 0;      ///< Leading components populated; at most k_max_components
  bool strict         = false;  ///< Drop full-tuple ties (strict) or keep them (inclusive)
};

/**
 * @brief Outcome of one fused boundary-filter pass
 */
struct boundary_filter_result {
  /// The gathered surviving rows, an empty table when nothing survived, or null when every row
  /// passed -- the caller then forwards the input batch unchanged, copy-free.
  std::unique_ptr<cudf::table> filtered;
  cudf::size_type rows_kept = 0;  ///< Always valid, including on the all-pass fast path
};

/**
 * @brief One fused pass: per-row lexicographic compare against the boundary, compacting passing
 * row indices
 *
 * For each row the kernel walks the boundary components in key order: a strictly better row is
 * kept, a strictly worse row dropped, ties fall through to the next component, and a full-tuple
 * tie is dropped exactly when `params.strict`. Null row values and disengaged boundary components
 * order by the component's output-order null placement. Passing row indices are stream-compacted
 * into a gather map (`cub::DeviceSelect::If`) and materialized with `cudf::gather`; reading back
 * the survivor count synchronizes @p stream, which also makes `rows_kept` valid on return.
 *
 * @pre `key_columns.size() >= params.count`; each engaged component's width matches its column's
 * physical representation.
 *
 * @throw std::invalid_argument when an engaged width-16 component's key column data is not
 * 16-byte aligned -- the natural-load contract cuDF's own fixed-point access relies on, checked
 * once per pass.
 *
 * @param[in] batch The rows to filter; all columns survive into the gathered result
 * @param[in] key_columns The ORDER BY keys' column indices in @p batch, in key order
 * @param[in] params By-value boundary description; no device state
 * @param[in] stream Stream the compare, compaction, gather, and count read-back are ordered on
 * @param[in] mr Allocator for the gather map, CUB temporary storage, and the gathered table
 */
[[nodiscard]] boundary_filter_result apply_boundary_filter(
  cudf::table_view const& batch,
  std::span<cudf::size_type const> key_columns,
  boundary_filter_params const& params,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Whether a key's nulls sort before every value in the final output order
 *
 * cuDF applies `null_order` in the ascending frame and reverses it for a DESCENDING key, so both
 * fields of @p key decide the output placement.
 */
[[nodiscard]] bool nulls_first_in_output(top_n_key_semantics const& key) noexcept;

/**
 * @brief Marshal the leading @p component_count components of a boundary into launch parameters
 *
 * Values widen exactly to `__int128_t` through `exact_host_scalar::widened()`; a disengaged
 * component stays disengaged; widths come from each key's storage type. A @p component_count beyond
 * `boundary_filter_params::k_max_components` is clamped and the marshal degrades to the
 * inclusive prefix form regardless of @p strict -- an inclusive prefix predicate is sound
 * standalone (prefix-worse rows are worse; prefix-ties are kept). Shared by the Top-N sink
 * prefilter and the Stage-3 boundary filter classes.
 *
 * @pre `boundary.size() >= component_count` and `keys.size() >= component_count`.
 */
[[nodiscard]] boundary_filter_params make_boundary_filter_params(
  exact_host_key_tuple const& boundary,
  std::span<top_n_key_semantics const> keys,
  std::size_t component_count,
  bool strict);

}  // namespace sirius::op::detail

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

// sirius
#include <op/dynamic_filter/top_n_boundary_filter.hpp>

// cudf
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

// cccl
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

// cucascade
#include <cucascade/error.hpp>

// rmm
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

/// @brief Per-row three-way lexicographic compare against the by-value boundary. One untemplated
/// functor: direction, null placement, strictness, and component count are uniform for the whole
/// launch (predictable branches, no divergence), and the type dimension collapses to a width
/// switch after `__int128_t` widening -- the type x direction x strictness template cross-product
/// would be instantiation bloat for no measured gain.
struct boundary_row_predicate {
  cudf::table_device_view keys;  ///< The key columns only, in component order
  sirius::op::detail::boundary_filter_params params;

  __device__ __forceinline__ static __int128_t load_widened(cudf::column_device_view const& col,
                                                            cudf::size_type row,
                                                            std::uint8_t width) noexcept
  {
    switch (width) {
      case 1: return static_cast<__int128_t>(col.data<std::int8_t>()[row]);
      case 2: return static_cast<__int128_t>(col.data<std::int16_t>()[row]);
      case 4: return static_cast<__int128_t>(col.data<std::int32_t>()[row]);
      case 8: return static_cast<__int128_t>(col.data<std::int64_t>()[row]);
      // A single natural 16-byte load, the same access cuDF's own fixed-point device code
      // performs; apply_boundary_filter refuses a misaligned buffer host-side, once per pass.
      case 16: return col.data<__int128_t>()[row];
      default:
        // Unreachable: the marshaller populates width 1/2/4/8/16 for every engaged component.
        assert(false);
        return 0;
    }
  }

  __device__ bool operator()(cudf::size_type row) const noexcept
  {
    for (std::uint32_t i = 0; i < params.count; ++i) {
      auto const& component = params.components[i];
      auto const& col       = keys.column(static_cast<cudf::size_type>(i));
      bool const row_null   = !col.is_valid(row);
      if (row_null || !component.engaged) {
        if (row_null && !component.engaged) { continue; }  // both null: sort-equal at this key
        // Exactly one side is null; the null side orders first iff nulls sort first.
        return row_null == component.nulls_first;
      }
      auto const value = load_widened(col, row, component.width);
      if (value == component.value) { continue; }
      bool const less = value < component.value;
      return component.descending ? !less : less;  // strictly better in output order -> keep
    }
    return !params.strict;  // full-tuple tie
  }
};

}  // namespace

namespace sirius::op::detail {

boundary_filter_result apply_boundary_filter(cudf::table_view const& batch,
                                             std::span<cudf::size_type const> key_columns,
                                             boundary_filter_params const& params,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const num_rows = batch.num_rows();
  if (num_rows == 0) { return {nullptr, 0}; }

  // Width-16 components are read with a single natural 16-byte load, so the buffer must honor
  // cuDF's own fixed-point alignment contract; a violating buffer fails loudly here, once per
  // pass, instead of feeding misread comparisons.
  for (std::uint32_t i = 0; i < params.count; ++i) {
    auto const& component = params.components[i];
    if (component.engaged && component.width == 16 &&
        reinterpret_cast<std::uintptr_t>(batch.column(key_columns[i]).data<__int128_t>()) %
            alignof(__int128_t) !=
          0) {
      throw std::invalid_argument(
        "[top_n boundary filter] width-16 key column is not 16-byte aligned");
    }
  }

  std::vector<cudf::column_view> key_views;
  key_views.reserve(params.count);
  for (std::uint32_t i = 0; i < params.count; ++i) {
    key_views.push_back(batch.column(key_columns[i]));
  }
  auto const keys_view   = cudf::table_view{key_views};
  auto const device_keys = cudf::table_device_view::create(keys_view, stream);

  boundary_row_predicate const predicate{*device_keys, params};

  // Fused pass: compact passing row indices directly into the gather map, with a device count.
  rmm::device_uvector<cudf::size_type> gather_map(static_cast<std::size_t>(num_rows), stream, mr);
  rmm::device_scalar<cudf::size_type> device_rows_kept(stream, mr);
  auto const row_indices = thrust::counting_iterator<cudf::size_type>{0};

  std::size_t temp_bytes = 0;
  CUCASCADE_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                           temp_bytes,
                                           row_indices,
                                           gather_map.begin(),
                                           device_rows_kept.data(),
                                           num_rows,
                                           predicate,
                                           stream.value()));
  rmm::device_buffer temp_storage{temp_bytes, stream, mr};
  CUCASCADE_CUDA_TRY(cub::DeviceSelect::If(temp_storage.data(),
                                           temp_bytes,
                                           row_indices,
                                           gather_map.begin(),
                                           device_rows_kept.data(),
                                           num_rows,
                                           predicate,
                                           stream.value()));

  // The documented count read-back: synchronizes the stream and enables the all-pass fast path.
  auto const rows_kept = device_rows_kept.value(stream);
  if (rows_kept == num_rows) { return {nullptr, rows_kept}; }
  if (rows_kept == 0) { return {cudf::empty_like(batch), 0}; }

  auto const map_view = cudf::column_view{
    cudf::data_type{cudf::type_to_id<cudf::size_type>()}, rows_kept, gather_map.data(), nullptr, 0};
  auto filtered = cudf::gather(batch, map_view, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return {std::move(filtered), rows_kept};
}

bool nulls_first_in_output(top_n_key_semantics const& key) noexcept
{
  return (key.null_order == cudf::null_order::BEFORE) == (key.order == cudf::order::ASCENDING);
}

boundary_filter_params make_boundary_filter_params(exact_host_key_tuple const& boundary,
                                                   std::span<top_n_key_semantics const> keys,
                                                   std::size_t component_count,
                                                   bool strict)
{
  boundary_filter_params params{};
  auto const count = std::min(component_count, boundary_filter_params::k_max_components);
  params.count     = static_cast<std::uint32_t>(count);
  params.strict    = strict && component_count == count;
  for (std::size_t i = 0; i < count; ++i) {
    auto const& key       = keys[i];
    auto& component       = params.components[i];
    component.descending  = key.order == cudf::order::DESCENDING;
    component.nulls_first = nulls_first_in_output(key);
    component.width       = static_cast<std::uint8_t>(cudf::size_of(key.storage_type));
    // The kernel reads components by width 1/2/4/8/16 and its `default:` branch is an assert,
    // which the release build compiles out into a silent zero. Every fixed-width cuDF type maps
    // to one of those widths, so this gate is unreachable today; it is kept as the
    // once-per-publication defence for the next widening -- the same explicitly-untestable-guard
    // status as the publication sync's race window (top_n_threshold_coordinator.cpp,
    // publish_revision).
    if (component.width != 1 && component.width != 2 && component.width != 4 &&
        component.width != 8 && component.width != 16) {
      throw std::invalid_argument(
        "[top_n boundary filter] boundary component width is outside the widths the comparison "
        "kernel can read");
    }
    if (auto const& bound = boundary.component(i)) {
      component.engaged = true;
      component.value   = bound->widened();
    }
  }
  return params;
}

}  // namespace sirius::op::detail

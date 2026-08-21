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
#include <op/aggregate/dense_count_join_impl.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

// rmm
#include <rmm/exec_policy.hpp>

// thrust
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

// cuda
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

constexpr int k_block_size      = 256;
constexpr int64_t k_max_blocks  = 4096;
constexpr uint64_t k_bigint_max = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

[[nodiscard]] unsigned grid_size_for(int64_t n)
{
  auto const blocks = std::min<int64_t>(n / k_block_size + (n % k_block_size != 0), k_max_blocks);
  return static_cast<unsigned>(std::max<int64_t>(blocks, 1));
}

[[nodiscard]] cudf::size_type checked_output_rows(int64_t num_groups, bool append_null_group)
{
  auto const max   = static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max());
  auto const extra = append_null_group ? int64_t{1} : int64_t{0};
  if (num_groups < 0 || num_groups > max - extra) {
    throw sirius::invalid_input_exception(
      "dense_count_join: output row count {} plus NULL-group row {} exceeds "
      "cudf::size_type max {}",
      num_groups,
      extra,
      max);
  }
  return static_cast<cudf::size_type>(num_groups + extra);
}

struct histogram_layout {
  std::size_t slots;
  std::size_t bytes_per_histogram;
};

[[nodiscard]] histogram_layout checked_histogram_layout(int64_t range, std::size_t slot_bytes)
{
  if (range <= 0) {
    throw sirius::internal_exception("dense_count_state: non-positive range {}", range);
  }
  auto const slots    = static_cast<uint64_t>(range);
  auto const size_max = std::numeric_limits<std::size_t>::max();
  if (slot_bytes == 0 || slot_bytes > size_max / 2 || slots > size_max / (2 * slot_bytes)) {
    throw sirius::invalid_input_exception(
      "dense_count_join: histogram range {} with {}-byte slots exceeds size_t allocation "
      "capacity",
      range,
      slot_bytes);
  }
  auto const slot_count = static_cast<std::size_t>(slots);
  return histogram_layout{slot_count, slot_count * slot_bytes};
}

__global__ void initialize_extrema_kernel(int64_t* extrema)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    extrema[0] = std::numeric_limits<int64_t>::max();
    extrema[1] = std::numeric_limits<int64_t>::min();
  }
}

template <typename KeyT>
__global__ void merge_extrema_kernel(KeyT const* batch_min, KeyT const* batch_max, int64_t* extrema)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    auto const lo = static_cast<int64_t>(*batch_min);
    auto const hi = static_cast<int64_t>(*batch_max);
    if (lo < extrema[0]) { extrema[0] = lo; }
    if (hi > extrema[1]) { extrema[1] = hi; }
  }
}

__device__ __forceinline__ void histogram_add(uint32_t* slot) { atomicAdd(slot, 1u); }

__device__ __forceinline__ void histogram_add(uint64_t* slot)
{
  static_assert(sizeof(unsigned long long) == sizeof(uint64_t));
  atomicAdd(reinterpret_cast<unsigned long long*>(slot), 1ULL);
}

/// Grid-stride histogram accumulation. A row contributes iff its key is valid, the optional
/// second validity mask (the COUNT(col) argument) is valid at that row, and — when
/// @p bounds_check — the key lies inside [min_key, min_key + range). Preserved-side keys skip
/// the bounds check (the histogram was sized from their global min/max).
template <typename KeyT, typename CountT>
__global__ void accumulate_kernel(KeyT const* __restrict__ keys,
                                  cudf::bitmask_type const* __restrict__ key_mask,
                                  cudf::size_type key_mask_offset,
                                  cudf::bitmask_type const* __restrict__ value_mask,
                                  cudf::size_type value_mask_offset,
                                  int64_t n,
                                  int64_t min_key,
                                  int64_t range,
                                  bool bounds_check,
                                  CountT* __restrict__ bins)
{
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    if (key_mask != nullptr &&
        !cudf::bit_is_set(key_mask, key_mask_offset + static_cast<cudf::size_type>(i))) {
      continue;
    }
    if (value_mask != nullptr &&
        !cudf::bit_is_set(value_mask, value_mask_offset + static_cast<cudf::size_type>(i))) {
      continue;
    }
    // Unsigned offset arithmetic: defined for ANY int64 key/min_key pair (a signed
    // subtraction could overflow for extreme counted-side keys). An in-domain key yields
    // offset < range exactly; every out-of-domain key — including ones whose signed
    // subtraction would have wrapped — lands at offset >= range and is dropped.
    auto const offset =
      static_cast<uint64_t>(static_cast<int64_t>(keys[i])) - static_cast<uint64_t>(min_key);
    if (bounds_check && offset >= static_cast<uint64_t>(range)) { continue; }
    histogram_add(&bins[offset]);
  }
}

template <typename CountT>
struct presence_positive {
  CountT const* presence;
  __device__ bool operator()(int64_t k) const { return presence[k] != CountT{0}; }
};

/// Write the output rows for the selected (presence > 0) slots, in ascending key order.
/// COUNT(col): value = presence * counts. COUNT(*): value = presence * max(counts, 1).
template <typename KeyT, typename CountT>
__global__ void emit_kernel(int64_t const* __restrict__ selected,
                            int64_t num_selected,
                            CountT const* __restrict__ presence,
                            CountT const* __restrict__ counts,
                            int64_t min_key,
                            bool count_star,
                            KeyT* __restrict__ out_keys,
                            int64_t* __restrict__ out_values,
                            int32_t* __restrict__ overflow_flag)
{
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_selected;
       i += stride) {
    auto const k = selected[i];
    auto const p = static_cast<uint64_t>(presence[k]);
    auto matched = static_cast<uint64_t>(counts[k]);
    if (count_star && matched == 0) { matched = 1; }
    out_keys[i] = static_cast<KeyT>(min_key + k);
    if (overflow_flag != nullptr && matched != 0 && p > k_bigint_max / matched) {
      atomicExch(overflow_flag, 1);
      out_values[i] = 0;
    } else {
      out_values[i] = static_cast<int64_t>(p * matched);
    }
  }
}

__global__ void validate_product_kernel(int64_t const* __restrict__ lhs,
                                        int64_t const* __restrict__ rhs,
                                        int64_t n,
                                        int32_t* __restrict__ status)
{
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    auto const left  = lhs[i];
    auto const right = rhs[i];
    if (left < 0 || right < 0) {
      atomicMax(status, int32_t{2});
      continue;
    }
    auto const left_u  = static_cast<uint64_t>(left);
    auto const right_u = static_cast<uint64_t>(right);
    if (right_u != 0 && left_u > k_bigint_max / right_u) { atomicMax(status, int32_t{1}); }
  }
}

template <typename CountT>
void accumulate_impl(cudf::column_view const& keys,
                     cudf::column_view const* count_validity_source,
                     int64_t min_key,
                     int64_t range,
                     bool bounds_check,
                     CountT* bins,
                     rmm::cuda_stream_view stream)
{
  if (count_validity_source != nullptr && count_validity_source->size() != keys.size()) {
    throw sirius::internal_exception(
      "dense_count_join: COUNT argument has {} rows but its key column has {}",
      count_validity_source->size(),
      keys.size());
  }
  auto const n = static_cast<int64_t>(keys.size());
  if (n == 0) { return; }

  cudf::bitmask_type const* key_mask = keys.null_count() > 0 ? keys.null_mask() : nullptr;
  cudf::bitmask_type const* val_mask = nullptr;
  cudf::size_type val_mask_offset    = 0;
  if (count_validity_source != nullptr && count_validity_source->null_count() > 0) {
    val_mask        = count_validity_source->null_mask();
    val_mask_offset = count_validity_source->offset();
  }

  auto const grid = grid_size_for(n);
  auto launch     = [&](auto key_tag) {
    using KeyT = decltype(key_tag);
    accumulate_kernel<KeyT, CountT>
      <<<grid, k_block_size, 0, stream.value()>>>(keys.template data<KeyT>(),
                                                  key_mask,
                                                  keys.offset(),
                                                  val_mask,
                                                  val_mask_offset,
                                                  n,
                                                  min_key,
                                                  range,
                                                  bounds_check,
                                                  bins);
  };
  switch (keys.type().id()) {
    case cudf::type_id::INT32: launch(int32_t{}); break;
    case cudf::type_id::INT64: launch(int64_t{}); break;
    default:
      throw sirius::internal_exception(
        "dense_count_join: unsupported key column type {} (expected INT32/INT64)",
        static_cast<int32_t>(keys.type().id()));
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

/// Zero-filled trailing NULL-group row: key storage is zeroed (deterministic bytes under the
/// null), the key null mask marks it NULL, and the value is `null_group_rows` for COUNT(*)
/// (each unmatched NULL-key preserved row survives the outer join) or 0 for COUNT(col).
template <typename KeyT>
__global__ void write_null_group_kernel(
  KeyT* key, int64_t* value, cudf::size_type row, bool count_star, int64_t null_group_rows)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    key[row]   = KeyT{0};
    value[row] = count_star ? null_group_rows : 0;
  }
}

void write_null_group_row(cudf::column& key_col,
                          cudf::column& value_col,
                          cudf::size_type row_idx,
                          bool count_star,
                          int64_t null_group_rows,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  if (row_idx < 0 || row_idx >= key_col.size() || row_idx >= value_col.size()) {
    throw sirius::internal_exception(
      "dense_count_join: NULL-group row {} is outside output column sizes {} and {}",
      row_idx,
      key_col.size(),
      value_col.size());
  }
  auto key_view   = key_col.mutable_view();
  auto value_view = value_col.mutable_view();
  auto launch     = [&](auto key_tag) {
    using KeyT = decltype(key_tag);
    write_null_group_kernel<KeyT><<<1, 1, 0, stream.value()>>>(key_view.template data<KeyT>(),
                                                               value_view.template data<int64_t>(),
                                                               row_idx,
                                                               count_star,
                                                               null_group_rows);
  };
  switch (key_view.type().id()) {
    case cudf::type_id::INT32: launch(int32_t{}); break;
    case cudf::type_id::INT64: launch(int64_t{}); break;
    default:
      throw sirius::internal_exception(
        "dense_count_join: unsupported output key type {} (expected INT32/INT64)",
        static_cast<int32_t>(key_view.type().id()));
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  auto mask = cudf::create_null_mask(key_col.size(), cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(
    static_cast<cudf::bitmask_type*>(mask.data()), row_idx, row_idx + 1, false, stream);
  key_col.set_null_mask(std::move(mask), 1);
}

template <typename CountT>
std::unique_ptr<cudf::table> emit_impl(CountT const* presence,
                                       CountT const* counts,
                                       int64_t min_key,
                                       int64_t range,
                                       cudf::data_type key_type,
                                       bool count_star,
                                       int64_t null_group_rows,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr,
                                       bool check_product_overflow)
{
  if (null_group_rows < 0) {
    throw sirius::internal_exception("dense_count_join: negative NULL-group row count");
  }
  // Route thrust/cub temporary storage through the memory space's resource so pool
  // accounting stays honest (the single-argument overload would use the device default).
  auto const policy = rmm::exec_policy(stream, mr);
  auto const begin  = thrust::make_counting_iterator<int64_t>(0);
  auto const end    = thrust::make_counting_iterator<int64_t>(range);

  int64_t const num_groups =
    thrust::count_if(policy, begin, end, presence_positive<CountT>{presence});
  auto const group_rows = checked_output_rows(num_groups, false);
  auto const total_rows = checked_output_rows(num_groups, null_group_rows > 0);

  auto key_col =
    cudf::make_fixed_width_column(key_type, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto value_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  std::optional<cudf::numeric_scalar<int32_t>> overflow_flag;
  if (check_product_overflow && num_groups > 0) { overflow_flag.emplace(0, true, stream, mr); }

  if (num_groups > 0) {
    rmm::device_uvector<int64_t> selected(static_cast<std::size_t>(group_rows), stream, mr);
    thrust::copy_if(policy, begin, end, selected.begin(), presence_positive<CountT>{presence});

    auto const grid = grid_size_for(num_groups);
    auto key_view   = key_col->mutable_view();
    auto value_view = value_col->mutable_view();
    auto launch     = [&](auto key_tag) {
      using KeyT = decltype(key_tag);
      emit_kernel<KeyT, CountT><<<grid, k_block_size, 0, stream.value()>>>(
        selected.data(),
        num_groups,
        presence,
        counts,
        min_key,
        count_star,
        key_view.template data<KeyT>(),
        value_view.template data<int64_t>(),
        overflow_flag ? overflow_flag->data() : nullptr);
    };
    switch (key_type.id()) {
      case cudf::type_id::INT32: launch(int32_t{}); break;
      case cudf::type_id::INT64: launch(int64_t{}); break;
      default:
        throw sirius::internal_exception(
          "dense_count_join: unsupported output key type {} (expected INT32/INT64)",
          static_cast<int32_t>(key_type.id()));
    }
    CUDF_CUDA_TRY(cudaGetLastError());
    // The scalar read synchronizes only on the rare path whose coarse host bound was inconclusive.
    if (overflow_flag && overflow_flag->value(stream) != 0) {
      throw sirius::invalid_input_exception("dense_count_join: COUNT result exceeds BIGINT max {}",
                                            k_bigint_max);
    }
  }

  if (null_group_rows > 0) {
    write_null_group_row(*key_col, *value_col, group_rows, count_star, null_group_rows, stream, mr);
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(key_col));
  columns.push_back(std::move(value_col));
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace

std::optional<std::pair<int64_t, int64_t>> dense_count_global_minmax(
  std::vector<cudf::column_view> const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<int64_t> extrema(2, stream, mr);
  initialize_extrema_kernel<<<1, 1, 0, stream.value()>>>(extrema.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  bool has_values = false;
  for (auto const& column : keys) {
    if (column.size() == 0 || column.size() == column.null_count()) { continue; }

    auto minmax = cudf::minmax(column, stream, mr);
    auto launch = [&](auto key_tag) {
      using KeyT            = decltype(key_tag);
      auto const& batch_min = static_cast<cudf::numeric_scalar<KeyT> const&>(*minmax.first);
      auto const& batch_max = static_cast<cudf::numeric_scalar<KeyT> const&>(*minmax.second);
      merge_extrema_kernel<KeyT>
        <<<1, 1, 0, stream.value()>>>(batch_min.data(), batch_max.data(), extrema.data());
    };
    switch (column.type().id()) {
      case cudf::type_id::INT32: launch(int32_t{}); break;
      case cudf::type_id::INT64: launch(int64_t{}); break;
      default:
        throw sirius::internal_exception(
          "dense_count_join: unsupported minmax key type {} (expected INT32/INT64)",
          static_cast<int32_t>(column.type().id()));
    }
    CUDF_CUDA_TRY(cudaGetLastError());
    has_values = true;
  }

  if (!has_values) { return std::nullopt; }

  std::array<int64_t, 2> host_extrema{};
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_extrema.data(),
                                extrema.data(),
                                sizeof(host_extrema),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  return std::pair{host_extrema[0], host_extrema[1]};
}

dense_count_state::dense_count_state(int64_t min_key,
                                     int64_t range,
                                     bool wide,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  : _min_key(min_key), _range(range), _wide(wide)
{
  auto const slot_bytes = _wide ? sizeof(uint64_t) : sizeof(uint32_t);
  auto const layout     = checked_histogram_layout(range, slot_bytes);
  if (_wide) {
    _presence64.emplace(layout.slots, stream, mr);
    _counts64.emplace(layout.slots, stream, mr);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(_presence64->data(), 0, layout.bytes_per_histogram, stream.value()));
    CUDF_CUDA_TRY(
      cudaMemsetAsync(_counts64->data(), 0, layout.bytes_per_histogram, stream.value()));
  } else {
    _presence32.emplace(layout.slots, stream, mr);
    _counts32.emplace(layout.slots, stream, mr);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(_presence32->data(), 0, layout.bytes_per_histogram, stream.value()));
    CUDF_CUDA_TRY(
      cudaMemsetAsync(_counts32->data(), 0, layout.bytes_per_histogram, stream.value()));
  }
}

void dense_count_state::accumulate_preserved(cudf::column_view const& keys,
                                             rmm::cuda_stream_view stream)
{
  // No bounds check: the histogram is sized from these columns' global min/max.
  if (_wide) {
    accumulate_impl<uint64_t>(
      keys, nullptr, _min_key, _range, /*bounds_check=*/false, _presence64->data(), stream);
  } else {
    accumulate_impl<uint32_t>(
      keys, nullptr, _min_key, _range, /*bounds_check=*/false, _presence32->data(), stream);
  }
}

void dense_count_state::accumulate_counted(cudf::column_view const& keys,
                                           cudf::column_view const* count_validity_source,
                                           rmm::cuda_stream_view stream)
{
  if (_wide) {
    accumulate_impl<uint64_t>(keys,
                              count_validity_source,
                              _min_key,
                              _range,
                              /*bounds_check=*/true,
                              _counts64->data(),
                              stream);
  } else {
    accumulate_impl<uint32_t>(keys,
                              count_validity_source,
                              _min_key,
                              _range,
                              /*bounds_check=*/true,
                              _counts32->data(),
                              stream);
  }
}

std::unique_ptr<cudf::table> dense_count_state::emit(cudf::data_type key_type,
                                                     bool count_star,
                                                     int64_t null_group_rows,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr,
                                                     bool check_product_overflow) const
{
  if (_wide) {
    return emit_impl<uint64_t>(_presence64->data(),
                               _counts64->data(),
                               _min_key,
                               _range,
                               key_type,
                               count_star,
                               null_group_rows,
                               stream,
                               mr,
                               check_product_overflow);
  }
  return emit_impl<uint32_t>(_presence32->data(),
                             _counts32->data(),
                             _min_key,
                             _range,
                             key_type,
                             count_star,
                             null_group_rows,
                             stream,
                             mr,
                             check_product_overflow);
}

void throw_if_count_product_overflows(cudf::column_view const& lhs,
                                      cudf::column_view const& rhs,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (lhs.type().id() != cudf::type_id::INT64 || rhs.type().id() != cudf::type_id::INT64) {
    throw sirius::internal_exception(
      "dense_count_join: overflow validation requires two INT64 columns, got {} and {}",
      static_cast<int32_t>(lhs.type().id()),
      static_cast<int32_t>(rhs.type().id()));
  }
  if (lhs.size() != rhs.size()) {
    throw sirius::internal_exception(
      "dense_count_join: overflow validation column sizes differ ({} versus {})",
      lhs.size(),
      rhs.size());
  }
  if (lhs.null_count() != 0 || rhs.null_count() != 0) {
    throw sirius::internal_exception(
      "dense_count_join: overflow validation requires non-NULL count columns");
  }
  if (lhs.size() == 0) { return; }

  cudf::numeric_scalar<int32_t> status(0, true, stream, mr);
  validate_product_kernel<<<grid_size_for(lhs.size()), k_block_size, 0, stream.value()>>>(
    lhs.data<int64_t>(), rhs.data<int64_t>(), static_cast<int64_t>(lhs.size()), status.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Deliberately synchronous: callers reach this function only when the cheap host upper bound
  // cannot prove safety, so it adds no synchronization to the normal path.
  auto const result = status.value(stream);
  if (result == 2) {
    throw sirius::internal_exception(
      "dense_count_join: aggregate multiplicities must be nonnegative");
  }
  if (result == 1) {
    throw sirius::invalid_input_exception("dense_count_join: COUNT result exceeds BIGINT max {}",
                                          k_bigint_max);
  }
}

std::unique_ptr<cudf::table> dense_count_empty_output(cudf::data_type key_type,
                                                      bool count_star,
                                                      int64_t null_group_rows,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  if (null_group_rows < 0) {
    throw sirius::internal_exception("dense_count_join: negative NULL-group row count");
  }
  cudf::size_type const total_rows = null_group_rows > 0 ? 1 : 0;
  auto key_col =
    cudf::make_fixed_width_column(key_type, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto value_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (total_rows == 1) {
    write_null_group_row(*key_col,
                         *value_col,
                         /*row_idx=*/0,
                         count_star,
                         null_group_rows,
                         stream,
                         mr);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(key_col));
  columns.push_back(std::move(value_col));
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op

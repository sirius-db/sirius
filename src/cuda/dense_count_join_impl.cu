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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

constexpr int k_block_size     = 256;
constexpr int64_t k_max_blocks = 4096;

[[nodiscard]] unsigned grid_size_for(int64_t n)
{
  auto const blocks = std::min<int64_t>((n + k_block_size - 1) / k_block_size, k_max_blocks);
  return static_cast<unsigned>(std::max<int64_t>(blocks, 1));
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
                            int64_t* __restrict__ out_values)
{
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_selected;
       i += stride) {
    auto const k = selected[i];
    auto const p = static_cast<uint64_t>(presence[k]);
    auto matched = static_cast<uint64_t>(counts[k]);
    if (count_star && matched == 0) { matched = 1; }
    out_keys[i]   = static_cast<KeyT>(min_key + k);
    out_values[i] = static_cast<int64_t>(p * matched);
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
void write_null_group_row(cudf::column& key_col,
                          cudf::column& value_col,
                          int64_t row_idx,
                          std::size_t key_size_bytes,
                          bool count_star,
                          int64_t null_group_rows,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  auto key_view   = key_col.mutable_view();
  auto value_view = value_col.mutable_view();
  CUDF_CUDA_TRY(
    cudaMemsetAsync(static_cast<char*>(key_view.head<void>()) + row_idx * key_size_bytes,
                    0,
                    key_size_bytes,
                    stream.value()));
  int64_t const null_value = count_star ? null_group_rows : 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(value_view.data<int64_t>() + row_idx,
                                &null_value,
                                sizeof(int64_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  // Synchronize before &null_value goes out of scope (host stack source of the async copy).
  stream.synchronize();

  auto mask = cudf::create_null_mask(key_col.size(), cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()),
                      static_cast<cudf::size_type>(row_idx),
                      static_cast<cudf::size_type>(row_idx + 1),
                      false,
                      stream);
  key_col.set_null_mask(std::move(mask), 1);
}

[[nodiscard]] std::size_t key_size_bytes_for(cudf::data_type key_type)
{
  switch (key_type.id()) {
    case cudf::type_id::INT32: return 4;
    case cudf::type_id::INT64: return 8;
    default:
      throw sirius::internal_exception(
        "dense_count_join: unsupported output key type {} (expected INT32/INT64)",
        static_cast<int32_t>(key_type.id()));
  }
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
                                       rmm::device_async_resource_ref mr)
{
  // Route thrust/cub temporary storage through the memory space's resource so pool
  // accounting stays honest (the single-argument overload would use the device default).
  auto const policy = rmm::exec_policy(stream, mr);
  auto const begin  = thrust::make_counting_iterator<int64_t>(0);
  auto const end    = thrust::make_counting_iterator<int64_t>(range);

  int64_t const num_groups =
    thrust::count_if(policy, begin, end, presence_positive<CountT>{presence});
  int64_t const total_rows = num_groups + (null_group_rows > 0 ? 1 : 0);

  auto key_col = cudf::make_fixed_width_column(
    key_type, static_cast<cudf::size_type>(total_rows), cudf::mask_state::UNALLOCATED, stream, mr);
  auto value_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT64},
                                                 static_cast<cudf::size_type>(total_rows),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);

  if (num_groups > 0) {
    rmm::device_uvector<int64_t> selected(static_cast<std::size_t>(num_groups), stream, mr);
    thrust::copy_if(policy, begin, end, selected.begin(), presence_positive<CountT>{presence});

    auto const grid = grid_size_for(num_groups);
    auto key_view   = key_col->mutable_view();
    auto value_view = value_col->mutable_view();
    auto launch     = [&](auto key_tag) {
      using KeyT = decltype(key_tag);
      emit_kernel<KeyT, CountT>
        <<<grid, k_block_size, 0, stream.value()>>>(selected.data(),
                                                    num_groups,
                                                    presence,
                                                    counts,
                                                    min_key,
                                                    count_star,
                                                    key_view.template data<KeyT>(),
                                                    value_view.template data<int64_t>());
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
  }

  if (null_group_rows > 0) {
    write_null_group_row(*key_col,
                         *value_col,
                         num_groups,
                         key_size_bytes_for(key_type),
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

}  // namespace

dense_count_state::dense_count_state(int64_t min_key,
                                     int64_t range,
                                     bool wide,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  : _min_key(min_key), _range(range), _wide(wide)
{
  if (range <= 0) {
    throw sirius::internal_exception("dense_count_state: non-positive range {}", range);
  }
  auto const n = static_cast<std::size_t>(range);
  if (_wide) {
    _presence64.emplace(n, stream, mr);
    _counts64.emplace(n, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(_presence64->data(), 0, n * sizeof(uint64_t), stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(_counts64->data(), 0, n * sizeof(uint64_t), stream.value()));
  } else {
    _presence32.emplace(n, stream, mr);
    _counts32.emplace(n, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(_presence32->data(), 0, n * sizeof(uint32_t), stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(_counts32->data(), 0, n * sizeof(uint32_t), stream.value()));
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
                                                     rmm::device_async_resource_ref mr) const
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
                               mr);
  }
  return emit_impl<uint32_t>(_presence32->data(),
                             _counts32->data(),
                             _min_key,
                             _range,
                             key_type,
                             count_star,
                             null_group_rows,
                             stream,
                             mr);
}

std::unique_ptr<cudf::table> dense_count_empty_output(cudf::data_type key_type,
                                                      bool count_star,
                                                      int64_t null_group_rows,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const total_rows = null_group_rows > 0 ? 1 : 0;
  auto key_col =
    cudf::make_fixed_width_column(key_type, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto value_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (total_rows == 1) {
    write_null_group_row(*key_col,
                         *value_col,
                         /*row_idx=*/0,
                         key_size_bytes_for(key_type),
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

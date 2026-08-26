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

// clang-format off
#include <helper/utils.hpp>
#include <op/aggregate/dense_count_join_impl.hpp>
#include <sirius/exception.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda/atomic>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>
// clang-format on

namespace sirius::op {

namespace {

constexpr int k_block_size      = 256;
constexpr uint64_t k_bigint_max = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
// Largest histogram, in bytes, that accumulate_impl privatizes into shared memory. 48 KiB is the
// default per-block dynamic-shared limit, so no cudaFuncSetAttribute opt-in is required.
constexpr std::size_t k_max_privatized_smem_bytes = 48 * 1024;

// Attribute of the current device, cached per thread; 0 when the query fails.
template <cudaDeviceAttr Attr>
[[nodiscard]] int device_attribute()
{
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
  thread_local int cached_device = -1;
  thread_local int cached_value  = 0;
  if (cached_device != device) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, Attr, device) != cudaSuccess) { return 0; }
    cached_value  = value;
    cached_device = device;
  }
  return cached_value;
}

// Dynamic shared memory a privatized histogram may occupy; 0 disables privatization.
[[nodiscard]] std::size_t smem_histogram_budget()
{
  return std::min(k_max_privatized_smem_bytes,
                  static_cast<std::size_t>(device_attribute<cudaDevAttrMaxSharedMemoryPerBlock>()));
}

// Every kernel here is grid-stride and so correct at any positive grid: size it to what the kernel
// can keep resident rather than to one block per k_block_size rows. Falls back to the full ceiling
// when the device queries fail.
template <typename KernelT>
[[nodiscard]] unsigned grid_size_for(int64_t n, KernelT kernel, std::size_t dynamic_smem_bytes = 0)
{
  auto const blocks   = std::max<int64_t>(sirius::utils::ceil_div<int64_t>(n, k_block_size), 1);
  auto const sm_count = device_attribute<cudaDevAttrMultiProcessorCount>();
  int blocks_per_sm   = 0;
  if (sm_count > 0 &&
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, k_block_size, dynamic_smem_bytes) == cudaSuccess &&
      blocks_per_sm > 0) {
    return static_cast<unsigned>(
      std::min<int64_t>(blocks, static_cast<int64_t>(sm_count) * blocks_per_sm));
  }
  return static_cast<unsigned>(blocks);
}

// Invokes fn with an INT32 or INT64 tag; what names the column in the diagnostic message.
template <typename Fn>
  requires std::invocable<Fn, int32_t> && std::invocable<Fn, int64_t>
decltype(auto) dispatch_key_type(cudf::type_id id, std::string_view what, Fn&& fn)
{
  switch (id) {
    case cudf::type_id::INT32: return std::forward<Fn>(fn)(int32_t{});
    case cudf::type_id::INT64: return std::forward<Fn>(fn)(int64_t{});
    default:
      throw sirius::internal_exception(
        "dense_count_join: unsupported {} type {} (expected INT32/INT64)",
        what,
        static_cast<int32_t>(id));
  }
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

// The scope is explicit because cudf::detail::atomic_add leaves cuda::atomic_ref at its default
// thread_scope_system, which would emit system-scoped atomics on this file's hottest instruction.
template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ void bin_add(T* slot, T delta)
{
  cuda::atomic_ref<T, Scope>{*slot}.fetch_add(delta, cuda::memory_order_relaxed);
}

// Per-row validity filter and domain offset, shared by both accumulate kernels.
template <typename KeyT>
__device__ __forceinline__ bool key_bin_offset(KeyT const* __restrict__ keys,
                                               cudf::bitmask_type const* __restrict__ key_mask,
                                               cudf::size_type key_mask_offset,
                                               cudf::bitmask_type const* __restrict__ value_mask,
                                               cudf::size_type value_mask_offset,
                                               int64_t i,
                                               int64_t min_key,
                                               int64_t slots,
                                               bool bounds_check,
                                               uint64_t& offset)
{
  if (key_mask != nullptr &&
      !cudf::bit_is_set(key_mask, key_mask_offset + static_cast<cudf::size_type>(i))) {
    return false;
  }
  if (value_mask != nullptr &&
      !cudf::bit_is_set(value_mask, value_mask_offset + static_cast<cudf::size_type>(i))) {
    return false;
  }
  // Unsigned subtraction avoids signed overflow; offset < slots exactly for in-domain keys.
  offset = static_cast<uint64_t>(static_cast<int64_t>(keys[i])) - static_cast<uint64_t>(min_key);
  return !bounds_check || offset < static_cast<uint64_t>(slots);
}

template <typename KeyT, typename CountT>
__global__ void accumulate_kernel(KeyT const* __restrict__ keys,
                                  cudf::bitmask_type const* __restrict__ key_mask,
                                  cudf::size_type key_mask_offset,
                                  cudf::bitmask_type const* __restrict__ value_mask,
                                  cudf::size_type value_mask_offset,
                                  int64_t n,
                                  int64_t min_key,
                                  int64_t slots,
                                  bool bounds_check,
                                  CountT* __restrict__ bins)
{
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    uint64_t offset = 0;
    if (key_bin_offset(keys,
                       key_mask,
                       key_mask_offset,
                       value_mask,
                       value_mask_offset,
                       i,
                       min_key,
                       slots,
                       bounds_check,
                       offset)) {
      bin_add<cuda::thread_scope_device>(&bins[offset], CountT{1});
    }
  }
}

// accumulate_kernel with a per-block histogram, flushed to `bins` once at the end, so that a small
// domain does not serialize every row on a handful of global addresses. Shared slots are uint32 for
// both CountT widths: a per-block partial is bounded by the block's share of the rows, itself
// bounded by cudf::size_type max. The launch gate guarantees `slots * sizeof(uint32_t)` dynamic
// shared bytes are available and that `slots` fits int32_t.
template <typename KeyT, typename CountT>
__global__ void accumulate_privatized_kernel(KeyT const* __restrict__ keys,
                                             cudf::bitmask_type const* __restrict__ key_mask,
                                             cudf::size_type key_mask_offset,
                                             cudf::bitmask_type const* __restrict__ value_mask,
                                             cudf::size_type value_mask_offset,
                                             int64_t n,
                                             int64_t min_key,
                                             int32_t slots,
                                             bool bounds_check,
                                             CountT* __restrict__ bins)
{
  // A typed extern __shared__ array would collide across this kernel's CountT instantiations.
  extern __shared__ __align__(alignof(uint32_t)) unsigned char smem_raw[];
  auto* const block_bins = reinterpret_cast<uint32_t*>(smem_raw);

  auto const slot_stride = static_cast<int32_t>(blockDim.x);
  for (int32_t j = static_cast<int32_t>(threadIdx.x); j < slots; j += slot_stride) {
    block_bins[j] = 0;
  }
  __syncthreads();

  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    uint64_t offset = 0;
    if (key_bin_offset(keys,
                       key_mask,
                       key_mask_offset,
                       value_mask,
                       value_mask_offset,
                       i,
                       min_key,
                       slots,
                       bounds_check,
                       offset)) {
      bin_add<cuda::thread_scope_block>(&block_bins[offset], uint32_t{1});
    }
  }
  __syncthreads();

  for (int32_t j = static_cast<int32_t>(threadIdx.x); j < slots; j += slot_stride) {
    auto const partial = block_bins[j];
    if (partial != 0) {
      bin_add<cuda::thread_scope_device>(&bins[j], static_cast<CountT>(partial));
    }
  }
}

template <typename CountT>
struct presence_positive {
  CountT const* presence;
  __device__ bool operator()(int64_t k) const { return presence[k] != CountT{0}; }
};

// A null @p selected is the identity permutation: group i is slot i, so no gather map has to be
// materialized and the presence/counts reads stream instead of gathering.
template <typename KeyT, typename CountT>
__global__ void emit_kernel(int64_t const* __restrict__ selected,
                            int64_t num_selected,
                            CountT const* __restrict__ presence,
                            CountT const* __restrict__ counts,
                            int64_t min_key,
                            int64_t unmatched_fill,
                            KeyT* __restrict__ out_keys,
                            int64_t* __restrict__ out_values,
                            int32_t* __restrict__ overflow_flag)
{
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_selected;
       i += stride) {
    auto const k = selected != nullptr ? selected[i] : i;
    auto const p = static_cast<uint64_t>(presence[k]);
    auto matched = static_cast<uint64_t>(counts[k]);
    if (matched == 0) { matched = static_cast<uint64_t>(unmatched_fill); }
    out_keys[i] = static_cast<KeyT>(min_key + k);
    // An overflowing product wraps in uint64_t, which is defined and discarded: the host throws.
    out_values[i] = static_cast<int64_t>(p * matched);
    if (overflow_flag != nullptr && matched != 0 && p > k_bigint_max / matched) {
      cuda::atomic_ref<int32_t, cuda::thread_scope_device>{*overflow_flag}.store(
        1, cuda::memory_order_relaxed);
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
    // Both operands are COUNT results, so nonnegative; a negative would widen to a huge unsigned
    // value and trip the check anyway.
    auto const left_u  = static_cast<uint64_t>(lhs[i]);
    auto const right_u = static_cast<uint64_t>(rhs[i]);
    if (right_u != 0 && left_u > k_bigint_max / right_u) {
      cuda::atomic_ref<int32_t, cuda::thread_scope_device>{*status}.store(
        1, cuda::memory_order_relaxed);
    }
  }
}

template <typename CountT>
void accumulate_impl(cudf::column_view const& keys,
                     std::optional<cudf::column_view> const& count_argument,
                     dense_count_layout const& layout,
                     bool bounds_check,
                     CountT* bins,
                     rmm::cuda_stream_view stream)
{
  if (count_argument && count_argument->size() != keys.size()) {
    throw sirius::internal_exception(
      "dense_count_join: COUNT argument has {} rows but its key column has {}",
      count_argument->size(),
      keys.size());
  }
  auto const n = static_cast<int64_t>(keys.size());
  if (n == 0) { return; }

  cudf::bitmask_type const* key_mask = keys.null_count() > 0 ? keys.null_mask() : nullptr;
  cudf::bitmask_type const* val_mask = nullptr;
  cudf::size_type val_mask_offset    = 0;
  if (count_argument && count_argument->null_count() > 0) {
    val_mask        = count_argument->null_mask();
    val_mask_offset = count_argument->offset();
  }

  auto const min_key = layout.min_key();
  auto const slots   = static_cast<int64_t>(layout.slots());
  // Privatize whenever the domain fits in shared memory, except when the batch is too short for
  // the zeroing and flush passes to pay for the contention they remove.
  auto const budget_slots = static_cast<int64_t>(smem_histogram_budget() / sizeof(uint32_t));
  auto const privatize    = slots <= budget_slots && n >= 8 * slots;
  auto const smem_bytes =
    privatize ? static_cast<std::size_t>(slots) * sizeof(uint32_t) : std::size_t{0};

  auto launch = [&](auto key_tag) {
    using KeyT = decltype(key_tag);
    if (privatize) {
      auto const grid = grid_size_for(n, accumulate_privatized_kernel<KeyT, CountT>, smem_bytes);
      accumulate_privatized_kernel<KeyT, CountT>
        <<<grid, k_block_size, smem_bytes, stream.value()>>>(keys.template data<KeyT>(),
                                                             key_mask,
                                                             keys.offset(),
                                                             val_mask,
                                                             val_mask_offset,
                                                             n,
                                                             min_key,
                                                             static_cast<int32_t>(slots),
                                                             bounds_check,
                                                             bins);
      return;
    }
    auto const grid = grid_size_for(n, accumulate_kernel<KeyT, CountT>);
    accumulate_kernel<KeyT, CountT>
      <<<grid, k_block_size, 0, stream.value()>>>(keys.template data<KeyT>(),
                                                  key_mask,
                                                  keys.offset(),
                                                  val_mask,
                                                  val_mask_offset,
                                                  n,
                                                  min_key,
                                                  slots,
                                                  bounds_check,
                                                  bins);
  };
  dispatch_key_type(keys.type().id(), "key column", launch);
  // Deliberately not CUDF_CHECK_CUDA: that macro synchronizes the stream in non-NDEBUG builds, and
  // this path runs once per input batch.
  CUDF_CUDA_TRY(cudaGetLastError());
}

void write_null_group_row(cudf::column& key_col,
                          cudf::column& value_col,
                          cudf::size_type row_idx,
                          dense_count_semantics semantics,
                          int64_t null_group_rows,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  auto key_view        = key_col.mutable_view();
  auto value_view      = value_col.mutable_view();
  auto const key_bytes = cudf::size_of(key_view.type());
  // Zero the key payload so the bytes beneath the NULL are deterministic; head<std::byte>() does
  // not participate in overload resolution, so offset the untyped overload instead.
  CUDF_CUDA_TRY(cudaMemsetAsync(static_cast<std::byte*>(key_view.head()) +
                                  static_cast<std::size_t>(key_view.offset() + row_idx) * key_bytes,
                                0,
                                key_bytes,
                                stream.value()));
  cudf::numeric_scalar<int64_t> const fill_value{
    semantics.null_group_value(null_group_rows), true, stream, mr};
  cudf::fill_in_place(value_view, row_idx, row_idx + 1, fill_value, stream);

  auto mask = cudf::create_null_mask(key_col.size(), cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(
    static_cast<cudf::bitmask_type*>(mask.data()), row_idx, row_idx + 1, false, stream);
  key_col.set_null_mask(std::move(mask), 1);
}

// Fills output rows [0, group_rows) of a freshly allocated `[key, BIGINT count]` pair.
template <typename Fn>
concept group_row_writer =
  std::invocable<Fn, cudf::mutable_column_view&, cudf::mutable_column_view&, cudf::size_type>;

// Sole owner of the `[key, BIGINT count]` output tail: the NULL-group row contract, the output size
// check, both column factories, the optional NULL-group row and the table assembly. fill_groups
// runs only when there is at least one group row.
template <group_row_writer Fn>
[[nodiscard]] std::unique_ptr<cudf::table> build_output_table(cudf::data_type key_type,
                                                              int64_t num_groups,
                                                              dense_count_semantics semantics,
                                                              int64_t null_group_rows,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr,
                                                              Fn&& fill_groups)
{
  if (null_group_rows < 0) {
    throw sirius::internal_exception("dense_count_join: negative NULL-group row count");
  }
  auto const append_null_group = null_group_rows > 0;
  auto const total_rows        = checked_output_rows(num_groups, append_null_group);
  auto const group_rows = static_cast<cudf::size_type>(total_rows - (append_null_group ? 1 : 0));

  auto key_col =
    cudf::make_fixed_width_column(key_type, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto value_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  if (group_rows > 0) {
    auto key_view   = key_col->mutable_view();
    auto value_view = value_col->mutable_view();
    std::forward<Fn>(fill_groups)(key_view, value_view, group_rows);
  }
  if (append_null_group) {
    write_null_group_row(*key_col, *value_col, group_rows, semantics, null_group_rows, stream, mr);
  }

  // Assembled locally rather than with sirius::make_table: its header transitively includes DuckDB
  // headers, which do not compile under nvcc.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(key_col));
  columns.push_back(std::move(value_col));
  return std::make_unique<cudf::table>(std::move(columns));
}

template <typename CountT>
std::unique_ptr<cudf::table> emit_impl(CountT const* presence,
                                       CountT const* counts,
                                       dense_count_layout const& layout,
                                       cudf::data_type key_type,
                                       dense_count_semantics semantics,
                                       int64_t null_group_rows,
                                       dense_count_bounds bounds,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const slots = static_cast<int64_t>(layout.slots());
  // Use this memory space's resource for Thrust/CUB temporaries so reservations account for them.
  auto const policy = rmm::exec_policy(stream, mr);
  auto const begin  = thrust::make_counting_iterator<int64_t>(0);
  auto const end    = thrust::make_counting_iterator<int64_t>(slots);

  auto const num_groups = thrust::count_if(policy, begin, end, presence_positive<CountT>{presence});
  std::optional<rmm::device_uvector<int64_t>> selected;
  // A fully occupied domain is the identity permutation, which emit_kernel walks without a gather
  // map. Otherwise the map is sized from the exact group count, never from the domain or the row
  // count: a duplicate-heavy preserved side occupies far fewer slots than either.
  if (num_groups != slots) {
    auto const selected_rows = checked_output_rows(num_groups, /*append_null_group=*/false);
    selected.emplace(static_cast<std::size_t>(selected_rows), stream, mr);
    thrust::copy_if(policy, begin, end, selected->begin(), presence_positive<CountT>{presence});
  }

  auto fill_groups = [&](cudf::mutable_column_view& key_view,
                         cudf::mutable_column_view& value_view,
                         cudf::size_type group_rows) {
    std::optional<cudf::numeric_scalar<int32_t>> overflow_flag;
    if (bounds.may_exceed_bigint()) { overflow_flag.emplace(0, true, stream, mr); }

    auto launch = [&](auto key_tag) {
      using KeyT      = decltype(key_tag);
      auto const grid = grid_size_for(group_rows, emit_kernel<KeyT, CountT>);
      emit_kernel<KeyT, CountT><<<grid, k_block_size, 0, stream.value()>>>(
        selected ? selected->data() : nullptr,
        group_rows,
        presence,
        counts,
        layout.min_key(),
        semantics.unmatched_fill,
        key_view.template data<KeyT>(),
        value_view.template data<int64_t>(),
        overflow_flag ? overflow_flag->data() : nullptr);
    };
    dispatch_key_type(key_type.id(), "output key", launch);
    CUDF_CUDA_TRY(cudaGetLastError());
    // The scalar read synchronizes only on the rare path whose coarse host bound was inconclusive.
    if (overflow_flag && overflow_flag->value(stream) != 0) {
      throw sirius::invalid_input_exception("dense_count_join: COUNT result exceeds BIGINT max {}",
                                            k_bigint_max);
    }
  };

  return build_output_table(
    key_type, num_groups, semantics, null_group_rows, stream, mr, fill_groups);
}

}  // namespace

std::optional<dense_count_layout> dense_count_layout::plan(int64_t min_key,
                                                           int64_t max_key,
                                                           int64_t preserved_rows,
                                                           int64_t counted_rows) noexcept
{
  if (max_key < min_key) { return std::nullopt; }
  // A zero unsigned range denotes the full 64-bit domain, which no histogram can represent.
  auto const range_u = static_cast<uint64_t>(max_key) - static_cast<uint64_t>(min_key) + 1;
  auto const slot_bytes =
    (std::cmp_greater_equal(preserved_rows, std::numeric_limits<uint32_t>::max()) ||
     std::cmp_greater_equal(counted_rows, std::numeric_limits<uint32_t>::max()))
      ? sizeof(uint64_t)
      : sizeof(uint32_t);
  constexpr auto size_max = std::numeric_limits<std::size_t>::max();
  if (range_u == 0 || range_u > size_max / (2 * slot_bytes) ||
      range_u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  return dense_count_layout{min_key, static_cast<std::size_t>(range_u), slot_bytes};
}

std::optional<std::pair<int64_t, int64_t>> dense_count_global_minmax(
  std::vector<cudf::column_view> const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  using scalar_pair = std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>>;
  std::vector<scalar_pair> batch_extrema;
  batch_extrema.reserve(keys.size());

  for (auto const& column : keys) {
    if (column.size() == 0 || column.size() == column.null_count()) { continue; }
    batch_extrema.push_back(cudf::minmax(column, stream, mr));
  }

  if (batch_extrema.empty()) { return std::nullopt; }

  // Every reduction is enqueued before the first scalar read, so only that read blocks.
  auto low  = std::numeric_limits<int64_t>::max();
  auto high = std::numeric_limits<int64_t>::min();
  for (auto const& extrema : batch_extrema) {
    auto fold = [&](auto key_tag) {
      using KeyT            = decltype(key_tag);
      auto const& batch_min = static_cast<cudf::numeric_scalar<KeyT> const&>(*extrema.first);
      auto const& batch_max = static_cast<cudf::numeric_scalar<KeyT> const&>(*extrema.second);
      low                   = std::min(low, static_cast<int64_t>(batch_min.value(stream)));
      high                  = std::max(high, static_cast<int64_t>(batch_max.value(stream)));
    };
    dispatch_key_type(extrema.first->type().id(), "minmax key", fold);
  }
  return std::pair{low, high};
}

template <typename CountT>
auto dense_count_state::make_bins(std::size_t slots,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr) -> histograms<CountT>
{
  // plan() proved 2 * slots * sizeof(CountT) fits size_t.
  histograms<CountT> result{rmm::device_uvector<CountT>(2 * slots, stream, mr)};
  CUDF_CUDA_TRY(cudaMemsetAsync(result.bins.data(), 0, 2 * slots * sizeof(CountT), stream.value()));
  return result;
}

dense_count_state::dense_count_state(dense_count_layout const& layout,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  : _layout(layout),
    _bins(layout.slot_bytes() == sizeof(uint64_t)
            ? bins_variant{make_bins<uint64_t>(layout.slots(), stream, mr)}
            : bins_variant{make_bins<uint32_t>(layout.slots(), stream, mr)})
{
}

void dense_count_state::accumulate_preserved(cudf::column_view const& keys,
                                             rmm::cuda_stream_view stream)
{
  std::visit(
    [&](auto& bins) {
      // No bounds check: the histogram is sized from these columns' global min/max.
      accumulate_impl(keys, std::nullopt, _layout, /*bounds_check=*/false, bins.presence(), stream);
    },
    _bins);
}

void dense_count_state::accumulate_counted(cudf::column_view const& keys,
                                           std::optional<cudf::column_view> const& count_argument,
                                           rmm::cuda_stream_view stream)
{
  std::visit(
    [&](auto& bins) {
      accumulate_impl(keys, count_argument, _layout, /*bounds_check=*/true, bins.counts(), stream);
    },
    _bins);
}

std::unique_ptr<cudf::table> dense_count_state::emit(cudf::data_type key_type,
                                                     dense_count_semantics semantics,
                                                     int64_t null_group_rows,
                                                     dense_count_bounds bounds,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr) const
{
  return std::visit(
    [&](auto const& bins) {
      return emit_impl(bins.presence(),
                       bins.counts(),
                       _layout,
                       key_type,
                       semantics,
                       null_group_rows,
                       bounds,
                       stream,
                       mr);
    },
    _bins);
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
  if (lhs.size() == 0) { return; }

  cudf::numeric_scalar<int32_t> status(0, true, stream, mr);
  validate_product_kernel<<<grid_size_for(lhs.size(), validate_product_kernel),
                            k_block_size,
                            0,
                            stream.value()>>>(
    lhs.data<int64_t>(), rhs.data<int64_t>(), static_cast<int64_t>(lhs.size()), status.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // This rare-path validation synchronizes only when the host bound cannot prove safety.
  if (status.value(stream) != 0) {
    throw sirius::invalid_input_exception("dense_count_join: COUNT result exceeds BIGINT max {}",
                                          k_bigint_max);
  }
}

std::unique_ptr<cudf::table> make_null_group_table(cudf::data_type key_type,
                                                   dense_count_semantics semantics,
                                                   int64_t null_group_rows,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  return build_output_table(
    key_type,
    /*num_groups=*/0,
    semantics,
    null_group_rows,
    stream,
    mr,
    [](cudf::mutable_column_view&, cudf::mutable_column_view&, cudf::size_type) {});
}

}  // namespace sirius::op

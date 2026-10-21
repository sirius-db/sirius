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
#include <op/aggregate/low_cardinality_groupby.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
// clang-format on

namespace sirius::op {

namespace {

constexpr int block_size              = 256;
constexpr int warps_per_block         = block_size / 32;
constexpr int code_space              = 1 << 16;  ///< Two one-byte keys form a 16-bit code.
constexpr int bitmap_words            = code_space / 32;
constexpr int max_groups              = 64;
constexpr int max_sum_columns         = 16;
constexpr std::size_t max_shared_size = 48 * 1024;
constexpr uint8_t absent_group        = 0xFF;

/// Device view of the up to two one-character key columns.
struct key_columns {
  int32_t const* offsets[2];
  char const* chars[2];
  int num_keys;
};

/// Returns the row's 16-bit key code, or -1 when a key is not exactly one character long.
__device__ inline int row_code(key_columns const& keys, int64_t row_idx)
{
  int code = 0;
  for (int key_idx = 0; key_idx < keys.num_keys; ++key_idx) {
    int32_t const begin = keys.offsets[key_idx][row_idx];
    if (keys.offsets[key_idx][row_idx + 1] - begin != 1) { return -1; }
    code |= static_cast<int>(static_cast<unsigned char>(keys.chars[key_idx][begin]))
            << (8 * key_idx);
  }
  return code;
}

/**
 * Returns the row's 16-bit key code once every key has been validated as one character long. All
 * values then have length 1, so row r's character is at chars[offsets[0] + r], and the offsets
 * array no longer needs to be read per row.
 */
__device__ inline int validated_row_code(key_columns const& keys, int64_t row_idx)
{
  int code = 0;
  for (int key_idx = 0; key_idx < keys.num_keys; ++key_idx) {
    int32_t const first = keys.offsets[key_idx][0];
    code |= static_cast<int>(static_cast<unsigned char>(keys.chars[key_idx][first + row_idx]))
            << (8 * key_idx);
  }
  return code;
}

/// Marks every key code present in the batch and records one representative row per code.
__global__ void find_groups_kernel(key_columns keys,
                                   int64_t num_rows,
                                   uint32_t* present_codes,
                                   int32_t* representative_rows,
                                   int* invalid_flag)
{
  __shared__ uint32_t block_codes[bitmap_words];
  for (int word_idx = threadIdx.x; word_idx < bitmap_words; word_idx += blockDim.x) {
    block_codes[word_idx] = 0;
  }
  __syncthreads();
  for (int64_t row_idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       row_idx < num_rows;
       row_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int const code = row_code(keys, row_idx);
    if (code < 0) {
      atomicOr(invalid_flag, 1);
      continue;
    }
    uint32_t const bit = 1u << (code & 31);
    if ((block_codes[code >> 5] & bit) == 0) {
      uint32_t const previous = atomicOr(&block_codes[code >> 5], bit);
      if ((previous & bit) == 0) {
        atomicCAS(&representative_rows[code], -1, static_cast<int32_t>(row_idx));
      }
    }
  }
  __syncthreads();
  for (int word_idx = threadIdx.x; word_idx < bitmap_words; word_idx += blockDim.x) {
    if (block_codes[word_idx] != 0) { atomicOr(&present_codes[word_idx], block_codes[word_idx]); }
  }
}

/**
 * Accumulates per-warp 64-bit partial sums and row counts in shared memory, then writes each
 * block's totals as 128-bit values. Slot layout per warp: `[group][column]`, where the last
 * column is the row count.
 */
__global__ void aggregate_kernel(key_columns keys,
                                 int64_t num_rows,
                                 uint8_t const* group_of_code,
                                 int num_groups,
                                 int64_t const* const* sum_columns,
                                 int num_sum_columns,
                                 int64_t max_abs_value,
                                 __int128_t* block_totals,
                                 int* overflow_flag)
{
  extern __shared__ unsigned long long warp_slots[];
  int const slots_per_copy = num_groups * (num_sum_columns + 1);
  for (int slot_idx = threadIdx.x; slot_idx < slots_per_copy * warps_per_block;
       slot_idx += blockDim.x) {
    warp_slots[slot_idx] = 0;
  }
  __syncthreads();
  unsigned long long* const warp_copy = warp_slots + (threadIdx.x / 32) * slots_per_copy;
  for (int64_t row_idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       row_idx < num_rows;
       row_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int const group                       = group_of_code[validated_row_code(keys, row_idx)];
    unsigned long long* const group_slots = warp_copy + group * (num_sum_columns + 1);
    for (int column_idx = 0; column_idx < num_sum_columns; ++column_idx) {
      int64_t const value = sum_columns[column_idx][row_idx];
      if (value > max_abs_value || value < -max_abs_value) { atomicOr(overflow_flag, 1); }
      atomicAdd(&group_slots[column_idx], static_cast<unsigned long long>(value));
    }
    atomicAdd(&group_slots[num_sum_columns], 1ull);
  }
  __syncthreads();
  for (int slot_idx = threadIdx.x; slot_idx < slots_per_copy; slot_idx += blockDim.x) {
    __int128_t total = 0;
    for (int warp_idx = 0; warp_idx < warps_per_block; ++warp_idx) {
      total += static_cast<int64_t>(warp_slots[warp_idx * slots_per_copy + slot_idx]);
    }
    block_totals[static_cast<int64_t>(blockIdx.x) * slots_per_copy + slot_idx] = total;
  }
}

/// Sums a 128-bit value across the warp; lane 0 holds the result.
__device__ inline __int128_t warp_sum(__int128_t value)
{
  for (int offset = 16; offset > 0; offset /= 2) {
    auto const low  = __shfl_down_sync(0xffffffffu, static_cast<uint64_t>(value), offset);
    auto const high = __shfl_down_sync(0xffffffffu, static_cast<int64_t>(value >> 64), offset);
    value += (static_cast<__int128_t>(high) << 64) | static_cast<__int128_t>(low);
  }
  return value;
}

/// Writes each row's dense group number, so the aggregation pass reads one byte per row of keys.
__global__ void group_ids_kernel(key_columns keys,
                                 int64_t num_rows,
                                 uint8_t const* group_of_code,
                                 uint8_t* group_ids)
{
  for (int64_t row_idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       row_idx < num_rows;
       row_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    group_ids[row_idx] = group_of_code[validated_row_code(keys, row_idx)];
  }
}

/**
 * Column-at-a-time aggregation for at most MaxGroups groups. For each sum column, and then for the
 * row count, every thread streams (group id, value) pairs and keeps only MaxGroups 64-bit
 * accumulators in registers, adding each value to all of them under a predicate. That keeps the
 * hot loop free of atomics and the register count low enough for full occupancy. Partials are
 * reduced as 128-bit values across the warp and then the block, and written in the slot layout
 * of aggregate_kernel: [group][column], with the row count after the sum columns.
 */
template <int MaxGroups>
__global__ void __launch_bounds__(block_size)
  aggregate_columns_kernel(uint8_t const* group_ids,
                           int64_t num_rows,
                           int num_groups,
                           int64_t const* const* sum_columns,
                           int num_sum_columns,
                           int64_t max_abs_value,
                           __int128_t* block_totals,
                           int* overflow_flag)
{
  __shared__ __int128_t warp_totals[warps_per_block][MaxGroups];
  int const warp_idx       = threadIdx.x / 32;
  int const lane_idx       = threadIdx.x % 32;
  int const slots_per_copy = num_groups * (num_sum_columns + 1);
  bool overflow            = false;
  for (int column_idx = 0; column_idx <= num_sum_columns; ++column_idx) {
    bool const is_count             = column_idx == num_sum_columns;
    int64_t const* const column     = is_count ? nullptr : sum_columns[column_idx];
    int64_t accumulators[MaxGroups] = {};
    for (int64_t row_idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         row_idx < num_rows;
         row_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      int const group     = group_ids[row_idx];
      int64_t const value = is_count ? 1 : column[row_idx];
      overflow |= value > max_abs_value || value < -max_abs_value;
#pragma unroll
      for (int group_idx = 0; group_idx < MaxGroups; ++group_idx) {
        accumulators[group_idx] += group_idx == group ? value : 0;
      }
    }
#pragma unroll
    for (int group_idx = 0; group_idx < MaxGroups; ++group_idx) {
      __int128_t const total = warp_sum(accumulators[group_idx]);
      if (lane_idx == 0) { warp_totals[warp_idx][group_idx] = total; }
    }
    __syncthreads();
    if (threadIdx.x < num_groups) {
      __int128_t total = 0;
      for (int warp = 0; warp < warps_per_block; ++warp) {
        total += warp_totals[warp][threadIdx.x];
      }
      block_totals[static_cast<int64_t>(blockIdx.x) * slots_per_copy +
                   threadIdx.x * (num_sum_columns + 1) + column_idx] = total;
    }
    __syncthreads();
  }
  if (overflow) { atomicOr(overflow_flag, 1); }
}

/// Sums the per-block totals of each slot. One block per slot.
__global__ void combine_kernel(__int128_t const* block_totals,
                               int num_blocks,
                               int slots_per_copy,
                               __int128_t* slot_totals)
{
  __shared__ __int128_t partial[block_size];
  __int128_t total = 0;
  for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
    total += block_totals[static_cast<int64_t>(block_idx) * slots_per_copy + blockIdx.x];
  }
  partial[threadIdx.x] = total;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { slot_totals[blockIdx.x] = partial[0]; }
}

/// Copies one output column's per-group values out of the slot totals.
template <typename Output>
__global__ void extract_kernel(
  __int128_t const* slot_totals, int num_groups, int slot_stride, int column_slot, Output* output)
{
  int const group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group < num_groups) {
    output[group] = static_cast<Output>(slot_totals[group * slot_stride + column_slot]);
  }
}

void check_cuda(cudaError_t status) { CUDF_CUDA_TRY(status); }

}  // namespace

std::unique_ptr<cudf::table> try_low_cardinality_groupby(
  cudf::table_view const& input,
  std::vector<int> const& group_idx,
  std::vector<cudf::aggregation::Kind> const& aggregates,
  std::vector<int> const& aggregate_idx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  int64_t const num_rows = input.num_rows();
  if (num_rows == 0 || num_rows >= std::numeric_limits<int32_t>::max() || group_idx.empty() ||
      group_idx.size() > 2 || aggregates.empty() || aggregates.size() != aggregate_idx.size()) {
    return nullptr;
  }

  key_columns keys{};
  keys.num_keys = static_cast<int>(group_idx.size());
  for (int key_idx = 0; key_idx < keys.num_keys; ++key_idx) {
    auto const& column = input.column(group_idx[key_idx]);
    if (column.type().id() != cudf::type_id::STRING || column.has_nulls()) { return nullptr; }
    cudf::strings_column_view const strings(column);
    if (strings.offsets().type().id() != cudf::type_id::INT32 ||
        strings.chars_size(stream) != num_rows) {
      return nullptr;
    }
    keys.offsets[key_idx] = strings.offsets().data<int32_t>() + column.offset();
    keys.chars[key_idx]   = strings.chars_begin(stream);
  }

  // Map each aggregate to a value slot: distinct SUM input columns, then the shared row count.
  std::vector<int> sum_column_ids;
  std::vector<int> aggregate_slot(aggregates.size());
  for (std::size_t agg_idx = 0; agg_idx < aggregates.size(); ++agg_idx) {
    auto const& column = input.column(aggregate_idx[agg_idx]);
    switch (aggregates[agg_idx]) {
      case cudf::aggregation::Kind::SUM: {
        auto const type_id = column.type().id();
        if ((type_id != cudf::type_id::DECIMAL64 && type_id != cudf::type_id::INT64) ||
            column.has_nulls()) {
          return nullptr;
        }
        auto const found =
          std::find(sum_column_ids.begin(), sum_column_ids.end(), aggregate_idx[agg_idx]);
        aggregate_slot[agg_idx] = static_cast<int>(found - sum_column_ids.begin());
        if (found == sum_column_ids.end()) { sum_column_ids.push_back(aggregate_idx[agg_idx]); }
        break;
      }
      case cudf::aggregation::Kind::COUNT_VALID:
        if (column.has_nulls()) { return nullptr; }
        aggregate_slot[agg_idx] = -1;
        break;
      case cudf::aggregation::Kind::COUNT_ALL: aggregate_slot[agg_idx] = -1; break;
      default: return nullptr;
    }
  }
  int const num_sum_columns = static_cast<int>(sum_column_ids.size());
  if (num_sum_columns > max_sum_columns) { return nullptr; }
  for (auto& slot : aggregate_slot) {
    if (slot < 0) { slot = num_sum_columns; }
  }

  // Pass 1: find the groups present and one representative row of each.
  rmm::device_uvector<uint32_t> present_codes(bitmap_words, stream, mr);
  rmm::device_uvector<int32_t> representative_rows(code_space, stream, mr);
  rmm::device_uvector<int> flags(2, stream, mr);
  check_cuda(cudaMemsetAsync(
    present_codes.data(), 0, present_codes.size() * sizeof(uint32_t), stream.value()));
  check_cuda(cudaMemsetAsync(representative_rows.data(),
                             0xFF,
                             representative_rows.size() * sizeof(int32_t),
                             stream.value()));
  check_cuda(cudaMemsetAsync(flags.data(), 0, flags.size() * sizeof(int), stream.value()));

  int device_id = 0;
  check_cuda(cudaGetDevice(&device_id));
  int num_sms = 0;
  check_cuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
  int64_t const max_blocks = (num_rows + block_size - 1) / block_size;
  int const num_blocks     = static_cast<int>(std::min<int64_t>(8L * num_sms, max_blocks));

  find_groups_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
    keys, num_rows, present_codes.data(), representative_rows.data(), flags.data());
  check_cuda(cudaGetLastError());

  std::vector<uint32_t> host_codes(bitmap_words);
  std::vector<int32_t> host_representatives(code_space);
  std::array<int, 2> host_flags{};
  check_cuda(cudaMemcpyAsync(host_codes.data(),
                             present_codes.data(),
                             bitmap_words * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost,
                             stream.value()));
  check_cuda(cudaMemcpyAsync(host_representatives.data(),
                             representative_rows.data(),
                             code_space * sizeof(int32_t),
                             cudaMemcpyDeviceToHost,
                             stream.value()));
  check_cuda(cudaMemcpyAsync(
    host_flags.data(), flags.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (host_flags[0] != 0) { return nullptr; }

  std::vector<uint8_t> host_group_of_code(code_space, absent_group);
  std::vector<int32_t> group_rows;
  for (int code = 0; code < code_space; ++code) {
    if ((host_codes[code >> 5] >> (code & 31)) & 1u) {
      if (static_cast<int>(group_rows.size()) == max_groups) { return nullptr; }
      host_group_of_code[code] = static_cast<uint8_t>(group_rows.size());
      group_rows.push_back(host_representatives[code]);
    }
  }
  int const num_groups     = static_cast<int>(group_rows.size());
  int const slots_per_copy = num_groups * (num_sum_columns + 1);
  std::size_t const shared_size =
    static_cast<std::size_t>(slots_per_copy) * warps_per_block * sizeof(unsigned long long);
  if (num_groups == 0 || shared_size > max_shared_size) { return nullptr; }

  // Pass 2: per-warp partial sums. A warp adds at most rows_per_warp values, so bounding every
  // value by 2^62 / rows_per_warp keeps each 64-bit partial sum from overflowing.
  int64_t const total_threads = static_cast<int64_t>(num_blocks) * block_size;
  int64_t const rows_per_warp = 32 * ((num_rows + total_threads - 1) / total_threads);
  int64_t const max_abs_value = (int64_t{1} << 62) / rows_per_warp;

  rmm::device_uvector<uint8_t> group_of_code(code_space, stream, mr);
  check_cuda(cudaMemcpyAsync(group_of_code.data(),
                             host_group_of_code.data(),
                             code_space,
                             cudaMemcpyHostToDevice,
                             stream.value()));
  std::vector<int64_t const*> host_sum_columns;
  for (int column_id : sum_column_ids) {
    auto const& column = input.column(column_id);
    host_sum_columns.push_back(column.data<int64_t>());
  }
  rmm::device_uvector<int64_t const*> sum_columns(num_sum_columns, stream, mr);
  if (num_sum_columns > 0) {
    check_cuda(cudaMemcpyAsync(sum_columns.data(),
                               host_sum_columns.data(),
                               num_sum_columns * sizeof(int64_t const*),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  }
  rmm::device_uvector<__int128_t> block_totals(
    static_cast<std::size_t>(num_blocks) * slots_per_copy, stream, mr);
  // Up to 8 groups: precompute one-byte group ids, then aggregate column by column with a few
  // register accumulators per thread. More groups fall back to per-warp shared-memory atomics.
  if (num_groups <= 8) {
    rmm::device_uvector<uint8_t> group_ids(num_rows, stream, mr);
    group_ids_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
      keys, num_rows, group_of_code.data(), group_ids.data());
    check_cuda(cudaGetLastError());
    if (num_groups <= 4) {
      aggregate_columns_kernel<4>
        <<<num_blocks, block_size, 0, stream.value()>>>(group_ids.data(),
                                                        num_rows,
                                                        num_groups,
                                                        sum_columns.data(),
                                                        num_sum_columns,
                                                        max_abs_value,
                                                        block_totals.data(),
                                                        flags.data() + 1);
    } else {
      aggregate_columns_kernel<8>
        <<<num_blocks, block_size, 0, stream.value()>>>(group_ids.data(),
                                                        num_rows,
                                                        num_groups,
                                                        sum_columns.data(),
                                                        num_sum_columns,
                                                        max_abs_value,
                                                        block_totals.data(),
                                                        flags.data() + 1);
    }
    check_cuda(cudaGetLastError());
    // group_ids is freed at the end of this scope; the stream orders the free after its readers.
  } else {
    aggregate_kernel<<<num_blocks, block_size, shared_size, stream.value()>>>(keys,
                                                                              num_rows,
                                                                              group_of_code.data(),
                                                                              num_groups,
                                                                              sum_columns.data(),
                                                                              num_sum_columns,
                                                                              max_abs_value,
                                                                              block_totals.data(),
                                                                              flags.data() + 1);
  }
  check_cuda(cudaGetLastError());
  rmm::device_uvector<__int128_t> slot_totals(slots_per_copy, stream, mr);
  combine_kernel<<<slots_per_copy, block_size, 0, stream.value()>>>(
    block_totals.data(), num_blocks, slots_per_copy, slot_totals.data());
  check_cuda(cudaGetLastError());
  check_cuda(cudaMemcpyAsync(
    host_flags.data() + 1, flags.data() + 1, sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (host_flags[1] != 0) { return nullptr; }

  // Keys: gather one representative row per group from the input key columns.
  std::vector<cudf::column_view> key_views;
  for (int column_id : group_idx) {
    key_views.push_back(input.column(column_id));
  }
  rmm::device_uvector<int32_t> gather_map(num_groups, stream, mr);
  check_cuda(cudaMemcpyAsync(gather_map.data(),
                             group_rows.data(),
                             num_groups * sizeof(int32_t),
                             cudaMemcpyHostToDevice,
                             stream.value()));
  cudf::column_view const gather_view(
    cudf::data_type{cudf::type_id::INT32}, num_groups, gather_map.data(), nullptr, 0);
  auto output_columns =
    cudf::gather(
      cudf::table_view(key_views), gather_view, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr)
      ->release();

  int const slot_stride = num_sum_columns + 1;
  for (std::size_t agg_idx = 0; agg_idx < aggregates.size(); ++agg_idx) {
    auto const input_type     = input.column(aggregate_idx[agg_idx]).type();
    bool const is_decimal_sum = aggregates[agg_idx] == cudf::aggregation::Kind::SUM &&
                                input_type.id() == cudf::type_id::DECIMAL64;
    cudf::data_type const output_type =
      is_decimal_sum ? cudf::data_type{cudf::type_id::DECIMAL128, input_type.scale()}
                     : cudf::data_type{cudf::type_id::INT64};
    auto output              = is_decimal_sum
                                 ? cudf::make_fixed_point_column(
                        output_type, num_groups, cudf::mask_state::UNALLOCATED, stream, mr)
                                 : cudf::make_numeric_column(
                        output_type, num_groups, cudf::mask_state::UNALLOCATED, stream, mr);
    int const extract_blocks = (num_groups + block_size - 1) / block_size;
    if (is_decimal_sum) {
      extract_kernel<__int128_t><<<extract_blocks, block_size, 0, stream.value()>>>(
        slot_totals.data(),
        num_groups,
        slot_stride,
        aggregate_slot[agg_idx],
        output->mutable_view().data<__int128_t>());
    } else {
      extract_kernel<int64_t>
        <<<extract_blocks, block_size, 0, stream.value()>>>(slot_totals.data(),
                                                            num_groups,
                                                            slot_stride,
                                                            aggregate_slot[agg_idx],
                                                            output->mutable_view().data<int64_t>());
    }
    check_cuda(cudaGetLastError());
    output_columns.push_back(std::move(output));
  }
  // The device buffers above are freed on return; make sure their last readers have finished.
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(output_columns));
}

}  // namespace sirius::op

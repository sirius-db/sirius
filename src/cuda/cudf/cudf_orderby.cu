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

#include "../operator/cuda_helper.cuh"
#include "config.hpp"
#include "cudf_utils.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_physical_order.hpp"
#include "log/logging.hpp"

#include <cub/cub.cuh>

#include <stdio.h>

#include <algorithm>

namespace duckdb {

// =================================================================================================
// 1. Debug Macros & Common Structs
// =================================================================================================
#define ENABLE_CUDA_DEBUG 1
#if ENABLE_CUDA_DEBUG
#define CUDA_CHECK_AND_SYNC(msg)                                                             \
  {                                                                                          \
    cudaDeviceSynchronize();                                                                 \
    cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) {                                                                \
      SIRIUS_LOG_DEBUG(                                                                      \
        "CUDA Error at [{}]: {} - {}", msg, cudaGetErrorName(err), cudaGetErrorString(err)); \
      return;                                                                                \
    }                                                                                        \
  }
#else
#define CUDA_CHECK_AND_SYNC(msg) \
  {                              \
  }
#endif

#define MAX_THREAD_TOP_K 32

struct str_top_n_record_type {
  uint32_t row_id;
  uint64_t key_prefix;
  __host__ __device__ str_top_n_record_type() : row_id(0xFFFFFFFF), key_prefix(0) {}
  __host__ __device__ str_top_n_record_type(uint32_t _row_id, uint64_t _key_prefix)
    : row_id(_row_id), key_prefix(_key_prefix)
  {
  }
};

enum class KernelColType : int {
  INT_64   = 0,
  INT_32   = 1,
  DOUBLE   = 2,
  STRING   = 3,
  INT_128  = 4,
  INT_16   = 5,
  FLOAT_32 = 6,
  UNKNOWN  = 99
};

struct DeviceKeyColumn {
  int type;
  uint32_t is_asc;
  uint8_t* data;
  uint64_t* offsets;
  uint32_t* null_mask;
};

// =================================================================================================
// 2. Functors for Radix Sort Path (Exact Re-sorting)
// =================================================================================================

struct CustomTopNStringLessThan {
  uint8_t* col_chars;
  uint64_t* col_offsets;
  __host__ __device__ CustomTopNStringLessThan(uint8_t* _col_chars_, uint64_t* _col_offsets_)
    : col_chars(_col_chars_), col_offsets(_col_offsets_)
  {
  }
  __forceinline__ __device__ bool operator()(const str_top_n_record_type& lhs,
                                             const str_top_n_record_type& rhs)
  {
    // Prefix comparison first
    if (lhs.key_prefix != rhs.key_prefix) return lhs.key_prefix < rhs.key_prefix;
    // Full string comparison
    uint32_t left_row_id          = lhs.row_id;
    uint64_t left_start_offset    = col_offsets[left_row_id];
    uint64_t left_len             = col_offsets[left_row_id + 1] - left_start_offset;
    uint8_t* left_chars           = col_chars + left_start_offset;
    uint32_t right_row_id         = rhs.row_id;
    uint64_t right_start_offset   = col_offsets[right_row_id];
    uint64_t right_len            = col_offsets[right_row_id + 1] - right_start_offset;
    uint8_t* right_chars          = col_chars + right_start_offset;
    const uint64_t num_cmp_values = min(left_len, right_len);
#pragma unroll
    for (uint64_t i = 0; i < num_cmp_values; i++) {
      if (left_chars[i] != right_chars[i]) return left_chars[i] < right_chars[i];
    }
    return left_len < right_len;
  }
};

struct CustomTopNStringGreaterThan {
  uint8_t* col_chars;
  uint64_t* col_offsets;
  __host__ __device__ CustomTopNStringGreaterThan(uint8_t* _col_chars_, uint64_t* _col_offsets_)
    : col_chars(_col_chars_), col_offsets(_col_offsets_)
  {
  }
  __forceinline__ __device__ bool operator()(const str_top_n_record_type& lhs,
                                             const str_top_n_record_type& rhs)
  {
    if (lhs.key_prefix != rhs.key_prefix) return lhs.key_prefix > rhs.key_prefix;
    uint32_t left_row_id          = lhs.row_id;
    uint64_t left_start_offset    = col_offsets[left_row_id];
    uint64_t left_len             = col_offsets[left_row_id + 1] - left_start_offset;
    uint8_t* left_chars           = col_chars + left_start_offset;
    uint32_t right_row_id         = rhs.row_id;
    uint64_t right_start_offset   = col_offsets[right_row_id];
    uint64_t right_len            = col_offsets[right_row_id + 1] - right_start_offset;
    uint8_t* right_chars          = col_chars + right_start_offset;
    const uint64_t num_cmp_values = min(left_len, right_len);
#pragma unroll
    for (uint64_t i = 0; i < num_cmp_values; i++) {
      if (left_chars[i] != right_chars[i]) return left_chars[i] > right_chars[i];
    }
    return left_len > right_len;
  }
};

// =================================================================================================
// 3. Device Helpers (Universal)
// =================================================================================================
__device__ __forceinline__ bool is_row_null(const uint32_t* mask, uint32_t row_id)
{
  if (mask == nullptr) return false;
  return !((mask[row_id / 32] >> (row_id % 32)) & 1);
}

template <typename T>
__device__ __forceinline__ T load_unaligned(const void* ptr)
{
  T val;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
  uint8_t* dst       = reinterpret_cast<uint8_t*>(&val);
#pragma unroll
  for (int i = 0; i < sizeof(T); ++i)
    dst[i] = src[i];
  return val;
}

__device__ __forceinline__ void load_int128_safe(uint8_t* base_ptr,
                                                 uint32_t row_id,
                                                 uint64_t& low,
                                                 int64_t& high)
{
  size_t offset = static_cast<size_t>(row_id) * 16;
  low           = load_unaligned<uint64_t>(base_ptr + offset);
  high          = load_unaligned<int64_t>(base_ptr + offset + 8);
}

// [Core] Generate 64-bit Key
__device__ __forceinline__ uint64_t load_primary_key_as_u64(const DeviceKeyColumn& col,
                                                            uint32_t row_id)
{
  if (is_row_null(col.null_mask, row_id)) return 0xFFFFFFFFFFFFFFFFULL;  // NULL is Infinity

  KernelColType type = static_cast<KernelColType>(col.type);
  if (type == KernelColType::INT_64) {
    int64_t val = load_unaligned<int64_t>(col.data + static_cast<size_t>(row_id) * 8);
    return static_cast<uint64_t>(val) ^ 0x8000000000000000ULL;
  } else if (type == KernelColType::INT_32) {
    int32_t val = load_unaligned<int32_t>(col.data + static_cast<size_t>(row_id) * 4);
    return (static_cast<uint64_t>(val) ^ 0x80000000) << 32;
  } else if (type == KernelColType::INT_16) {
    int16_t val = load_unaligned<int16_t>(col.data + static_cast<size_t>(row_id) * 2);
    return (static_cast<uint64_t>(val) ^ 0x8000) << 48;
  } else if (type == KernelColType::DOUBLE) {
    double val    = load_unaligned<double>(col.data + static_cast<size_t>(row_id) * 8);
    uint64_t bits = *reinterpret_cast<uint64_t*>(&val);
    return bits ^ ((static_cast<int64_t>(bits) >> 63) | 0x8000000000000000ULL);
  } else if (type == KernelColType::FLOAT_32) {
    float val     = load_unaligned<float>(col.data + static_cast<size_t>(row_id) * 4);
    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    return (static_cast<uint64_t>(bits ^ ((static_cast<int32_t>(bits) >> 31) | 0x80000000))) << 32;
  } else if (type == KernelColType::INT_128) {
    uint64_t low;
    int64_t high;
    load_int128_safe(col.data, row_id, low, high);
    return static_cast<uint64_t>(high) ^ 0x8000000000000000ULL;
  } else if (type == KernelColType::STRING) {
    uint64_t start     = col.offsets[row_id];
    uint64_t len       = col.offsets[row_id + 1] - start;
    uint64_t prefix    = 0;
    const uint8_t* ptr = col.data + start;
    uint64_t bytes     = min(len, static_cast<uint64_t>(8));
    uint32_t shift     = 56;
    for (int i = 0; i < bytes; ++i) {
      prefix |= static_cast<uint64_t>(ptr[i]) << shift;
      shift -= 8;
    }
    return prefix;
  }
  return 0;
}

// [Core] Generic multi-column comparison
__device__ int compare_rows_multi_col(uint32_t row_a,
                                      uint32_t row_b,
                                      const DeviceKeyColumn* columns,
                                      int num_columns)
{
  for (int i = 0; i < num_columns; ++i) {
    const DeviceKeyColumn& col = columns[i];
    bool null_a                = is_row_null(col.null_mask, row_a);
    bool null_b                = is_row_null(col.null_mask, row_b);
    int cmp                    = 0;
    if (null_a != null_b)
      cmp = null_a ? 1 : -1;
    else if (null_a)
      cmp = 0;
    else {
      KernelColType type = static_cast<KernelColType>(col.type);
      if (type == KernelColType::INT_64) {
        int64_t val_a = load_unaligned<int64_t>(col.data + static_cast<size_t>(row_a) * 8);
        int64_t val_b = load_unaligned<int64_t>(col.data + static_cast<size_t>(row_b) * 8);
        cmp           = (val_a > val_b) - (val_a < val_b);
      } else if (type == KernelColType::INT_32) {
        int32_t val_a = load_unaligned<int32_t>(col.data + static_cast<size_t>(row_a) * 4);
        int32_t val_b = load_unaligned<int32_t>(col.data + static_cast<size_t>(row_b) * 4);
        cmp           = (val_a > val_b) - (val_a < val_b);
      } else if (type == KernelColType::INT_16) {
        int16_t val_a = load_unaligned<int16_t>(col.data + static_cast<size_t>(row_a) * 2);
        int16_t val_b = load_unaligned<int16_t>(col.data + static_cast<size_t>(row_b) * 2);
        cmp           = (val_a > val_b) - (val_a < val_b);
      } else if (type == KernelColType::DOUBLE) {
        double val_a = load_unaligned<double>(col.data + static_cast<size_t>(row_a) * 8);
        double val_b = load_unaligned<double>(col.data + static_cast<size_t>(row_b) * 8);
        cmp          = (val_a > val_b) - (val_a < val_b);
      } else if (type == KernelColType::FLOAT_32) {
        float val_a = load_unaligned<float>(col.data + static_cast<size_t>(row_a) * 4);
        float val_b = load_unaligned<float>(col.data + static_cast<size_t>(row_b) * 4);
        cmp         = (val_a > val_b) - (val_a < val_b);
      } else if (type == KernelColType::INT_128) {
        uint64_t la, lb;
        int64_t ha, hb;
        load_int128_safe(col.data, row_a, la, ha);
        load_int128_safe(col.data, row_b, lb, hb);
        if (ha > hb)
          cmp = 1;
        else if (ha < hb)
          cmp = -1;
        else
          cmp = (la > lb) - (la < lb);
      } else if (type == KernelColType::STRING) {
        uint64_t off_a   = col.offsets[row_a];
        uint64_t len_a   = col.offsets[row_a + 1] - off_a;
        uint8_t* ptr_a   = col.data + off_a;
        uint64_t off_b   = col.offsets[row_b];
        uint64_t len_b   = col.offsets[row_b + 1] - off_b;
        uint8_t* ptr_b   = col.data + off_b;
        uint64_t min_len = min(len_a, len_b);
        for (uint64_t k = 0; k < min_len; ++k) {
          if (ptr_a[k] != ptr_b[k]) {
            cmp = (ptr_a[k] > ptr_b[k]) ? 1 : -1;
            break;
          }
        }
        if (cmp == 0) cmp = (len_a > len_b) - (len_a < len_b);
      }
    }
    if (cmp != 0) return (col.is_asc == 1) ? cmp : -cmp;
  }
  return 0;
}

// Functor for CUB Sort (Generalized for any single column)
struct CustomExactSortComparator {
  DeviceKeyColumn* d_cols;
  int num_cols;
  bool is_asc;

  __host__ __device__ CustomExactSortComparator(DeviceKeyColumn* _d_cols,
                                                int _num_cols,
                                                bool _is_asc)
    : d_cols(_d_cols), num_cols(_num_cols), is_asc(_is_asc)
  {
  }

  __device__ bool operator()(const str_top_n_record_type& lhs, const str_top_n_record_type& rhs)
  {
    // First check Prefix (Radix Key)
    if (lhs.key_prefix != rhs.key_prefix) {
      return is_asc ? (lhs.key_prefix < rhs.key_prefix) : (lhs.key_prefix > rhs.key_prefix);
    }
    int cmp = compare_rows_multi_col(lhs.row_id, rhs.row_id, d_cols, num_cols);
    return cmp < 0;
  }
};

// =================================================================================================
// 3. Kernels: Engine A (Heap Based - Small Limit / Multi-Col)
// =================================================================================================

// Helper: Insert into local heap
__device__ void insert_into_heap(uint32_t row_id,
                                 uint64_t pk,
                                 uint32_t* top_ids,
                                 uint64_t* top_pks,
                                 uint32_t& current_k,
                                 uint32_t limit,
                                 bool asc,
                                 DeviceKeyColumn* cols,
                                 int num_cols)
{
  if (current_k < limit) {
    int pos = current_k;
    current_k++;
    while (pos > 0) {
      bool swap        = false;
      uint64_t prev_pk = top_pks[pos - 1];
      if (pk != prev_pk) {
        if (asc) {
          if (pk < prev_pk) swap = true;
        } else {
          if (pk > prev_pk) swap = true;
        }
      } else {
        if (compare_rows_multi_col(row_id, top_ids[pos - 1], cols, num_cols) < 0) swap = true;
      }
      if (swap) {
        top_pks[pos] = top_pks[pos - 1];
        top_ids[pos] = top_ids[pos - 1];
        pos--;
      } else
        break;
    }
    top_pks[pos] = pk;
    top_ids[pos] = row_id;
    return;
  }
  uint64_t worst_pk = top_pks[limit - 1];
  bool better       = false;
  if (pk != worst_pk) {
    if (asc) {
      if (pk < worst_pk) better = true;
    } else {
      if (pk > worst_pk) better = true;
    }
  } else {
    if (compare_rows_multi_col(row_id, top_ids[limit - 1], cols, num_cols) < 0) better = true;
  }
  if (better) {
    int pos = limit - 1;
    while (pos > 0) {
      bool swap        = false;
      uint64_t prev_pk = top_pks[pos - 1];
      if (pk != prev_pk) {
        if (asc) {
          if (pk < prev_pk) swap = true;
        } else {
          if (pk > prev_pk) swap = true;
        }
      } else {
        if (compare_rows_multi_col(row_id, top_ids[pos - 1], cols, num_cols) < 0) swap = true;
      }
      if (swap) {
        top_pks[pos] = top_pks[pos - 1];
        top_ids[pos] = top_ids[pos - 1];
        pos--;
      } else
        break;
    }
    top_pks[pos] = pk;
    top_ids[pos] = row_id;
  }
}

__global__ void per_thread_multi_col_top_k_kernel(DeviceKeyColumn* device_cols,
                                                  int num_cols,
                                                  uint32_t num_records,
                                                  uint32_t limit,
                                                  str_top_n_record_type* output_candidates)
{
  if (num_records == 0) return;
  uint32_t top_ids[MAX_THREAD_TOP_K];
  uint64_t top_pks[MAX_THREAD_TOP_K];
  uint32_t current_k = 0;
  bool asc           = (device_cols[0].is_asc == 1);
  uint64_t sentinel  = asc ? 0xFFFFFFFFFFFFFFFFULL : 0;
  uint32_t tid       = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t stride    = blockDim.x * gridDim.x;
  for (uint32_t row_id = tid; row_id < num_records; row_id += stride) {
    uint64_t curr_pk = load_primary_key_as_u64(device_cols[0], row_id);
    insert_into_heap(
      row_id, curr_pk, top_ids, top_pks, current_k, limit, asc, device_cols, num_cols);
  }
  uint32_t out_offset = tid * limit;
  for (int i = 0; i < current_k; ++i)
    output_candidates[out_offset + i] = str_top_n_record_type(top_ids[i], top_pks[i]);
  for (int i = current_k; i < limit; ++i)
    output_candidates[out_offset + i] = str_top_n_record_type(0xFFFFFFFF, sentinel);
}

__global__ void per_block_multi_col_top_k_kernel(str_top_n_record_type* input_candidates,
                                                 uint32_t num_thread_outputs,
                                                 DeviceKeyColumn* device_cols,
                                                 int num_cols,
                                                 uint32_t limit,
                                                 str_top_n_record_type* block_output,
                                                 uint32_t input_threads_per_block)
{
  if (threadIdx.x != 0) return;
  uint32_t top_ids[MAX_THREAD_TOP_K];
  uint64_t top_pks[MAX_THREAD_TOP_K];
  uint32_t current_k             = 0;
  bool asc                       = (device_cols[0].is_asc == 1);
  uint64_t sentinel              = asc ? 0xFFFFFFFFFFFFFFFFULL : 0;
  uint32_t items_per_thread      = limit;
  uint32_t items_per_block_input = input_threads_per_block * items_per_thread;
  uint32_t start_idx             = blockIdx.x * items_per_block_input;
  uint32_t end_idx               = min(start_idx + items_per_block_input, num_thread_outputs);
  for (uint32_t i = start_idx; i < end_idx; ++i) {
    uint32_t row_id = input_candidates[i].row_id;
    if (row_id == 0xFFFFFFFF) continue;
    uint64_t pk = input_candidates[i].key_prefix;
    insert_into_heap(row_id, pk, top_ids, top_pks, current_k, limit, asc, device_cols, num_cols);
  }
  uint32_t out_offset = blockIdx.x * limit;
  for (int i = 0; i < current_k; ++i)
    block_output[out_offset + i] = str_top_n_record_type(top_ids[i], top_pks[i]);
  for (int i = current_k; i < limit; ++i)
    block_output[out_offset + i] = str_top_n_record_type(0xFFFFFFFF, sentinel);
}

__global__ void global_final_multi_col_top_k_kernel(str_top_n_record_type* input_candidates,
                                                    uint32_t num_candidates,
                                                    DeviceKeyColumn* device_cols,
                                                    int num_cols,
                                                    uint32_t limit,
                                                    str_top_n_record_type* final_records)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t top_ids[MAX_THREAD_TOP_K];
  uint64_t top_pks[MAX_THREAD_TOP_K];
  uint32_t current_k = 0;
  bool asc           = (device_cols[0].is_asc == 1);
  uint64_t sentinel  = asc ? 0xFFFFFFFFFFFFFFFFULL : 0;
  for (uint32_t i = 0; i < num_candidates; ++i) {
    uint32_t row_id = input_candidates[i].row_id;
    if (row_id == 0xFFFFFFFF) continue;
    uint64_t pk = input_candidates[i].key_prefix;
    insert_into_heap(row_id, pk, top_ids, top_pks, current_k, limit, asc, device_cols, num_cols);
  }
  for (int i = 0; i < current_k; ++i)
    final_records[i] = str_top_n_record_type(top_ids[i], top_pks[i]);
  for (int i = current_k; i < limit; ++i)
    final_records[i] = str_top_n_record_type(0xFFFFFFFF, sentinel);
}

// =================================================================================================
// 4. Kernels: Engine B (Radix Sort - Single Col / Large Limit)
// =================================================================================================

__global__ void create_radix_keys(uint64_t* key_prefixes,
                                  uint32_t* row_ids,
                                  DeviceKeyColumn* d_cols,
                                  uint32_t num_records)
{
  uint32_t row_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (row_id < num_records) {
    key_prefixes[row_id] = load_primary_key_as_u64(d_cols[0], row_id);
    row_ids[row_id]      = row_id;
  }
}

__global__ void find_candidate_cutoff(uint64_t* sorted_prefixes,
                                      uint32_t num_records,
                                      uint32_t limit,
                                      uint32_t* cutoff_index)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (limit >= num_records) {
      *cutoff_index = num_records;
      return;
    }
    uint64_t boundary_val = sorted_prefixes[limit - 1];
    uint32_t idx          = limit;
    while (idx < num_records && sorted_prefixes[idx] == boundary_val) {
      idx++;
    }
    *cutoff_index = idx;
  }
}

__global__ void combine_to_records(str_top_n_record_type* records,
                                   uint64_t* sorted_prefixes,
                                   uint32_t* sorted_row_ids,
                                   uint32_t num_records)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_records) {
    records[idx].key_prefix = sorted_prefixes[idx];
    records[idx].row_id     = sorted_row_ids[idx];
  }
}

// =================================================================================================
// 5. Materialize Kernels (Common)
// =================================================================================================

__global__ void gather_validity_mask_kernel(str_top_n_record_type* records,
                                            const cudf::bitmask_type* src_mask,
                                            cudf::bitmask_type* dst_mask,
                                            uint32_t num_records)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_records) return;
  uint32_t row_id = records[idx].row_id;
  if (row_id == 0xFFFFFFFF) return;
  bool is_valid = true;
  if (src_mask != nullptr) {
    uint32_t word = src_mask[row_id / 32];
    is_valid      = (word >> (row_id % 32)) & 1;
  }
  if (is_valid) atomicOr(&dst_mask[idx / 32], (1u << (idx % 32)));
}

__global__ void gather_fixed_width_128_kernel(str_top_n_record_type* records,
                                              uint8_t* src_data,
                                              uint8_t* dst_data,
                                              uint32_t num_records)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_records) {
    uint32_t row_id = records[idx].row_id;
    if (row_id == 0xFFFFFFFF) return;
    size_t src_offset = static_cast<size_t>(row_id) * 16;
    size_t dst_offset = static_cast<size_t>(idx) * 16;
    for (int i = 0; i < 16; ++i)
      dst_data[dst_offset + i] = src_data[src_offset + i];
  }
}

template <typename T>
__global__ void gather_fixed_width_kernel(str_top_n_record_type* records,
                                          uint8_t* src_data,
                                          uint8_t* dst_data,
                                          uint32_t num_records)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_records) {
    uint32_t row_id = records[idx].row_id;
    if (row_id == 0xFFFFFFFF) return;
    T val        = load_unaligned<T>(src_data + static_cast<size_t>(row_id) * sizeof(T));
    T* dst_ptr   = reinterpret_cast<T*>(dst_data);
    dst_ptr[idx] = val;
  }
}

__global__ void materialize_determine_lengths(str_top_n_record_type* ordered_records,
                                              uint64_t* src_col_offsets,
                                              uint64_t* result_lengths,
                                              uint64_t num_records)
{
  uint32_t curr_record = threadIdx.x + blockIdx.x * blockDim.x;
  if (curr_record < num_records) {
    uint32_t row_id = ordered_records[curr_record].row_id;
    if (row_id == 0xFFFFFFFF) {
      result_lengths[curr_record] = 0;
      return;
    }
    result_lengths[curr_record] = src_col_offsets[row_id + 1] - src_col_offsets[row_id];
  } else if (curr_record == num_records)
    result_lengths[curr_record] = 0;
}

__global__ void materialize_copy_string(str_top_n_record_type* ordered_records,
                                        uint8_t* src_chars,
                                        uint64_t* src_offsets,
                                        uint8_t* dst_chars,
                                        uint64_t* dst_offsets,
                                        uint64_t num_records)
{
  uint32_t curr_record = threadIdx.x + blockIdx.x * blockDim.x;
  if (curr_record < num_records) {
    uint32_t row_id = ordered_records[curr_record].row_id;
    if (row_id == 0xFFFFFFFF) return;
    uint64_t src_start_offset    = src_offsets[row_id];
    const uint64_t record_length = src_offsets[row_id + 1] - src_start_offset;
    uint8_t* read_ptr            = src_chars + src_start_offset;
    uint8_t* write_ptr           = dst_chars + dst_offsets[curr_record];
#pragma unroll
    for (uint64_t i = 0; i < record_length; i++)
      write_ptr[i] = read_ptr[i];
  }
}

// =================================================================================================
// 6. Host Logic
// =================================================================================================

KernelColType MapSiriusTypeToKernelType(GPUColumnTypeId type_id)
{
  switch (type_id) {
    case GPUColumnTypeId::INT64:
    case GPUColumnTypeId::TIMESTAMP_SEC:
    case GPUColumnTypeId::TIMESTAMP_MS:
    case GPUColumnTypeId::TIMESTAMP_US:
    case GPUColumnTypeId::TIMESTAMP_NS: return KernelColType::INT_64;
    case GPUColumnTypeId::INT32:
    case GPUColumnTypeId::DATE:
    case GPUColumnTypeId::BOOLEAN: return KernelColType::INT_32;
    case GPUColumnTypeId::INT16: return KernelColType::INT_16;
    case GPUColumnTypeId::FLOAT64: return KernelColType::DOUBLE;
    case GPUColumnTypeId::FLOAT32: return KernelColType::FLOAT_32;
    case GPUColumnTypeId::VARCHAR: return KernelColType::STRING;
    case GPUColumnTypeId::INT128:
    case GPUColumnTypeId::DECIMAL: return KernelColType::INT_128;
    default: return KernelColType::UNKNOWN;
  }
}

void PrepareDeviceColumns(vector<shared_ptr<GPUColumn>>& keys,
                          OrderByType* order_by_type,
                          idx_t num_keys,
                          DeviceKeyColumn* d_cols_ptr,
                          GPUBufferManager* gpuBufferManager)
{
  std::vector<DeviceKeyColumn> h_cols(num_keys);
  for (int i = 0; i < num_keys; ++i) {
    h_cols[i].type = (int)MapSiriusTypeToKernelType(keys[i]->data_wrapper.type.id());
    if (h_cols[i].type == (int)KernelColType::UNKNOWN) throw std::runtime_error("Unsupported type");
    h_cols[i].data      = keys[i]->data_wrapper.data;
    h_cols[i].offsets   = keys[i]->data_wrapper.offset;
    h_cols[i].null_mask = reinterpret_cast<uint32_t*>(keys[i]->data_wrapper.validity_mask);
    h_cols[i].is_asc    = (order_by_type[i] == OrderByType::ASCENDING) ? 1 : 0;
  }
  cudaMemcpy(d_cols_ptr, h_cols.data(), num_keys * sizeof(DeviceKeyColumn), cudaMemcpyHostToDevice);
}

// Materialization Helper
void MaterializeResults(vector<shared_ptr<GPUColumn>>& projection,
                        str_top_n_record_type* d_records,
                        uint32_t final_count,
                        GPUBufferManager* gpuBufferManager,
                        idx_t num_projections)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  uint32_t gather_block_size = 256;
  uint32_t gather_grid_size  = (final_count + gather_block_size - 1) / gather_block_size;
  if (gather_grid_size == 0) gather_grid_size = 1;

  for (int col_idx = 0; col_idx < num_projections; ++col_idx) {
    DataWrapper& src_wrapper  = projection[col_idx]->data_wrapper;
    KernelColType kernel_type = MapSiriusTypeToKernelType(src_wrapper.type.id());

    size_t mask_bytes              = getMaskBytesSize(final_count);
    cudf::bitmask_type* d_new_mask = reinterpret_cast<cudf::bitmask_type*>(
      gpuBufferManager->customCudaMalloc<uint8_t>(mask_bytes, 0, 0));
    cudaMemset(d_new_mask, 0, mask_bytes);
    gather_validity_mask_kernel<<<gather_grid_size, gather_block_size>>>(
      d_records, src_wrapper.validity_mask, d_new_mask, final_count);

    if (kernel_type == KernelColType::STRING) {
      uint64_t* d_new_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(final_count + 1, 0, 0);
      materialize_determine_lengths<<<gather_grid_size, gather_block_size>>>(
        d_records, src_wrapper.offset, d_new_offsets, final_count);
      void* d_temp      = nullptr;
      size_t temp_bytes = 0;
      cub::DeviceScan::ExclusiveSum(
        d_temp, temp_bytes, d_new_offsets, d_new_offsets, final_count + 1);
      d_temp = gpuBufferManager->customCudaMalloc<uint8_t>(temp_bytes, 0, 0);
      cub::DeviceScan::ExclusiveSum(
        d_temp, temp_bytes, d_new_offsets, d_new_offsets, final_count + 1);
      gpuBufferManager->customCudaFree(static_cast<uint8_t*>(d_temp), 0);
      uint64_t total_bytes;
      cudaMemcpy(
        &total_bytes, d_new_offsets + final_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
      uint8_t* d_new_chars = gpuBufferManager->customCudaMalloc<uint8_t>(total_bytes, 0, 0);
      materialize_copy_string<<<gather_grid_size, gather_block_size>>>(
        d_records, src_wrapper.data, src_wrapper.offset, d_new_chars, d_new_offsets, final_count);
      projection[col_idx] = make_shared_ptr<GPUColumn>(
        final_count, src_wrapper.type, d_new_chars, d_new_offsets, total_bytes, true, d_new_mask);
    } else {
      size_t type_size = src_wrapper.getColumnTypeSize();
      if (type_size == 0) {
        switch (src_wrapper.type.id()) {
          case GPUColumnTypeId::INT128:
          case GPUColumnTypeId::DECIMAL: type_size = 16; break;
          case GPUColumnTypeId::INT64:
          case GPUColumnTypeId::FLOAT64:
          case GPUColumnTypeId::TIMESTAMP_SEC:
          case GPUColumnTypeId::TIMESTAMP_MS:
          case GPUColumnTypeId::TIMESTAMP_US:
          case GPUColumnTypeId::TIMESTAMP_NS: type_size = 8; break;
          case GPUColumnTypeId::INT32:
          case GPUColumnTypeId::FLOAT32:
          case GPUColumnTypeId::DATE: type_size = 4; break;
          case GPUColumnTypeId::INT16: type_size = 2; break;
          case GPUColumnTypeId::BOOLEAN: type_size = 1; break;
          default: type_size = 0;
        }
      }
      uint8_t* d_new_data =
        gpuBufferManager->customCudaMalloc<uint8_t>(final_count * type_size, 0, 0);
      if (type_size == 16)
        gather_fixed_width_128_kernel<<<gather_grid_size, gather_block_size>>>(
          d_records, src_wrapper.data, d_new_data, final_count);
      else if (type_size == 8)
        gather_fixed_width_kernel<uint64_t><<<gather_grid_size, gather_block_size>>>(
          d_records, src_wrapper.data, d_new_data, final_count);
      else if (type_size == 4)
        gather_fixed_width_kernel<uint32_t><<<gather_grid_size, gather_block_size>>>(
          d_records, src_wrapper.data, d_new_data, final_count);
      else if (type_size == 2)
        gather_fixed_width_kernel<uint16_t><<<gather_grid_size, gather_block_size>>>(
          d_records, src_wrapper.data, d_new_data, final_count);
      else if (type_size == 1)
        gather_fixed_width_kernel<uint8_t><<<gather_grid_size, gather_block_size>>>(
          d_records, src_wrapper.data, d_new_data, final_count);
      projection[col_idx] =
        make_shared_ptr<GPUColumn>(final_count, src_wrapper.type, d_new_data, d_new_mask);
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  SIRIUS_LOG_DEBUG("MULTI-COL TOP N Result Write Time : {} ms", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Engine A: Heap Based
void CustomMultiColumnTopN(vector<shared_ptr<GPUColumn>>& keys,
                           vector<shared_ptr<GPUColumn>>& projection,
                           idx_t num_keys,
                           idx_t num_projections,
                           OrderByType* order_by_type,
                           idx_t num_results)
{
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  const uint32_t num_records = keys[0]->column_length;

  DeviceKeyColumn* d_cols = reinterpret_cast<DeviceKeyColumn*>(
    gpuBufferManager->customCudaMalloc<uint8_t>(num_keys * sizeof(DeviceKeyColumn), 0, 0));
  PrepareDeviceColumns(keys, order_by_type, num_keys, d_cols, gpuBufferManager);
  CUDA_CHECK_AND_SYNC("Memcpy Cols");

  // P1: Thread Top K
  uint32_t num_threads = 256;
  uint32_t num_blocks  = min((num_records + num_threads - 1) / num_threads, (uint32_t)256);
  if (num_blocks == 0) num_blocks = 1;
  uint32_t total_threads     = num_threads * num_blocks;
  uint64_t num_candidates_p1 = total_threads * num_results;
  str_top_n_record_type* d_candidates_p1 =
    reinterpret_cast<str_top_n_record_type*>(gpuBufferManager->customCudaMalloc<uint8_t>(
      num_candidates_p1 * sizeof(str_top_n_record_type), 0, 0));
  per_thread_multi_col_top_k_kernel<<<num_blocks, num_threads>>>(
    d_cols, num_keys, num_records, num_results, d_candidates_p1);
  CUDA_CHECK_AND_SYNC("TopK Phase 1");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  SIRIUS_LOG_DEBUG("MULTI-COL TOP N Per-Thread Filter Time : {} ms", elapsedTime);
  cudaEventRecord(start, 0);

  // P2: Block Merge
  uint32_t num_block_candidates = num_blocks * num_results;
  str_top_n_record_type* d_candidates_p2 =
    reinterpret_cast<str_top_n_record_type*>(gpuBufferManager->customCudaMalloc<uint8_t>(
      num_block_candidates * sizeof(str_top_n_record_type), 0, 0));
  per_block_multi_col_top_k_kernel<<<num_blocks, 1>>>(d_candidates_p1,
                                                      num_candidates_p1,
                                                      d_cols,
                                                      num_keys,
                                                      num_results,
                                                      d_candidates_p2,
                                                      num_threads);
  CUDA_CHECK_AND_SYNC("TopK Phase 2");

  // P3: Global Merge
  uint32_t final_count = (uint32_t)std::min((uint64_t)num_results, (uint64_t)num_records);
  str_top_n_record_type* d_records = reinterpret_cast<str_top_n_record_type*>(
    gpuBufferManager->customCudaMalloc<uint8_t>(final_count * sizeof(str_top_n_record_type), 0, 0));
  global_final_multi_col_top_k_kernel<<<1, 1>>>(
    d_candidates_p2, num_block_candidates, d_cols, num_keys, final_count, d_records);
  CUDA_CHECK_AND_SYNC("TopK Phase 3");

  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_candidates_p1), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_candidates_p2), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_cols), 0);

  MaterializeResults(projection, d_records, final_count, gpuBufferManager, num_projections);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_records), 0);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Engine B: Radix Sort Based (Supports Any Single Column)
void CustomSingleColumnRadixTopN(vector<shared_ptr<GPUColumn>>& keys,
                                 vector<shared_ptr<GPUColumn>>& projection,
                                 idx_t num_keys,
                                 idx_t num_projections,
                                 OrderByType* order_by_type,
                                 idx_t num_results)
{
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  const uint32_t num_records = keys[0]->column_length;

  // 1. Prepare Cols (Reuse logic)
  DeviceKeyColumn* d_cols = reinterpret_cast<DeviceKeyColumn*>(
    gpuBufferManager->customCudaMalloc<uint8_t>(num_keys * sizeof(DeviceKeyColumn), 0, 0));
  PrepareDeviceColumns(keys, order_by_type, num_keys, d_cols, gpuBufferManager);

  // 2. Generate Radix Keys
  uint32_t num_create_workers = (num_records + BLOCK_THREADS - 1) / BLOCK_THREADS;
  uint64_t* d_key_prefixes    = gpuBufferManager->customCudaMalloc<uint64_t>(num_records, 0, 0);
  uint32_t* d_row_ids         = gpuBufferManager->customCudaMalloc<uint32_t>(num_records, 0, 0);
  create_radix_keys<<<num_create_workers, BLOCK_THREADS>>>(
    d_key_prefixes, d_row_ids, d_cols, num_records);
  CUDA_CHECK_AND_SYNC("Radix Key Gen");

  // 3. Radix Sort
  uint64_t* d_sorted_prefixes = gpuBufferManager->customCudaMalloc<uint64_t>(num_records, 0, 0);
  uint32_t* d_sorted_row_ids  = gpuBufferManager->customCudaMalloc<uint32_t>(num_records, 0, 0);
  void* d_temp                = nullptr;
  size_t temp_bytes           = 0;

  if (order_by_type[0] == OrderByType::ASCENDING) {
    cub::DeviceRadixSort::SortPairs(d_temp,
                                    temp_bytes,
                                    d_key_prefixes,
                                    d_sorted_prefixes,
                                    d_row_ids,
                                    d_sorted_row_ids,
                                    num_records);
    d_temp = gpuBufferManager->customCudaMalloc<uint8_t>(temp_bytes, 0, 0);
    cub::DeviceRadixSort::SortPairs(d_temp,
                                    temp_bytes,
                                    d_key_prefixes,
                                    d_sorted_prefixes,
                                    d_row_ids,
                                    d_sorted_row_ids,
                                    num_records);
  } else {
    cub::DeviceRadixSort::SortPairsDescending(d_temp,
                                              temp_bytes,
                                              d_key_prefixes,
                                              d_sorted_prefixes,
                                              d_row_ids,
                                              d_sorted_row_ids,
                                              num_records);
    d_temp = gpuBufferManager->customCudaMalloc<uint8_t>(temp_bytes, 0, 0);
    cub::DeviceRadixSort::SortPairsDescending(d_temp,
                                              temp_bytes,
                                              d_key_prefixes,
                                              d_sorted_prefixes,
                                              d_row_ids,
                                              d_sorted_row_ids,
                                              num_records);
  }
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_key_prefixes), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_row_ids), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp), 0);
  CUDA_CHECK_AND_SYNC("Radix Sort");

  // 4. Find Cutoff (Optimization: Only sort valid candidates)
  uint32_t* d_cutoff_idx = gpuBufferManager->customCudaMalloc<uint32_t>(1, 0, 0);
  find_candidate_cutoff<<<1, 1>>>(d_sorted_prefixes, num_records, num_results, d_cutoff_idx);
  uint32_t cutoff_idx = 0;
  cudaMemcpy(&cutoff_idx, d_cutoff_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_cutoff_idx), 0);

  // 5. Combine to Records
  str_top_n_record_type* d_records = reinterpret_cast<str_top_n_record_type*>(
    gpuBufferManager->customCudaMalloc<uint8_t>(cutoff_idx * sizeof(str_top_n_record_type), 0, 0));
  uint32_t combine_workers = (cutoff_idx + BLOCK_THREADS - 1) / BLOCK_THREADS;
  combine_to_records<<<combine_workers, BLOCK_THREADS>>>(
    d_records, d_sorted_prefixes, d_sorted_row_ids, cutoff_idx);

  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_sorted_prefixes), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_sorted_row_ids), 0);

  // 6. Exact Re-Sort (Handling collisions)
  CustomExactSortComparator comparator(
    d_cols, num_keys, (order_by_type[0] == OrderByType::ASCENDING));
  d_temp     = nullptr;
  temp_bytes = 0;
  cub::DeviceMergeSort::SortKeys(d_temp, temp_bytes, d_records, cutoff_idx, comparator);
  d_temp = gpuBufferManager->customCudaMalloc<uint8_t>(temp_bytes, 0, 0);
  cub::DeviceMergeSort::SortKeys(d_temp, temp_bytes, d_records, cutoff_idx, comparator);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_cols), 0);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  SIRIUS_LOG_DEBUG("RADIX TOP N Total Time : {} ms", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // 7. Materialize
  uint32_t final_count = (uint32_t)std::min((uint64_t)num_results, (uint64_t)cutoff_idx);
  MaterializeResults(projection, d_records, final_count, gpuBufferManager, num_projections);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_records), 0);
}

// Router
void cudf_orderby(vector<shared_ptr<GPUColumn>>& keys,
                  vector<shared_ptr<GPUColumn>>& projection,
                  idx_t num_keys,
                  idx_t num_projections,
                  OrderByType* order_by_type,
                  idx_t num_results)
{
  // 1. Try using custom optimization engine
  if (Config::USE_CUSTOM_TOP_N && num_results > 0) {
    bool type_supported = true;
    // Check if all column types are supported
    for (size_t col = 0; col < num_keys; col++) {
      if (MapSiriusTypeToKernelType(keys[col]->data_wrapper.type.id()) == KernelColType::UNKNOWN) {
        type_supported = false;
        break;
      }
    }
    for (size_t col = 0; col < num_projections; col++) {
      if (MapSiriusTypeToKernelType(projection[col]->data_wrapper.type.id()) ==
          KernelColType::UNKNOWN) {
        type_supported = false;
        break;
      }
    }

    if (type_supported) {
      // [Routing Strategy Optimization]

      // Strategy A: For small limits (<= 32), prefer Heap Sort (Engine A).
      // Rationale: Even for single column, Heap Sort only needs one-pass scan, lower latency than
      // Radix Sort (Multi-Pass). This significantly speeds up queries like Q25, Q11 with LIMIT 10.
      if (num_results <= MAX_THREAD_TOP_K) {
        SIRIUS_LOG_DEBUG("Using Heap Sort Engine (Small Limit)");
        CustomMultiColumnTopN(
          keys, projection, num_keys, num_projections, order_by_type, num_results);
        return;
      }

      // Strategy B: For large limit but single column, use Radix Sort (Engine B).
      // Rationale: As limit grows, heap maintenance cost increases, Radix Sort's high throughput
      // advantage shows.
      else if (num_keys == 1) {
        SIRIUS_LOG_DEBUG("Using Radix Sort Engine (Single Column Large Limit)");
        CustomSingleColumnRadixTopN(
          keys, projection, num_keys, num_projections, order_by_type, num_results);
        return;
      }
    }
  }

  // 2. Handle empty data
  if (keys[0]->column_length == 0) {
    for (idx_t col = 0; col < num_projections; col++) {
      bool old_unique = projection[col]->is_unique;
      if (projection[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
        projection[col] = make_shared_ptr<GPUColumn>(0,
                                                     projection[col]->data_wrapper.type,
                                                     projection[col]->data_wrapper.data,
                                                     projection[col]->data_wrapper.offset,
                                                     0,
                                                     true,
                                                     nullptr);
      } else {
        projection[col] = make_shared_ptr<GPUColumn>(
          0, projection[col]->data_wrapper.type, projection[col]->data_wrapper.data, nullptr);
      }
      projection[col]->is_unique = old_unique;
    }
    return;
  }
  SIRIUS_LOG_DEBUG("Cudf order using custom top n of {} has val {}", num_results, false);

  // 3. Fallback: libcudf full sorting
  // Applicable for: multi-column with Limit > 32, or unsupported types, or complex Offset handling
  // (though currently logic doesn't pass Offset)
  SIRIUS_LOG_DEBUG("CUDF Order By (Fallback)");
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  cudf::set_current_device_resource(gpuBufferManager->mr);

  std::vector<cudf::column_view> columns_cudf;
  for (int key = 0; key < num_keys; key++)
    columns_cudf.push_back(keys[key]->convertToCudfColumn());

  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> null_orders;
  for (int i = 0; i < num_keys; i++) {
    if (order_by_type[i] == OrderByType::ASCENDING) {
      orders.push_back(cudf::order::ASCENDING);
      null_orders.push_back(cudf::null_order::AFTER);
    } else {
      orders.push_back(cudf::order::DESCENDING);
      null_orders.push_back(cudf::null_order::BEFORE);
    }
  }

  auto keys_table        = cudf::table_view(columns_cudf);
  auto sorted_order      = cudf::sorted_order(keys_table, orders, null_orders);
  auto sorted_order_view = sorted_order->view();

  std::vector<cudf::column_view> projection_cudf;
  for (int col = 0; col < num_projections; col++)
    projection_cudf.push_back(projection[col]->convertToCudfColumn());

  auto projection_table = cudf::table_view(projection_cudf);
  auto gathered_table   = cudf::gather(projection_table, sorted_order_view);

  for (int col = 0; col < num_projections; col++) {
    projection[col]->setFromCudfColumn(
      gathered_table->get_column(col), projection[col]->is_unique, nullptr, 0, gpuBufferManager);
  }
}

}  // namespace duckdb

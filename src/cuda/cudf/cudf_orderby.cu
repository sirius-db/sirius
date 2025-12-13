/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include "config.hpp"
 #include "cudf_utils.hpp"
 #include "../operator/cuda_helper.cuh"
 #include "gpu_physical_order.hpp"
 #include "gpu_buffer_manager.hpp"
 #include "log/logging.hpp"
 #include <cub/cub.cuh>
 #include <stdio.h>
 
 namespace duckdb {
 
 // =================================================================================================
 // Debug Macros (only effective in Debug mode or when explicitly enabled)
 // =================================================================================================
 
 #define ENABLE_CUDA_DEBUG 1 
 
 #if ENABLE_CUDA_DEBUG
     #define CUDA_CHECK_AND_SYNC(msg) \
     { \
         cudaDeviceSynchronize(); \
         cudaError_t err = cudaGetLastError(); \
         if (err != cudaSuccess) { \
             SIRIUS_LOG_DEBUG("CUDA Error at [{}]: {} - {}", msg, cudaGetErrorName(err), cudaGetErrorString(err)); \
             /* Don't throw, let it fall back to CPU */ \
             return; \
         } \
     }
 #else
     #define CUDA_CHECK_AND_SYNC(msg) {}
 #endif
 
 // =================================================================================================
 // Struct Definitions
 // =================================================================================================
 
 #define MAX_THREAD_TOP_K 32
 
 struct str_top_n_record_type { 
     uint32_t row_id;
     uint64_t key_prefix; 
 
     __host__ __device__ str_top_n_record_type() : row_id(0), key_prefix(0) {}
     __host__ __device__ str_top_n_record_type(uint32_t _row_id, uint64_t _key_prefix) : row_id(_row_id), key_prefix(_key_prefix) {}
 
     __device__ __forceinline__ bool operator==(const str_top_n_record_type& other) const { return key_prefix == other.key_prefix; }
     __device__ __forceinline__ bool operator<(const str_top_n_record_type& other) const { return key_prefix < other.key_prefix; }
 };
 
 enum class KernelColType : int {
     INT_64 = 0, 
     INT_32 = 1, 
     DOUBLE = 2, 
     STRING = 3, 
     INT_128 = 4, 
     UNKNOWN = 99
 };
 
 // Explicitly define struct to avoid padding inconsistencies
 struct DeviceKeyColumn {
     int type;           // enum KernelColType
     uint32_t is_asc;    // 1 = asc, 0 = desc (use uint32 instead of bool for alignment)
     uint8_t* data;
     uint64_t* offsets;  
 };
 
 // =================================================================================================
 // Device Helper Functions (Safe Memory Access)
 // =================================================================================================
 
 // Unaligned memory read helper
 // Works correctly even when address is not aligned
 template <typename T>
 __device__ __forceinline__ T load_unaligned(const void* ptr) {
     // Byte-by-byte copy for safety
     // CUDA compiler typically optimizes to byte-aligned load
     T val;
     const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
     uint8_t* dst = reinterpret_cast<uint8_t*>(&val);
     #pragma unroll
     for (int i = 0; i < sizeof(T); ++i) {
         dst[i] = src[i];
     }
     return val;
 }
 
 // Safely read high and low bits of 128-bit data
 __device__ __forceinline__ void load_int128_safe(uint8_t* base_ptr, uint32_t row_id, uint64_t& low, int64_t& high) {
     size_t offset = static_cast<size_t>(row_id) * 16;
     const uint8_t* addr = base_ptr + offset;
     
     // DuckDB hugeint_t layout: low 64 bits first, high 64 bits second (Little Endian)
     low = load_unaligned<uint64_t>(addr);
     high = load_unaligned<int64_t>(addr + 8);
 }
 
 // Primary key generation (Order Preserving)
 __device__ __forceinline__ uint64_t load_primary_key_as_u64(const DeviceKeyColumn& col, uint32_t row_id) {
     KernelColType type = static_cast<KernelColType>(col.type);
     
     if (type == KernelColType::INT_64) {
         // Use load_unaligned to prevent crashes from data pointer offset
         int64_t val = load_unaligned<int64_t>(col.data + static_cast<size_t>(row_id) * 8);
         return static_cast<uint64_t>(val) ^ 0x8000000000000000ULL;
     } 
     else if (type == KernelColType::INT_32) {
         int32_t val = load_unaligned<int32_t>(col.data + static_cast<size_t>(row_id) * 4);
         return (static_cast<uint64_t>(val) ^ 0x80000000) << 32;
     } 
     else if (type == KernelColType::DOUBLE) {
         double val = load_unaligned<double>(col.data + static_cast<size_t>(row_id) * 8);
         uint64_t bits = *reinterpret_cast<uint64_t*>(&val);
         uint64_t mask = (static_cast<int64_t>(bits) >> 63) | 0x8000000000000000ULL;
         return bits ^ mask;
     } 
     else if (type == KernelColType::INT_128) {
         uint64_t low; int64_t high;
         load_int128_safe(col.data, row_id, low, high);
         return static_cast<uint64_t>(high) ^ 0x8000000000000000ULL;
     }
     else if (type == KernelColType::STRING) {
         uint64_t start = col.offsets[row_id];
         uint64_t len = col.offsets[row_id + 1] - start;
         uint64_t prefix = 0;
         const uint8_t* ptr = col.data + start;
         uint64_t bytes = min(len, static_cast<uint64_t>(8));
         uint32_t shift = 56;
         for(int i=0; i<bytes; ++i) {
             prefix |= static_cast<uint64_t>(ptr[i]) << shift;
             shift -= 8;
         }
         return prefix;
     }
     return 0;
 }
 
 // Multi-column comparison
 __device__ int compare_rows_multi_col(uint32_t row_a, uint32_t row_b, const DeviceKeyColumn* columns, int num_columns) {
     for (int i = 0; i < num_columns; ++i) {
         const DeviceKeyColumn& col = columns[i];
         KernelColType type = static_cast<KernelColType>(col.type);
         int cmp = 0; 
 
         if (type == KernelColType::INT_64) {
             int64_t val_a = load_unaligned<int64_t>(col.data + static_cast<size_t>(row_a) * 8);
             int64_t val_b = load_unaligned<int64_t>(col.data + static_cast<size_t>(row_b) * 8);
             cmp = (val_a > val_b) - (val_a < val_b);
         } else if (type == KernelColType::INT_32) {
             int32_t val_a = load_unaligned<int32_t>(col.data + static_cast<size_t>(row_a) * 4);
             int32_t val_b = load_unaligned<int32_t>(col.data + static_cast<size_t>(row_b) * 4);
             cmp = (val_a > val_b) - (val_a < val_b);
         } else if (type == KernelColType::DOUBLE) {
             double val_a = load_unaligned<double>(col.data + static_cast<size_t>(row_a) * 8);
             double val_b = load_unaligned<double>(col.data + static_cast<size_t>(row_b) * 8);
             cmp = (val_a > val_b) - (val_a < val_b);
         } else if (type == KernelColType::INT_128) {
             uint64_t low_a, low_b; int64_t high_a, high_b;
             load_int128_safe(col.data, row_a, low_a, high_a);
             load_int128_safe(col.data, row_b, low_b, high_b);
             if (high_a > high_b) cmp = 1;
             else if (high_a < high_b) cmp = -1;
             else {
                 if (low_a > low_b) cmp = 1;
                 else if (low_a < low_b) cmp = -1;
             }
         } else if (type == KernelColType::STRING) {
             uint64_t off_a = col.offsets[row_a];
             uint64_t len_a = col.offsets[row_a + 1] - off_a;
             uint8_t* ptr_a = col.data + off_a;
             
             uint64_t off_b = col.offsets[row_b];
             uint64_t len_b = col.offsets[row_b + 1] - off_b;
             uint8_t* ptr_b = col.data + off_b;
 
             uint64_t min_len = min(len_a, len_b);
             bool diff_found = false;
             for(uint64_t k=0; k < min_len; ++k) {
                 if (ptr_a[k] != ptr_b[k]) {
                     cmp = (ptr_a[k] > ptr_b[k]) ? 1 : -1;
                     diff_found = true;
                     break;
                 }
             }
             if (!diff_found) {
                 cmp = (len_a > len_b) - (len_a < len_b);
             }
         } 
 
         if (cmp != 0) {
             return (col.is_asc == 1) ? cmp : -cmp;
         }
     }
     return 0;
 }
 
 // =================================================================================================
 // Kernels
 // =================================================================================================
 
 __global__ void per_thread_multi_col_top_k_kernel(
     DeviceKeyColumn* device_cols,
     int num_cols,
     uint32_t num_records,
     uint32_t limit,
     str_top_n_record_type* output_candidates
 ) {
     if (num_records == 0) return;
 
     // [Debug] Only first thread prints to verify parameters are passed correctly
     // if (threadIdx.x == 0 && blockIdx.x == 0) {
     //     printf("Kernel Start: num_records=%u, limit=%u, cols=%d, first_col_type=%d\n", num_records, limit, num_cols, device_cols[0].type);
     // }
 
     uint32_t top_ids[MAX_THREAD_TOP_K];
     uint64_t top_primary_keys[MAX_THREAD_TOP_K];
     uint32_t current_k = 0;
 
     bool primary_asc = (device_cols[0].is_asc == 1);
     uint64_t sentinel = primary_asc ? 0xFFFFFFFFFFFFFFFFULL : 0;
     
     #pragma unroll
     for (int i = 0; i < MAX_THREAD_TOP_K; ++i) {
         top_primary_keys[i] = sentinel;
     }
 
     uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
     uint32_t stride = blockDim.x * gridDim.x;
 
     for (uint32_t row_id = tid; row_id < num_records; row_id += stride) {
         uint64_t curr_pk = load_primary_key_as_u64(device_cols[0], row_id);
         bool is_candidate = false;
         
         if (current_k < limit) {
             is_candidate = true;
         } else {
             uint64_t worst_pk = top_primary_keys[limit - 1];
             if (curr_pk != worst_pk) {
                 if (primary_asc) { if (curr_pk < worst_pk) is_candidate = true; }
                 else { if (curr_pk > worst_pk) is_candidate = true; }
             } else {
                 uint32_t worst_row_id = top_ids[limit - 1];
                 int cmp = compare_rows_multi_col(row_id, worst_row_id, device_cols, num_cols);
                 if (cmp < 0) is_candidate = true;
             }
         }
 
         if (is_candidate) {
             int insert_pos = (current_k < limit) ? current_k : limit - 1;
             if (current_k < limit) current_k++;
 
             while (insert_pos > 0) {
                 bool should_swap = false;
                 uint64_t prev_pk = top_primary_keys[insert_pos - 1];
                 
                 if (curr_pk != prev_pk) {
                     if (primary_asc) { if (curr_pk < prev_pk) should_swap = true; }
                     else { if (curr_pk > prev_pk) should_swap = true; }
                 } else {
                     uint32_t prev_row_id = top_ids[insert_pos - 1];
                     int cmp = compare_rows_multi_col(row_id, prev_row_id, device_cols, num_cols);
                     if (cmp < 0) should_swap = true;
                 }
 
                 if (should_swap) {
                     top_primary_keys[insert_pos] = top_primary_keys[insert_pos - 1];
                     top_ids[insert_pos] = top_ids[insert_pos - 1];
                     insert_pos--;
                 } else { break; }
             }
             top_primary_keys[insert_pos] = curr_pk;
             top_ids[insert_pos] = row_id;
         }
     }
 
     uint32_t output_offset = tid * limit;
     for (int i = 0; i < current_k; ++i) {
         output_candidates[output_offset + i] = str_top_n_record_type(top_ids[i], top_primary_keys[i]);
     }
     for (int i = current_k; i < limit; ++i) {
         output_candidates[output_offset + i] = str_top_n_record_type(0xFFFFFFFF, sentinel);
     }
 }
 
 __global__ void unpack_candidates_kernel(str_top_n_record_type* candidates, uint64_t* prefixes, uint32_t* row_ids, uint32_t num_items) {
     uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
     if (idx < num_items) {
         prefixes[idx] = candidates[idx].key_prefix;
         row_ids[idx] = candidates[idx].row_id;
     }
 }
 
 __global__ void combine_to_records(str_top_n_record_type* records, uint64_t* sorted_prefixes, uint32_t* sorted_row_ids, uint32_t num_records) {
     uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
     if (idx < num_records) {
         records[idx].key_prefix = sorted_prefixes[idx];
         records[idx].row_id = sorted_row_ids[idx];
     }
 }
 
 // 128-bit specialized Gather Kernel (safe version)
 __global__ void gather_fixed_width_128_kernel(
     str_top_n_record_type* records,
     uint8_t* src_data,
     uint8_t* dst_data,
     uint32_t num_records) 
 {
     uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
     if (idx < num_records) {
         uint32_t row_id = records[idx].row_id;
         size_t src_offset = static_cast<size_t>(row_id) * 16;
         size_t dst_offset = static_cast<size_t>(idx) * 16;
         
         // Byte-by-byte copy, completely safe
         for(int i=0; i<16; ++i) {
             dst_data[dst_offset + i] = src_data[src_offset + i];
         }
     }
 }
 
 // Generic fixed-width data copy Kernel
 template <typename T>
 __global__ void gather_fixed_width_kernel(
     str_top_n_record_type* records,
     uint8_t* src_data,
     uint8_t* dst_data,
     uint32_t num_records) 
 {
     uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
     if (idx < num_records) {
         uint32_t row_id = records[idx].row_id;
         // Safe read
         T val = load_unaligned<T>(src_data + static_cast<size_t>(row_id) * sizeof(T));
         // Write result (output buffer is usually aligned, but using safe method anyway)
         T* dst_ptr = reinterpret_cast<T*>(dst_data);
         dst_ptr[idx] = val; 
     }
 }
 
 __global__ void materialize_determine_lengths(str_top_n_record_type* ordered_records, uint64_t* src_col_offsets, uint64_t* result_lengths, uint64_t num_records) { 
     uint32_t curr_record = threadIdx.x + blockIdx.x * blockDim.x;
     if(curr_record < num_records) { 
         uint32_t row_id = ordered_records[curr_record].row_id;
         result_lengths[curr_record] = src_col_offsets[row_id + 1] - src_col_offsets[row_id];
     } else if(curr_record == num_records) { 
         result_lengths[curr_record] = 0;
     }
 }
 
 __global__ void materialize_copy_string(str_top_n_record_type* ordered_records, uint8_t* src_chars, uint64_t* src_offsets, uint8_t* dst_chars, uint64_t* dst_offsets, uint64_t num_records) { 
     uint32_t curr_record = threadIdx.x + blockIdx.x * blockDim.x;
     if(curr_record < num_records) {
         uint32_t row_id = ordered_records[curr_record].row_id;
         uint64_t src_start_offset = src_offsets[row_id];
         const uint64_t record_length = src_offsets[row_id + 1] - src_start_offset;
         uint8_t* read_ptr = src_chars + src_start_offset;
         uint8_t* write_ptr = dst_chars + dst_offsets[curr_record];
         #pragma unroll
         for(uint64_t i = 0; i < record_length; i++) { 
             write_ptr[i] = read_ptr[i];
         }
     }
 }
 
 // =================================================================================================
 // Host Logic
 // =================================================================================================
 
 KernelColType MapSiriusTypeToKernelType(GPUColumnTypeId type_id) {
     switch (type_id) {
         case GPUColumnTypeId::INT64:
         case GPUColumnTypeId::TIMESTAMP_SEC:
         case GPUColumnTypeId::TIMESTAMP_MS:
         case GPUColumnTypeId::TIMESTAMP_US:
         case GPUColumnTypeId::TIMESTAMP_NS:
             return KernelColType::INT_64;
         case GPUColumnTypeId::INT32:
         case GPUColumnTypeId::INT16:
         case GPUColumnTypeId::DATE:
         case GPUColumnTypeId::BOOLEAN:
             return KernelColType::INT_32;
         case GPUColumnTypeId::FLOAT64:
         case GPUColumnTypeId::FLOAT32:
             return KernelColType::DOUBLE;
         case GPUColumnTypeId::VARCHAR:
             return KernelColType::STRING;
         case GPUColumnTypeId::INT128:
         case GPUColumnTypeId::DECIMAL:
             return KernelColType::INT_128;
         default:
             return KernelColType::UNKNOWN;
     }
 }
 
 void CustomMultiColumnTopN(
     vector<shared_ptr<GPUColumn>>& keys, 
     vector<shared_ptr<GPUColumn>>& projection, 
     idx_t num_keys, 
     idx_t num_projections, 
     OrderByType* order_by_type, 
     idx_t num_results
 ) {
     GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
     SETUP_TIMING();
     START_TIMER();
 
     const uint32_t num_records = keys[0]->column_length;
     if (num_records == 0) return;
 
     // 1. Prepare sorting column metadata
     std::vector<DeviceKeyColumn> h_cols(num_keys);
     for(int i=0; i<num_keys; ++i) {
         auto type_id = keys[i]->data_wrapper.type.id();
         h_cols[i].type = (int)MapSiriusTypeToKernelType(type_id);
         
         if (h_cols[i].type == (int)KernelColType::UNKNOWN) throw std::runtime_error("Unsupported type");
 
         h_cols[i].data = keys[i]->data_wrapper.data;
         h_cols[i].offsets = keys[i]->data_wrapper.offset; 
         h_cols[i].is_asc = (order_by_type[i] == OrderByType::ASCENDING) ? 1 : 0;
     }
 
     DeviceKeyColumn* d_cols = reinterpret_cast<DeviceKeyColumn*>(
         gpuBufferManager->customCudaMalloc<uint8_t>(num_keys * sizeof(DeviceKeyColumn), 0, 0)
     );
     cudaMemcpy(d_cols, h_cols.data(), num_keys * sizeof(DeviceKeyColumn), cudaMemcpyHostToDevice);
     CUDA_CHECK_AND_SYNC("Memcpy Cols");
 
     // 2. Run Per-Thread Top-K Filter
     uint32_t num_threads = 256;
     uint32_t num_blocks = min((num_records + num_threads - 1) / num_threads, (uint32_t)256);
     if (num_blocks == 0) num_blocks = 1;
 
     uint32_t total_threads = num_threads * num_blocks;
     uint64_t num_candidates = total_threads * num_results;
     
     str_top_n_record_type* d_candidates = reinterpret_cast<str_top_n_record_type*>(
         gpuBufferManager->customCudaMalloc<uint8_t>(num_candidates * sizeof(str_top_n_record_type), 0, 0)
     );
 
     per_thread_multi_col_top_k_kernel<<<num_blocks, num_threads>>>(d_cols, num_keys, num_records, num_results, d_candidates);
     CUDA_CHECK_AND_SYNC("TopK Kernel");
     
     RECORD_TIMER("MULTI-COL TOP N Per-Thread Filter Time");
 
     // 3. Global candidate sorting
     uint64_t* d_cand_prefixes = gpuBufferManager->customCudaMalloc<uint64_t>(num_candidates, 0, 0);
     uint32_t* d_cand_row_ids = gpuBufferManager->customCudaMalloc<uint32_t>(num_candidates, 0, 0);
     uint64_t* d_sorted_prefixes = gpuBufferManager->customCudaMalloc<uint64_t>(num_candidates, 0, 0);
     uint32_t* d_sorted_row_ids = gpuBufferManager->customCudaMalloc<uint32_t>(num_candidates, 0, 0);
 
     uint32_t unpack_workers = (num_candidates + BLOCK_THREADS - 1) / BLOCK_THREADS;
     if (unpack_workers == 0) unpack_workers = 1;
     unpack_candidates_kernel<<<unpack_workers, BLOCK_THREADS>>>(d_candidates, d_cand_prefixes, d_cand_row_ids, num_candidates);
     CUDA_CHECK_AND_SYNC("Unpack Kernel");
 
     void* d_radix_sort_temp_storage = nullptr;
     size_t radix_sort_temp_storage_bytes = 0;
     
     if(order_by_type[0] == OrderByType::ASCENDING) { 
         cub::DeviceRadixSort::SortPairs(d_radix_sort_temp_storage, radix_sort_temp_storage_bytes, d_cand_prefixes, d_sorted_prefixes, d_cand_row_ids, d_sorted_row_ids, num_candidates);
         d_radix_sort_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(radix_sort_temp_storage_bytes, 0, 0);
         cub::DeviceRadixSort::SortPairs(d_radix_sort_temp_storage, radix_sort_temp_storage_bytes, d_cand_prefixes, d_sorted_prefixes, d_cand_row_ids, d_sorted_row_ids, num_candidates);
     } else { 
         cub::DeviceRadixSort::SortPairsDescending(d_radix_sort_temp_storage, radix_sort_temp_storage_bytes, d_cand_prefixes, d_sorted_prefixes, d_cand_row_ids, d_sorted_row_ids, num_candidates);
         d_radix_sort_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(radix_sort_temp_storage_bytes, 0, 0);
         cub::DeviceRadixSort::SortPairsDescending(d_radix_sort_temp_storage, radix_sort_temp_storage_bytes, d_cand_prefixes, d_sorted_prefixes, d_cand_row_ids, d_sorted_row_ids, num_candidates);
     }
     CUDA_CHECK_AND_SYNC("Radix Sort");
 
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_candidates), 0);
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_cand_prefixes), 0);
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_cand_row_ids), 0);
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_radix_sort_temp_storage), 0);
 
     // 4. Extract results
     uint32_t final_count = min((uint64_t)num_results, num_candidates);
     str_top_n_record_type* d_records = reinterpret_cast<str_top_n_record_type*>(
         gpuBufferManager->customCudaMalloc<uint8_t>(final_count * sizeof(str_top_n_record_type), 0, 0)
     );
     uint32_t combine_workers = (final_count + BLOCK_THREADS - 1) / BLOCK_THREADS;
     if (combine_workers == 0) combine_workers = 1;
     combine_to_records<<<combine_workers, BLOCK_THREADS>>>(d_records, d_sorted_prefixes, d_sorted_row_ids, final_count);
     CUDA_CHECK_AND_SYNC("Combine Kernel");
 
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_sorted_prefixes), 0);
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_sorted_row_ids), 0);
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_cols), 0);
 
     // 5. Materialization
     START_TIMER();
     
     uint32_t gather_block_size = 256;
     uint32_t gather_grid_size = (num_results + gather_block_size - 1) / gather_block_size;
     if (gather_grid_size == 0) gather_grid_size = 1;
 
     for (int col_idx = 0; col_idx < num_projections; ++col_idx) {
         DataWrapper& src_wrapper = projection[col_idx]->data_wrapper;
         GPUColumnTypeId type_id = src_wrapper.type.id();
         KernelColType kernel_type = MapSiriusTypeToKernelType(type_id);
 
         if (kernel_type == KernelColType::STRING) {
             uint8_t* src_chars = src_wrapper.data;
             uint64_t* src_offsets = src_wrapper.offset;
 
             uint64_t* d_new_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(num_results + 1, 0, 0);
             materialize_determine_lengths<<<gather_grid_size, gather_block_size>>>(
                 d_records, src_offsets, d_new_offsets, num_results
             );
             
             void* d_temp = nullptr; size_t temp_bytes = 0;
             cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_new_offsets, d_new_offsets, num_results + 1);
             d_temp = gpuBufferManager->customCudaMalloc<uint8_t>(temp_bytes, 0, 0);
             cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_new_offsets, d_new_offsets, num_results + 1);
             gpuBufferManager->customCudaFree(static_cast<uint8_t*>(d_temp), 0);
 
             uint64_t total_bytes;
             cudaMemcpy(&total_bytes, d_new_offsets + num_results, sizeof(uint64_t), cudaMemcpyDeviceToHost);
             uint8_t* d_new_chars = gpuBufferManager->customCudaMalloc<uint8_t>(total_bytes, 0, 0);
             
             materialize_copy_string<<<gather_grid_size, gather_block_size>>>(
                 d_records, src_chars, src_offsets, d_new_chars, d_new_offsets, num_results
             );
             
             CUDA_CHECK_AND_SYNC("Gather String");
 
             auto validity = createNullMask(num_results, cudf::mask_state::ALL_VALID); 
             projection[col_idx] = make_shared_ptr<GPUColumn>(
                 num_results, src_wrapper.type, d_new_chars, d_new_offsets, total_bytes, true, validity
             );
 
         } else {
             size_t type_size = src_wrapper.getColumnTypeSize();
             if (type_size == 0) {
                  if (kernel_type == KernelColType::INT_128) type_size = 16;
                  else if (kernel_type == KernelColType::INT_64 || kernel_type == KernelColType::DOUBLE) type_size = 8;
                  else if (kernel_type == KernelColType::INT_32) type_size = 4;
             }
 
             uint8_t* d_new_data = gpuBufferManager->customCudaMalloc<uint8_t>(num_results * type_size, 0, 0);
             
             if (type_size == 16) {
                 gather_fixed_width_128_kernel<<<gather_grid_size, gather_block_size>>>(
                     d_records, src_wrapper.data, d_new_data, num_results
                 );
             } else if (type_size == 8) {
                 gather_fixed_width_kernel<uint64_t><<<gather_grid_size, gather_block_size>>>(
                     d_records, src_wrapper.data, d_new_data, num_results
                 );
             } else if (type_size == 4) {
                 gather_fixed_width_kernel<uint32_t><<<gather_grid_size, gather_block_size>>>(
                     d_records, src_wrapper.data, d_new_data, num_results
                 );
             } else if (type_size == 1) {
                 gather_fixed_width_kernel<uint8_t><<<gather_grid_size, gather_block_size>>>(
                     d_records, src_wrapper.data, d_new_data, num_results
                 );
             }
             CUDA_CHECK_AND_SYNC("Gather Fixed");
 
             auto validity = createNullMask(num_results, cudf::mask_state::ALL_VALID);
             projection[col_idx] = make_shared_ptr<GPUColumn>(
                 num_results, src_wrapper.type, d_new_data, validity
             );
         }
     }
     
     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_records), 0);
     RECORD_TIMER("MULTI-COL TOP N Result Write Time");
 }
 
 void cudf_orderby(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& projection, idx_t num_keys, idx_t num_projections, OrderByType* order_by_type, idx_t num_results) 
 {
     // =================================================================================
     // 1. Optimization path check: Multi-Column Top-K
     // =================================================================================
     if (Config::USE_CUSTOM_TOP_N && num_results > 0 && num_results <= MAX_THREAD_TOP_K) {
         
         bool can_use_optimization = true;
         
         for (size_t col = 0; col < num_keys; col++) {
             auto type_id = keys[col]->data_wrapper.type.id();
             if (MapSiriusTypeToKernelType(type_id) == KernelColType::UNKNOWN) {
                 can_use_optimization = false;
                 break;
             }
         }
 
         if (can_use_optimization) {
             for (size_t col = 0; col < num_projections; col++) {
                 auto type_id = projection[col]->data_wrapper.type.id();
                 KernelColType ktype = MapSiriusTypeToKernelType(type_id);
                 if (ktype == KernelColType::UNKNOWN) {
                     can_use_optimization = false;
                     break;
                 }
             }
         }
 
         if (can_use_optimization) {
             CustomMultiColumnTopN(keys, projection, num_keys, num_projections, order_by_type, num_results);
             return;
         }
     }
 
     // =================================================================================
     // 2. Fallback Logic
     // =================================================================================
     if (keys[0]->column_length == 0) {
         for (idx_t col = 0; col < num_projections; col++) {
             bool old_unique = projection[col]->is_unique;
             if (projection[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
                 projection[col] = make_shared_ptr<GPUColumn>(0, projection[col]->data_wrapper.type, projection[col]->data_wrapper.data, projection[col]->data_wrapper.offset, 0, true, nullptr);
             } else {
                 projection[col] = make_shared_ptr<GPUColumn>(0, projection[col]->data_wrapper.type, projection[col]->data_wrapper.data, nullptr);
             }
             projection[col]->is_unique = old_unique;
         }
         return;
     }
 
     GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
     cudf::set_current_device_resource(gpuBufferManager->mr);
 
     std::vector<cudf::column_view> columns_cudf;
     for (int key = 0; key < num_keys; key++) {
         auto cudf_column_view = keys[key]->convertToCudfColumn();
         columns_cudf.push_back(cudf_column_view);
     }
 
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
 
     auto keys_table = cudf::table_view(columns_cudf);
     auto sorted_order = cudf::sorted_order(keys_table, orders, null_orders);
     auto sorted_order_view = sorted_order->view();
 
     std::vector<cudf::column_view> projection_cudf;
     for (int col = 0; col < num_projections; col++) {
         auto cudf_column = projection[col]->convertToCudfColumn();
         projection_cudf.push_back(cudf_column);
     }
     auto projection_table = cudf::table_view(projection_cudf);
     auto gathered_table = cudf::gather(projection_table, sorted_order_view);
 
     for (int col = 0; col < num_projections; col++) {
         auto sorted_column = gathered_table->get_column(col);
         projection[col]->setFromCudfColumn(sorted_column, projection[col]->is_unique, nullptr, 0, gpuBufferManager);
     }
 }
 
 } //namespace duckdb
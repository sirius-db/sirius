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
#include "cudf_utils.hpp"
#include "helper/common.h"
#include "gpu_columns.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "utils.hpp"

namespace duckdb {

// Declaration of the CUDA kernel
template <typename T> T* callCudaMalloc(size_t size, int gpu);
template <typename T> T* callCudaHostAlloc(size_t size, bool return_dev_ptr);
template <typename T> void callCudaFree(T* ptr, int gpu);
template <typename T> void callCudaMemcpyHostToDevice(T* dest, T* src, size_t size, int gpu);
template <typename T> void callCudaMemcpyDeviceToHost(T* dest, T* src, size_t size, int gpu);
void cudaMemmove(uint8_t* destination, uint8_t* source, size_t num);
uint8_t* allocatePinnedCPUMemory(size_t size);
uint8_t* allocatePageableCPUMemory(size_t size);
void freePinnedCPUMemory(uint8_t* ptr);
void freePageableCPUMemory(uint8_t* ptr);
size_t getFreeGPUMemorySize(int gpu);
void warmup_gpu();

struct pointer_and_key {
	uint64_t* pointer;
	uint64_t num_key;
};

struct string_group_by_metadata_type {
    void* all_keys;
    void* offsets;
    uint64_t num_keys;
};

struct string_top_n_record_type {
	uint32_t row_id;
	uint32_t key_prefix;
};

struct string_group_by_record_type {
	string_group_by_metadata_type* group_by_metadata;
	uint64_t row_id;
	uint64_t row_signature;
};

struct duckdb_string_type {
	union {
		struct {
			uint32_t length;
			char prefix[4];
			char *ptr;
		} pointer;
		struct {
			uint32_t length;
			char inlined[12];
		} inlined;
	} value;
};

// Currently a singleton class, would not work for multiple GPUs
class GPUBufferManager {
public:
    // Static method to get the singleton instance
    static GPUBufferManager& GetInstance(size_t cache_size_per_gpu = 0, size_t processing_size_per_gpu = 0, size_t processing_size_per_cpu = 0) {
        static GPUBufferManager instance(cache_size_per_gpu, processing_size_per_gpu, processing_size_per_cpu);
        return instance;
    }

    // Delete copy constructor and assignment operator to prevent copying
    GPUBufferManager(const GPUBufferManager&) = delete;
    GPUBufferManager& operator=(const GPUBufferManager&) = delete;

    void ResetBuffer();
    void ResetCache();
	uint8_t** gpuCache, **cpuCache; //each gpu has one, `cpuCache` will be used if `gpuCache` is full
	uint8_t** gpuProcessing, *cpuProcessing;
	size_t* gpuProcessingPointer, *gpuCachingPointer, *cpuCachingPointer; //each gpu has one
	size_t cpuProcessingPointer; 

	size_t cache_size_per_gpu;
	size_t processing_size_per_gpu;
	size_t processing_size_per_cpu;

	vector<size_t> available_gpu_cache_size;

	rmm::mr::cuda_memory_resource* cuda_mr;
	rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>* mr;

	template <typename T>
	T* customCudaMalloc(size_t size, int gpu, bool caching);

	template <typename T>
	T* customCudaHostAlloc(size_t size);

	void customCudaFree(uint8_t* ptr, int gpu);

	void Print();

	map<string, shared_ptr<GPUIntermediateRelation>> tables;

	void lockAllocation(void* ptr, int gpu);

	void createTableAndColumnInGPU(Catalog& catalog, ClientContext& context, string table_name, string column_name);
	void createTable(string table_name, size_t column_count);
	void createColumn(string table_name, string column_name, GPUColumnType column_type, size_t column_id, vector<size_t> unique_columns);
	bool checkIfColumnCached(string table_name, string column_name);

	std::vector<std::unique_ptr<rmm::device_buffer>> rmm_stored_buffers;

	//create an allocation table that keep tracks of the allocation of the memory, it stores the pointer, size, and the gpu id
	vector<map<void*, uint64_t>> allocation_table;
	vector<map<void*, uint64_t>> locked_allocation_table;
private:
    // Private constructor
   	GPUBufferManager(size_t cache_size_per_gpu, size_t processing_size_per_gpu, size_t processing_size_per_cpu);
    ~GPUBufferManager();

};


} // namespace duckdb   
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
template <typename T> void materializeExpression(T *a, T*& result, uint64_t *row_ids, uint64_t result_len, uint64_t input_len);
void materializeString(uint8_t* data, uint64_t* offset, uint8_t* &result, uint64_t* &result_offset, uint64_t* row_ids, uint64_t* &result_bytes, uint64_t result_len, uint64_t input_size, uint64_t input_bytes);
template <typename T> void printGPUColumn(T* a, size_t N, int gpu);
void cudaMemmove(uint8_t* destination, uint8_t* source, size_t num);
void warmup_gpu();

struct pointer_and_key {
	uint64_t* pointer;
	uint64_t num_key;
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
	uint8_t** gpuCache; //each gpu has one
	uint8_t** gpuProcessing, *cpuProcessing;
	size_t* gpuProcessingPointer, *gpuCachingPointer; //each gpu has one
	size_t cpuProcessingPointer; 

	size_t cache_size_per_gpu;
	size_t processing_size_per_gpu;
    size_t processing_size_per_cpu;

	rmm::mr::cuda_memory_resource* cuda_mr;
	rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>* mr;

	template <typename T>
	T* customCudaMalloc(size_t size, int gpu, bool caching);

	template <typename T>
	T* customCudaHostAlloc(size_t size);

	template <typename T>
	void customCudaFree(T* ptr, size_t size, int gpu);

	void customCudaFree(uint8_t* ptr, int gpu);

	void Print();

    map<string, GPUIntermediateRelation*> tables;

	DataWrapper allocateChunk(DataChunk &input);
	DataWrapper allocateColumnBufferInCPU(unique_ptr<MaterializedQueryResult> input);
	DataWrapper allocateStringChunk(DataChunk &input_chunk, size_t row_count, DataWrapper &prev_data);
	void cacheDataInGPU(DataWrapper cpu_data, string table_name, string column_name, int gpu);
	DataWrapper allocateColumnBufferInGPU(DataWrapper cpu_data, int gpu);
	DataWrapper allocateStrColumnInGPU(DataWrapper cpu_data, int gpu);

	void createTableAndColumnInGPU(Catalog& catalog, ClientContext& context, string table_name, string column_name);
	void createTable(string table_name, size_t column_count);
	void createColumn(string table_name, string column_name, ColumnType column_type, size_t column_id, vector<size_t> unique_columns);
	bool checkIfColumnCached(string table_name, string column_name);
private:
    // Private constructor
   	GPUBufferManager(size_t cache_size_per_gpu, size_t processing_size_per_gpu, size_t processing_size_per_cpu);
    ~GPUBufferManager();
	//create an allocation table that keep tracks of the allocation of the memory, it stores the pointer, size, and the gpu id
	vector<map<uintptr_t, uint64_t>> allocation_table;
	
};


} // namespace duckdb   
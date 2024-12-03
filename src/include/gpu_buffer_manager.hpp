#pragma once

#include "helper/common.h"
#include "gpu_columns.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/catalog/catalog.hpp"

namespace duckdb {

// Declaration of the CUDA kernel
template <typename T> T* callCudaMalloc(size_t size, int gpu);
template <typename T> void callCudaFree(T* ptr, int gpu);
template <typename T> void callCudaMemcpyHostToDevice(T* dest, T* src, size_t size, int gpu);
template <typename T> void callCudaMemcpyDeviceToHost(T* dest, T* src, size_t size, int gpu);
template <typename T> void materializeExpression(T *a, T* result, uint64_t *row_ids, uint64_t N);
int strMateralizeOffsets(int* materalized_offsets, int* input_offsets, uint64_t* row_ids, size_t num_rows);
void strMateralizeChars(uint8_t* materalized_chars, uint8_t* input_chars, int* materalized_offsets, 
	int* input_offsets, uint64_t* row_ids, size_t num_rows);

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

	uint8_t** gpuCache; //each gpu has one
	uint8_t** gpuProcessing, *cpuProcessing;
	size_t* gpuProcessingPointer, *gpuCachingPointer; //each gpu has one
	size_t cpuProcessingPointer; 

	size_t cache_size_per_gpu;
	size_t processing_size_per_gpu;
    size_t processing_size_per_cpu;

	template <typename T>
	T* customCudaMalloc(size_t size, int gpu, bool caching);

	template <typename T>
	T* customCudaHostAlloc(size_t size);

	void Print();

    map<string, GPUIntermediateRelation*> tables;

	DataWrapper allocateStringChunk(Vector &input,	size_t chunk_size);
	DataWrapper allocateChunk(DataChunk &input);
	DataWrapper mergeWrappers(DataWrapper first, DataWrapper second);
	DataWrapper allocateColumnBufferInCPU(unique_ptr<MaterializedQueryResult> input);
	void cacheDataInGPU(DataWrapper cpu_data, string table_name, string column_name, int gpu);
	DataWrapper allocateStrColumnInGPU(DataWrapper cpu_data, int gpu);
	DataWrapper allocateColumnBufferInGPU(DataWrapper cpu_data, int gpu);
	void createTableAndColumnInGPU(Catalog& catalog, ClientContext& context, string table_name, string column_name);
	void createTable(string table_name, size_t column_count);
	void createColumn(string table_name, string column_name, ColumnType column_type, size_t column_id);

private:
    // Private constructor
   	GPUBufferManager(size_t cache_size_per_gpu, size_t processing_size_per_gpu, size_t processing_size_per_cpu);
    ~GPUBufferManager();
};


} // namespace duckdb   
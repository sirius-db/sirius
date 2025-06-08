#include "gpu_buffer_manager.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "utils.hpp"
#include "log/logging.hpp"
#include "helper/types.hpp"
#include "operator/gpu_physical_table_scan.hpp"

#define NUM_GPUS 1

namespace duckdb {

template int*
GPUBufferManager::customCudaMalloc<int>(size_t size, int gpu, bool caching);

template uint64_t*
GPUBufferManager::customCudaMalloc<uint64_t>(size_t size, int gpu, bool caching);

template uint8_t*
GPUBufferManager::customCudaMalloc<uint8_t>(size_t size, int gpu, bool caching);

template float*
GPUBufferManager::customCudaMalloc<float>(size_t size, int gpu, bool caching);

template double*
GPUBufferManager::customCudaMalloc<double>(size_t size, int gpu, bool caching);

template char*
GPUBufferManager::customCudaMalloc<char>(size_t size, int gpu, bool caching);

template bool*
GPUBufferManager::customCudaMalloc<bool>(size_t size, int gpu, bool caching);

template duckdb_string_type*
GPUBufferManager::customCudaMalloc<duckdb_string_type>(size_t size, int gpu, bool caching);

template pointer_and_key*
GPUBufferManager::customCudaMalloc<pointer_and_key>(size_t size, int gpu, bool caching);

template string_group_by_metadata_type*
GPUBufferManager::customCudaMalloc<string_group_by_metadata_type>(size_t size, int gpu, bool caching);

template void**
GPUBufferManager::customCudaMalloc<void*>(size_t size, int gpu, bool caching);

template string_group_by_record_type*
GPUBufferManager::customCudaMalloc<string_group_by_record_type>(size_t size, int gpu, bool caching);

template uint8_t**
GPUBufferManager::customCudaMalloc<uint8_t*>(size_t size, int gpu, bool caching);

template uint64_t**
GPUBufferManager::customCudaMalloc<uint64_t*>(size_t size, int gpu, bool caching);

template double**
GPUBufferManager::customCudaMalloc<double*>(size_t size, int gpu, bool caching);

template int*
GPUBufferManager::customCudaHostAlloc<int>(size_t size);

template uint64_t*
GPUBufferManager::customCudaHostAlloc<uint64_t>(size_t size);

template uint8_t*
GPUBufferManager::customCudaHostAlloc<uint8_t>(size_t size);

template float*
GPUBufferManager::customCudaHostAlloc<float>(size_t size);

template double*
GPUBufferManager::customCudaHostAlloc<double>(size_t size);

template char*
GPUBufferManager::customCudaHostAlloc<char>(size_t size);

template bool*
GPUBufferManager::customCudaHostAlloc<bool>(size_t size);

template string_t*
GPUBufferManager::customCudaHostAlloc<string_t>(size_t size);

template AggregationType*
GPUBufferManager::customCudaHostAlloc<AggregationType>(size_t size);

template OrderByType*
GPUBufferManager::customCudaHostAlloc<OrderByType>(size_t size);

template ScanDataType*
GPUBufferManager::customCudaHostAlloc<ScanDataType>(size_t size);

template CompareType*
GPUBufferManager::customCudaHostAlloc<CompareType>(size_t size);

template int**
GPUBufferManager::customCudaHostAlloc<int*>(size_t size);

template uint64_t**
GPUBufferManager::customCudaHostAlloc<uint64_t*>(size_t size);

template uint8_t**
GPUBufferManager::customCudaHostAlloc<uint8_t*>(size_t size);

template float**
GPUBufferManager::customCudaHostAlloc<float*>(size_t size);

template double**
GPUBufferManager::customCudaHostAlloc<double*>(size_t size);

template char**
GPUBufferManager::customCudaHostAlloc<char*>(size_t size);

template string_t**
GPUBufferManager::customCudaHostAlloc<string_t*>(size_t size);

template ConstantFilter**
GPUBufferManager::customCudaHostAlloc<ConstantFilter*>(size_t size);

GPUBufferManager::GPUBufferManager(size_t cache_size_per_gpu, size_t processing_size_per_gpu, size_t processing_size_per_cpu) : 
    cache_size_per_gpu(cache_size_per_gpu), processing_size_per_gpu(processing_size_per_gpu), processing_size_per_cpu(processing_size_per_cpu) {
    SIRIUS_LOG_INFO("Initializing GPU buffer manager");
    gpuCache = new uint8_t*[NUM_GPUS];
    gpuProcessing = new uint8_t*[NUM_GPUS];
    cpuProcessing = allocatePinnedCPUMemory(processing_size_per_cpu);
    gpuProcessingPointer = new size_t[NUM_GPUS];
    gpuCachingPointer = new size_t[NUM_GPUS];
    cpuProcessingPointer = 0;

    cuda_mr = new rmm::mr::cuda_memory_resource();
    SIRIUS_LOG_INFO("Allocating cache size {} in GPU 0", cache_size_per_gpu);
    SIRIUS_LOG_INFO("Allocating processing size {} in GPU 0", processing_size_per_gpu);
    mr = new rmm::mr::pool_memory_resource(cuda_mr, processing_size_per_gpu, processing_size_per_cpu);
    cudf::set_current_device_resource(mr);
    allocation_table.resize(NUM_GPUS);
    locked_allocation_table.resize(NUM_GPUS);

    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        gpuCache[gpu] = callCudaMalloc<uint8_t>(cache_size_per_gpu, gpu);
        // gpuProcessing[gpu] = callCudaMalloc<uint8_t>(processing_size_per_gpu, gpu);
        // gpuCache[gpu] = callCudaHostAlloc<uint8_t>(cache_size_per_gpu, 1);
        // gpuProcessing[gpu] = callCudaHostAlloc<uint8_t>(processing_size_per_gpu, 1);
        gpuProcessingPointer[gpu] = 0;
        gpuCachingPointer[gpu] = 0;
    }

    warmup_gpu();
}

GPUBufferManager::~GPUBufferManager() {
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        callCudaFree<uint8_t>(gpuCache[gpu], gpu);
        // callCudaFree<uint8_t>(gpuProcessing[gpu], gpu);
        mr->deallocate((void*) gpuProcessing[gpu], processing_size_per_gpu);
    }
    freePinnedCPUMemory(cpuProcessing);
    delete[] gpuCache;
    delete[] gpuProcessing;
    delete[] gpuProcessingPointer;
    delete[] gpuCachingPointer;
    delete mr;
    delete cuda_mr;
}

void GPUBufferManager::ResetBuffer() {
    cudf::set_current_device_resource(mr);
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        SIRIUS_LOG_DEBUG("Resetting buffer for GPU {}", gpu);
        gpuProcessingPointer[gpu] = 0;
        //write a program to free all allocation in the allocation table
        for (auto it = allocation_table[gpu].begin(); it != allocation_table[gpu].end(); ++it) {
            auto ptr = it->first;
            auto size = it->second;
            if (ptr != nullptr) {
                // customCudaFree<uint8_t>(reinterpret_cast<uint8_t*>(ptr), size, 0);
                mr->deallocate((void*) ptr, size);
                // SIRIUS_LOG_DEBUG("Deallocating Pointer {} size {}", static_cast<void*>(ptr), size);
                // allocation_table[gpu].erase(it);
            }
        }
        allocation_table[gpu].clear();
        if (!allocation_table[gpu].empty()) {
            throw InvalidInputException("Allocation table is not empty");
        }
        // SIRIUS_LOG_DEBUG("Locked allocation table size {}", locked_allocation_table[gpu].size());
        for (auto it = locked_allocation_table[gpu].begin(); it != locked_allocation_table[gpu].end(); ++it) {
            auto ptr = it->first;
            auto size = it->second;
            if (ptr != nullptr) {
                // SIRIUS_LOG_DEBUG("Deallocating Locked Pointer {} size {}", static_cast<void*>(ptr), size);
                mr->deallocate((void*) ptr, size);
            }
        }
        locked_allocation_table[gpu].clear();
        if (!locked_allocation_table[gpu].empty()) {
            throw InvalidInputException("Locked allocation table is not empty");
        }
        rmm_stored_buffers.clear();
        // SIRIUS_LOG_DEBUG("pool size {}", mr->pool_size());

        // size_t allocated_size = mr->pool_size();
        // SIRIUS_LOG_DEBUG("Allocating {} bytes", allocated_size);
        // void* ptr = mr->allocate(allocated_size);
        // mr->deallocate(ptr, allocated_size);
    }
    cpuProcessingPointer = 0;
    for (auto it = tables.begin(); it != tables.end(); it++) {
        shared_ptr<GPUIntermediateRelation> table = it->second;
        for (int col = 0; col < table->columns.size(); col++) {
            if (table->columns[col] != nullptr) {
                table->columns[col]->row_ids = nullptr;
                table->columns[col]->row_id_count = 0;
            }
        }
    }
}

void GPUBufferManager::ResetCache() {
    SIRIUS_LOG_DEBUG("Resetting cache");
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        gpuCachingPointer[gpu] = 0;
    }
    cpuProcessingPointer = 0;
    for (auto it = tables.begin(); it != tables.end(); it++) {
        shared_ptr<GPUIntermediateRelation> table = it->second;
        for (int col = 0; col < table->columns.size(); col++) {
            table->columns[col] = nullptr;
        }
        table->column_names.clear();
        table->column_names.resize(table->column_count);
    }
}

template <typename T>
T*
GPUBufferManager::customCudaMalloc(size_t size, int gpu, bool caching) {
	size_t alloc = (size * sizeof(T));
    //always ensure that it aligns with 8B
    // int alignment = alignof(double);
    int alignment = 256;
    alloc = alloc + (alignment - alloc % alignment);
    if (caching) {
        size_t start = __atomic_fetch_add(&gpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
        assert((start + alloc) < cache_size_per_gpu);
        if (start + alloc >= cache_size_per_gpu) {
            throw InvalidInputException("Out of GPU caching memory");
        }
        T* ptr = reinterpret_cast<T*>(gpuCache[gpu] + start);
        if (reinterpret_cast<uintptr_t>(ptr) % alignof(double) != 0) {
            throw InvalidInputException("Memory is not properly aligned");
        } 
        return ptr;
    } else {
        cudf::set_current_device_resource(mr);
        void* ptr = mr->allocate(alloc);
        // size_t start = __atomic_fetch_add(&gpuProcessingPointer[gpu], alloc, __ATOMIC_RELAXED);
        // // SIRIUS_LOG_DEBUG("Current pointer {}", gpuProcessingPointer[gpu]);
        // assert((start + alloc) < processing_size_per_gpu);
        // if (start + alloc >= processing_size_per_gpu) {
        //     throw InvalidInputException("Out of GPU processing memory");
        // }
        // // SIRIUS_LOG_DEBUG("Allocating {} bytes at {}", alloc, start);
        // // SIRIUS_LOG_DEBUG("Current pointer {}", gpuProcessingPointer[gpu]);
        // T* ptr = reinterpret_cast<T*>(gpuProcessing[gpu] + start);
        // if (reinterpret_cast<uintptr_t>(ptr) % alignof(double) != 0) {
        //     throw InvalidInputException("Memory is not properly aligned");
        // }
        // SIRIUS_LOG_DEBUG("Allocating Pointer {} size {}", static_cast<void*>(ptr), alloc);
        if (ptr == nullptr) throw InvalidInputException("Pointer is nullptr");
        if (allocation_table[gpu].find(ptr) != allocation_table[gpu].end()) {
            throw InvalidInputException("Pointer already exists in allocation table");
        }
        allocation_table[gpu][ptr] = alloc;
        return reinterpret_cast<T*>(ptr);
    }
};

shared_ptr<GPUColumn> 
GPUBufferManager::copyDataFromcuDFColumn(cudf::column_view& column, int gpu) {
    //copy the data to the gpu
    //create a column
    //return the column
    uint8_t* data = const_cast<uint8_t*>(column.data<uint8_t>());

    if (column.type() == cudf::data_type(cudf::type_id::STRING)) {

        int32_t temp_num_bytes;
        int32_t* temp_offset = const_cast<int32_t*>(column.child(0).data<int32_t>());
        callCudaMemcpyDeviceToHost<int32_t>(&temp_num_bytes, temp_offset + column.size(), 1, 0);
        uint8_t* temp_column = customCudaMalloc<uint8_t>(temp_num_bytes, 0, false);
        callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, data, temp_num_bytes, 0);

        shared_ptr<GPUColumn> column_ptr = make_shared_ptr<GPUColumn>(column.size(), GPUColumnType(GPUColumnTypeId::VARCHAR), temp_column);
        column_ptr->convertCudfOffsetToSiriusOffset(temp_offset);
        column_ptr->data_wrapper.num_bytes = temp_num_bytes;
        column_ptr->data_wrapper.is_string_data = true;
        return column_ptr;
    } else if (column.type() == cudf::data_type(cudf::type_id::UINT64)) {
        uint8_t* temp_column = customCudaMalloc<uint8_t>(column.size() * sizeof(uint64_t), 0, false);
        callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, data, column.size() * sizeof(uint64_t), 0);
        return make_shared_ptr<GPUColumn>(column.size(), GPUColumnType(GPUColumnTypeId::INT64), temp_column);
    } else if (column.type() == cudf::data_type(cudf::type_id::INT32)) {
        uint8_t* temp_column = customCudaMalloc<uint8_t>(column.size() * sizeof(int32_t), 0, false);
        callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, data, column.size() * sizeof(int32_t), 0);
        return make_shared_ptr<GPUColumn>(column.size(), GPUColumnType(GPUColumnTypeId::INT32), temp_column);
    } else if (column.type() == cudf::data_type(cudf::type_id::FLOAT32)) {
        uint8_t* temp_column = customCudaMalloc<uint8_t>(column.size() * sizeof(float), 0, false);
        callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, data, column.size() * sizeof(float), 0);
        return make_shared_ptr<GPUColumn>(column.size(), GPUColumnType(GPUColumnTypeId::FLOAT32), temp_column);
    } else if (column.type() == cudf::data_type(cudf::type_id::FLOAT64)) {
        uint8_t* temp_column = customCudaMalloc<uint8_t>(column.size() * sizeof(double), 0, false);
        callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, data, column.size() * sizeof(double), 0);
        return make_shared_ptr<GPUColumn>(column.size(), GPUColumnType(GPUColumnTypeId::FLOAT64), temp_column);
    } else if (column.type() == cudf::data_type(cudf::type_id::BOOL8)) {
        uint8_t* temp_column = customCudaMalloc<uint8_t>(column.size() * sizeof(bool), 0, false);
        callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, data, column.size() * sizeof(bool), 0);
        return make_shared_ptr<GPUColumn>(column.size(), GPUColumnType(GPUColumnTypeId::BOOLEAN), temp_column);
    }
    throw duckdb::InternalException("Unsupported cuDF column: {}", static_cast<int>(column.type().id()));
}

void
GPUBufferManager::lockAllocation(void* ptr, int gpu) {
    //move entries from the allocation table to the locked table
    auto it = allocation_table[gpu].find(ptr);
    if (it != allocation_table[gpu].end()) {
        // SIRIUS_LOG_DEBUG("Locking Pointer {}", static_cast<void*>(ptr));
        locked_allocation_table[gpu][ptr] = it->second;
        allocation_table[gpu].erase(it);
    }
}

void
GPUBufferManager::customCudaFree(uint8_t* ptr, int gpu) {
    //check if ptr is not in gpuCaching
    cudf::set_current_device_resource(mr);
    if (ptr != nullptr && (ptr < gpuCache[gpu] || ptr >= gpuCache[gpu] + cache_size_per_gpu)) {
        auto it = allocation_table[gpu].find(reinterpret_cast<void*>(ptr));
        if (it != allocation_table[gpu].end()) {
            // SIRIUS_LOG_DEBUG("Deallocating Pointer {} size {}", static_cast<void*>(ptr), it->second);
            mr->deallocate((void*) ptr, it->second);
            allocation_table[gpu].erase(it);
        } else {
            auto locked_it = locked_allocation_table[gpu].find(reinterpret_cast<void*>(ptr));
            if (locked_it == locked_allocation_table[gpu].end()) {
                // check if in rmm_stored_buffer
                bool found = 0;
                for (int it = 0; it < rmm_stored_buffers.size(); it++) {
                    if (ptr == reinterpret_cast<uint8_t*>(rmm_stored_buffers[it]->data())) {
                        found = 1; break;
                    }
                }
                if (!found) {
                    SIRIUS_LOG_DEBUG("Invalid Pointer {}", static_cast<void*>(ptr));
                    throw InvalidInputException("Pointer not found in allocation table");
                }
            }
        }
    }
}

template <typename T>
T*
GPUBufferManager::customCudaHostAlloc(size_t size) {
	size_t alloc = (size * sizeof(T));
	size_t start = __atomic_fetch_add(&cpuProcessingPointer, alloc, __ATOMIC_RELAXED);
	assert((start + alloc) < processing_size_per_cpu);
    if (start + alloc >= processing_size_per_cpu) {
        throw InvalidInputException("Out of CPU memory");
    }
	return reinterpret_cast<T*>(cpuProcessing + start);
};

DataWrapper GPUBufferManager::allocateStringChunk(DataChunk &input_chunk, size_t row_count, DataWrapper &prev_data) {
	Vector input = input_chunk.data[0];
    size_t chunk_size = input_chunk.size();
    input.Flatten(chunk_size);
    LogicalType vector_type = input.GetType();
    if(vector_type.id() != LogicalTypeId::VARCHAR) {
        throw InvalidInputException("Wrong type");
    }

    DataWrapper result;
    result.type = GPUColumnType(GPUColumnTypeId::VARCHAR);
    // SIRIUS_LOG_DEBUG("chunk size {}", chunk_size);
    result.size = prev_data.size + chunk_size;

    // First iteration, allocate the offset array
    if (prev_data.size == 0) {
        result.offset = customCudaHostAlloc<uint64_t>(row_count + 1);
    } else {
        result.offset = prev_data.offset;
    }

    uint64_t curr_offset = 0;
    for(uint64_t i = 0; i < chunk_size; i++) {
        std::string curr_string = input.GetValue(i).ToString();
        result.offset[prev_data.size + i] = curr_offset + prev_data.num_bytes;
        curr_offset += curr_string.length();
    }

    result.offset[prev_data.size + chunk_size] = curr_offset + prev_data.num_bytes;

    // Now do the same for the chars
    result.num_bytes = (size_t) (curr_offset + prev_data.num_bytes);
    uint64_t copy_offset = 0;

    //assuming its contiguous with prev_data
    uint8_t* ptr = customCudaHostAlloc<uint8_t>(curr_offset);

    //TODO: Need to optimize this part
    for(uint64_t i = 0; i < chunk_size; i++) {
        std::string curr_string = input.GetValue(i).ToString();
        uint64_t str_length = curr_string.length();
        memcpy(ptr + copy_offset, reinterpret_cast<uint8_t*>(curr_string.data()), str_length * sizeof(uint8_t));
        copy_offset += str_length;
    }

    if (prev_data.size == 0) {
        result.data = ptr;
    } else {
        result.data = prev_data.data;
    }
    
    result.is_string_data = true;
    return result;
}

DataWrapper
GPUBufferManager::allocateChunk(DataChunk &input){
	size_t chunk_size = input.size();
	LogicalType vector_type = input.data[0].GetType();
    uint8_t* ptr = nullptr;
    GPUColumnType column_type;

    //the allocation below is assuming its contiguous with prev_data
    switch (vector_type.id()) {
        case LogicalTypeId::INTEGER: {
            int* ptr_int = customCudaHostAlloc<int>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_int);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(int));
            column_type = GPUColumnType(GPUColumnTypeId::INT32);
            break;
        }
        case LogicalTypeId::BIGINT: {
            uint64_t* ptr_int64 = customCudaHostAlloc<uint64_t>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_int64);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(uint64_t));
            column_type = GPUColumnType(GPUColumnTypeId::INT64);
            break;
        }
        case LogicalTypeId::FLOAT: {
            float* ptr_float = customCudaHostAlloc<float>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_float);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(float));
            column_type = GPUColumnType(GPUColumnTypeId::FLOAT32);
            break;
        }
        case LogicalTypeId::DOUBLE: {
            double* ptr_double = customCudaHostAlloc<double>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_double);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(double));
            column_type = GPUColumnType(GPUColumnTypeId::FLOAT64);
            break;
        }
        case LogicalTypeId::BOOLEAN: {
            uint8_t* ptr_bool = customCudaHostAlloc<uint8_t>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_bool);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(uint8_t));
            column_type = GPUColumnType(GPUColumnTypeId::BOOLEAN);
            break;
        }
        case LogicalTypeId::VARCHAR: {
            throw InvalidInputException("String type not supported");
            //FIX TODO: Need to handle the case where the string is larger than 128 characters
            char* ptr_varchar = customCudaHostAlloc<char>(chunk_size * 128);
            ptr = reinterpret_cast<uint8_t*>(ptr_varchar);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(double));
            column_type = GPUColumnType(GPUColumnTypeId::VARCHAR);
            break;
        }
        default:
            throw InvalidInputException("Unsupported type");
    }

    return DataWrapper(column_type, ptr, chunk_size);
}

//TODO: We have to lock the CPU buffer before calling bufferChunkInCPU to ensure contiguous memory allocation
DataWrapper
GPUBufferManager::allocateColumnBufferInCPU(unique_ptr<MaterializedQueryResult> input) {
    auto row_count = input->RowCount();
    SIRIUS_LOG_DEBUG("Row count {}", row_count);
	auto input_chunk = input->Fetch();
	if (!input_chunk) {
		throw InvalidInputException("No data in input chunk");
	}

    DataWrapper result_wrapper(GPUColumnType(GPUColumnTypeId::INT32), nullptr, 0);
    if (input_chunk->data[0].GetType().id() == LogicalTypeId::VARCHAR) {
        result_wrapper = allocateStringChunk(*input_chunk, row_count, result_wrapper);
    } else {
        result_wrapper = allocateChunk(*input_chunk);
    }
    input_chunk = input->Fetch();
    //TODO: Need to handle merging data_wrapper in a better way, currently assuming contiguous memory allocation
    //Better way to do this is to lock the buffer manager during this call
	while (input_chunk) {
		// auto wrapper = allocateChunk(*input_chunk);
        if (input_chunk->data[0].GetType().id() == LogicalTypeId::VARCHAR) {
            result_wrapper = allocateStringChunk(*input_chunk, row_count, result_wrapper);
        } else {
            auto wrapper = allocateChunk(*input_chunk);
            result_wrapper.size += wrapper.size;
        }
		input_chunk = input->Fetch();
	}
    SIRIUS_LOG_DEBUG("Done allocating column buffer in CPU");
    return result_wrapper;
}

DataWrapper GPUBufferManager::allocateStrColumnInGPU(DataWrapper cpu_data, int gpu) {

    DataWrapper result;
    result.is_string_data = cpu_data.is_string_data;
    result.type = GPUColumnType(GPUColumnTypeId::VARCHAR);

    // First allocate and copy the offset buffer
    result.size = cpu_data.size;
    result.offset = customCudaMalloc<uint64_t>((cpu_data.size + 1), 0, true);
    SIRIUS_LOG_DEBUG("Copying offset with {} strings", result.size);
    callCudaMemcpyHostToDevice<uint64_t>(result.offset, cpu_data.offset, (cpu_data.size + 1), 0);

    // Do the same for the characeters
    result.num_bytes = cpu_data.num_bytes;
    result.data = customCudaMalloc<uint8_t>(cpu_data.num_bytes, 0, true);
    SIRIUS_LOG_DEBUG("Copying sizes with {} chars", result.num_bytes);
    callCudaMemcpyHostToDevice<uint8_t>(result.data, cpu_data.data, cpu_data.num_bytes, 0);

    SIRIUS_LOG_DEBUG("Returning wrapper of size {} and {}", result.size, result.num_bytes);
    return result;
}


DataWrapper
GPUBufferManager::allocateColumnBufferInGPU(DataWrapper cpu_data, int gpu) {
    if(cpu_data.is_string_data) {
        SIRIUS_LOG_DEBUG("Calling allocateStrColumnInGPU");
        return allocateStrColumnInGPU(cpu_data, gpu);
    }
    
    uint8_t* ptr = nullptr;
    GPUColumnType column_type;

	switch (cpu_data.type.id()) {
		case GPUColumnTypeId::INT32: {
            int* ptr_int = customCudaMalloc<int>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_int);
            column_type = GPUColumnType(GPUColumnTypeId::INT32);
			break;
        }
		case GPUColumnTypeId::INT64: {
            uint64_t* ptr_int64 = customCudaMalloc<uint64_t>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_int64);
            column_type = GPUColumnType(GPUColumnTypeId::INT64);
			break;
        }
		case GPUColumnTypeId::FLOAT32: {
            float* ptr_float = customCudaMalloc<float>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_float);
            column_type = GPUColumnType(GPUColumnTypeId::FLOAT32);
			break;
        }
		case GPUColumnTypeId::FLOAT64: {
            double* ptr_double = customCudaMalloc<double>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_double);
            column_type = GPUColumnType(GPUColumnTypeId::FLOAT64);
			break;
        }
		case GPUColumnTypeId::BOOLEAN: {
            uint8_t* ptr_bool = customCudaMalloc<uint8_t>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_bool);
            column_type = GPUColumnType(GPUColumnTypeId::BOOLEAN);
			break;
        }
		case GPUColumnTypeId::VARCHAR: {
            char* ptr_char = customCudaMalloc<char>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_char);
            column_type = GPUColumnType(GPUColumnTypeId::VARCHAR);
			break;
        }
        default:
            throw InvalidInputException("Unsupported type");
	}
    return DataWrapper(column_type, ptr, cpu_data.size);
}

void
GPUBufferManager::cacheDataInGPU(DataWrapper cpu_data, string table_name, string column_name, int gpu) {
    string up_column_name = column_name;
    string up_table_name = table_name;
    transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
    transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
    auto column_it = find(tables[up_table_name]->column_names.begin(), tables[up_table_name]->column_names.end(), up_column_name);
    if (column_it == tables[up_table_name]->column_names.end()) {
        throw InvalidInputException("Column not found");
    }
    DataWrapper gpu_allocated_buffer = allocateColumnBufferInGPU(cpu_data, gpu);
    if(!gpu_allocated_buffer.is_string_data) {
        callCudaMemcpyHostToDevice<uint8_t>(gpu_allocated_buffer.data, cpu_data.data, cpu_data.size * cpu_data.getColumnTypeSize(), 0);
    } 
    int column_idx = column_it - tables[up_table_name]->column_names.begin(); 
    tables[up_table_name]->columns[column_idx]->data_wrapper = gpu_allocated_buffer;
    tables[up_table_name]->columns[column_idx]->column_length = gpu_allocated_buffer.size;
    SIRIUS_LOG_DEBUG("Data cached in GPU");
}

void
GPUBufferManager::createTableAndColumnInGPU(Catalog& catalog, ClientContext& context, string table_name, string column_name) {
	TableCatalogEntry &table = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA, table_name).Cast<TableCatalogEntry>();
    auto column_names = table.GetColumns().GetColumnNames();
    auto& constraints = table.GetConstraints();
    vector<size_t> unique_columns;

	// for (auto &constraint : constraints) {
	// 	if (constraint->type == ConstraintType::NOT_NULL) {
	// 		auto &not_null = constraint->Cast<NotNullConstraint>();
	// 		not_null_columns.insert(not_null.index);
	// 	} else if (constraint->type == ConstraintType::UNIQUE) {
	// 		auto &pk = constraint->Cast<UniqueConstraint>();
	// 		if (pk.HasIndex()) {
	// 			// no columns specified: single column constraint
	// 			if (pk.IsPrimaryKey()) {
	// 				pk_columns.insert(pk.GetIndex());
	// 			} else {
	// 				unique_columns.insert(pk.GetIndex());
	// 			}
	// 		} else {
	// 			// multi-column constraint, this constraint needs to go at the end after all columns
	// 			if (pk.IsPrimaryKey()) {
	// 				// multi key pk column: insert set of columns into multi_key_pks
	// 				for (auto &col : pk.GetColumnNames()) {
	// 					multi_key_pks.insert(col);
	// 				}
	// 			}
	// 			extra_constraints.push_back(constraint->ToString());
	// 		}
	// 	} else if (constraint->type == ConstraintType::FOREIGN_KEY) {
	// 		auto &fk = constraint->Cast<ForeignKeyConstraint>();
	// 		if (fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE ||
	// 		    fk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
	// 			extra_constraints.push_back(constraint->ToString());
	// 		}
	// 	} else {
	// 		extra_constraints.push_back(constraint->ToString());
	// 	}
	// }
    
	for (auto &constraint : constraints) {
		if (constraint->type == ConstraintType::UNIQUE) {
			auto &pk = constraint->Cast<UniqueConstraint>();
			if (pk.HasIndex()) {
                SIRIUS_LOG_DEBUG("Unique constraint on index {}", pk.GetIndex().index);
                for (auto &col : pk.GetColumnNames()) {
                    SIRIUS_LOG_DEBUG("Unique constraint on column {}", col);
                }
                unique_columns.push_back(pk.GetIndex().index);
			} else {
                for (auto &col : pk.GetColumnNames()) {
                    SIRIUS_LOG_DEBUG("Unique constraint on column {}", col);
                }
            }
		}
	}
    
    //finding column_name in column_names
    //convert column_name to uppercase
    string up_column_name = column_name;
    //when caching table, it has to be exaclty the same as the column name in the table (case sensitive)
    transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
    if (find(column_names.begin(), column_names.end(), column_name) != column_names.end()) {
        // convert table_name to uppercase
        size_t column_id = table.GetColumnIndex(column_name, false).index;
        string up_table_name = table_name;
        transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
        createTable(up_table_name, table.GetTypes().size());
        GPUColumnType column_type = convertLogicalTypeToColumnType(table.GetColumn(column_name).GetType());
        SIRIUS_LOG_DEBUG("Creating column {}", up_column_name);
        createColumn(up_table_name, up_column_name, column_type, column_id, unique_columns);
    } else {
        throw InvalidInputException("Column does not exists");
    }
    SIRIUS_LOG_DEBUG("Table and column created in GPU");
}

void
GPUBufferManager::createTable(string up_table_name, size_t column_count) {
    //we will update the length later
    //check if table already exists
    if (tables.find(up_table_name) == tables.end()) {
        tables[up_table_name] = make_shared_ptr<GPUIntermediateRelation>(column_count);
        tables[up_table_name]->names = up_table_name;
    }
}

bool
GPUBufferManager::checkIfColumnCached(string table_name, string column_name) {
    string up_column_name = column_name;
    string up_table_name = table_name;
    transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
    transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
    auto table_it = tables.find(up_table_name);
    if (table_it == tables.end()) {
        return false;
    }
    auto column_it = find(tables[table_name]->column_names.begin(), tables[table_name]->column_names.end(), up_column_name);
    if (column_it == tables[up_table_name]->column_names.end()) {
        return false;
    }
    return true;
}

void
GPUBufferManager::createColumn(string up_table_name, string up_column_name, GPUColumnType column_type, size_t column_id, vector<size_t> unique_columns) {
    shared_ptr<GPUIntermediateRelation> table = tables[up_table_name];
    table->column_names[column_id] = up_column_name;
    if (find(unique_columns.begin(), unique_columns.end(), column_id) != unique_columns.end()) {
        table->columns[column_id] = make_shared_ptr<GPUColumn>(0, column_type, nullptr);
        table->columns[column_id]->is_unique = true;
    } else {
        table->columns[column_id] = make_shared_ptr<GPUColumn>(0, column_type, nullptr);
        table->columns[column_id]->is_unique = false;
    }
}

}; // namespace duckdb
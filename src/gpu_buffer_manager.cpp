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

#include "gpu_buffer_manager.hpp"
#include "config.hpp"
#include <rmm/aligned.hpp>
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

template int16_t*
GPUBufferManager::customCudaMalloc<int16_t>(size_t size, int gpu, bool caching);

template int*
GPUBufferManager::customCudaMalloc<int>(size_t size, int gpu, bool caching);

template uint64_t*
GPUBufferManager::customCudaMalloc<uint64_t>(size_t size, int gpu, bool caching);

template int64_t*
GPUBufferManager::customCudaMalloc<int64_t>(size_t size, int gpu, bool caching);

template uint32_t*
GPUBufferManager::customCudaMalloc<uint32_t>(size_t size, int gpu, bool caching);

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

template __int128_t*
GPUBufferManager::customCudaMalloc<__int128_t>(size_t size, int gpu, bool caching);

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

template string_top_n_record_type*
GPUBufferManager::customCudaMalloc<string_top_n_record_type>(size_t size, int gpu, bool caching);

template uint8_t**
GPUBufferManager::customCudaMalloc<uint8_t*>(size_t size, int gpu, bool caching);

template uint64_t**
GPUBufferManager::customCudaMalloc<uint64_t*>(size_t size, int gpu, bool caching);

template uint32_t**
GPUBufferManager::customCudaMalloc<uint32_t*>(size_t size, int gpu, bool caching);

template int32_t**
GPUBufferManager::customCudaMalloc<int32_t*>(size_t size, int gpu, bool caching);

template int64_t**
GPUBufferManager::customCudaMalloc<int64_t*>(size_t size, int gpu, bool caching);

template double**
GPUBufferManager::customCudaMalloc<double*>(size_t size, int gpu, bool caching);

template int*
GPUBufferManager::customCudaHostAlloc<int>(size_t size);

template int64_t*
GPUBufferManager::customCudaHostAlloc<int64_t>(size_t size);

template uint64_t*
GPUBufferManager::customCudaHostAlloc<uint64_t>(size_t size);

template uint32_t*
GPUBufferManager::customCudaHostAlloc<uint32_t>(size_t size);

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

template int64_t**
GPUBufferManager::customCudaHostAlloc<int64_t*>(size_t size);

template uint64_t**
GPUBufferManager::customCudaHostAlloc<uint64_t*>(size_t size);

template uint8_t**
GPUBufferManager::customCudaHostAlloc<uint8_t*>(size_t size);

template uint32_t**
GPUBufferManager::customCudaHostAlloc<uint32_t*>(size_t size);

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
    SIRIUS_LOG_INFO("Initializing GPU buffer manager with params: Use Pin - {}, GPU Cache Size - {}, GPU Processing Size - {}, CPU Processing Size - {}", 
        Config::USE_PIN_MEM_FOR_CPU_PROCESSING, cache_size_per_gpu, processing_size_per_gpu, processing_size_per_cpu);
    gpuCache = new uint8_t*[NUM_GPUS];
    cpuCache = new uint8_t*[NUM_GPUS];
    gpuProcessing = new uint8_t*[NUM_GPUS];
    cpuProcessing = Config::USE_PIN_MEM_FOR_CPU_PROCESSING
        ? allocatePinnedCPUMemory(processing_size_per_cpu) : allocatePageableCPUMemory(processing_size_per_cpu);
    gpuProcessingPointer = new size_t[NUM_GPUS];
    gpuCachingPointer = new size_t[NUM_GPUS];
    cpuCachingPointer = new size_t[NUM_GPUS];
    cpuProcessingPointer = 0;
    available_gpu_cache_size.resize(NUM_GPUS);

    cuda_mr = new rmm::mr::cuda_memory_resource();
    mr = new rmm::mr::pool_memory_resource(cuda_mr, processing_size_per_gpu, processing_size_per_cpu);
    cudf::set_current_device_resource(mr);
    allocation_table.resize(NUM_GPUS);
    locked_allocation_table.resize(NUM_GPUS);
    SIRIUS_LOG_INFO("Allocated processing size {} in GPU 0", processing_size_per_gpu);

    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        // We cannot allocate exactly all free memory using `cudaMalloc()`
        size_t free_gpu_mem_size = getFreeGPUMemorySize(gpu) * 0.99;
        if (free_gpu_mem_size >= cache_size_per_gpu) {
            gpuCache[gpu] = callCudaMalloc<uint8_t>(cache_size_per_gpu, gpu);
            cpuCache[gpu] = nullptr;
            available_gpu_cache_size[gpu] = cache_size_per_gpu;
            SIRIUS_LOG_INFO("Allocated cache size {} in GPU 0", cache_size_per_gpu);
        } else {
            gpuCache[gpu] = callCudaMalloc<uint8_t>(free_gpu_mem_size, gpu);
            cpuCache[gpu] = allocatePinnedCPUMemory(cache_size_per_gpu - free_gpu_mem_size);
            available_gpu_cache_size[gpu] = free_gpu_mem_size;
            SIRIUS_LOG_INFO("Allocated cache size {} for GPU 0 ({} in GPU, {} in CPU)",
                cache_size_per_gpu, free_gpu_mem_size, cache_size_per_gpu - free_gpu_mem_size);
        }

        // gpuProcessing[gpu] = callCudaMalloc<uint8_t>(processing_size_per_gpu, gpu);
        // gpuCache[gpu] = callCudaHostAlloc<uint8_t>(cache_size_per_gpu, 1);
        // gpuProcessing[gpu] = callCudaHostAlloc<uint8_t>(processing_size_per_gpu, 1);
        gpuProcessingPointer[gpu] = 0;
        gpuCachingPointer[gpu] = 0;
        cpuCachingPointer[gpu] = 0;
    }

    warmup_gpu();
}

GPUBufferManager::~GPUBufferManager() {
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        callCudaFree<uint8_t>(gpuCache[gpu], gpu);
        if (cpuCache[gpu] != nullptr) {
            freePinnedCPUMemory(cpuCache[gpu]);
        }
        // callCudaFree<uint8_t>(gpuProcessing[gpu], gpu);
        mr->deallocate((void*) gpuProcessing[gpu], processing_size_per_gpu);
    }
    Config::USE_PIN_MEM_FOR_CPU_PROCESSING
        ? freePinnedCPUMemory(cpuProcessing) : freePageableCPUMemory(cpuProcessing);
    delete[] gpuCache;
    delete[] cpuCache;
    delete[] gpuProcessing;
    delete[] gpuProcessingPointer;
    delete[] gpuCachingPointer;
    delete[] cpuCachingPointer;
    rmm_stored_buffers.clear();
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
        cpuCachingPointer[gpu] = 0;
    }
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
    //always ensure that it aligns with RMM's CUDA allocation alignment
    // int alignment = alignof(double);
    int alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    alloc = alloc + (alignment - alloc % alignment);
    if (caching) {
        size_t start = __atomic_fetch_add(&gpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
        T* ptr = nullptr;
        if (start + alloc <= available_gpu_cache_size[gpu]) {
            ptr = reinterpret_cast<T*>(gpuCache[gpu] + start);
        } else {
            __atomic_fetch_sub(&gpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
            start = __atomic_fetch_add(&cpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
            if (start + alloc > cache_size_per_gpu - available_gpu_cache_size[gpu]) {
                __atomic_fetch_sub(&cpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
                throw InvalidInputException("Out of caching memory");
            }
            ptr = reinterpret_cast<T*>(cpuCache[gpu] + start);
        }
        if (reinterpret_cast<uintptr_t>(ptr) % alignof(double) != 0) {
            throw InvalidInputException("Memory is not properly aligned");
        } 
        return ptr;
    } else {
        cudf::set_current_device_resource(mr);
        void* ptr = mr->allocate(alloc);
        // SIRIUS_LOG_DEBUG("Allocating Pointer {} size {}", static_cast<void*>(ptr), alloc);
        if (ptr == nullptr) throw InvalidInputException("Pointer is nullptr");
        if (allocation_table[gpu].find(ptr) != allocation_table[gpu].end()) {
            throw InvalidInputException("Pointer already exists in allocation table");
        }
        allocation_table[gpu][ptr] = alloc;
        return reinterpret_cast<T*>(ptr);
    }
};

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
    if (ptr == nullptr) {
        return;
    }
    if (ptr >= gpuCache[gpu] && ptr < gpuCache[gpu] + available_gpu_cache_size[gpu]) {
        return;
    }
    if (cpuCache[gpu] != nullptr && ptr >= cpuCache[gpu] &&
        ptr < cpuCache[gpu] + cache_size_per_gpu - available_gpu_cache_size[gpu]) {
        return;
    }
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

void
GPUBufferManager::createTableAndColumnInGPU(Catalog& catalog, ClientContext& context, string table_name, string column_name) {
	SIRIUS_LOG_DEBUG("CreateTable and Column called for table {} and col {}", table_name, column_name);
    TableCatalogEntry &table = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA, table_name).Cast<TableCatalogEntry>();
    auto column_names = table.GetColumns().GetColumnNames();
    auto& constraints = table.GetConstraints();
    vector<size_t> unique_columns;
    
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
    SIRIUS_LOG_DEBUG("Crate Table called for table {} with {} cols", up_table_name, column_count);
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
    const auto& table = table_it->second;
    auto column_it = find(table->column_names.begin(), table->column_names.end(), up_column_name);
    if (column_it == table->column_names.end()) {
        return false;
    }
    return true;
}

void
GPUBufferManager::createColumn(string up_table_name, string up_column_name, GPUColumnType column_type, size_t column_id, vector<size_t> unique_columns) {
    shared_ptr<GPUIntermediateRelation> table = tables[up_table_name];
    table->column_names[column_id] = up_column_name;
    if (find(unique_columns.begin(), unique_columns.end(), column_id) != unique_columns.end()) {
        table->columns[column_id] = make_shared_ptr<GPUColumn>(0, column_type, nullptr, nullptr);
        table->columns[column_id]->is_unique = true;
    } else {
        table->columns[column_id] = make_shared_ptr<GPUColumn>(0, column_type, nullptr, nullptr);
        table->columns[column_id]->is_unique = false;
    }
}

}; // namespace duckdb
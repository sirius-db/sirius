#include "gpu_buffer_manager.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types.hpp"

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

template pointer_and_key*
GPUBufferManager::customCudaMalloc<pointer_and_key>(size_t size, int gpu, bool caching);

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

GPUBufferManager::GPUBufferManager(size_t cache_size_per_gpu, size_t processing_size_per_gpu, size_t processing_size_per_cpu) : 
    cache_size_per_gpu(cache_size_per_gpu), processing_size_per_gpu(processing_size_per_gpu), processing_size_per_cpu(processing_size_per_cpu) {
    printf("Initializing GPU buffer manager\n");
    gpuCache = new uint8_t*[NUM_GPUS];
    gpuProcessing = new uint8_t*[NUM_GPUS];
    cpuProcessing = new uint8_t[processing_size_per_cpu];
    gpuProcessingPointer = new size_t[NUM_GPUS];
    gpuCachingPointer = new size_t[NUM_GPUS];
    cpuProcessingPointer = 0;

    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        gpuCache[gpu] = callCudaMalloc<uint8_t>(cache_size_per_gpu, gpu);
        gpuProcessing[gpu] = callCudaMalloc<uint8_t>(processing_size_per_gpu, gpu);
        // if (reinterpret_cast<uintptr_t>(gpuCache[gpu]) % alignof(double) == 0) {
        //     printf("Memory is not properly aligned 1\n");
        // } else if (reinterpret_cast<uintptr_t>(gpuCache[gpu]) % alignof(int) == 0) {
        //     printf("Memory is not properly aligned 2\n");
        // } else if (reinterpret_cast<uintptr_t>(gpuCache[gpu]) % alignof(char) == 0) {
        //     printf("Memory is not properly aligned 3\n");
        // }
        gpuProcessingPointer[gpu] = 0;
        gpuCachingPointer[gpu] = 0;
    }
}

GPUBufferManager::~GPUBufferManager() {
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        callCudaFree<uint8_t>(gpuCache[gpu], gpu);
        callCudaFree<uint8_t>(gpuProcessing[gpu], gpu);
    }
    delete[] cpuProcessing;
    delete[] gpuProcessingPointer;
    delete[] gpuCachingPointer;
}

void GPUBufferManager::ResetBuffer() {
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        gpuProcessingPointer[gpu] = 0;
    }
    cpuProcessingPointer = 0;
    for (auto it = tables.begin(); it != tables.end(); it++) {
        GPUIntermediateRelation* table = it->second;
        for (int col = 0; col < table->columns.size(); col++) {
            if (table->columns[col] != nullptr) {
                table->columns[col]->row_ids = nullptr;
                table->columns[col]->row_id_count = 0;
            }
        }
    }
}

template <typename T>
T*
GPUBufferManager::customCudaMalloc(size_t size, int gpu, bool caching) {
	size_t alloc = (size * sizeof(T));
    //always ensure that it aligns with 8B
    alloc = alloc + (alignof(double) - alloc % alignof(double));
    if (caching) {
        size_t start = __atomic_fetch_add(&gpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
        assert((start + alloc) < cache_size_per_gpu);
        T* ptr = reinterpret_cast<T*>(gpuCache[gpu] + start);
        if (reinterpret_cast<uintptr_t>(ptr) % alignof(double) != 0) {
            throw InvalidInputException("Memory is not properly aligned");
        } 
        return ptr;
    } else {
        // printf("Allocating %d bytes\n", alloc);
        // printf("Current pointer %d\n", gpuProcessingPointer[gpu]);
        size_t start = __atomic_fetch_add(&gpuProcessingPointer[gpu], alloc, __ATOMIC_RELAXED);
        assert((start + alloc) < processing_size_per_gpu);
        // printf("Allocating %d bytes at %d\n", alloc, start);
        // printf("Current pointer %d\n", gpuProcessingPointer[gpu]);
        T* ptr = reinterpret_cast<T*>(gpuProcessing[gpu] + start);
        if (reinterpret_cast<uintptr_t>(ptr) % alignof(double) != 0) {
            throw InvalidInputException("Memory is not properly aligned");
        } 
        return ptr;
    }
};

template <typename T>
T*
GPUBufferManager::customCudaHostAlloc(size_t size) {
	size_t alloc = (size * sizeof(T));
	size_t start = __atomic_fetch_add(&cpuProcessingPointer, alloc, __ATOMIC_RELAXED);
	assert((start + alloc) < processing_size_per_cpu);
	return reinterpret_cast<T*>(cpuProcessing + start);
};

void 
GPUBufferManager::Print() {
    printf("I am inside GPU buffer manager\n");
}

DataWrapper GPUBufferManager::allocateStringChunk(Vector &input, size_t chunk_size) {
    DataWrapper result;
    result.type = ColumnType::VARCHAR;
    result.num_strings = chunk_size;

    // First iterate through and set the offsets
    result.offsets = customCudaHostAlloc<int>(result.num_strings + 1);
    int curr_offset = 0;
    for(int i = 0; i < result.num_strings; i++) {
        std::string curr_string = input.GetValue(i).ToString();
        result.offsets[i] = curr_offset;
        curr_offset += curr_string.length();
    }
    result.offsets[result.num_strings] = curr_offset;

    // Now do the same for the chars
    result.size = (size_t) curr_offset;
    int copy_offset = 0;
    result.data = customCudaHostAlloc<uint8_t>(curr_offset);
    for(int i = 0; i < result.num_strings; i++) {
        std::string curr_string = input.GetValue(i).ToString();
        int str_length = curr_string.length();
        memcpy(result.data + copy_offset, reinterpret_cast<uint8_t*>(curr_string.data()), str_length * sizeof(uint8_t));
        copy_offset += str_length;
    }
    
    result.is_string_data = true;
    return result;
}

DataWrapper GPUBufferManager::allocateChunk(DataChunk &input) {
    // Get the input vector
    size_t chunk_size = input.size();
	auto input_vector = input.data[0];
    input_vector.Flatten(chunk_size);

	LogicalType vector_type = input_vector.GetType();
    uint8_t* ptr = nullptr;
    ColumnType column_type;

    if(vector_type.id() == LogicalTypeId::VARCHAR) {
        return allocateStringChunk(input_vector, chunk_size);
    }
    
    switch (vector_type.id()) {
        case LogicalTypeId::INTEGER: {
            int* ptr_int = customCudaHostAlloc<int>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_int);
            memcpy(ptr, input_vector.GetData(), chunk_size * sizeof(int));
            column_type = ColumnType::INT32;
            break;
        }
        case LogicalTypeId::BIGINT: {
            uint64_t* ptr_int64 = customCudaHostAlloc<uint64_t>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_int64);
            memcpy(ptr, input_vector.GetData(), chunk_size * sizeof(uint64_t));
            column_type = ColumnType::INT64;
            break;
        }
        case LogicalTypeId::FLOAT: {
            float* ptr_float = customCudaHostAlloc<float>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_float);
            memcpy(ptr, input_vector.GetData(), chunk_size * sizeof(float));
            column_type = ColumnType::FLOAT32;
            break;
        }
        case LogicalTypeId::DOUBLE: {
            double* ptr_double = customCudaHostAlloc<double>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_double);
            memcpy(ptr, input_vector.GetData(), chunk_size * sizeof(double));
            column_type = ColumnType::FLOAT64;
            break;
        }
        default:
            throw InvalidInputException("Unsupported type");
    }

    return DataWrapper(column_type, ptr, chunk_size);
}

DataWrapper GPUBufferManager::mergeWrappers(DataWrapper first, DataWrapper second) {
    DataWrapper result;

    // If one of them has null data then just use the other
    if(first.data == nullptr) {
        result = second;
    } else if(second.data == nullptr) {
        result = first;
    } else {
        // First copy over the metatada
        assert(first.type == second.type && "Can only merge columns of the same size");
        assert(first.is_string_data == second.is_string_data && "Need both columns to be of type string");
        result.type = first.type; 
        result.is_string_data = first.is_string_data;

        // Now combine the data
        result.size = first.size + second.size;
        result.data = customCudaHostAlloc<uint8_t>(result.size * sizeof(uint8_t));
        memcpy(result.data, first.data, first.size * sizeof(uint8_t));
        memcpy(result.data + first.size, second.data, second.size * sizeof(uint8_t));
        free(first.data); free(second.data);

        // For string columns also need to combine offsets
        if(result.is_string_data) {
            // First increment the second offset
            int second_offset_increment = first.size;
            for(int i = 0; i < second.num_strings; i++) {
                second.offsets[i] += second_offset_increment;
            }

            // Now combine the offsets
            result.num_strings = first.num_strings + second.num_strings;
            result.offsets = customCudaHostAlloc<int>((result.num_strings + 1) * sizeof(int));
            memcpy(result.offsets, first.offsets, first.num_strings * sizeof(int));
            memcpy(result.offsets + first.num_strings, second.offsets, second.num_strings * sizeof(int));
            result.offsets[result.num_strings] = result.size;
            free(first.offsets); free(second.offsets);
        }
    }

    return result;
}

//TODO: We have to lock the CPU buffer before calling bufferChunkInCPU to ensure contiguous memory allocation
DataWrapper
GPUBufferManager::allocateColumnBufferInCPU(unique_ptr<MaterializedQueryResult> input) {
	auto input_chunk = input->Fetch();
	if (!input_chunk) {
		throw InvalidInputException("No data in input chunk");
	}

    DataWrapper result_wrapper = allocateChunk(*input_chunk);
    input_chunk = input->Fetch();
	while (input_chunk) {
        // Get the wrapper for this chunk and merge it
		auto wrapper = allocateChunk(*input_chunk);
        result_wrapper = mergeWrappers(result_wrapper, wrapper);        
		input_chunk = input->Fetch();
	}

    std::cout << "Returning result wrapper with " << result_wrapper.num_strings << " strings and " << result_wrapper.size << " chars" << std::endl;
    return result_wrapper;
}

DataWrapper GPUBufferManager::allocateStrColumnInGPU(DataWrapper cpu_data, int gpu) {
    // First copy the data
    std::cout << "CPU data called with " << cpu_data.size << " chars and " << cpu_data.num_strings << " strings" << std::endl;

    DataWrapper result;
    result.is_string_data = cpu_data.is_string_data;
    result.type = ColumnType::VARCHAR;

    // First allocate and copy the offsets buffer
    result.num_strings = cpu_data.num_strings;
    result.offsets = customCudaMalloc<int>(cpu_data.num_strings + 1, 0, true);
    std::cout << "Copying offsets with " << result.num_strings << " strings" << std::endl;
    callCudaMemcpyHostToDevice<int>(result.offsets, cpu_data.offsets, cpu_data.num_strings * sizeof(int), 0);

    // Do the same for the characeters
    result.size = cpu_data.size;
    result.data = customCudaMalloc<uint8_t>(cpu_data.size, 0, true);
    std::cout << "Copying sizes with " << result.size << " chars" << std::endl;
    callCudaMemcpyHostToDevice<uint8_t>(result.data, cpu_data.data, cpu_data.size * sizeof(uint8_t), 0);

    std::cout << "Returning wrapper of size " << result.num_strings << " and " << result.size << std::endl;
    return result;
}

DataWrapper GPUBufferManager::allocateColumnBufferInGPU(DataWrapper cpu_data, int gpu) {
    if(cpu_data.is_string_data) {
        std::cout << "Calling allocateStrColumnInGPU" << std::endl;
        return allocateStrColumnInGPU(cpu_data, gpu);
    }

    uint8_t* ptr = nullptr;
    ColumnType column_type;

	switch (cpu_data.type) {
		case ColumnType::INT32: {
            int* ptr_int = customCudaMalloc<int>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_int);
            column_type = ColumnType::INT32;
			break;
        }
		case ColumnType::INT64: {
            uint64_t* ptr_int64 = customCudaMalloc<uint64_t>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_int64);
            column_type = ColumnType::INT64;
			break;
        }
		case ColumnType::FLOAT32: {
            float* ptr_float = customCudaMalloc<float>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_float);
            column_type = ColumnType::INT64;
			break;
        }
		case ColumnType::FLOAT64: {
            double* ptr_double = customCudaMalloc<double>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_double);
            column_type = ColumnType::FLOAT64;
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
    if(!gpu_allocated_buffer.is_string_data) {
        tables[up_table_name]->columns[column_idx]->column_length = gpu_allocated_buffer.size;
    } else {
        tables[up_table_name]->columns[column_idx]->column_length = gpu_allocated_buffer.num_strings;
    }
    std::cout << "cacheDataInGPU set column length of " << tables[up_table_name]->columns[column_idx]->column_length << std::endl;
    
    tables[up_table_name]->length = gpu_allocated_buffer.size;

    char* curr_data = reinterpret_cast<char*>(cpu_data.data);
    
}

void
GPUBufferManager::createTableAndColumnInGPU(Catalog& catalog, ClientContext& context, string table_name, string column_name) {
    std::cout << "Create Table created with table " << table_name << " and col name " << column_name << std::endl;

	TableCatalogEntry &table = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA, table_name).Cast<TableCatalogEntry>();
    auto column_names = table.GetColumns().GetColumnNames();
    for (int i = 0; i < column_names.size(); i++) {
        printf("GPU Caching Existing Column names: %s\n", column_names[i].c_str());
    }
    std::string upper_col_name = column_name;
    transform(upper_col_name.begin(), upper_col_name.end(), upper_col_name.begin(), ::toupper);

    //finding column_name in column_names
    if (find(column_names.begin(), column_names.end(), upper_col_name) != column_names.end()) {
        // convert table_name to uppercase
        size_t column_id = table.GetColumnIndex(upper_col_name, false).index;
        string up_table_name = table_name;
        transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
        createTable(up_table_name, table.GetTypes().size());

        // printf("logical type %d %d %s\n", column_id, table.GetTypes()[column_id].id(), table.GetColumn(column_name).GetName().c_str());
        ColumnType column_type = convertLogicalTypetoColumnType(table.GetColumn(column_name).GetType());

        // convert table_name to uppercase
        createColumn(up_table_name, upper_col_name, column_type, column_id);
    } else {
        throw InvalidInputException("createTableAndColumnInGPU Couldn't find column " + column_name);
    }
}

void
GPUBufferManager::createTable(string up_table_name, size_t column_count) {
    //we will update the length later
    //check if table already exists
    if (tables.find(up_table_name) == tables.end()) {
        GPUIntermediateRelation* table = new GPUIntermediateRelation(column_count);
        table->names = up_table_name;
        tables[up_table_name] = table;
        std::cout << "Inserted table " << up_table_name << " into buffer manager " << std::endl;
    }
}

void
GPUBufferManager::createColumn(string up_table_name, string up_column_name, ColumnType column_type, size_t column_id) {
    GPUIntermediateRelation* table = tables[up_table_name];
    table->column_names[column_id] = up_column_name;
    table->columns[column_id] = new GPUColumn(up_column_name, 0, column_type, nullptr);
    std::cout << "Added column " << up_column_name << " to table " << up_table_name << std::endl;
}

}; // namespace duckdb
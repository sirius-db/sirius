#include "gpu_buffer_manager.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "utils.hpp"

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
        if (start + alloc >= cache_size_per_gpu) {
            throw InvalidInputException("Out of GPU caching memory");
        }
        T* ptr = reinterpret_cast<T*>(gpuCache[gpu] + start);
        if (reinterpret_cast<uintptr_t>(ptr) % alignof(double) != 0) {
            throw InvalidInputException("Memory is not properly aligned");
        } 
        return ptr;
    } else {
        // printf("Allocating %ld bytes\n", alloc);
        size_t start = __atomic_fetch_add(&gpuProcessingPointer[gpu], alloc, __ATOMIC_RELAXED);
        // printf("Current pointer %ld\n", gpuProcessingPointer[gpu]);
        assert((start + alloc) < processing_size_per_gpu);
        if (start + alloc >= processing_size_per_gpu) {
            throw InvalidInputException("Out of GPU processing memory");
        }
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
    if (start + alloc >= processing_size_per_cpu) {
        throw InvalidInputException("Out of CPU memory");
    }
	return reinterpret_cast<T*>(cpuProcessing + start);
};

void 
GPUBufferManager::Print() {
    printf("I am inside GPU buffer manager\n");
}

DataWrapper GPUBufferManager::allocateStringChunk(DataChunk &input_chunk, size_t row_count, DataWrapper &prev_data) {
	Vector input = input_chunk.data[0];
    size_t chunk_size = input_chunk.size();
    input.Flatten(chunk_size);
    LogicalType vector_type = input.GetType();
    if(vector_type.id() != LogicalTypeId::VARCHAR) {
        throw InvalidInputException("Wrong type");
    }

    DataWrapper result;
    result.type = ColumnType::VARCHAR;
    // printf("chunk size %ld\n", chunk_size);
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
    // printf("Num bytes %d\n", result.num_bytes);
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
    ColumnType column_type;

    //the allocation below is assuming its contiguous with prev_data
    switch (vector_type.id()) {
        case LogicalTypeId::INTEGER: {
            int* ptr_int = customCudaHostAlloc<int>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_int);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(int));
            column_type = ColumnType::INT32;
            break;
        }
        case LogicalTypeId::BIGINT: {
            uint64_t* ptr_int64 = customCudaHostAlloc<uint64_t>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_int64);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(uint64_t));
            column_type = ColumnType::INT64;
            break;
        }
        case LogicalTypeId::FLOAT: {
            float* ptr_float = customCudaHostAlloc<float>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_float);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(float));
            column_type = ColumnType::FLOAT32;
            break;
        }
        case LogicalTypeId::DOUBLE: {
            double* ptr_double = customCudaHostAlloc<double>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_double);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(double));
            column_type = ColumnType::FLOAT64;
            break;
        }
        case LogicalTypeId::BOOLEAN: {
            uint8_t* ptr_bool = customCudaHostAlloc<uint8_t>(chunk_size);
            ptr = reinterpret_cast<uint8_t*>(ptr_bool);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(uint8_t));
            column_type = ColumnType::BOOLEAN;
            break;
        }
        case LogicalTypeId::VARCHAR: {
            char* ptr_varchar = customCudaHostAlloc<char>(chunk_size * 128);
            ptr = reinterpret_cast<uint8_t*>(ptr_varchar);
            memcpy(ptr, input.data[0].GetData(), input.size() * sizeof(double));
            column_type = ColumnType::VARCHAR;
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
    printf("Row count %d\n", row_count);
	auto input_chunk = input->Fetch();
	if (!input_chunk) {
		throw InvalidInputException("No data in input chunk");
	}

    DataWrapper result_wrapper(ColumnType::INT32, nullptr, 0);
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
    printf("Done allocating column buffer in CPU\n");
    return result_wrapper;
}

DataWrapper GPUBufferManager::allocateStrColumnInGPU(DataWrapper cpu_data, int gpu) {
    // First copy the data
    std::cout << "CPU data called with " << cpu_data.num_bytes << " chars and " << cpu_data.size << " strings" << std::endl;

    DataWrapper result;
    result.is_string_data = cpu_data.is_string_data;
    result.type = ColumnType::VARCHAR;

    // First allocate and copy the offset buffer
    result.size = cpu_data.size;
    result.offset = customCudaMalloc<uint64_t>((cpu_data.size + 1), 0, true);
    std::cout << "Copying offset with " << result.size << " strings" << std::endl;
    callCudaMemcpyHostToDevice<uint64_t>(result.offset, cpu_data.offset, (cpu_data.size + 1), 0);
    //     std::string output_str(cpu_data.data + cpu_data.offset[i], cpu_data.data + cpu_data.offset[i + 1]);
    //     Value output_value(output_str);
    //     std::cout << "Recording value " << output_value.ToString() << " for idx " << i << std::endl;
    // }

    // Do the same for the characeters
    result.num_bytes = cpu_data.num_bytes;
    result.data = customCudaMalloc<uint8_t>(cpu_data.num_bytes, 0, true);
    std::cout << "Copying sizes with " << result.num_bytes << " chars" << std::endl;
    callCudaMemcpyHostToDevice<uint8_t>(result.data, cpu_data.data, cpu_data.num_bytes, 0);

    std::cout << "Returning wrapper of size " << result.size << " and " << result.num_bytes << std::endl;
    return result;
}


DataWrapper
GPUBufferManager::allocateColumnBufferInGPU(DataWrapper cpu_data, int gpu) {
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
            column_type = ColumnType::FLOAT32;
			break;
        }
		case ColumnType::FLOAT64: {
            double* ptr_double = customCudaMalloc<double>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_double);
            column_type = ColumnType::FLOAT64;
			break;
        }
		case ColumnType::BOOLEAN: {
            uint8_t* ptr_bool = customCudaMalloc<uint8_t>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_bool);
            column_type = ColumnType::BOOLEAN;
			break;
        }
		case ColumnType::VARCHAR: {
            char* ptr_char = customCudaMalloc<char>(cpu_data.size, 0, true);
            ptr = reinterpret_cast<uint8_t*>(ptr_char);
            column_type = ColumnType::VARCHAR;
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
    // for (int i = 0; i < tables[table_name]->column_names.size(); i++) {
    //     printf("Column name: %s\n", tables[table_name]->column_names[i].c_str());
    // }
    if (column_it == tables[up_table_name]->column_names.end()) {
        throw InvalidInputException("Column not found");
    }
    DataWrapper gpu_allocated_buffer = allocateColumnBufferInGPU(cpu_data, gpu);
    // callCudaMemcpyHostToDevice<uint8_t>(gpu_allocated_buffer.data, cpu_data.data, cpu_data.size * cpu_data.getColumnTypeSize(), 0);
    if(!gpu_allocated_buffer.is_string_data) {
        callCudaMemcpyHostToDevice<uint8_t>(gpu_allocated_buffer.data, cpu_data.data, cpu_data.size * cpu_data.getColumnTypeSize(), 0);
    } 
    // else {
    //     callCudaMemcpyHostToDevice<uint8_t>(gpu_allocated_buffer.data, cpu_data.data, cpu_data.num_bytes * sizeof(uint8_t), 0);
    // }
    int column_idx = column_it - tables[up_table_name]->column_names.begin(); 
    tables[up_table_name]->columns[column_idx]->data_wrapper = gpu_allocated_buffer;
    tables[up_table_name]->columns[column_idx]->column_length = gpu_allocated_buffer.size;
    tables[up_table_name]->length = gpu_allocated_buffer.size;
    printf("Data cached in GPU\n");
}

void
GPUBufferManager::createTableAndColumnInGPU(Catalog& catalog, ClientContext& context, string table_name, string column_name) {
	TableCatalogEntry &table = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA, table_name).Cast<TableCatalogEntry>();
    auto column_names = table.GetColumns().GetColumnNames();
    std::unordered_set<std::string> up_column_name_set;
    for (auto name: column_names) {
        transform(name.begin(), name.end(), name.begin(), ::toupper);
        up_column_name_set.emplace(name);
    }
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
                printf("Unique constraint on index %d\n", pk.GetIndex().index);
                for (auto &col : pk.GetColumnNames()) {
                    printf("Unique constraint on column %s\n", col.c_str());
                }
                unique_columns.push_back(pk.GetIndex().index);
			} else {
                for (auto &col : pk.GetColumnNames()) {
                    printf("Unique constraint on column %s\n", col.c_str());
                }
            }
		}
	}
    
    // convert table_name to uppercase
    string up_table_name = table_name;
    transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
    // convert column_name to uppercase
    string up_column_name = column_name;
    transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
    // finding up_column_name in up_column_name_set
    if (up_column_name_set.find(up_column_name) != up_column_name_set.end()) {
        // GetColumnIndex() will modify `up_column_name` so make a copy
        auto up_column_name_copy = up_column_name;
        size_t column_id = table.GetColumnIndex(up_column_name_copy, false).index;
        createTable(up_table_name, table.GetTypes().size());
        // printf("logical type %d %d %s\n", column_id, table.GetTypes()[column_id].id(), table.GetColumn(column_name).GetName().c_str());
        ColumnType column_type = convertLogicalTypetoColumnType(table.GetColumn(up_column_name).GetType());
        createColumn(up_table_name, up_column_name, column_type, column_id, unique_columns);
    } else {
        throw InvalidInputException("Column '" + up_column_name + "' does not exist in table '" + up_table_name + "'");
    }
    printf("Table and column created in GPU\n");
}

void
GPUBufferManager::createTable(string up_table_name, size_t column_count) {
    //we will update the length later
    //check if table already exists
    if (tables.find(up_table_name) == tables.end()) {
        GPUIntermediateRelation* table = new GPUIntermediateRelation(column_count);
        table->names = up_table_name;
        tables[up_table_name] = table;
    }
}

void
GPUBufferManager::createColumn(string up_table_name, string up_column_name, ColumnType column_type, size_t column_id, vector<size_t> unique_columns) {
    GPUIntermediateRelation* table = tables[up_table_name];
    table->column_names[column_id] = up_column_name;
    if (find(unique_columns.begin(), unique_columns.end(), column_id) != unique_columns.end()) {
        table->columns[column_id] = new GPUColumn(up_column_name, 0, column_type, nullptr);
        table->columns[column_id]->is_unique = true;
    } else {
        table->columns[column_id] = new GPUColumn(up_column_name, 0, column_type, nullptr);
        table->columns[column_id]->is_unique = false;
    }
}

}; // namespace duckdb
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"
#include "cpu_cache.hpp"
#include "test_utils.hpp"

#include <memory>

using namespace std;

namespace duckdb { 

TEST_CASE("Test caching simple integer column", "[cpu_cache]") {
    // Initialize the buffer manager
    size_t num_records = 1024;
    size_t column_bytes = num_records * sizeof(int32_t);
    size_t memory_buffer_sizes = 2 * column_bytes;
    GPUBufferManager::GetInstance(memory_buffer_sizes, memory_buffer_sizes, memory_buffer_sizes);

    // Create a cpu cache with enough pinned memory and one stream
    MallocCPUCache cpu_cache(memory_buffer_sizes, 1);
    
    // Now create a GPU column representing a single integer column
    int32_t* column_data = callCudaMalloc<int32_t>(num_records, 0);
    shared_ptr<GPUColumn> column = make_shared_ptr<GPUColumn>(num_records, GPUColumnType(GPUColumnTypeId::INT32), (uint8_t*) column_data, nullptr);
    shared_ptr<GPUIntermediateRelation> relationship = make_shared_ptr<GPUIntermediateRelation>(1);
    relationship->columns[0] = column;

    // Now cache the column to CPU
    uint32_t chunk_id = cpu_cache.moveDataToCPU(relationship);
    REQUIRE(chunk_id == 0);

    // Now load the column back from the CPU cache to the GPU
    shared_ptr<GPUIntermediateRelation> loaded_relationship = cpu_cache.moveDataToGPU(chunk_id, true);
    REQUIRE(loaded_relationship->columns.size() == 1);
    shared_ptr<GPUColumn> loaded_column = loaded_relationship->columns[0];

    // Verify there were no cuda errors
    cudaCheckErrors("CUDA Errors in Caching Test");

    // Also verify that the loaded column and the original column are equal
    CheckGPUColumnsEquality(column, loaded_column);

    // Cleanup all allocated memory
    callCudaFree<int32_t>(column_data, 0);
}

}
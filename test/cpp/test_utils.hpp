#include "catch.hpp"
#include "gpu_columns.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

namespace duckdb { 

void cudaCheckErrors(const char *msg) {
    cudaError_t __err = cudaGetLastError();
    if (__err != cudaSuccess) {
        printf("Fatal error: %s (%s at %s:%d)\n",
                msg, cudaGetErrorString(__err),
                __FILE__, __LINE__);
        REQUIRE(1 == 2);
    }   
}

void CheckGPUBuffers(uint8_t* buffer_1, uint8_t* buffer_2, size_t num_bytes) {
    // If the first buffer is null then verify that the second one is null as well
    if(buffer_1 == nullptr) {
        REQUIRE(buffer_2 == nullptr);
        return;
    }

    // Allocate temporary host buffers to copy the data back
    uint8_t* host_buffer_1 = (uint8_t*) malloc(num_bytes);
    uint8_t* host_buffer_2 = (uint8_t*) malloc(num_bytes);

    // Copy the data back to the host
    cudaMemcpy(host_buffer_1, buffer_1, num_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_buffer_2, buffer_2, num_bytes, cudaMemcpyDeviceToHost);

    // Now compare the two buffers
    REQUIRE(memcmp(host_buffer_1, host_buffer_2, num_bytes) == 0);

    // Free the temporary host buffers
    free(host_buffer_1);
    free(host_buffer_2);
}

void CheckGPUColumnsEquality(shared_ptr<GPUColumn> col1, shared_ptr<GPUColumn> col2) { 
    // First verify that all of the metadata is the same
    REQUIRE(col1->column_length == col2->column_length);
    REQUIRE(col1->row_id_count == col2->row_id_count);
    REQUIRE(col1->is_unique == col2->is_unique);

    DataWrapper col1_data = col1->data_wrapper;
    DataWrapper col2_data = col2->data_wrapper;
    REQUIRE(col1_data.type.id() == col2_data.type.id());
    REQUIRE(col1_data.size == col2_data.size);
    REQUIRE(col1_data.num_bytes == col2_data.num_bytes);
    REQUIRE(col1_data.is_string_data == col2_data.is_string_data);
    REQUIRE(col1_data.mask_bytes == col2_data.mask_bytes);

    // Now verify all of the buffers are the same
    CheckGPUBuffers((uint8_t*) col1->row_ids, (uint8_t*) col2->row_ids, col1->row_id_count * sizeof(uint64_t));
    CheckGPUBuffers(col1_data.data, col2_data.data, col1_data.num_bytes);
    CheckGPUBuffers((uint8_t*) col1_data.validity_mask, (uint8_t*) col2_data.validity_mask, col1_data.mask_bytes);
    if(col1_data.is_string_data) {
        CheckGPUBuffers((uint8_t*) col1_data.offset, (uint8_t*) col2_data.offset, col1_data.size * sizeof(uint64_t));
    }
}

}
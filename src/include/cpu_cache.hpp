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

#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"

#include <atomic>
#include <vector>
#include <utility>
#include <unordered_map>
#include <shared_mutex>

using namespace std;

namespace duckdb {

class CPUCache {
public:
    // Copies over the columns in the relationship to CPU and returns an id for the saved chunk. Note 
    // that this method just copies the data to CPU memory and does not free the allocated GPU memory
    virtual uint32_t moveDataToCPU(shared_ptr<GPUIntermediateRelation> relationship) = 0;

    // Copies over the columns in the relationship onto the GPU based on the chunk id. The chunk id should
    // have been previously returned by moveDataToCPU. Note that the method only frees the allocated CPU memory
    // if evict_from_cpu is set to true
    virtual shared_ptr<GPUIntermediateRelation> moveDataToGPU(uint32_t chunk_id, bool evict_from_cpu) = 0;
};

class SegmentMetadata {
public:
    SegmentMetadata() = default;
    ~SegmentMetadata() = default;

    uint8_t* segment_start_ptr;
    size_t segment_size;
    int segment_id; 
    std::atomic<bool> occupied;
};

// In Memory Cache that first tries to cache GPU data in pinned memory and then falls back to pageable memory
// if pinned memory is full. It utilizes multiple streams to copy parallelize copying across columns
class MallocCPUCache : public CPUCache {
public:
    MallocCPUCache(size_t pinned_memory_size, size_t num_streams = 1);
    ~MallocCPUCache();

    uint32_t moveDataToCPU(shared_ptr<GPUIntermediateRelation> relationship) override;
    shared_ptr<GPUIntermediateRelation> moveDataToGPU(uint32_t chunk_id, bool evict_from_cpu) override;

private:
    // Helper method to get a buffer of the specified size from the pinned memory pool. Returns a pointer to the
    // buffer as well as the segment id associated with that buffer
    std::pair<uint8_t*, int> get_cache_buffer(size_t size);

    // Helper method to cache a column with data on the GPU into the CPU cache using the provided stream. 
    // It returns a GPUColumn object representing the cached column with data now on the CPU. Also sets the
    // event that can be used to track when the copy is complete
    shared_ptr<GPUColumn> cache_column_to_cpu(shared_ptr<GPUColumn> gpu_column, cudaStream_t& copy_stream, cudaEvent_t& copy_complete_event);

    /// Helper method to move a cached column from CPU back to GPU memory using the provided stream. If evict is specified
    /// then we also free the CPU memory associated with caching this column. ALso sets the event that can be used to track when the copy is complete
    shared_ptr<GPUColumn> move_cached_column_to_gpu(shared_ptr<GPUColumn> cpu_column, cudaStream_t& copy_stream, cudaEvent_t& copy_complete_event);

    uint8_t* pinned_memory_buffer; // Pointer to the pinned memory buffer
    size_t pinned_memory_capacity; // Total capacity of the pinned memory buffer
    std::atomic<int32_t> pinned_memory_offset; // Offset on the amount of pinned memory consumed so far
    size_t num_streams; // Number of streams to use for copying
    cudaStream_t* streams; // Pointer to streams that we can use
    std::atomic<uint32_t> next_chunk_id; // Atomic counter to generate unique chunk ids
    std::atomic<uint32_t> copy_stream_sequence; // Atomic counter to determine which stream to use for the next copy operation
    std::unordered_map<uint32_t, shared_ptr<GPUIntermediateRelation>> all_cached_relationships; // Map storing the cached chunks
    std::vector<shared_ptr<SegmentMetadata>> segments; // Vector storing the segments in the pinned memory pool
    std::shared_mutex segment_lock; // Mutex to protect access to the segments vector
};

} // namespace duckdb
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

#include "cpu_cache.hpp"

#include "duckdb/common/assert.hpp"

namespace duckdb {
    
MallocCPUCache::MallocCPUCache(size_t pinned_memory_size, size_t num_streams) : pinned_memory_capacity(pinned_memory_size), 
    pinned_memory_offset(0), num_streams(num_streams), next_chunk_id(0), copy_stream_sequence(0) {

    // Allocate the specified amount of pinned memory    
    pinned_memory_buffer = allocatePinnedCPUMemory(pinned_memory_capacity);
    if (pinned_memory_buffer == nullptr) {
        throw InternalException("Failed to allocate pinned memory for CPU cache");
    }

    // Create multiple streams that can be used to copy data
    streams = new cudaStream_t[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
}

MallocCPUCache::~MallocCPUCache() {
    // Free the pinned memory buffer
    if (pinned_memory_buffer) {
        cudaFreeHost(pinned_memory_buffer);
        pinned_memory_buffer = nullptr;
    }

    // Destroy the streams
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

std::pair<uint8_t*, int> MallocCPUCache::get_cache_buffer(size_t size) { 
    // First see if there is space in the pinned memory to create a new segment
    if(pinned_memory_offset.load(std::memory_order_relaxed) < pinned_memory_capacity) { 
        size_t current_offset = pinned_memory_offset.fetch_add(size, std::memory_order_relaxed);
        if(current_offset + size <= pinned_memory_capacity) {
            // Create a new occupied segment in the pinned memory pool
            shared_ptr<SegmentMetadata> segment = make_shared_ptr<SegmentMetadata>();
            segment->segment_start_ptr = pinned_memory_buffer + current_offset;
            segment->segment_size = size;
            segment->occupied.store(true, std::memory_order_relaxed);
            
            // Save this new segment to segments vector
            int segment_id = 0;
            {
                std::unique_lock lock(segment_lock);

                segment_id = segments.size();
                segment->segment_id = segment_id;
                segments.push_back(segment);
            }

            return std::make_pair(segment->segment_start_ptr, segment_id);
        } 
    }

    // If not find an empty segment large enough to hold the requested buffer
    bool unoccupied_segment_value = false;
    bool occupied_segment_value = true;
    {
        std::shared_lock lock(segment_lock);

        for(auto &segment : segments) { 
            // Check if the segment is free and large enough to hold the requested buffer
            if(segment->segment_size < size) {
                continue;
            }

            // See if this segment is already occupied
            if(segment->occupied.load(std::memory_order_relaxed)) {
                continue;
            }

            // If not try to occupy the segment using a CAS
            if(segment->occupied.compare_exchange_strong(unoccupied_segment_value, occupied_segment_value, std::memory_order_relaxed)) {
                return std::make_pair(segment->segment_start_ptr, segment->segment_id);
            }
        }
    }

    // If no such segement exists then malloc a page in pinned memory
    return std::make_pair(allocatePageableCPUMemory(size), -1);
}

shared_ptr<GPUColumn> MallocCPUCache::cache_column_to_cpu(shared_ptr<GPUColumn> gpu_column, cudaStream_t& copy_stream) { 
    // First determine the number of bytes needed to cache this column on the CPU
    shared_ptr<GPUColumn> cpu_cached_column = make_shared_ptr<GPUColumn>(gpu_column);
    size_t cpu_buffer_size = 0; 
    if(cpu_cached_column->row_ids != nullptr) {
        cpu_buffer_size += cpu_cached_column->row_id_count * sizeof(uint64_t);
    }
    if(cpu_cached_column->data_wrapper.validity_mask != nullptr) {
        cpu_buffer_size += cpu_cached_column->data_wrapper.mask_bytes;
    }
    if(cpu_cached_column->data_wrapper.is_string_data) {
        cpu_buffer_size += cpu_cached_column->data_wrapper.size * sizeof(uint64_t); // Add space to store string offset
    }
    cpu_buffer_size += cpu_cached_column->data_wrapper.num_bytes;

    // Allocate a buffer of the requested size from the buffer pool
    std::pair<uint8_t*, int> cpu_buffer_info = get_cache_buffer(cpu_buffer_size);
    uint8_t* cpu_store_buffer = cpu_buffer_info.first;
    cpu_cached_column->segment_start_ptr = cpu_store_buffer;
    cpu_cached_column->segment_id = cpu_buffer_info.second;

    // Now copy over the data from GPU to CPU asychronously using the provided stream
    if(cpu_cached_column->row_ids != nullptr) { 
        size_t row_id_bytes = cpu_cached_column->row_id_count * sizeof(uint64_t);
        cudaMemcpyAsync(cpu_store_buffer, (uint8_t*) cpu_cached_column->row_ids, row_id_bytes, cudaMemcpyDeviceToHost, copy_stream);
        cpu_cached_column->row_ids = reinterpret_cast<uint64_t*>(cpu_store_buffer);
        cpu_store_buffer += row_id_bytes;
    }

    if(cpu_cached_column->data_wrapper.validity_mask != nullptr) { 
        size_t mask_bytes = cpu_cached_column->data_wrapper.mask_bytes;
        cudaMemcpyAsync(cpu_store_buffer, (uint8_t*) cpu_cached_column->data_wrapper.validity_mask, mask_bytes, cudaMemcpyDeviceToHost, copy_stream);
        cpu_cached_column->data_wrapper.validity_mask = reinterpret_cast<cudf::bitmask_type*>(cpu_store_buffer);
        cpu_store_buffer += mask_bytes;
    }

    if(cpu_cached_column->data_wrapper.is_string_data) { 
        size_t offset_bytes = cpu_cached_column->data_wrapper.size * sizeof(uint64_t);
        cudaMemcpyAsync(cpu_store_buffer, (uint8_t*) cpu_cached_column->data_wrapper.offset, offset_bytes, cudaMemcpyDeviceToHost, copy_stream);
        cpu_cached_column->data_wrapper.offset = reinterpret_cast<uint64_t*>(cpu_store_buffer);
        cpu_store_buffer += offset_bytes;
    }

    size_t data_bytes = cpu_cached_column->data_wrapper.num_bytes;
    cudaMemcpyAsync(cpu_store_buffer, (uint8_t*) cpu_cached_column->data_wrapper.data, data_bytes, cudaMemcpyDeviceToHost, copy_stream);
    cpu_cached_column->data_wrapper.data = cpu_store_buffer;

    // Return the cached column
    return cpu_cached_column;
}

uint32_t MallocCPUCache::moveDataToCPU(shared_ptr<GPUIntermediateRelation> relationship) { 
    // Generate a unique id for this chunk
    uint32_t chunk_id = next_chunk_id.fetch_add(1);

    // For each column in the relationship, copy it over from gpu memory using a different stream for each column
    uint32_t num_columns = relationship->columns.size();
    uint32_t stream_sequence_num = copy_stream_sequence.fetch_add(num_columns);
    shared_ptr<GPUIntermediateRelation> cpu_rel = make_shared_ptr<GPUIntermediateRelation>(num_columns);
    #pragma unroll
    for(uint32_t i = 0; i < num_columns; i++) { 
        uint32_t column_stream_idx = (stream_sequence_num + i) % num_streams;
        cudaStream_t& col_copy_stream = streams[column_stream_idx];
        cpu_rel->columns[i] = cache_column_to_cpu(relationship->columns[i], col_copy_stream);
    }

    // Now synchronize all the streams to ensure that the copy is complete before we return
    #pragma unroll
    for (size_t i = 0; i < num_columns; i++) {
        uint32_t column_stream_idx = (stream_sequence_num + i) % num_streams;
        cudaStreamSynchronize(streams[column_stream_idx]);
    }

    // Save the cached relationship in the map and return the key to the caller
    all_cached_relationships[chunk_id] = cpu_rel;
    return chunk_id;
}

shared_ptr<GPUColumn> MallocCPUCache::move_cached_column_to_gpu(shared_ptr<GPUColumn> cpu_column, cudaStream_t& copy_stream, bool evict_from_cpu) { 
    // Create the gpu column by copying over all of the metadata
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    shared_ptr<GPUColumn> gpu_column = make_shared_ptr<GPUColumn>(cpu_column);
    gpu_column->segment_id = -1; 

    // First copy over the row ids if they exist
    if(gpu_column->row_ids != nullptr) {
        size_t row_id_bytes = gpu_column->row_id_count * sizeof(uint64_t);
        gpu_column->row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(gpu_column->row_id_count, 0, 0);
        cudaMemcpyAsync(gpu_column->row_ids, cpu_column->row_ids, row_id_bytes, cudaMemcpyHostToDevice, copy_stream);
    }

    // Now copy over the validity mask if it exists
    if(gpu_column->data_wrapper.validity_mask != nullptr) {
        size_t mask_bytes = gpu_column->data_wrapper.mask_bytes;
        gpu_column->data_wrapper.validity_mask = reinterpret_cast<cudf::bitmask_type*>(gpuBufferManager->customCudaMalloc<uint8_t>(mask_bytes, 0, 0));
        cudaMemcpyAsync(gpu_column->data_wrapper.validity_mask, cpu_column->data_wrapper.validity_mask, mask_bytes, cudaMemcpyHostToDevice, copy_stream);
    }

    // If it is a string column also copy over the offsets
    if(gpu_column->data_wrapper.is_string_data) {
        size_t offset_bytes = gpu_column->data_wrapper.size * sizeof(uint64_t);
        gpu_column->data_wrapper.offset = gpuBufferManager->customCudaMalloc<uint64_t>(gpu_column->data_wrapper.size, 0, 0);
        cudaMemcpyAsync(gpu_column->data_wrapper.offset, cpu_column->data_wrapper.offset, offset_bytes, cudaMemcpyHostToDevice, copy_stream);
    }

    // Finally copy over the actual data
    size_t data_bytes = cpu_column->data_wrapper.num_bytes;
    gpu_column->data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(data_bytes, 0, 0);
    cudaMemcpyAsync(gpu_column->data_wrapper.data, cpu_column->data_wrapper.data, data_bytes, cudaMemcpyHostToDevice, copy_stream);

    // If evict is specified then also clear the CPU memory associated with this column
    if(evict_from_cpu) {
        if(cpu_column->segment_id != -1) { 
            // If the column was cached in a segment in the pinned memory pool then mark that segment as free
            std::shared_lock lock(segment_lock);
            segments[cpu_column->segment_id]->occupied.store(false, std::memory_order_relaxed);
        } else {
            // Just free the pageable memory that was allocated for this segment
            freePageableCPUMemory(cpu_column->segment_start_ptr);
        }
    }

    return gpu_column;
}

shared_ptr<GPUIntermediateRelation> MallocCPUCache::moveDataToGPU(uint32_t chunk_id, bool evict_from_cpu) { 
    // First load the chunk from the map and if specified evict it from the map
    D_ASSERT(all_cached_relationships.find(chunk_id) != all_cached_relationships.end());
    shared_ptr<GPUIntermediateRelation> cpu_rel = all_cached_relationships[chunk_id];
    if(evict_from_cpu) { 
        all_cached_relationships.erase(chunk_id);
    }

    // For each column in the relationship, copy it over to gpu memory using a different stream for each column
    uint32_t num_columns = cpu_rel->columns.size();
    uint32_t stream_sequence_num = copy_stream_sequence.fetch_add(num_columns);
    shared_ptr<GPUIntermediateRelation> gpu_rel = make_shared_ptr<GPUIntermediateRelation>(num_columns);
    #pragma unroll
    for(uint32_t i = 0; i < num_columns; i++) { 
        uint32_t column_stream_idx = (stream_sequence_num + i) % num_streams;
        cudaStream_t& col_copy_stream = streams[column_stream_idx];
        gpu_rel->columns[i] = move_cached_column_to_gpu(cpu_rel->columns[i], col_copy_stream, evict_from_cpu);
    }

    // Now synchronize all the streams to ensure that the copy is complete before we return
    #pragma unroll
    for (size_t i = 0; i < num_columns; i++) {
        uint32_t column_stream_idx = (stream_sequence_num + i) % num_streams;
        cudaStreamSynchronize(streams[column_stream_idx]);
    }

    return gpu_rel;
}

}
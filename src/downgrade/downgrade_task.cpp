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

#include "downgrade/downgrade_task.hpp"
//#include "downgrade/downgrade_executor.hpp"
#include "memory/memory_reservation.hpp"
#include "data/cpu_data_representation.hpp"
#include "data/gpu_data_representation.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include <rmm/cuda_stream_view.hpp>
#include "cudf/contiguous_split.hpp"

namespace sirius {
namespace parallel {
namespace detail {


    sirius::memory::fixed_size_host_memory_resource::multiple_blocks_allocation copy_data_to_host(const rmm::device_buffer* gpu_data, 
        sirius::memory::fixed_size_host_memory_resource* mr,
        std::size_t& data_size,
        rmm::cuda_stream_view stream) {
       /* nvtx3::scoped_range range{"cudf_table_converter::copy_data_to_host"};

        data_size = gpu_data->size();

        sirius::memory::fixed_size_host_memory_resource::multiple_blocks_allocation allocation = mr->allocate_multiple_blocks(data_size);

        if (allocation.size() == 0) {
        return allocation;
        }

        nvtx3::scoped_range copy_range{"gpu_to_host_copy_loop"};
        std::size_t remaining_bytes = data_size;
        std::size_t block_index = 0;
        std::size_t block_offset = 0;
        const std::size_t block_size = mr->get_block_size();

        const uint8_t* gpu_data_ptr = static_cast<const uint8_t*>(gpu_data->data());

        while (remaining_bytes > 0) {
        std::size_t bytes_to_copy = std::min(remaining_bytes, block_size - block_offset);

        void* block_ptr = allocation.getBlock(block_index);
        void* dest_ptr = static_cast<char*>(block_ptr) + block_offset;

        std::size_t source_offset = data_size - remaining_bytes;

        cudaMemcpyAsync(dest_ptr, gpu_data_ptr + source_offset, bytes_to_copy, cudaMemcpyDeviceToHost, stream.value());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
        }

        remaining_bytes -= bytes_to_copy;
        block_offset += bytes_to_copy;

        if (block_offset >= block_size) {
        block_index++;
        block_offset = 0;
        }
        }

        stream.synchronize();

        return allocation;*/
    }

    sirius::unique_ptr<sirius::host_table_representation>
     convert_to_host_representation(const sirius::unique_ptr<gpu_table_representation>& table, 
                                          sirius::memory::fixed_size_host_memory_resource* mr,
                                          rmm::cuda_stream_view stream) {
     /*   nvtx3::scoped_range range{"cudf_table_converter::convert_to_host", nvtx3::rgb{0, 0, 255}};
    
        if (table->get_table().num_columns() == 0) {
            auto empty_allocation = mr->allocate_multiple_blocks(0ull);
            auto empty_metadata = sirius::make_unique<sirius::vector<uint8_t>>();
            return sirius::make_unique<host_table_representation>(std::move(empty_allocation), std::move(empty_metadata), 0);
        }
    
        nvtx3::scoped_range pack_range{"cudf::pack"};
        auto packed_data = cudf::pack(table->get_table(), stream);
    
        auto metadata = sirius::make_unique<sirius::vector<uint8_t>>(*packed_data.metadata);
    
        nvtx3::scoped_range copy_range{"copy_data_to_host"};
        std::size_t data_size = packed_data.gpu_data->size();
        try{
        auto allocation = mr->allocate_multiple_blocks(data_size);
        }catch(const rmm::out_of_memory& e){
            throw std::runtime_error("Failed to allocate memory for data: " + std::string(e.what()));
        }
        auto allocation = copy_data_to_host(packed_data.gpu_data.get(), mr, data_size, stream);
    
        return sirius::make_unique<host_table_representation>(std::move(allocation), std::move(metadata), data_size);
    }

    
*/
}
}
void downgrade_task::execute() {
    //TODO: store this in local state i think
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    //get the memory_space and check that its gpu
    auto memory_space = _local_state->cast<downgrade_task_local_state>()._batch->get_memory_space();
    if (memory_space->get_tier() != memory::Tier::GPU) {
        mark_task_completion();
        return;
    }else{
        auto& batch = _local_state->cast<downgrade_task_local_state>()._batch;
        auto data_size = batch->get_data()->get_size_in_bytes();

        try{
          

            auto& mr_manager = sirius::memory::memory_reservation_manager::get_instance();
            auto reservation = mr_manager.request_reservation(sirius::memory::any_memory_space_in_tier{sirius::memory::Tier::HOST}, data_size);
            if(!reservation){
                throw rmm::out_of_memory("Failed to allocate host memory for downgrade task.");
            }
            // Reservation identifies a memory_space (tier + device). Fetch its default allocator.
            auto mem_space = mr_manager.get_memory_space(reservation->tier, reservation->device_id);
            if (!mem_space) {
                throw std::runtime_error("Invalid reservation memory_space for HOST tier");
            }
            auto fixed_mr = mem_space->get_default_allocator_as<sirius::memory::fixed_size_host_memory_resource>();
            if (fixed_mr == nullptr) {
                throw std::runtime_error("Default HOST allocator is not fixed_size_host_memory_resource");
            }

            batch->convert_to_memory_space(mem_space, stream);

            mark_task_completion();
            return;

        }catch(const rmm::out_of_memory& e){
            throw std::runtime_error("Failed to allocate gpu_memory");
        }
        
        // Obtain HOST-tier memory resource from the memory manager
        // auto& mr_manager = sirius::memory::memory_reservation_manager::get_instance();
        // auto host_spaces = mr_manager.get_memory_spaces_for_tier(sirius::memory::Tier::HOST);
        // if (host_spaces.empty()) {
        //     mark_task_completion();
        //     return;
        // }
        // auto host_allocator_ref = host_spaces[0]->get_default_allocator();
        // auto* host_fixed_mr = dynamic_cast<sirius::memory::fixed_size_host_memory_resource*>(&host_allocator_ref.get());

        // // Fallback: if cast fails, complete task without conversion
        // if (host_fixed_mr == nullptr) {
        //     mark_task_completion();
        //     return;
        // }

        // // Use default CUDA stream for the conversion
        // rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  
        // auto host_table = detail::convert_to_host_representation(table, host_fixed_mr, stream);
        // _local_state->cast<downgrade_task_local_state>()._batch->set_data(std::move(host_table));
        // mark_task_completion();
        // return;
    }

    mark_task_completion();
}

void downgrade_task::mark_task_completion() {
    // notify task_creator about task completion
    uint64_t task_id = _local_state->cast<downgrade_task_local_state>()._task_id;
    uint64_t pipeline_id = _local_state->cast<downgrade_task_local_state>()._pipeline_id;
    auto message = sirius::make_unique<sirius::task_completion_message>();
    message->task_id = task_id;
    message->pipeline_id = pipeline_id;
    message->source = sirius::Source::PIPELINE;
    _global_state->cast<downgrade_task_global_state>()._message_queue.EnqueueMessage(std::move(message));
}

uint64_t downgrade_task::get_task_id() const {
    return _local_state->cast<downgrade_task_local_state>()._task_id;
}

} // namespace parallel
} // namespace sirius
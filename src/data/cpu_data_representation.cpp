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

#include "data/cpu_data_representation.hpp"

namespace sirius {

host_table_representation::host_table_representation(sirius::unique_ptr<sirius::memory::table_allocation> host_table, sirius::memory_space& memory_space)
    : idata_representation(memory_space), _host_table(std::move(host_table)) {}

std::size_t host_table_representation::get_size_in_bytes() const {
    return _host_table->data_size;
}

sirius::unique_ptr<idata_representation> host_table_representation::convert_to_memory_space(sirius::memory_space& target_memory_space, rmm::cuda_stream_view stream) {
    // TODO: Implement conversion to GPU representation
    // This should use data_representation_converter::convert_to_gpu_representation
    throw std::runtime_error("host_table_representation::convert_to_memory_space not yet implemented");
}

} // namespace sirius


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

#include "data/gpu_data_representation.hpp"
#include <cudf/utilities/traits.hpp>

namespace sirius {

GPUTableRepresentation::GPUTableRepresentation(cudf::table table)
    : table_(std::move(table)) {}

Tier GPUTableRepresentation::GetCurrentTier() const {
    return Tier::GPU;
}

std::size_t GPUTableRepresentation::GetSizeInBytes() const {
    // TODO: Implement proper size calculation
    // This should return the total size of all columns in the table
    std::size_t total_size = 0;
    for (auto const& col : table_.view()) {
        // For now, we can calculate a rough estimate based on column size
        // This will need to be refined to account for all buffers (data, validity, offsets, etc.)
        total_size += col.size() * cudf::size_of(col.type());
    }
    return total_size;
}

const cudf::table& GPUTableRepresentation::GetTable() const {
    return table_;
}

sirius::unique_ptr<IDataRepresentation> GPUTableRepresentation::ConvertToTier(Tier target_tier,
                                                         rmm::mr::device_memory_resource* mr,
                                                         rmm::cuda_stream_view stream) {
    // TODO: Implement conversion to GPU representation
    // This should use DataRepresentationConverter::ConvertToGPURepresentation
    throw std::runtime_error("GPUTableRepresentation::ConvertToTier not yet implemented");
}

} // namespace sirius


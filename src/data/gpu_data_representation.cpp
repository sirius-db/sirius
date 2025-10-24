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
#include "data/representation_converter.hpp"
#include "helper/helper.hpp"
#include "cudf/cudf_utils.hpp"
#include <cudf/contiguous_split.hpp>

namespace sirius {

std::size_t GPUTableRepresentation::GetSizeInBytes() const {
    // Use cuDF's pack to get accurate memory size
    auto packed = cudf::pack(table_);
    return packed.gpu_data->size();
}

sirius::unique_ptr<IDataRepresentation>
GPUTableRepresentation::ConvertToTier(Tier target_tier,
                                     FixedSizeHostMemoryResource* host_mr,
                                     rmm::mr::device_memory_resource* device_mr,
                                     rmm::cuda_stream_view stream) {
    if (target_tier == Tier::GPU) {
        // Already in GPU tier, return a move of this object
        return sirius::make_unique<GPUTableRepresentation>(std::move(table_));
    } else if (target_tier == Tier::HOST) {
        // Convert to host representation
        if (host_mr == nullptr) {
            throw std::runtime_error("Host memory resource required for conversion from GPU to HOST");
        }
        
        // Create a temporary unique_ptr to this object for conversion
        auto this_ptr = sirius::make_unique<GPUTableRepresentation>(std::move(table_));
        
        return DataRepresentationConverter::ConvertToHostRepresentation(
            this_ptr, host_mr, stream);
    } else {
        throw std::runtime_error("Unsupported target tier for conversion from GPU");
    }
}

} // namespace sirius
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
#include "data/representation_converter.hpp"
#include "helper/helper.hpp"
#include "cudf/cudf_utils.hpp"

namespace sirius {

sirius::unique_ptr<IDataRepresentation>
HostTableRepresentation::ConvertToTier(Tier target_tier, 
                                      FixedSizeHostMemoryResource* host_mr,
                                      rmm::mr::device_memory_resource* device_mr,
                                      rmm::cuda_stream_view stream) {
    if (target_tier == Tier::HOST) {
        // Already in host tier, return a copy
        return sirius::make_unique<HostTableRepresentation>(
            std::move(allocation_),
            std::move(metadata_),
            data_size_
        );
    } else if (target_tier == Tier::GPU) {
        // Convert to GPU representation
        if (device_mr == nullptr) {
            // Use default device memory resource if not provided
            device_mr = rmm::mr::get_current_device_resource();
        }
        
        // Create a temporary unique_ptr to this object for conversion
        auto this_ptr = sirius::make_unique<HostTableRepresentation>(
            std::move(allocation_),
            std::move(metadata_),
            data_size_
        );
        
        return DataRepresentationConverter::ConvertToGPURepresentation(
            this_ptr, device_mr, stream);
    } else {
        throw std::runtime_error("Unsupported target tier for conversion from HOST");
    }
}

} // namespace sirius
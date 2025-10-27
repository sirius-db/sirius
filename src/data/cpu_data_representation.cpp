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

HostTableRepresentation::HostTableRepresentation(sirius::unique_ptr<MultipleBlocksAllocation> allocation_blocks,
                                                 sirius::unique_ptr<sirius::vector<uint8_t>> meta,
                                                 std::size_t size)
    : allocation_(std::move(allocation_blocks)), metadata_(std::move(meta)), data_size_(size) {}

Tier HostTableRepresentation::GetCurrentTier() const {
    return Tier::HOST;
}

std::size_t HostTableRepresentation::GetSizeInBytes() const {
    return data_size_;
}

sirius::unique_ptr<IDataRepresentation> HostTableRepresentation::ConvertToTier(Tier target_tier,
                                                         rmm::mr::device_memory_resource* mr,
                                                         rmm::cuda_stream_view stream) {
    // TODO: Implement conversion to GPU representation
    // This should use DataRepresentationConverter::ConvertToGPURepresentation
    throw std::runtime_error("HostTableRepresentation::ConvertToTier not yet implemented");
}

} // namespace sirius


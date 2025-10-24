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

#include "data/representation_converter.hpp"
#include "helper/helper.hpp"
#include "cudf/cudf_utils.hpp"

namespace sirius {

sirius::unique_ptr<HostTableRepresentation>
DataRepresentationConverter::ConvertToHostRepresentation(const sirius::unique_ptr<GPUTableRepresentation>& table, FixedSizeHostMemoryResource* mr,
                               rmm::cuda_stream_view stream) {
}

sirius::unique_ptr<GPUTableRepresentation> 
DataRepresentationConverter::ConvertToGPURepresentation(const sirius::unique_ptr<HostTableRepresentation>& table, 
                              rmm::mr::device_memory_resource* mr, // TODO: Replace eventually with actual allocator type 
                              rmm::cuda_stream_view stream) {
}

} // namespace sirius
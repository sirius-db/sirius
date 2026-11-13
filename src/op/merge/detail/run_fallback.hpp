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

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius::op::merge::detail {

/**
 * @brief Whole-batch fallback for a run batch the windowed merge cannot read by row range.
 *
 * For a batch whose representation has no row-range reader (or whose read threw
 * row_range_unsupported_error): convert it ONCE to the GPU tier in `gpu_space` through the
 * converter registry, under the batch's exclusive lock, and leave it in its repository.
 *
 * @param batch The run batch to convert in place.
 * @param gpu_space The GPU memory space of the merge task.
 * @param stream CUDA stream used for the conversion.
 * @return true if a conversion happened, false if it was already GPU-resident in `gpu_space`.
 */
bool convert_run_batch_to_gpu(cucascade::data_batch& batch,
                              cucascade::memory::memory_space& gpu_space,
                              rmm::cuda_stream_view stream);

}  // namespace sirius::op::merge::detail

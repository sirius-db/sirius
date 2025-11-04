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

#include "data/data_batch_view.hpp"
#include "data/data_repository_manager.hpp"
#include "memory/memory_space.hpp"

namespace sirius {
namespace op {

/**
 * @brief Functionalities for mergeing multiple data batches into a single one.
 * 
 * Provide functionalities including:
 * - Concatenate multiple data batches;
 * - Merge aggregation over multiple data batches (presumebaly each input data batch is a local aggregation result);
 * - Merge sort over multiple sorted data batches.
 * 
 * Require caller to have already upgraded input data batches into `gpu_table_representation`
 * (the input data batch views are pinned).
 */
class gpu_merge {
public:
    /**
     * @brief Concatenate multiple data batches.
     * 
     * @param input The input batches to be concatenated.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * 
     * @return The concatenated data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> concat(
        const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
        rmm::cuda_stream_view stream,
        sirius::memory::memory_space& memory_space,
        sirius::data_repository_manager& data_repository_mgr);
};

} // namespace op
} // namespace sirius

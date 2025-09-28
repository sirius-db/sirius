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
#include "spilling/downgrade_executor.hpp"

namespace sirius {

class GPUMemoryManager {
public:
    GPUMemoryManager(DataRepository &data_repository, DowngradeExecutor &downgrade_executor);
    ~GPUMemoryManager();

    // scan data repository for data batches residing in GPU memory and schedule downgrade tasks
    void scanDowngradeTask();

private:
    DataRepository &data_repository_;
    DowngradeExecutor &downgrade_executor_;
};

} // namespace sirius
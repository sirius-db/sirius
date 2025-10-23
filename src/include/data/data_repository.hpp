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
#include "data_batch.hpp"
#include "helper/helper.hpp"

namespace sirius {

class IDataRepository {
public:
    virtual void AddNewDataBatch(sirius::unique_ptr<DataBatchView> data_batch) = 0;

    virtual sirius::unique_ptr<DataBatchView> EvictDataBatch() = 0;

    virtual std::vector<uint64_t> GetDowngradableDataBatches(size_t num_data_batches) = 0;
private:
    sirius::mutex mutex_;                                      ///< Mutex for thread-safe access to repository operations
    sirius::vector<sirius::unique_ptr<DataBatchView>> data_batches_;  ///< Map of pipeline source to DataBatchView
};

} // namespace sirius
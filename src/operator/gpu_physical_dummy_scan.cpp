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

#include "gpu_physical_dummy_scan.hpp"
#include "log/logging.hpp"

namespace duckdb {

SourceResultType 
GPUPhysicalDummyScan::GetData(GPUIntermediateRelation &output_relation) const {

    SIRIUS_LOG_DEBUG("Reading data from dummy scan");
    for (int col_idx = 0; col_idx < output_relation.columns.size(); col_idx++) {
        output_relation.columns[col_idx] = nullptr;
    }

	return SourceResultType::FINISHED;
}

} // namespace duckdb

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

#include "operator/gpu_physical_empty_result.hpp"
#include "log/logging.hpp"

namespace duckdb {

SourceResultType 
GPUPhysicalEmptyResult::GetData(GPUIntermediateRelation &output_relation) const {
    SIRIUS_LOG_DEBUG("Reading data from empty result");
    for (int col = 0; col < types.size(); col++) {
        output_relation.columns[col] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr);
    }
	return SourceResultType::FINISHED;
}

} // namespace duckdb

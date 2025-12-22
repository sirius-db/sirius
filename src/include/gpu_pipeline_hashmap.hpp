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
#include "gpu_pipeline.hpp"
#include "helper/helper.hpp"

namespace sirius {

class gpu_pipeline_hashmap {
public:
    gpu_pipeline_hashmap(duckdb::vector<duckdb::shared_ptr<duckdb::GPUPipeline>> vec) : _vec(std::move(vec)) {};
    ~gpu_pipeline_hashmap() = default;
    duckdb::vector<duckdb::shared_ptr<duckdb::GPUPipeline>> _vec;
};

} //namespace sirius
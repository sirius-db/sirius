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
#include "helper/helper.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {

/**
 * @brief Helper container that indexes GPU pipelines by their operators.
 *
 * sirius_pipeline_hashmap owns a collection of sirius_pipeline instances and builds
 * an unordered map from each sirius_physical_operator* (both intermediate operators
 * and the sink operator) to the sirius_pipeline that contains it. This allows
 * efficient lookup of the pipeline associated with a given GPU physical
 * operator.
 *
 * The constructor expects a vector of pipelines that are already fully
 * constructed. During initialization, all operators from every pipeline are
 * inserted into the internal hashmap. If the same operator pointer is found
 * in more than one pipeline, a std::runtime_error is thrown, enforcing the
 * invariant that each operator belongs to at most one pipeline.
 *
 * Typical usage is to construct this class once for a set of pipelines and
 * then use the exposed _map and _vec members elsewhere to resolve
 * sirius_physical_operator instances back to their owning sirius_pipeline.
 */
class sirius_pipeline_hashmap {
 public:
  sirius_pipeline_hashmap(duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> vec)
    : _vec(std::move(vec)) {};
  ~sirius_pipeline_hashmap() = default;
  duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> _vec;
};

}  // namespace sirius

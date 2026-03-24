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

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalUnion : public GPUPhysicalOperator {
 public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::UNION;

 public:
  GPUPhysicalUnion(vector<LogicalType> types,
                   unique_ptr<GPUPhysicalOperator> top,
                   unique_ptr<GPUPhysicalOperator> bottom,
                   idx_t estimated_cardinality);
  ~GPUPhysicalUnion() override;

 public:
  // Pipeline construction
  void BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline) override;

  vector<const_reference<GPUPhysicalOperator>> GetSources() const override;
};

}  // namespace duckdb

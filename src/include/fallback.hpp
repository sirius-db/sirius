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
#include "gpu_physical_projection.hpp"
#include "duckdb/common/exception.hpp"

#include <string>

namespace duckdb {

/**
 * Use to check if we need to fall back a query to duckdb execution.
 */
class FallbackChecker {
public:
  explicit FallbackChecker(const vector<shared_ptr<GPUPipeline>> pipelines_p)
    : pipelines(pipelines_p) {}
  
  void Check() const;

private:
  void CheckOperator(const GPUPhysicalOperator& op) const;
  void CheckOperator(const GPUPhysicalProjection& op) const;

  void CheckExpression(const Expression& expr) const;
  void CheckExpression(const BoundBetweenExpression& expr) const;
  void CheckExpression(const BoundCaseExpression& expr) const;
  void CheckExpression(const BoundCastExpression& expr) const;
  void CheckExpression(const BoundComparisonExpression& expr) const;
  void CheckExpression(const BoundConjunctionExpression& expr) const;
  void CheckExpression(const BoundFunctionExpression& expr) const;
  void CheckExpression(const BoundOperatorExpression& expr) const;

  vector<shared_ptr<GPUPipeline>> pipelines;
};

}

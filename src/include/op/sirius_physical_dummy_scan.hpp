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

#include "duckdb/execution/physical_operator.hpp"
#include "op/sirius_physical_operator.hpp"

namespace sirius {
namespace op {

class sirius_physical_dummy_scan : public sirius_physical_operator {
 public:
  static constexpr const duckdb::PhysicalOperatorType TYPE =
    duckdb::PhysicalOperatorType::DUMMY_SCAN;

 public:
  explicit sirius_physical_dummy_scan(duckdb::vector<duckdb::LogicalType> types,
                                      duckdb::idx_t estimated_cardinality)
    : sirius_physical_operator(
        duckdb::PhysicalOperatorType::DUMMY_SCAN, std::move(types), estimated_cardinality)
  {
  }

 public:
  bool is_source() const override { return true; }
};

}  // namespace op
}  // namespace sirius

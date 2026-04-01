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

#include "op/sirius_physical_cpu_source.hpp"

namespace sirius::op {

sirius_physical_cpu_source::sirius_physical_cpu_source(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::idx_t estimated_cardinality,
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CPU_SOURCE, std::move(types), estimated_cardinality),
    collection(std::move(collection)),
    produce_single_row(false)
{
}

sirius_physical_cpu_source::sirius_physical_cpu_source(duckdb::vector<duckdb::LogicalType> types,
                                                       duckdb::idx_t estimated_cardinality,
                                                       bool produce_single_row)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CPU_SOURCE, std::move(types), estimated_cardinality),
    collection(nullptr),
    produce_single_row(produce_single_row)
{
}

}  // namespace sirius::op
